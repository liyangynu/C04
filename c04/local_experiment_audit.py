"""Audit local final models and five-fold cleaning evidence without publishing assets."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from .check_expanded_predictions import select_records, verify_predictions
from .cleaning_variant_results import rows_sha256
from .common import file_sha256, read_json, read_jsonl, write_json
from .confident_learning import estimate_label_issues, make_folds
from .model import BiLSTMClassifier
from .runtime import choose_device, configure_runtime, load_official_split, make_loader, predict
from .text import build_vocab


def resolve_run(record):
    if record.get("run_dir"):
        return Path(record["run_dir"].replace("\\", "/"))
    comparison = Path(record["reused_from"].replace("\\", "/"))
    method = "cl_ce" if record["method"] == "confident_learning" else record["method"]
    original = read_json(Path("artifacts") / ("stage3_" + comparison.parent.name) / "comparison.json")
    row = next(r for r in original["records"] if r["method"] == method and r["seed"] == record["seed"])
    return Path(row["run_dir"].replace("\\", "/"))


def collect_runs(completion=Path("artifacts/completion_v1"), require_complete=True):
    records = []
    for name in ("idn", "sym"):
        for row in read_json(f"artifacts/stage3_formal_{name}10_v2/comparison.json")["records"]:
            records.append(("formal_" + name, row))
    noisy, clean = select_records(read_json("artifacts/stage5_expanded_runs/progress.json"))
    records += [("stage5", row) for row in [*noisy.values(), *clean.values()]]
    records += [("ablation", row) for row in read_json("artifacts/stage4_ablations_idn10_v2/ablation.json")["records"]]
    records += [("cleaning", row) for row in read_json("artifacts/stage4_cleaning_variants_idn10_v2/comparison.json")["records"]]
    if (completion / "progress.json").exists():
        records += [("completion_" + row["kind"], row) for row in read_json(completion / "progress.json")]
    unique = {}
    for group, row in records:
        run = resolve_run(row)
        ident = run.resolve()
        if ident not in unique:
            unique[ident] = {"run": run, "record": row, "group": group}
    if require_complete and (not (completion / "completed.json").exists() or len(unique) != 213):
        raise ValueError(f"Expected completed supplementary matrix and 213 final runs, found {len(unique)}")
    return list(unique.values())


def expected_config(item):
    row, group = item["record"], item["group"]
    method = row.get("method", "ce")
    if group == "ablation":
        config = read_json("configs/formal_gce.json")
        variant = next(v for v in read_json("configs/ablation_idn10_v2.json")["variants"] if v["name"] == row["variant"])
        for key, value in variant["changes"].items():
            section, name = key.split(".")
            config[section][name] = value
        return config
    config = copy.deepcopy(read_json("configs/formal_gce.json" if method == "gce" else "configs/formal_ce.json"))
    if group == "completion_q":
        config["training"]["q"] = row["q"]
    return config


def training_inputs(completion):
    paths = list(Path("data/processed/fewshot").rglob("*.jsonl"))
    paths += [Path(r["train_path"]) for r in read_json("data/stage5_expanded/matrix.json")["records"]]
    paths += [Path(r["train_path"]) for r in read_json(completion / "inputs/matrix.json")["records"]]
    paths += list(Path("artifacts").glob("stage3_formal_*/*/k10_s*/oof/cleaned_train.jsonl"))
    paths += list(Path("artifacts").glob("stage5_expanded_runs/*/*/k*_s*/oof/cleaned_train.jsonl"))
    paths += list(completion.glob("noise/*/*/seed_*/oof/cleaned_train.jsonl"))
    paths += list(Path("artifacts/stage4_cleaning_variants_idn10_v2").glob("*/k10_s*/input/train.jsonl"))
    # Some stage3 direct paths use the old layout; the stage5 matrix contains
    # byte-identical copies, so fingerprints intentionally join across locations.
    return {file_sha256(path): path for path in paths}


def audit_metadata(item, expected, labels, inputs):
    root, record = item["run"], item["record"]
    config = expected_config(item)
    run = read_json(root / "training/run_from_epoch_1.json")
    summary = read_json(root / "training/summary.json")
    metrics = read_json(root / "evaluation/metrics.json")
    values = verify_predictions(root / "evaluation", expected, labels, record)
    if run["config"] != config or run["seed"] != record["seed"]:
        raise ValueError(f"Configuration/seed differs: {root}")
    fingerprints = run["fingerprints"]
    if (summary["fingerprints"] != fingerprints or metrics["train_sha256"] != fingerprints["train"] or
            file_sha256(root / "training/best.pt") != metrics["checkpoint_sha256"] or
            fingerprints["validation"] != file_sha256("data/processed/clean/val.jsonl")):
        raise ValueError(f"Training/checkpoint fingerprint differs: {root}")
    if record.get("train_sha256", fingerprints["train"]) != fingerprints["train"]:
        raise ValueError(f"Record training hash differs: {root}")
    history = summary["history"]
    if not history or [h["epoch"] for h in history] != list(range(1, len(history) + 1)):
        raise ValueError(f"Epoch sequence differs: {root}")
    best = max(history, key=lambda h: h["val_macro_f1"])
    if (best["epoch"] != summary["best_epoch"] or best["epoch"] != metrics["selected_epoch"] or
            best["val_macro_f1"] != summary["best_val_macro_f1"] or metrics["history"] != history or
            len(history) != summary["epochs_completed"] or len(history) > config["training"]["max_epochs"]):
        raise ValueError(f"Validation checkpoint selection differs: {root}")
    vocab = read_json(root / "training/assets/vocab.json")
    train_path = inputs.get(fingerprints["train"])
    if train_path is None:
        raise ValueError(f"Cannot locate frozen training data: {root}")
    training = read_jsonl(train_path)
    if (file_sha256(root / "training/assets/vocab.json") != run["embedding"]["vocab_sha256"] or
            vocab != build_vocab(training, config["text"]["min_frequency"]) or run["train_rows"] != len(training)):
        raise ValueError(f"Vocabulary changed: {root}")
    if "sample_weights" in fingerprints:
        if file_sha256(root / "input/weights.json") != fingerprints["sample_weights"]:
            raise ValueError(f"Training weights changed: {root}")
    if (fingerprints["labels"] != file_sha256("data/processed/labels.json") or
            metrics["test_sha256"] != file_sha256("data/processed/clean/test.jsonl") or
            metrics["oos_test_sha256"] != file_sha256("data/processed/clean/oos_test.jsonl")):
        raise ValueError(f"Label/evaluation fingerprint differs: {root}")
    return {"run": root.as_posix(), "group": item["group"], "seed": record["seed"], **values,
            "checkpoint_sha256": metrics["checkpoint_sha256"], "training_git": run["git"],
            "training_environment": run["environment"], "vocab_size": len(vocab),
            "epoch_seconds": summary["total_epoch_seconds"], "epochs_completed": len(history),
            "peak_reserved_mb": max(h["peak_gpu_reserved_mb"] or 0 for h in history)}


def replay_final(item, rows, device_name):
    root = item["run"]
    metrics = read_json(root / "evaluation/metrics.json")
    checkpoint = torch.load(root / "training/best.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config != expected_config(item) or checkpoint["seed"] != item["record"]["seed"]:
        raise ValueError(f"Checkpoint config/seed differs: {root}")
    configure_runtime(checkpoint["seed"], config["training"])
    device = choose_device(device_name)
    model = BiLSTMClassifier(checkpoint["model"]["embedding.weight"],
        **{k: v for k, v in config["model"].items() if k != "embedding_dim"}).to(device)
    model.load_state_dict(checkpoint["model"])
    logits, _ = predict(model, make_loader(rows, checkpoint["vocab"], config["text"]["max_length"],
                        config["training"]["batch_size"]), device)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    scores = 1 - probabilities.max(axis=1)
    saved = read_jsonl(root / "evaluation/predictions.jsonl")
    error = float(np.max(np.abs(scores - [r["oos_score"] for r in saved])))
    if (file_sha256(root / "training/best.pt") != metrics["checkpoint_sha256"] or
            not np.array_equal(logits.argmax(axis=1), [r["predicted_intent"] for r in saved]) or error > 1e-7):
        raise ValueError(f"Weight inference differs: {root}, error={error}")
    return {"run": root.as_posix(), "rows": len(saved), "classes_equal": True,
            "max_msp_error": error, "checkpoint_sha256": metrics["checkpoint_sha256"]}


def audit_oof(root, replay=False, device_name="cuda"):
    report = read_json(root / "cleaning_report.json")
    if report["uses_clean_labels"] or report["uses_test_labels"] or not report["coverage_exactly_once"]:
        raise ValueError(f"OOF isolation statement differs: {root}")
    if file_sha256(root / "oof_probabilities.npz") != report["oof_sha256"]:
        raise ValueError(f"OOF probability hash differs: {root}")
    with np.load(root / "oof_probabilities.npz", allow_pickle=False) as data:
        ids, observed = data["ids"].tolist(), data["observed_labels"].tolist()
        probabilities, fold_ids = data["probabilities"], data["fold"]
    official, _ = load_official_split("data/processed/clean/train.jsonl", "train")
    by_id = {r["id"]: r for r in official}
    rows = [dict(by_id[sid], label=int(label)) for sid, label in zip(ids, observed)]
    if rows_sha256(rows) != report["input_sha256"] or len(set(ids)) != len(ids):
        raise ValueError(f"OOF observed input differs: {root}")
    regenerated_folds, generated_report = make_folds(rows, 150, 5, report["folds"]["seed"])
    if generated_report != report["folds"]:
        raise ValueError(f"OOF split generator differs: {root}")
    coverage, errors, seconds, peaks = [], [], [], []
    for fold, generated in zip(report["fold_records"], regenerated_folds):
        heldout, training = set(fold["heldout_ids"]), set(fold["training_ids"])
        if (heldout & training or heldout | training != set(ids) or
                heldout != {ids[i] for i in generated} or
                heldout != {sid for i, sid in enumerate(ids) if fold_ids[i] == fold["fold"] - 1}):
            raise ValueError(f"OOF fold overlap/assignment differs: {root}")
        coverage.extend(fold["heldout_ids"])
        folder = root / f"fold_{fold['fold']}"
        train_rows = [row for row in rows if row["id"] in training]
        if (read_jsonl(folder / "train.jsonl") != train_rows or rows_sha256(train_rows) != fold["train_sha256"] or
                read_json(folder / "training/assets/vocab.json") != build_vocab(train_rows) or
                file_sha256(folder / "training/best.pt") != fold["checkpoint_sha256"]):
            raise ValueError(f"OOF fold input/vocabulary/weight differs: {root}")
        summary = read_json(folder / "training/summary.json")
        seconds.append(summary["total_epoch_seconds"])
        peaks.append(max(h["peak_gpu_reserved_mb"] or 0 for h in summary["history"]))
        if replay:
            checkpoint = torch.load(folder / "training/best.pt", map_location="cpu", weights_only=True)
            config = checkpoint["config"]
            configure_runtime(checkpoint["seed"], config["training"])
            device = choose_device(device_name)
            model = BiLSTMClassifier(checkpoint["model"]["embedding.weight"],
                **{k: v for k, v in config["model"].items() if k != "embedding_dim"}).to(device)
            model.load_state_dict(checkpoint["model"])
            indices = [i for i, row in enumerate(rows) if row["id"] in heldout]
            logits, _ = predict(model, make_loader([rows[i] for i in indices], checkpoint["vocab"],
                                config["text"]["max_length"], config["training"]["batch_size"]), device)
            replayed = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
            error = float(np.max(np.abs(replayed - probabilities[indices])))
            if error > 1e-6:
                raise ValueError(f"OOF weight replay differs: {root}, fold={fold['fold']}, error={error}")
            errors.append(error)
    if sorted(coverage) != sorted(ids) or len(coverage) != len(set(coverage)):
        raise ValueError(f"OOF coverage is not exactly once: {root}")
    estimated = estimate_label_issues(rows, probabilities, 150)
    kept = [row for i, row in enumerate(rows) if estimated["keep_mask"][i]]
    if (read_jsonl(root / "cleaned_train.jsonl") != kept or rows_sha256(kept) != report["output_sha256"] or
            len(kept) != report["rows_kept"] or
            not np.array_equal(estimated["confident_joint"], report["confident_joint"])):
        raise ValueError(f"OOF cleaning cannot be replayed: {root}")
    return {"oof": root.as_posix(), "rows": len(rows), "kept": len(kept), "folds": 5,
            "weights_replayed": replay, "maximum_probability_error": max(errors, default=None),
            "epoch_seconds": sum(seconds), "peak_reserved_mb": max(peaks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", type=Path, default=Path("artifacts/completion_v1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/final_local_audit.json"))
    parser.add_argument("--weights", action="store_true")
    parser.add_argument("--oof-weights", action="store_true")
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    items = collect_runs(args.completion, not args.partial)
    test, labels = load_official_split("data/processed/clean/test.jsonl", "test")
    oos, _ = load_official_split("data/processed/clean/oos_test.jsonl", "oos_test")
    rows = test + oos
    expected = [(r["id"], r["label"]) for r in rows]
    inputs = training_inputs(args.completion)
    finals, replays = [], []
    for i, item in enumerate(items):
        finals.append(audit_metadata(item, expected, labels, inputs))
        if args.weights:
            replays.append(replay_final(item, rows, args.device))
        if (i + 1) % 10 == 0:
            print(f"Verified final runs {i+1}/{len(items)}", flush=True)
    oof = []
    for item in items:
        if (item["run"] / "oof/cleaning_report.json").exists():
            oof.append(audit_oof(item["run"] / "oof", args.oof_weights, args.device))
            print(f"Verified OOF routes {len(oof)}", flush=True)
    write_json(args.output, {"all_passed": True, "partial": args.partial,
               "final_model_count": len(finals), "prediction_rows": len(rows)*len(finals),
               "final_runs": finals, "weight_replays": replays, "oof": oof,
               "scope": "Local formal final models, ablations, cleaning variants, expanded and supplementary runs; exploratory pilots excluded."})
    print(f"PASS: {len(finals)} final runs; {len(replays)} final weight replays; {len(oof)} OOF routes", flush=True)


if __name__ == "__main__":
    main()
