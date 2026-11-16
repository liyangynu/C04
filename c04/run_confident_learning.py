"""Train five OOF models and produce an observed-label-only clean set."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .common import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from .confident_learning import estimate_label_issues, make_folds
from .model import BiLSTMClassifier
from .runtime import check_rows, choose_device, configure_runtime, git_state, make_loader, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--glove", type=Path, default=Path("data/embeddings/glove.6B.100d.txt"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    if config["training"].get("loss", "ce") != "ce":
        raise ValueError("OOF models use the fixed CE configuration")
    started = time.perf_counter()
    configure_runtime(args.seed, config["training"])
    classes = config["model"]["classes"]
    rows = read_jsonl(args.train)
    check_rows(rows, classes, "train")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output already contains files; use a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    folds, fold_report = make_folds(rows, classes, args.folds, args.seed)
    probabilities = np.full((len(rows), classes), np.nan, dtype=np.float64)
    fold_ids = np.full(len(rows), -1, dtype=np.int64)
    device = choose_device(args.device)
    common = [sys.executable, "-m", "c04.train", "--glove", str(args.glove),
              "--config", str(args.config), "--device", args.device]
    fold_records = []
    coverage = np.zeros(len(rows), dtype=np.int64)
    for fold_number, heldout in enumerate(folds):
        heldout_set = set(heldout.tolist())
        training_rows = [row for index, row in enumerate(rows) if index not in heldout_set]
        fold_dir = args.output/f"fold_{fold_number+1}"
        fold_dir.mkdir()
        fold_train = fold_dir/"train.jsonl"
        write_jsonl(fold_train, training_rows)
        fold_seed = (args.seed * 1009 + fold_number) % (2**31 - 1)
        command = common + ["--train", str(fold_train), "--seed", str(fold_seed),
                           "--output", str(fold_dir/"training")]
        log_path = fold_dir/"train.log"
        with log_path.open("w", encoding="utf-8", newline="\n") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            raise RuntimeError(f"OOF fold {fold_number+1} failed; see {log_path}")
        checkpoint_path = fold_dir/"training/best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = BiLSTMClassifier(checkpoint["model"]["embedding.weight"],
            **{k: v for k, v in checkpoint["config"]["model"].items() if k != "embedding_dim"}).to(device)
        model.load_state_dict(checkpoint["model"])
        heldout_rows = [rows[index] for index in heldout]
        if {row["id"] for row in training_rows} & {row["id"] for row in heldout_rows}:
            raise AssertionError("OOF train/heldout overlap")
        loader = make_loader(heldout_rows, checkpoint["vocab"], checkpoint["config"]["text"]["max_length"],
                             checkpoint["config"]["training"]["batch_size"])
        logits, _ = predict(model, loader, device)
        probabilities[heldout] = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        fold_ids[heldout] = fold_number
        coverage[heldout] += 1
        fold_summary = read_json(fold_dir/"training/summary.json")
        fold_records.append({"fold": fold_number + 1, "seed": fold_seed,
                             "heldout_rows": len(heldout), "train_rows": len(training_rows),
                             "train_sha256": file_sha256(fold_train),
                             "checkpoint_sha256": file_sha256(checkpoint_path),
                             "heldout_ids": [row["id"] for row in heldout_rows],
                             "training_ids": [row["id"] for row in training_rows],
                             "best_epoch": fold_summary["best_epoch"],
                             "total_epoch_seconds": fold_summary["total_epoch_seconds"]})
        print(f"fold={fold_number+1}/{args.folds}; heldout={len(heldout)}", flush=True)
        del model, checkpoint
    if not np.all(coverage == 1) or not np.isfinite(probabilities).all():
        raise AssertionError("OOF output is incomplete")
    result = estimate_label_issues(rows, probabilities, classes)
    clean_rows = [row for index, row in enumerate(rows) if result["keep_mask"][index]]
    np.savez_compressed(args.output/"oof_probabilities.npz", ids=np.array([r["id"] for r in rows]),
                        probabilities=probabilities, observed_labels=np.array([r["label"] for r in rows]),
                        fold=np.array(fold_ids))
    write_jsonl(args.output/"cleaned_train.jsonl", clean_rows)
    write_json(args.output/"cleaning_report.json", {
        "input_sha256": file_sha256(args.train), "output_sha256": file_sha256(args.output/"cleaned_train.jsonl"),
        "classes": classes, "rows": len(rows), "rows_kept": len(clean_rows),
        "rows_flagged": int(result["issue_mask"].sum()), "thresholds":
            [None if not np.isfinite(value) else float(value) for value in result["thresholds"]],
        "protected_observed_classes": result["protected_observed_classes"],
        "confident_joint": result["confident_joint"].tolist(), "folds": fold_report,
        "implementation": result["implementation"],
        "absent_observed_classes": result["absent_observed_classes"],
        "coverage_exactly_once": bool(np.all(coverage == 1)),
        "oof_sha256": file_sha256(args.output/"oof_probabilities.npz"),
        "wall_seconds": time.perf_counter()-started, "git": git_state(),
        "fold_records": fold_records, "config_sha256": file_sha256(args.config),
        "uses_clean_labels": False, "uses_test_labels": False})
    print(f"OOF complete: {len(rows)} rows; flagged={int(result['issue_mask'].sum())}; kept={len(clean_rows)}")


if __name__ == "__main__":
    main()
