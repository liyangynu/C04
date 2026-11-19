"""Recompute the expanded matrix from local predictions and earlier released runs.

This check requires completed local stage-5 runs. It does not train models or
replace checkpoint replay, OOF auditing, or a fresh dependency environment.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from .common import file_sha256, read_json, read_jsonl, write_json
from .metrics import binary_auroc, classification_metrics
from .paired_statistics import holm_adjust, summarize_pair
from .run_expanded_matrix import key
from .runtime import load_official_split
from .stage5_results import expected_conditions, summarize


METRICS = {"macro_f1": "test_macro_f1", "accuracy": "test_accuracy", "oos_auroc": "oos_auroc"}


def require_close(actual, expected, description):
    if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual - expected) > 1e-12:
        raise ValueError(f"{description} differs from predictions")


def verify_predictions(directory, expected, labels, record):
    rows = read_jsonl(directory / "predictions.jsonl")
    if [(r["id"], r["true_label"]) for r in rows] != expected:
        raise ValueError("Prediction IDs/truth differ from official test splits")
    truth = np.array([r["true_label"] for r in rows])
    predicted = np.array([r["predicted_intent"] for r in rows])
    scores = np.array([r["oos_score"] for r in rows], dtype=float)
    if (not np.issubdtype(predicted.dtype, np.integer) or
            np.any(predicted < 0) or np.any(predicted >= len(labels)) or
            not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError("Invalid predicted intents or OOS scores")
    is_id = truth >= 0
    result = classification_metrics(truth[is_id], predicted[is_id], len(labels))
    values = {"macro_f1": result["macro_f1"], "accuracy": result["accuracy"],
              "oos_auroc": binary_auroc((~is_id).astype(int), scores)}
    metrics = read_json(directory / "metrics.json")
    for metric, field in METRICS.items():
        require_close(values[metric], metrics[metric], f"Saved {metric}")
        require_close(values[metric], record[field], f"Progress {field}")
    with (directory / "confusion_matrix.csv").open(encoding="utf-8", newline="") as stream:
        table = list(csv.reader(stream))
    if (table[0] != ["true_id/predicted_id", *map(str, range(len(labels)))] or
            [r[0] for r in table[1:]] != list(map(str, range(len(labels)))) or
            not np.array_equal(np.array([r[1:] for r in table[1:]], dtype=int), result["confusion_matrix"])):
        raise ValueError("Confusion matrix differs from predictions")
    with (directory / "per_class.csv").open(encoding="utf-8", newline="") as stream:
        per_class = list(csv.DictReader(stream))
    if [int(r["class_id"]) for r in per_class] != list(range(len(labels))):
        raise ValueError("Per-class IDs differ")
    by_id = {v: k for k, v in labels.items()}
    for i, row in enumerate(per_class):
        if row["intent"] != by_id[i]:
            raise ValueError("Per-class intent differs from label mapping")
        for metric in ("precision", "recall", "f1", "support"):
            require_close(float(row[metric]), float(result[metric][i]), f"Per-class {metric}")
    return {**{METRICS[k]: v for k, v in values.items()}, "prediction_rows": len(rows),
            "predictions_sha256": file_sha256(directory / "predictions.jsonl")}


def select_records(progress):
    chosen, clean = {}, {}
    for record in progress:
        if record["method"] == "clean_ce":
            ident = (record["shots_per_class"], record["seed"])
            previous = clean.get(ident)
            if previous:
                for field in METRICS.values():
                    require_close(record[field], previous[field], f"Repeated clean reference {field}")
            else:
                clean[ident] = record
            continue
        ident = key(record)
        previous = chosen.get(ident)
        if previous:
            for field in METRICS.values():
                require_close(record[field], previous[field], f"Repeated route {field}")
        if previous is None or record.get("reused", False):
            chosen[ident] = record
    expected = {(noise, rate, shots, seed, method)
                for noise, rate, shots in expected_conditions() for seed in (17, 42, 2026)
                for method in ("ce", "gce", "confident_learning")}
    if set(chosen) != expected or set(clean) != {(k, s) for k in (5, 10, 20, 100) for s in (17, 42, 2026)}:
        raise ValueError("Expected 90 noisy results and 12 clean references")
    return chosen, clean


def run_directory(record):
    if record.get("reused", False):
        comparison = Path(record["reused_from"].replace("\\", "/"))
        method = "cl_ce" if record["method"] in {"cl_ce", "confident_learning"} else record["method"]
        directory = comparison.parent / method / f"k{record['shots_per_class']}_s{record['seed']}"
        return directory, directory / "run.json", directory / "training_summary.json"
    directory = Path(record["run_dir"].replace("\\", "/"))
    return directory / "evaluation", directory / "training/run_from_epoch_1.json", directory / "training/summary.json"


def verify_training(directory, run_path, summary_path, record):
    run, summary = read_json(run_path), read_json(summary_path)
    metrics = read_json(directory / "metrics.json")
    config = read_json("configs/formal_gce.json" if record["method"] == "gce" else "configs/formal_ce.json")
    if run["config"] != config or run["seed"] != record["seed"]:
        raise ValueError("Training configuration or seed differs from frozen protocol")
    history = summary["history"]
    if [h["epoch"] for h in history] != list(range(1, len(history) + 1)) or not history:
        raise ValueError("Incomplete epoch history")
    if not all(np.isfinite(h["val_macro_f1"]) for h in history):
        raise ValueError("Invalid validation history")
    best = max(history, key=lambda h: h["val_macro_f1"])
    if (summary["best_epoch"] != best["epoch"] or metrics["selected_epoch"] != best["epoch"] or
            summary["epochs_completed"] != len(history) or len(history) > config["training"]["max_epochs"] or
            metrics["history"] != history):
        raise ValueError("Selected epoch differs from validation history")
    require_close(summary["best_val_macro_f1"], best["val_macro_f1"], "Best validation F1")
    _, labels = load_official_split("data/processed/clean/val.jsonl", "val")
    fingerprints = run["fingerprints"]
    if (fingerprints != summary["fingerprints"] or
            fingerprints["train"] != metrics["train_sha256"] or
            fingerprints["validation"] != file_sha256("data/processed/clean/val.jsonl") or
            fingerprints["labels"] != file_sha256("data/processed/labels.json") or
            run["config"]["model"]["classes"] != len(labels)):
        raise ValueError("Training fingerprints differ")
    if "train_sha256" in record and fingerprints["train"] != record["train_sha256"]:
        raise ValueError("Training input differs from progress record")
    for field, split in (("test_sha256", "test"), ("oos_test_sha256", "oos_test")):
        if metrics[field] != file_sha256(f"data/processed/clean/{split}.jsonl"):
            raise ValueError("Evaluation split fingerprint differs")


def verify(progress_path, summary_path):
    progress = read_json(progress_path)
    chosen, clean = select_records(progress)
    test, labels = load_official_split("data/processed/clean/test.jsonl", "test")
    oos, _ = load_official_split("data/processed/clean/oos_test.jsonl", "oos_test")
    expected = [(r["id"], r["label"]) for r in test + oos]
    checked, clean_values, noisy_values = [], {}, {}
    for ident, record in list(clean.items()) + list(chosen.items()):
        directory, run_path, training_path = run_directory(record)
        values = verify_predictions(directory, expected, labels, record)
        verify_training(directory, run_path, training_path, record)
        checked.append({"key": list(ident), "directory": directory.as_posix(), **values})
        if record["method"] == "clean_ce":
            clean_values[ident] = values
        else:
            noisy_values[ident] = values
    summaries, pairs = [], []
    for noise, rate, shots in expected_conditions():
        methods = {}
        for method in ("ce", "gce", "confident_learning"):
            records = []
            for seed in (17, 42, 2026):
                ident = (noise, rate, shots, seed, method)
                values = noisy_values[ident]
                retention = values["test_macro_f1"] / clean_values[(shots, seed)]["test_macro_f1"]
                require_close(chosen[ident]["f1_retention"], retention, "F1 retention")
                records.append({**values, "f1_retention": retention})
            methods[method] = {metric: summarize([r[metric] for r in records])
                               for metric in (*METRICS.values(), "f1_retention")}
        summaries.append({"noise": noise, "rate": rate, "shots_per_class": shots,
                          "clean_ce": summarize([clean_values[(shots, s)]["test_macro_f1"] for s in (17, 42, 2026)]),
                          "methods": methods})
        for method in ("gce", "confident_learning"):
            for metric in ("test_macro_f1", "oos_auroc"):
                pair = summarize_pair(methods[method][metric]["values"], methods["ce"][metric]["values"])
                pairs.append({"family": metric, "noise": noise, "rate": rate, "shots_per_class": shots,
                              "comparison": f"{method} - noisy CE", "metric": metric,
                              "seeds": [17, 42, 2026], **pair})
    for metric in ("test_macro_f1", "oos_auroc"):
        indices = [i for i, p in enumerate(pairs) if p["metric"] == metric]
        for test_name in ("paired_t", "wilcoxon_exact"):
            adjusted = holm_adjust([pairs[i][test_name]["p_two_sided"] for i in indices])
            for i, value in zip(indices, adjusted):
                pairs[i][test_name]["p_holm"] = value
    published = read_json(summary_path)
    if (published["progress_sha256"] != file_sha256(progress_path) or
            published["protocol_sha256"] != file_sha256("configs/expanded_matrix_v1.json") or
            published["selected_model_count"] != 90 or
            published["condition_summaries"] != summaries or published["pairwise_statistics"] != pairs):
        raise ValueError("Published aggregate/statistics differ from recomputed predictions")
    return {"all_passed": True, "progress_sha256": file_sha256(progress_path),
            "summary_sha256": file_sha256(summary_path), "unique_prediction_runs": len(checked),
            "prediction_rows": sum(r["prediction_rows"] for r in checked), "paired_comparisons": len(pairs),
            "runs": checked, "scope": "Local predictions, official labels, matrices, per-class tables, training metadata, aggregates and paired statistics.",
            "limitations": ["Requires ignored local stage-5 training artifacts; these are not published by this command.",
                            "Does not replay model weights, OOF probabilities or IDN encoder inference.",
                            "Does not reinstall dependencies or retrain in a new environment."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, default=Path("artifacts/stage5_expanded_runs/progress.json"))
    parser.add_argument("--summary", type=Path, default=Path("docs/阶段5扩展实验结果.json"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/stage5_prediction_check.json"))
    args = parser.parse_args()
    report = verify(args.progress, args.summary)
    write_json(args.report, report)
    print(f"PASS: {report['unique_prediction_runs']} runs; {report['prediction_rows']} predictions; "
          f"{report['paired_comparisons']} paired comparisons")


if __name__ == "__main__":
    main()
