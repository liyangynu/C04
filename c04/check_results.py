"""Recompute exported scores from saved predictions and verify official test labels."""
import argparse
import csv
from pathlib import Path

import numpy as np

from .common import file_sha256, read_json, read_jsonl, write_json
from .metrics import binary_auroc, classification_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    for name, checksum in read_json(args.results/"checksums.json").items():
        if file_sha256(args.results/name) != checksum:
            raise ValueError(f"Result file changed: {name}")
    official = read_jsonl("data/processed/clean/test.jsonl")+read_jsonl("data/processed/clean/oos_test.jsonl")
    expected = [(r["id"], r["label"]) for r in official]
    protocol = read_json("configs/protocol.json")
    manifest = read_json("data/processed/manifest.json")
    for split in ["test", "oos_test"]:
        if file_sha256(f"data/processed/clean/{split}.jsonl") != manifest["files"][f"clean/{split}.jsonl"]:
            raise ValueError("Official evaluation data changed")
    if manifest["source_sha256"] != protocol["dataset"]["sha256"]:
        raise ValueError("Unexpected evaluation data version")
    matrix = read_json(args.results/"matrix.json")
    reports = []
    for record in matrix["records"]:
        name = f"k{record['shots_per_class']}_s{record['seed']}"
        path = args.results/name
        rows = read_jsonl(path/"predictions.jsonl")
        if [(r["id"], r["true_label"]) for r in rows] != expected:
            raise ValueError(f"{name}: prediction IDs or true labels differ from official data")
        truth = np.array([r["true_label"] for r in rows])
        is_id = truth >= 0
        predicted = np.array([r["predicted_intent"] for r in rows])
        result = classification_metrics(truth[is_id], predicted[is_id], protocol["dataset"]["classes"])
        values = {"macro_f1": result["macro_f1"], "accuracy": result["accuracy"],
                  "oos_auroc": binary_auroc((~is_id).astype(int), [r["oos_score"] for r in rows])}
        metrics = read_json(path/"metrics.json")
        for metric, value in values.items():
            if abs(value-metrics[metric]) > 1e-12:
                raise ValueError(f"{name}: {metric} does not match predictions")
            matrix_key = {"macro_f1": "test_macro_f1", "accuracy": "test_accuracy", "oos_auroc": "oos_auroc"}[metric]
            if abs(value-record[matrix_key]) > 1e-12:
                raise ValueError(f"{name}: {metric} differs from matrix summary")
        with (path/"confusion_matrix.csv").open(encoding="utf-8", newline="") as stream:
            table = list(csv.reader(stream))
        if not np.array_equal(np.array([r[1:] for r in table[1:]], dtype=int), result["confusion_matrix"]):
            raise ValueError(f"{name}: confusion matrix mismatch")
        reports.append({"run": name, "rows": len(rows), "metrics_recomputed": values, "passed": True})
    for shots in matrix["shots_per_class"]:
        records = [r for r in matrix["records"] if r["shots_per_class"] == shots]
        for metric in ["best_val_macro_f1", "test_macro_f1", "test_accuracy", "oos_auroc"]:
            values = np.array([r[metric] for r in records])
            summary = matrix["summary_by_shots"][str(shots)][metric]
            if abs(float(values.mean())-summary["mean"]) > 1e-12:
                raise ValueError("Aggregate mean mismatch")
            if len(values)>1 and abs(float(values.std(ddof=1))-summary["std_sample"]) > 1e-12:
                raise ValueError("Aggregate sample deviation mismatch")
    report = {"matrix_sha256": file_sha256(args.results/"matrix.json"),
              "runs": reports, "all_passed": True}
    write_json("reports/stage2_prediction_checks.json", report)
    print(f"PASS: {len(reports)} exported runs match official labels, predictions, matrices and summaries.")


if __name__ == "__main__":
    main()
