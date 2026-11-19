"""Choose among four predeclared paper-guided SVM implementations using validation."""
import argparse
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.svm import LinearSVC, SVC

from .common import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from .evaluate import write_csv
from .metrics import classification_metrics
from .public_cnn import official_rows
from .public_svm import infer, make_vectorizer, metrics_from_rows
from .runtime import git_state


CONFIG = Path("configs/public_svm_validation_v2.json")


def classifier_for(candidate, plan, seed):
    if candidate["solver"] == "svc":
        return SVC(C=plan["C"], kernel="linear", cache_size=512, random_state=seed)
    if candidate["solver"] == "linear_svc":
        return LinearSVC(C=plan["C"], random_state=seed, **plan["linear_svc"])
    raise ValueError("Unknown frozen SVM solver")


def select_candidate(records):
    if not records or any(not np.isfinite(r["validation_accuracy"]) for r in records):
        raise ValueError("Finite validation scores required")
    return max(records, key=lambda r: r["validation_accuracy"])["candidate"]


def run(output):
    plan = read_json(CONFIG)
    if output.exists():
        raise ValueError("Use a new SVM-v2 directory")
    output.mkdir(parents=True)
    train, labels = official_rows("train")
    oos_train, _ = official_rows("oos_train")
    validation, _ = official_rows("val")
    oos_val, _ = official_rows("oos_val")
    train_rows, val_rows = train + oos_train, validation + oos_val
    settings = read_json("configs/public_svm_v1.json")
    candidates = []
    for candidate in plan["candidates"]:
        started = time.perf_counter()
        config = {**settings, "vectorizer": dict(settings["vectorizer"], binary=candidate["binary"])}
        vectorizer = make_vectorizer(config)
        features = vectorizer.fit_transform([r["text"] for r in train_rows])
        classifier = classifier_for(candidate, plan, 17)
        classifier.fit(features, [r["label"] for r in train_rows])
        predicted = infer(vectorizer, classifier, val_rows)
        accuracy = float(np.mean([r["predicted_class"] == r["true_label"] for r in predicted]))
        row = {"candidate": candidate, "validation_accuracy": accuracy,
               "wall_seconds": time.perf_counter() - started, "test_read": False}
        candidates.append(row)
        joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, output / (candidate["name"] + ".joblib"))
        write_json(output / "candidate_progress.json", candidates)
        print(f"candidate={candidate['name']}; validation_accuracy={accuracy:.6f}", flush=True)
    selected = select_candidate(candidates)
    write_json(output / "selection.json", {"config_sha256": file_sha256(CONFIG), "candidates": candidates,
               "selected": selected, "selected_before_test_read": True, "git": git_state()})
    # No test file has been opened above. Candidate selection is now immutable on disk.
    test, _ = official_rows("test")
    oos, _ = official_rows("oos_test")
    records = []
    for seed in plan["seeds"]:
        dest = output / f"seed_{seed}"
        dest.mkdir()
        started = time.perf_counter()
        if seed == 17:
            model = joblib.load(output / (selected["name"] + ".joblib"))
            vectorizer, classifier = model["vectorizer"], model["classifier"]
        else:
            config = {**settings, "vectorizer": dict(settings["vectorizer"], binary=selected["binary"])}
            vectorizer = make_vectorizer(config)
            features = vectorizer.fit_transform([r["text"] for r in train_rows])
            classifier = classifier_for(selected, plan, seed)
            classifier.fit(features, [r["label"] for r in train_rows])
        predictions = infer(vectorizer, classifier, test + oos)
        write_jsonl(dest / "predictions.jsonl", predictions)
        write_json(dest / "vocab.json", {k: int(v) for k, v in vectorizer.vocabulary_.items()})
        joblib.dump({"vectorizer": vectorizer, "classifier": classifier, "selected": selected,
                     "seed": seed, "config_sha256": file_sha256(CONFIG)}, dest / "model.joblib")
        result = classification_metrics(np.array([r["true_label"] for r in predictions]),
                                        np.array([r["predicted_class"] for r in predictions]), 151)
        write_csv(dest / "confusion_matrix.csv", ["true_id/predicted_id", *range(151)],
                  [[i, *row] for i, row in enumerate(result["confusion_matrix"])])
        names = {v: k for k, v in labels.items()}
        write_csv(dest / "per_class.csv", ["id", "intent", "precision", "recall", "f1", "support"],
                  [[i, names.get(i, "OOS"), *[result[k][i] for k in ("precision", "recall", "f1", "support")]]
                   for i in range(151)])
        record = {"seed": seed, "metrics": metrics_from_rows(predictions), "selected": selected,
                  "wall_seconds": time.perf_counter() - started,
                  "model_sha256": file_sha256(dest / "model.joblib"),
                  "predictions_sha256": file_sha256(dest / "predictions.jsonl"),
                  "selection_sha256": file_sha256(output / "selection.json"),
                  "official_sha256": {s: file_sha256(f"data/processed/clean/{s}.jsonl") for s in
                                        ("train", "oos_train", "val", "oos_val", "test", "oos_test")}}
        write_json(dest / "metrics.json", record)
        records.append(record)
        print(f"SVM v2 seed={seed}: {record['metrics']}", flush=True)
    summary = {k: {"mean": float(np.mean([r["metrics"][k] for r in records])),
                   "std_sample": float(np.std([r["metrics"][k] for r in records], ddof=1))}
               for k in records[0]["metrics"]}
    delta = 100 * (summary["id_accuracy"]["mean"] - plan["reference"]["id_accuracy"])
    write_json(output / "results.json", {"records": records, "summary": summary,
               "reference": plan["reference"], "id_accuracy_difference_pp": delta,
               "within_two_percentage_points": abs(delta) <= 2,
               "limitations": plan["limitations"]})
    print(f"SVM v2 accuracy difference={delta:+.2f}pp; tolerance={abs(delta) <= 2}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/public_svm_validation_v2"))
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
