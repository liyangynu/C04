"""Choose among four predeclared paper-guided SVM implementations using validation."""
import argparse
import csv
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


def verify(output):
    plan = read_json(CONFIG)
    selection = read_json(output / "selection.json")
    results = read_json(output / "results.json")
    if (selection["config_sha256"] != file_sha256(CONFIG) or
            [r["candidate"] for r in selection["candidates"]] != plan["candidates"] or
            select_candidate(selection["candidates"]) != selection["selected"] or
            selection["selected_before_test_read"] is not True):
        raise ValueError("Validation candidate selection differs from frozen protocol")
    train, labels = official_rows("train")
    oos_train, _ = official_rows("oos_train")
    val, _ = official_rows("val")
    oos_val, _ = official_rows("oos_val")
    test, _ = official_rows("test")
    oos, _ = official_rows("oos_test")
    settings = read_json("configs/public_svm_v1.json")
    vocabs = {}
    for row in selection["candidates"]:
        candidate = row["candidate"]
        config = {**settings, "vectorizer": dict(settings["vectorizer"], binary=candidate["binary"])}
        expected_vocab = make_vectorizer(config).fit([r["text"] for r in train + oos_train]).vocabulary_
        vocabs[candidate["binary"]] = expected_vocab
        model = joblib.load(output / (candidate["name"] + ".joblib"))
        if (model["vectorizer"].vocabulary_ != expected_vocab or
                model["vectorizer"].binary != candidate["binary"] or
                model["classifier"].get_params() != classifier_for(candidate, plan, 17).get_params()):
            raise ValueError("Candidate vocabulary/config differs")
        rows = infer(model["vectorizer"], model["classifier"], val + oos_val)
        accuracy = float(np.mean([r["true_label"] == r["predicted_class"] for r in rows]))
        if accuracy != row["validation_accuracy"] or row["test_read"]:
            raise ValueError("Candidate validation accuracy differs")
    verified = []
    for seed in plan["seeds"]:
        dest = output / f"seed_{seed}"
        record = read_json(dest / "metrics.json")
        if (record["seed"] != seed or record["selection_sha256"] != file_sha256(output / "selection.json") or
                record["selected"] != selection["selected"] or
                read_json(dest / "vocab.json") != vocabs[selection["selected"]["binary"]]):
            raise ValueError("Selected model metadata differs")
        for split, digest in record["official_sha256"].items():
            if file_sha256(f"data/processed/clean/{split}.jsonl") != digest:
                raise ValueError("SVM official split fingerprint differs")
        if file_sha256(dest / "model.joblib") != record["model_sha256"]:
            raise ValueError("SVM model fingerprint differs")
        model = joblib.load(dest / "model.joblib")
        if (model["config_sha256"] != file_sha256(CONFIG) or model["seed"] != seed or
                model["selected"] != selection["selected"] or
                model["classifier"].get_params() != classifier_for(selection["selected"], plan, seed).get_params()):
            raise ValueError("SVM selected model config/seed differs")
        rows = read_jsonl(dest / "predictions.jsonl")
        if ([(r["id"], r["true_label"]) for r in rows] != [(r["id"], r["label"]) for r in test + oos] or
                file_sha256(dest / "predictions.jsonl") != record["predictions_sha256"] or
                metrics_from_rows(rows) != record["metrics"] or
                infer(model["vectorizer"], model["classifier"], test + oos) != rows):
            raise ValueError("SVM predictions/metrics/inference replay differs")
        result = classification_metrics(np.array([r["true_label"] for r in rows]),
                                        np.array([r["predicted_class"] for r in rows]), 151)
        with (dest / "confusion_matrix.csv").open(encoding="utf-8", newline="") as stream:
            table = list(csv.reader(stream))
        if not np.array_equal(np.array([r[1:] for r in table[1:]], dtype=int), result["confusion_matrix"]):
            raise ValueError("SVM matrix differs")
        with (dest / "per_class.csv").open(encoding="utf-8", newline="") as stream:
            per_class = list(csv.DictReader(stream))
        if len(per_class) != 151 or any(int(r["id"]) != i or any(
                abs(float(r[k]) - result[k][i]) > 1e-12 for k in ("precision", "recall", "f1", "support"))
                for i, r in enumerate(per_class)):
            raise ValueError("SVM per-class table differs")
        if record != next(r for r in results["records"] if r["seed"] == seed):
            raise ValueError("SVM aggregate record differs")
        verified.append({"seed": seed, "rows": len(rows), "classes_and_scores_equal": True})
    summary = {k: {"mean": float(np.mean([r["metrics"][k] for r in results["records"]])),
                   "std_sample": float(np.std([r["metrics"][k] for r in results["records"]], ddof=1))}
               for k in results["records"][0]["metrics"]}
    delta = 100 * (summary["id_accuracy"]["mean"] - plan["reference"]["id_accuracy"])
    if summary != results["summary"] or results["id_accuracy_difference_pp"] != delta:
        raise ValueError("SVM aggregate statistics differ")
    report = {"all_passed": True, "validation_candidates_replayed": 4, "final_models_replayed": verified,
              "summary": summary, "id_accuracy_difference_pp": delta, "within_two_percentage_points": abs(delta) <= 2,
              "implementation_limitations": plan["limitations"]}
    write_json(output / "verified.json", report)
    print("PASS: four validation candidates and 16500 final predictions independently replayed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/public_svm_validation_v2"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        run(args.output)
    verify(args.output)


if __name__ == "__main__":
    main()
