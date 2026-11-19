"""Paper-guided CLINC150 SVM with frozen binary bag-of-words features."""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

from .common import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from .evaluate import write_csv
from .metrics import binary_auroc, classification_metrics
from .public_cnn import official_rows
from .runtime import environment, git_state


CONFIG = Path("configs/public_svm_v1.json")


def make_vectorizer(config):
    settings = dict(config["vectorizer"])
    settings["ngram_range"] = tuple(settings["ngram_range"])
    return CountVectorizer(**settings)


def metrics_from_rows(rows):
    truth = np.array([r["true_label"] for r in rows])
    predicted = np.array([r["predicted_class"] for r in rows])
    is_id = truth != 150
    closed = classification_metrics(truth[is_id], predicted[is_id], 151)
    return {"id_accuracy": closed["accuracy"], "id_macro_f1": float(closed["f1"][:150].mean()),
            "oos_recall": float(np.mean(predicted[~is_id] == 150)),
            "oos_auroc_margin": binary_auroc((~is_id).astype(int), [r["oos_margin"] for r in rows])}


def infer(vectorizer, classifier, rows):
    matrix = vectorizer.transform([r["text"] for r in rows])
    prediction = classifier.predict(matrix)
    scores = classifier.decision_function(matrix)
    if classifier.classes_.tolist() != list(range(151)):
        raise ValueError("Expected the frozen 151-class label order")
    margin = scores[:, 150] - scores[:, :150].max(axis=1)
    return [{"id": r["id"], "true_label": r["label"], "predicted_class": int(prediction[i]),
             "oos_margin": float(margin[i])} for i, r in enumerate(rows)]


def run(output):
    import torch
    config = read_json(CONFIG)
    output.mkdir(parents=True, exist_ok=True)
    train, labels = official_rows("train")
    oos_train, _ = official_rows("oos_train")
    validation, _ = official_rows("val")
    oos_val, _ = official_rows("oos_val")
    train_rows = train + oos_train
    records = []
    for seed in config["seeds"]:
        dest = output / f"seed_{seed}"
        if (dest / "metrics.json").exists():
            record = read_json(dest / "metrics.json")
            if record["config_sha256"] != file_sha256(CONFIG):
                raise ValueError("Existing SVM configuration changed")
            records.append(record)
            continue
        if dest.exists():
            raise ValueError("Partial SVM run exists; use a new output directory")
        dest.mkdir()
        started = time.perf_counter()
        vectorizer = make_vectorizer(config)
        features = vectorizer.fit_transform([r["text"] for r in train_rows])
        classifier = SVC(**config["classifier"], random_state=seed)
        classifier.fit(features, [r["label"] for r in train_rows])
        training_seconds = time.perf_counter() - started
        val_predictions = infer(vectorizer, classifier, validation + oos_val)
        validation_accuracy = float(np.mean([r["predicted_class"] == r["true_label"] for r in val_predictions]))
        # Frozen classifier; test is only loaded after training and validation evaluation.
        test, _ = official_rows("test")
        oos, _ = official_rows("oos_test")
        predictions = infer(vectorizer, classifier, test + oos)
        write_jsonl(dest / "predictions.jsonl", predictions)
        write_json(dest / "vocab.json", {k: int(v) for k, v in vectorizer.vocabulary_.items()})
        joblib.dump({"vectorizer": vectorizer, "classifier": classifier, "config": config,
                     "seed": seed}, dest / "model.joblib")
        full = classification_metrics(np.array([r["true_label"] for r in predictions]),
                                      np.array([r["predicted_class"] for r in predictions]), 151)
        write_csv(dest / "confusion_matrix.csv", ["true_id/predicted_id", *range(151)],
                  [[i, *row] for i, row in enumerate(full["confusion_matrix"])])
        names = {v: k for k, v in labels.items()}
        write_csv(dest / "per_class.csv", ["id", "intent", "precision", "recall", "f1", "support"],
                  [[i, names.get(i, "OOS"), *[full[k][i] for k in ("precision", "recall", "f1", "support")]]
                   for i in range(151)])
        record = {"seed": seed, "metrics": metrics_from_rows(predictions),
                  "validation_accuracy": validation_accuracy, "training_seconds": training_seconds,
                  "wall_seconds": time.perf_counter() - started, "train_rows": len(train_rows),
                  "config_sha256": file_sha256(CONFIG), "model_sha256": file_sha256(dest / "model.joblib"),
                  "predictions_sha256": file_sha256(dest / "predictions.jsonl"),
                  "official_sha256": {s: file_sha256(f"data/processed/clean/{s}.jsonl") for s in
                                        ("train", "oos_train", "val", "oos_val", "test", "oos_test")},
                  "environment": {**environment(torch.device("cpu")), "scikit_learn": sklearn.__version__},
                  "git": git_state()}
        write_json(dest / "metrics.json", record)
        records.append(record)
        print(f"SVM seed={seed}: {record['metrics']}", flush=True)
    write_json(output / "config.json", config)
    verify(output, replay=True)


def verify(output, replay=False):
    config = read_json(CONFIG)
    if read_json(output / "config.json") != config:
        raise ValueError("SVM config changed")
    train, _ = official_rows("train")
    oos_train, _ = official_rows("oos_train")
    train_rows = train + oos_train
    expected_vocab = make_vectorizer(config).fit([r["text"] for r in train_rows]).vocabulary_
    test, _ = official_rows("test")
    oos, _ = official_rows("oos_test")
    expected = [(r["id"], r["label"]) for r in test + oos]
    records = []
    for seed in config["seeds"]:
        dest = output / f"seed_{seed}"
        record = read_json(dest / "metrics.json")
        if record["seed"] != seed or record["config_sha256"] != file_sha256(CONFIG):
            raise ValueError("SVM seed/config differs")
        for split, digest in record["official_sha256"].items():
            if file_sha256(f"data/processed/clean/{split}.jsonl") != digest:
                raise ValueError("SVM official data fingerprint differs")
        rows = read_jsonl(dest / "predictions.jsonl")
        if [(r["id"], r["true_label"]) for r in rows] != expected:
            raise ValueError("SVM test IDs/truth differ")
        if (file_sha256(dest / "predictions.jsonl") != record["predictions_sha256"] or
                metrics_from_rows(rows) != record["metrics"] or
                read_json(dest / "vocab.json") != expected_vocab):
            raise ValueError("SVM predictions, vocabulary or metrics differ")
        if replay:
            if file_sha256(dest / "model.joblib") != record["model_sha256"]:
                raise ValueError("SVM model changed")
            model = joblib.load(dest / "model.joblib")
            if model["config"] != config or model["seed"] != seed:
                raise ValueError("SVM model config/seed differs")
            if infer(model["vectorizer"], model["classifier"], test + oos) != rows:
                raise ValueError("SVM inference replay differs")
        records.append(record)
    summary = {k: {"mean": float(np.mean([r["metrics"][k] for r in records])),
                   "std_sample": float(np.std([r["metrics"][k] for r in records], ddof=1))}
               for k in records[0]["metrics"]}
    delta = 100 * (summary["id_accuracy"]["mean"] - config["reference"]["id_accuracy"])
    result = {"all_passed": True, "records": records, "summary": summary,
              "id_accuracy_difference_pp": delta, "within_two_percentage_points": abs(delta) <= 2,
              "reference": config["reference"], "weights_replayed": replay,
              "note": "Deterministic SVC results may be identical; three runs do not imply stochastic uncertainty."}
    write_json(output / "verified.json", result)
    print(f"SVM verification passed; accuracy delta={delta:+.2f}pp; tolerance={abs(delta) <= 2}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/public_svm_v1"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.output, args.replay)
    else:
        run(args.output)


if __name__ == "__main__":
    main()
