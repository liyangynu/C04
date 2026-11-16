"""Evaluate a saved classifier on the official closed-set and OOS test splits."""
import argparse
import csv
import os
from pathlib import Path

from .runtime import (choose_device, configure_runtime, environment, load_official_split,
                      make_loader, predict)
import numpy as np
import torch

from .common import file_sha256, write_json, write_jsonl
from .metrics import binary_auroc, classification_metrics, roc_points
from .model import BiLSTMClassifier


def write_csv(path, header, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def plot_results(output, result, oos_truth, scores, history, loss_name="ce"):
    cache = Path("tmp/matplotlib").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr = roc_points(oos_truth, scores)
    ax.plot(fpr, tpr, label=f"MSP AUROC={binary_auroc(oos_truth, scores):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set(xlabel="False positive rate (ID detected as OOS)", ylabel="True positive rate (OOS)", title="OOS detection")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output/"oos_roc.png", dpi=160)
    plt.close(fig)
    matrix = result["confusion_matrix"]
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ticks = list(range(0, len(matrix), 10))
    ax.set(xticks=ticks, yticks=ticks, xlabel="Predicted intent ID", ylabel="True intent ID",
           title="CLINC150 row-normalized confusion matrix")
    fig.colorbar(im, ax=ax, label="Fraction of true-class examples")
    fig.tight_layout()
    fig.savefig(output/"confusion_matrix.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["train_loss"] for r in history], marker="o", label=f"Train ({loss_name.upper()})")
    axes[0].plot(epochs, [r["val_loss"] for r in history], marker="o", label="Validation (CE)")
    axes[0].set(xlabel="Epoch", ylabel="Objective value", title="Learning curves")
    axes[0].legend()
    axes[1].plot(epochs, [r["val_macro_f1"] for r in history], marker="o")
    axes[1].set(xlabel="Epoch", ylabel="Macro F1", ylim=(0, 1), title="Validation (150 classes)")
    fig.tight_layout()
    fig.savefig(output/"learning_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=Path("data/processed/clean/test.jsonl"))
    parser.add_argument("--oos", type=Path, default=Path("data/processed/clean/oos_test.jsonl"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config, vocab = checkpoint["config"], checkpoint["vocab"]
    configure_runtime(checkpoint["seed"], config["training"])
    device = choose_device(args.device)
    test, labels = load_official_split(args.test, "test")
    oos, oos_labels = load_official_split(args.oos, "oos_test")
    if labels != checkpoint["labels"] or labels != oos_labels:
        raise ValueError("Checkpoint and evaluation label mappings differ")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Evaluation output already exists; use a new directory")
    model = BiLSTMClassifier(checkpoint["model"]["embedding.weight"],
        **{k: v for k, v in config["model"].items() if k != "embedding_dim"}).to(device)
    model.load_state_dict(checkpoint["model"])
    rows = test+oos
    loader = make_loader(rows, vocab, config["text"]["max_length"], config["training"]["batch_size"])
    logits, all_truth = predict(model, loader, device)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    scores = 1-probabilities.max(axis=1)
    predicted = logits[:len(test)].argmax(axis=1)
    result = classification_metrics(all_truth[:len(test)], predicted, config["model"]["classes"])
    oos_truth = np.r_[np.zeros(len(test), dtype=np.int64), np.ones(len(oos), dtype=np.int64)]
    summary = {"seed": checkpoint["seed"], "selected_epoch": checkpoint["epoch"],
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "train_sha256": checkpoint["fingerprints"]["train"], "test_sha256": file_sha256(args.test),
        "oos_test_sha256": file_sha256(args.oos), "test_rows": len(test), "oos_rows": len(oos),
        "macro_f1": result["macro_f1"], "accuracy": result["accuracy"],
        "oos_auroc": binary_auroc(oos_truth, scores),
        "oos_positive_class": "OOS", "oos_score": "1 - max softmax probability",
        "history": checkpoint["history"], "training_git": checkpoint["git"],
        "training_environment": checkpoint["environment"], "evaluation_environment": environment(device)}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/"metrics.json", summary)
    by_id = {index: name for name, index in labels.items()}
    write_csv(args.output/"per_class.csv", ["class_id", "intent", "precision", "recall", "f1", "support"],
              [[i, by_id[i], result["precision"][i], result["recall"][i], result["f1"][i], result["support"][i]]
               for i in range(len(labels))])
    write_csv(args.output/"confusion_matrix.csv", ["true_id/predicted_id", *range(len(labels))],
              [[i, *row] for i, row in enumerate(result["confusion_matrix"])])
    write_jsonl(args.output/"predictions.jsonl", [
        {"id": row["id"], "true_label": row["label"], "predicted_intent": int(logits[i].argmax()),
         "oos_score": float(scores[i])} for i, row in enumerate(rows)])
    np.savez_compressed(args.output/"logits.npz", ids=np.array([r["id"] for r in rows]), logits=logits)
    plot_results(args.output, result, oos_truth, scores, checkpoint["history"], config["training"].get("loss", "ce"))
    print(f"macro_f1={summary['macro_f1']:.4f}; accuracy={summary['accuracy']:.4f}; OOS_AUROC={summary['oos_auroc']:.4f}")


if __name__ == "__main__":
    main()
