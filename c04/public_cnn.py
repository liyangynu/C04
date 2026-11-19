"""Paper-guided CNN with explicit choices for unspecified implementation details."""
import argparse
import time
from pathlib import Path

from .runtime import (configure_runtime, choose_device, environment, git_state,
                      load_official_split, make_loader, predict, save_checkpoint)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import file_sha256, read_json, write_json, write_jsonl
from .embeddings import build_assets
from .metrics import binary_auroc, classification_metrics
from .train import train_epoch
from .evaluate import write_csv


class PublicCNN(nn.Module):
    def __init__(self, matrix, filters=200, kernel_size=3, dense_dim=300, dropout=.6,
                 classes=151, conv_activation="softmax_channels"):
        super().__init__()
        if kernel_size < 1 or conv_activation != "softmax_channels":
            raise ValueError("Expected positive kernel width and the frozen convolution activation")
        self.kernel_size = kernel_size
        self.embedding = nn.Embedding.from_pretrained(matrix.clone(), freeze=False, padding_idx=0)
        self.conv = nn.Conv1d(matrix.shape[1], filters, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(filters, dense_dim)
        self.classifier = nn.Linear(dense_dim, classes)

    def forward(self, tokens, lengths):
        if tokens.shape[1] < self.kernel_size:
            tokens = F.pad(tokens, (0, self.kernel_size - tokens.shape[1]))
        values = torch.softmax(self.conv(self.embedding(tokens).transpose(1, 2)), dim=1)
        valid_count = (lengths.to(tokens.device) - self.kernel_size + 1).clamp(min=1)
        valid = torch.arange(values.shape[-1], device=tokens.device)[None, :] < valid_count[:, None]
        pooled = values.masked_fill(~valid[:, None, :], -torch.inf).amax(dim=2)
        return self.classifier(self.dropout(F.relu(self.dense(self.dropout(pooled)))))


def official_rows(split):
    """Verify official files before mapping OOS=-1 to the 151st class."""
    rows, labels = load_official_split(Path(f"data/processed/clean/{split}.jsonl"), split)
    if split.startswith("oos_"):
        rows = [dict(r, label=150) for r in rows]
    return rows, labels


def prediction_metrics(rows):
    truth = np.asarray([r["true_label"] for r in rows], dtype=np.int64)
    prediction = np.asarray([r["predicted_class"] for r in rows], dtype=np.int64)
    is_id = truth != 150
    id_result = classification_metrics(truth[is_id], prediction[is_id], 151)
    return {"id_accuracy": float(np.mean(truth[is_id] == prediction[is_id])),
            "oos_recall": float(np.mean(prediction[~is_id] == 150)),
            "id_macro_f1": float(id_result["f1"][:150].mean()),
            "oos_auroc_p_oos": binary_auroc((~is_id).astype(int), [r["p_oos"] for r in rows])}


def train_one(config_path, output, seed, requested_device):
    config = read_json(config_path)
    if seed not in config["seeds"]:
        raise ValueError("Seed must be one of the frozen three seeds")
    if output.exists():
        raise ValueError(f"Use a new output directory: {output}")
    configure_runtime(seed, config["training"])
    device = choose_device(requested_device)
    started = time.perf_counter()
    train_id, labels = official_rows("train")
    train_oos, _ = official_rows("oos_train")
    val_id, _ = official_rows("val")
    val_oos, _ = official_rows("oos_val")
    # Both ID and OOS training text belong to the training-only vocabulary.
    rows = train_id + [dict(r, id="train:" + r["id"]) for r in train_oos]
    output.mkdir(parents=True)
    train_path = output / "train.jsonl"
    write_jsonl(train_path, rows)
    glove = Path("data/embeddings/glove.6B.100d.txt")
    if file_sha256(glove) != read_json("configs/glove.json")["member_sha256"]:
        raise ValueError("Pinned GloVe checksum mismatch")
    assets = build_assets(train_path, glove, output / "assets", seed,
                          config["model"]["embedding_dim"], config["text"]["max_length"])
    vocab = read_json(output / "assets/vocab.json")
    matrix = torch.from_numpy(np.load(output / "assets/embeddings.npy", allow_pickle=False))
    model_args = {k: v for k, v in config["model"].items() if k != "embedding_dim"}
    model = PublicCNN(matrix, **model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"],
                                 weight_decay=config["training"]["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = make_loader(rows, vocab, config["text"]["max_length"], 16, True, generator)
    val_loader = make_loader(val_id + val_oos, vocab, config["text"]["max_length"], 16)
    provenance = {"seed": seed, "config": config, "config_sha256": file_sha256(config_path),
                  "git": git_state(), "environment": environment(device), "assets": assets,
                  "official_input_sha256": {s: file_sha256(f"data/processed/clean/{s}.jsonl")
                                             for s in ("train", "oos_train", "val", "oos_val")},
                  "train_rows": len(rows), "validation_rows": len(val_id) + len(val_oos)}
    write_json(output / "run.json", provenance)
    history, best, bad, best_epoch = [], -1.0, 0, 0
    for epoch in range(1, config["training"]["max_epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        clock = time.perf_counter()
        loss = train_epoch(model, loader, optimizer, device, 5.0, config["training"])
        logits, truth = predict(model, val_loader, device)
        score = float(np.mean(logits.argmax(axis=1) == truth))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append({"epoch": epoch, "train_loss": loss, "val_accuracy": score,
                        "epoch_seconds": time.perf_counter() - clock,
                        "peak_allocated_mb": torch.cuda.max_memory_allocated(device)/1024**2 if device.type == "cuda" else 0,
                        "peak_reserved_mb": torch.cuda.max_memory_reserved(device)/1024**2 if device.type == "cuda" else 0})
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            save_checkpoint(output / "best.pt", {"model": model.state_dict(), "vocab": vocab,
                            "config": config, "epoch": epoch, "provenance": provenance})
        else:
            bad += 1
        write_json(output / "history.json", history)
        print(f"seed={seed} epoch={epoch} val_accuracy={score:.4f} loss={loss:.4f}", flush=True)
        if bad >= config["training"]["patience"]:
            break
    # Test data is first read after validation has selected the checkpoint.
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test, _ = official_rows("test")
    oos, _ = official_rows("oos_test")
    eval_rows = test + oos
    logits, truth = predict(model, make_loader(eval_rows, vocab, config["text"]["max_length"], 16), device)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = [{"id": r["id"], "true_label": r["label"], "predicted_class": int(logits[i].argmax()),
                    "p_oos": float(probabilities[i, 150])} for i, r in enumerate(eval_rows)]
    write_jsonl(output / "predictions.jsonl", predictions)
    values = prediction_metrics(predictions)
    result = classification_metrics(truth, logits.argmax(axis=1), 151)
    write_csv(output / "confusion_matrix.csv", ["true_id/predicted_id", *range(151)],
              [[i, *r] for i, r in enumerate(result["confusion_matrix"])])
    by_id = {v: k for k, v in labels.items()}
    write_csv(output / "per_class.csv", ["id", "intent", "precision", "recall", "f1", "support"],
              [[i, by_id.get(i, "OOS"), result["precision"][i], result["recall"][i],
                result["f1"][i], result["support"][i]] for i in range(151)])
    report = {"seed": seed, "best_epoch": best_epoch, "best_validation_accuracy": best,
              "epochs_completed": len(history), "metrics": values,
              "checkpoint_sha256": file_sha256(output / "best.pt"), "train_sha256": file_sha256(train_path),
              "predictions_sha256": file_sha256(output / "predictions.jsonl"),
              "test_sha256": file_sha256("data/processed/clean/test.jsonl"),
              "oos_test_sha256": file_sha256("data/processed/clean/oos_test.jsonl"),
              "wall_seconds": time.perf_counter() - started,
              "total_epoch_seconds": sum(h["epoch_seconds"] for h in history),
              "peak_allocated_mb": max(h["peak_allocated_mb"] for h in history),
              "peak_reserved_mb": max(h["peak_reserved_mb"] for h in history), "git": provenance["git"]}
    write_json(output / "metrics.json", report)
    print(f"Completed seed={seed}: {values}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/public_cnn_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    train_one(args.config, args.output, args.seed, args.device)


if __name__ == "__main__":
    main()
