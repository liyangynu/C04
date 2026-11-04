"""Train a clean CE BiLSTM baseline; select checkpoints using validation F1."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import random
import time

from .runtime import (check_rows, choose_device, configure_runtime, environment, git_state,
                      load_official_split, make_loader, predict, save_checkpoint)
import torch

from .common import file_sha256, read_json, read_jsonl, write_json
from .embeddings import build_assets
from .metrics import classification_metrics
from .model import BiLSTMClassifier
from .text import text_stats


def train_epoch(model, loader, optimizer, device, gradient_clip):
    model.train()
    total, seen = 0.0, 0
    for tokens, lengths, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens.to(device), lengths)
        loss = torch.nn.functional.cross_entropy(logits, labels.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach())*len(labels)
        seen += len(labels)
    return total/seen


def config_without_epochs(config):
    result = copy.deepcopy(config)
    result["training"].pop("max_epochs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, default=Path("data/processed/clean/val.jsonl"))
    parser.add_argument("--glove", type=Path, default=Path("data/embeddings/glove.6B.100d.txt"))
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, help="Total target epochs, including already completed epochs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Continue output/last.pt with its RNG and optimizer state")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.epochs is not None:
        config["training"]["max_epochs"] = args.epochs
    training = config["training"]
    if training["max_epochs"] < 1 or training["patience"] < 1:
        raise ValueError("Positive epoch count and patience required")
    configure_runtime(args.seed, training)
    device = choose_device(args.device)
    rows = read_jsonl(args.train)
    check_rows(rows, config["model"]["classes"], "train")
    validation, labels = load_official_split(args.val, "val")
    if len(labels) != config["model"]["classes"]:
        raise ValueError("Model class count differs from official label mapping")
    fingerprints = {"train": file_sha256(args.train), "validation": file_sha256(args.val),
                    "labels": file_sha256(args.val.parent.parent/"labels.json")}
    output = args.output
    previous = None
    if args.resume:
        previous = torch.load(output/"last.pt", map_location="cpu", weights_only=True)
        if previous["seed"] != args.seed or previous["fingerprints"] != fingerprints:
            raise ValueError("Resume seed or data changed")
        if config_without_epochs(previous["config"]) != config_without_epochs(config):
            raise ValueError("Resume configuration changed beyond target epoch count")
        if previous["environment"] != environment(device):
            raise ValueError("Exact resume requires the same runtime and device")
        if not (output/"best.pt").exists():
            raise ValueError("Resume requires the existing best checkpoint")
        if previous["bad_epochs"] >= training["patience"]:
            raise ValueError("Early stopping already reached; use a new run for a different protocol")
        if training["max_epochs"] <= previous["epoch"]:
            raise ValueError("Target epochs must exceed the saved epoch")
        vocab = previous["vocab"]
        matrix = previous["model"]["embedding.weight"]
        asset_report = previous["embedding_report"]
    else:
        if file_sha256(args.glove) != read_json("configs/glove.json")["member_sha256"]:
            raise ValueError("Expected the pinned Stanford GloVe 6B 100d vectors")
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output already contains files; use a new directory or --resume")
        output.mkdir(parents=True, exist_ok=True)
        asset_report = build_assets(args.train, args.glove, output/"assets", args.seed,
                                    config["model"]["embedding_dim"], config["text"]["max_length"],
                                    config["text"]["min_frequency"])
        vocab = read_json(output/"assets/vocab.json")
        import numpy as np
        matrix = torch.from_numpy(np.load(output/"assets/embeddings.npy", allow_pickle=False))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(rows, vocab, config["text"]["max_length"], training["batch_size"], True, generator)
    val_loader = make_loader(validation, vocab, config["text"]["max_length"], training["batch_size"])
    model_args = {k: v for k, v in config["model"].items() if k != "embedding_dim"}
    model = BiLSTMClassifier(matrix, **model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    history, start_epoch, best_f1, best_epoch, bad_epochs = [], 1, -1.0, 0, 0
    if previous:
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        history = previous["history"]
        start_epoch, best_f1, best_epoch, bad_epochs = (previous["epoch"]+1, previous["best_f1"],
                                                     previous["best_epoch"], previous["bad_epochs"])
        generator.set_state(previous["loader_rng"])
        random.setstate(previous["python_rng"])
        torch.set_rng_state(previous["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
    run = {"seed": args.seed, "config": config, "fingerprints": fingerprints,
           "git": git_state(), "environment": environment(device),
           "train_rows": len(rows), "val_rows": len(validation), "embedding": asset_report,
           "validation_text": text_stats(validation, vocab, config["text"]["max_length"]),
           "parameters": sum(p.numel() for p in model.parameters()),
           "resumed_from_epoch": start_epoch-1, "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output/f"run_from_epoch_{start_epoch}.json", run)
    print(f"device={device}; rows={len(rows)}; vocab={len(vocab)}; start_epoch={start_epoch}", flush=True)
    for epoch in range(start_epoch, training["max_epochs"]+1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        loss = train_epoch(model, train_loader, optimizer, device, training["gradient_clip"])
        logits, truth = predict(model, val_loader, device)
        result = classification_metrics(truth, logits.argmax(axis=1), config["model"]["classes"])
        val_loss = float(torch.nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(truth)))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        entry = {"epoch": epoch, "train_loss": loss, "val_loss": val_loss,
                 "val_macro_f1": result["macro_f1"], "val_accuracy": result["accuracy"],
                 "epoch_seconds": time.perf_counter()-start,
                 "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device)/1024**2 if device.type == "cuda" else None,
                 "peak_gpu_reserved_mb": torch.cuda.max_memory_reserved(device)/1024**2 if device.type == "cuda" else None}
        history.append(entry)
        improved = entry["val_macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, bad_epochs = entry["val_macro_f1"], epoch, 0
        else:
            bad_epochs += 1
        checkpoint = {"format_version": 1, "seed": args.seed, "config": config,
                      "fingerprints": fingerprints, "vocab": vocab, "labels": labels,
                      "embedding_report": asset_report, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "epoch": epoch,
                      "best_f1": best_f1, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
                      "history": history, "environment": environment(device), "git": run["git"],
                      "loader_rng": generator.get_state(), "python_rng": random.getstate(),
                      "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        if improved:
            save_checkpoint(output/"best.pt", checkpoint)
        save_checkpoint(output/"last.pt", checkpoint)
        write_json(output/"history.json", history)
        print(f"epoch={epoch}; loss={loss:.4f}; val_f1={result['macro_f1']:.4f}; "
              f"val_acc={result['accuracy']:.4f}; seconds={entry['epoch_seconds']:.2f}", flush=True)
        if bad_epochs >= training["patience"]:
            break
    best_checkpoint = torch.load(output/"best.pt", map_location="cpu", weights_only=True)
    best_checkpoint["history"] = history
    save_checkpoint(output/"best.pt", best_checkpoint)
    write_json(output/"summary.json", {"best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "epochs_completed": history[-1]["epoch"], "early_stopped": bad_epochs >= training["patience"],
        "total_epoch_seconds": sum(h["epoch_seconds"] for h in history),
        "history": history, "fingerprints": fingerprints, "git": run["git"], "environment": run["environment"]})


if __name__ == "__main__":
    main()
