"""Deterministic runtime, text batches, provenance and prediction helpers."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from functools import partial
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .common import file_sha256, read_json, read_jsonl
from .text import encode


def configure_runtime(seed, config):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(config["deterministic"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config["deterministic"]


def choose_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def check_rows(rows, classes, prefix, oos=False):
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty split or duplicate IDs")
    for row in rows:
        if set(row) != {"id", "text", "label"}:
            raise ValueError("Text data requires exactly id, text, label")
        if not isinstance(row["text"], str) or not row["id"].startswith(prefix+":"):
            raise ValueError(f"Expected text rows with {prefix}: IDs")
        label = row["label"]
        if not isinstance(label, int) or (label != -1 if oos else not 0 <= label < classes):
            raise ValueError("Invalid labels")


def load_official_split(path, split):
    path = Path(path)
    root = path.parent.parent
    manifest = read_json(root/"manifest.json")
    protocol = read_json("configs/protocol.json")
    if manifest["source_sha256"] != protocol["dataset"]["sha256"]:
        raise ValueError("Unexpected CLINC150 source")
    if manifest["labels_sha256"] != file_sha256(root/"labels.json"):
        raise ValueError("Label mapping checksum mismatch")
    if file_sha256(path) != manifest["files"].get(f"clean/{split}.jsonl"):
        raise ValueError(f"Official {split} checksum mismatch")
    rows = read_jsonl(path)
    check_rows(rows, protocol["dataset"]["classes"], split, split.startswith("oos"))
    return rows, read_json(root/"labels.json")


def collate(rows, vocab, max_length, sample_weights=None):
    sequences = [torch.tensor(encode(r["text"], vocab, max_length), dtype=torch.long) for r in rows]
    batch = (pad_sequence(sequences, batch_first=True, padding_value=0),
            torch.tensor([len(s) for s in sequences]),
            torch.tensor([r["label"] for r in rows], dtype=torch.long))
    if sample_weights is not None:
        batch += (torch.tensor([sample_weights[r['id']] for r in rows], dtype=torch.float32),)
    return batch


def make_loader(rows, vocab, max_length, batch_size, shuffle=False, generator=None, sample_weights=None):
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=0, collate_fn=partial(collate, vocab=vocab, max_length=max_length,
                                                       sample_weights=sample_weights))


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    for tokens, lengths, truth in loader:
        logits.append(model(tokens.to(device), lengths).cpu())
        labels.append(truth)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def environment(device):
    return {"python": __import__("sys").version.split()[0], "numpy": np.__version__,
            "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def numpy_rng_state():
    name, keys, position, has_gauss, cached_gauss = np.random.get_state()
    return {"name": name, "keys": keys.tolist(), "position": position,
            "has_gauss": has_gauss, "cached_gauss": cached_gauss}


def restore_numpy_rng(state):
    np.random.set_state((state["name"], np.asarray(state["keys"], dtype=np.uint32),
                         state["position"], state["has_gauss"], state["cached_gauss"]))


def git_state():
    def run(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"revision": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def save_checkpoint(path, contents):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(contents, temporary)
    temporary.replace(path)
