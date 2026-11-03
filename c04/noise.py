"""Inject auditable training-only noise; clean truth is never in the model input.

Instance-dependent noise requires *real* frozen sentence-encoder embeddings.
The generator is a study-specific protocol, not a reproduction of a published IDN generator.
"""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np

from .common import file_sha256, read_json, read_jsonl, stable_seed, write_json, write_jsonl


def validate_rows(rows, classes):
    if len(rows) < 2 or classes < 2:
        raise ValueError("At least two training rows and classes are required")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate training IDs")
    if any(not r["id"].startswith("train:") for r in rows):
        raise ValueError("Noise injection accepts train: IDs only, never validation/test/OOS")
    labels = np.array([r["label"] for r in rows], dtype=np.int64)
    if np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError("Invalid in-scope training labels")
    return labels


def align_embeddings(path, rows):
    """NPZ contains ids, embeddings, encoder_id, and exact encoder_revision."""
    with np.load(path, allow_pickle=False) as data:
        needed = {"ids", "embeddings", "encoder_id", "encoder_revision"}
        if not needed.issubset(data.files):
            raise ValueError(f"Embedding NPZ requires {sorted(needed)}")
        ids = data["ids"].astype(str).tolist()
        embeddings = np.asarray(data["embeddings"], dtype=np.float64)
        provenance = {k: str(data[k].item()) for k in ["encoder_id", "encoder_revision"]}
    if len(ids) != len(set(ids)) or embeddings.ndim != 2 or len(ids) != len(embeddings):
        raise ValueError("Invalid embedding IDs or shape")
    if not all(provenance.values()) or not np.isfinite(embeddings).all():
        raise ValueError("Missing encoder provenance or nonfinite vectors")
    positions = {sid: i for i, sid in enumerate(ids)}
    if any(not sid.startswith("train:") for sid in ids):
        raise ValueError("Embedding file must contain only training rows")
    missing = [r["id"] for r in rows if r["id"] not in positions]
    if missing:
        raise ValueError(f"Embeddings missing {len(missing)} selected rows")
    aligned = embeddings[[positions[r["id"]] for r in rows]]
    if np.any(np.linalg.norm(aligned, axis=1) <= 1e-12):
        raise ValueError("Zero embedding vectors")
    return aligned, provenance


def inject(rows, classes, rate, seed, kind="symmetric", embeddings=None, temperature=0.2):
    if not 0 <= rate <= 1:
        raise ValueError("Noise rate must lie in [0, 1]")
    if kind not in {"symmetric", "instance"} or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid noise type or temperature")
    labels = validate_rows(rows, classes)
    # Separate namespaces prevent a change in selection code from changing flip-label draws.
    select_rng = np.random.default_rng(stable_seed(seed, f"noise-selection:{kind}"))
    flip_rng = np.random.default_rng(stable_seed(seed, f"noise-target:{kind}"))
    count = int(math.floor(len(rows) * rate + 0.5))
    probabilities = None
    if kind == "symmetric":
        ordering = select_rng.permutation(len(rows))
        targets = flip_rng.integers(0, classes - 1, len(rows))
        targets = targets + (targets >= labels)
    else:
        if embeddings is None:
            raise ValueError("Instance noise requires actual sentence embeddings; no random fallback")
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] != len(rows) or not np.isfinite(x).all():
            raise ValueError("Invalid aligned embeddings")
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("Zero vectors")
        x = x / norms
        counts = np.bincount(labels, minlength=classes)
        if np.any(counts < 2):
            raise ValueError("IDN leave-one-out centroid requires >=2 clean training rows per class")
        sums = np.zeros((classes, x.shape[1]), dtype=np.float64)
        np.add.at(sums, labels, x)
        centroids = sums / counts[:, None]
        centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        if np.any(centroid_norms <= 1e-12):
            raise ValueError("Degenerate class centroid")
        centroids /= centroid_norms
        similarities = x @ centroids.T
        # Own-class leave-one-out similarity avoids trivially including the query in its own prototype.
        own = (sums[labels] - x) / (counts[labels, None] - 1)
        own_norms = np.linalg.norm(own, axis=1, keepdims=True)
        if np.any(own_norms <= 1e-12):
            raise ValueError("Degenerate leave-one-out centroid")
        own /= own_norms
        own_similarity = np.sum(x * own, axis=1)
        similarities[np.arange(len(rows)), labels] = -np.inf
        ambiguity = (np.max(similarities, axis=1) - own_similarity) / temperature
        # Gumbel top-k is weighted sampling without replacement. More ambiguous instances
        # receive greater corruption propensity; exact total count remains controlled.
        ordering = np.argsort(-(ambiguity + select_rng.gumbel(size=len(rows))), kind="stable")
        scores = similarities / temperature
        scores -= np.max(scores, axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        targets = np.array([flip_rng.choice(classes, p=p) for p in probabilities])
    flipped = np.zeros(len(rows), dtype=bool)
    flipped[ordering[:count]] = True
    observed = np.where(flipped, targets, labels)
    if int(np.sum(observed != labels)) != count:
        raise AssertionError("Noise protocol produced a self flip")
    training = [{"id": r["id"], "text": r["text"], "label": int(observed[i])}
                for i, r in enumerate(rows)]
    audit = [{"id": r["id"], "clean_label": int(labels[i]),
              "observed_label": int(observed[i]), "flipped": bool(flipped[i])}
             for i, r in enumerate(rows)]
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (labels, observed), 1)
    diagnostics = {"requested_rate": rate, "actual_rate": count / len(rows),
                   "changed": count, "total": len(rows),
                   "class_counts_before": np.bincount(labels, minlength=classes).tolist(),
                   "class_counts_after": np.bincount(observed, minlength=classes).tolist(),
                   "transition_counts": matrix.tolist()}
    if probabilities is not None:
        diagnostics["mean_target_entropy"] = float(
            -np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, 1)), axis=1).mean())
    return training, audit, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--kind", choices=["symmetric", "instance"], required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    classes = read_json("configs/protocol.json")["dataset"]["classes"]
    rows = read_jsonl(args.input)
    vectors, provenance = None, None
    if args.kind == "instance":
        if args.embeddings is None:
            parser.error("--embeddings is required for real instance-dependent noise")
        vectors, provenance = align_embeddings(args.embeddings, rows)
    training, audit, diagnostics = inject(rows, classes, args.rate, args.seed,
                                         args.kind, vectors, args.temperature)
    write_jsonl(args.output / "train.jsonl", training)
    write_jsonl(args.output / "audit" / "labels.jsonl", audit)
    write_json(args.output / "audit" / "diagnostics.json", diagnostics)
    manifest = {"kind": args.kind, "rate": args.rate, "seed": args.seed,
                "temperature": args.temperature if args.kind == "instance" else None,
                "source_sha256": file_sha256(args.input),
                "protocol_sha256": file_sha256("configs/protocol.json"),
                "train_sha256": file_sha256(args.output / "train.jsonl"),
                "audit_sha256": file_sha256(args.output / "audit" / "labels.jsonl"),
                "encoder": provenance,
                "embeddings_sha256": file_sha256(args.embeddings) if args.kind == "instance" else None,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "numpy_version": np.__version__}
    write_json(args.output / "manifest.json", manifest)
    print(f"{args.kind}: {diagnostics['changed']}/{diagnostics['total']} flipped; "
          f"actual={diagnostics['actual_rate']:.6f}; train SHA256={manifest['train_sha256']}")


if __name__ == "__main__":
    main()
