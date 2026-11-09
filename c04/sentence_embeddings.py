"""Export and validate frozen sentence-encoder embeddings for IDN."""

import argparse
from pathlib import Path

import numpy as np

from .common import file_sha256, read_jsonl, write_json


def write_embedding_artifact(path, ids, embeddings, encoder_id, encoder_revision, batch_size, device):
    """Write a non-pickle NPZ with explicit encoder provenance."""
    path = Path(path)
    ids = np.asarray(ids, dtype=str)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if ids.ndim != 1 or embeddings.ndim != 2 or len(ids) != len(embeddings):
        raise ValueError("IDs and embeddings must have matching first dimensions")
    if len(set(ids.tolist())) != len(ids) or not np.isfinite(embeddings).all():
        raise ValueError("Embedding IDs must be unique and vectors finite")
    if not str(encoder_id).strip() or not str(encoder_revision).strip():
        raise ValueError("Encoder ID and immutable revision are required")
    if np.any(np.linalg.norm(embeddings, axis=1) <= 1e-12):
        raise ValueError("Zero vectors are not valid sentence embeddings")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=ids, embeddings=embeddings,
                        encoder_id=np.array(str(encoder_id)),
                        encoder_revision=np.array(str(encoder_revision)))
    manifest = {"path": str(path), "sha256": file_sha256(path), "rows": len(ids),
                "dimension": int(embeddings.shape[1]), "encoder_id": str(encoder_id),
                "encoder_revision": str(encoder_revision), "batch_size": int(batch_size),
                "device": str(device), "format": "npz_no_pickle"}
    write_json(path.with_suffix(".manifest.json"), manifest)
    return manifest


def export_with_sentence_transformers(rows, output, model_id, revision, batch_size, device, max_length):
    """Encode rows with a frozen sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install requirements-idn.txt before exporting real embeddings") from exc
    local_model = Path(model_id).exists()
    model = SentenceTransformer(str(Path(model_id).resolve()) if local_model else model_id,
                                **({} if local_model else {"revision": revision}), device=device)
    model.eval()
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_length
    texts = [row["text"] for row in rows]
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=False)
    resolved_revision = getattr(model[0].auto_model.config, "_commit_hash", None) or revision
    return write_embedding_artifact(output, [row["id"] for row in rows], vectors,
                                    model_id, resolved_revision, batch_size, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--revision", required=True, help="Immutable Hugging Face commit hash")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("Batch size and max length must be positive")
    rows = read_jsonl(args.input)
    if not rows or any(not row["id"].startswith("train:") for row in rows):
        raise ValueError("Embedding export accepts nonempty training rows only")
    manifest = export_with_sentence_transformers(rows, args.output, args.model, args.revision,
                                                 args.batch_size, args.device, args.max_length)
    print(f"Wrote {manifest['rows']} embeddings ({manifest['dimension']}d); SHA256={manifest['sha256']}")


if __name__ == "__main__":
    main()
