"""Replay the pinned local MiniLM encoder and reconstruct historical noise labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .common import file_sha256, read_json, read_jsonl, write_json
from .noise import align_embeddings, inject


def rows_digest(rows):
    return hashlib.sha256("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")).hexdigest()


def replay(output, device):
    from sentence_transformers import SentenceTransformer
    protocol = read_json("configs/idn_encoder.json")
    model_root = Path("tmp/hf_all-MiniLM-L6-v2")
    expected = read_json("reports/stage3_idn_asset_pilot.json")["encoder"]["model_safetensors_sha256"]
    if file_sha256(model_root / "model.safetensors") != expected:
        raise ValueError("Pinned MiniLM weights changed")
    model = SentenceTransformer(str(model_root.resolve()), device=device)
    model.eval()
    model.max_seq_length = protocol["max_length"]
    matrix = read_json("data/stage5_expanded/matrix.json")
    additional = read_json("artifacts/completion_v1/inputs/matrix.json")
    records = []
    for shots in (5, 10, 20, 100):
        for seed in (17, 42, 2026):
            rows = read_jsonl(f"data/processed/fewshot/seed_{seed}/k_{shots}.jsonl")
            path = (Path(f"artifacts/stage3_pilot/idn_embeddings_s{seed}_k10.npz") if shots == 10 else
                    Path(f"artifacts/stage5_embeddings/idn_s{seed}_k{shots}.npz"))
            original, provenance = align_embeddings(path, rows)
            if provenance["encoder_revision"] != protocol["revision"]:
                raise ValueError("Historical encoder revision differs")
            vectors = model.encode([r["text"] for r in rows], batch_size=64,
                                   convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
            error = float(np.max(np.abs(vectors - original)))
            if error > 1e-5:
                raise ValueError(f"Encoder replay differs: k={shots}, seed={seed}, error={error}")
            inputs = [r for r in matrix["records"] + additional["records"] if
                      r["noise"] == "instance" and r["shots_per_class"] == shots and r["seed"] == seed]
            verified_rates = []
            for source in inputs:
                regenerated, _, _ = inject(rows, 150, source["rate"], seed, "instance", vectors, .2)
                if rows_digest(regenerated) != source["train_sha256"]:
                    raise ValueError(f"Replayed encoder changes historical IDN labels: {source}")
                verified_rates.append(source["rate"])
            records.append({"shots_per_class": shots, "seed": seed, "rows": len(rows),
                            "max_vector_error": error, "source_sha256": file_sha256(path),
                            "noise_rates_reconstructed": verified_rates})
            print(f"Encoder replay k={shots} seed={seed}: error={error:.3g}; rates={verified_rates}", flush=True)
    write_json(output, {"all_passed": True, "model_weights_sha256": expected, "revision": protocol["revision"],
                       "device": device, "records": records, "total_rows_encoded": sum(r["rows"] for r in records)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/encoder_replay.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    replay(args.output, args.device)


if __name__ == "__main__":
    main()
