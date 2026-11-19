"""Prepare the frozen stage-5 noise inputs without starting model training."""
import argparse
from pathlib import Path

import numpy as np

from .common import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from .noise import align_embeddings, inject


def conditions():
    result = [(10, "symmetric", rate) for rate in (.1, .2, .3)]
    result += [(shots, "symmetric", .3) for shots in (5, 20, 100)]
    result += [(shots, "instance", .3) for shots in (5, 10, 20, 100)]
    return result


def align_frozen_embeddings(path, rows, instance_protocol):
    vectors, provenance = align_embeddings(path, rows)
    for field in ("encoder_id", "encoder_revision"):
        if provenance[field] != instance_protocol[field]:
            raise ValueError(f"Embedding {field} differs from the frozen stage-5 protocol")
    return vectors, provenance


def prepare(output, embeddings_template):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output already exists; choose a new path: {output}")
    protocol = read_json("configs/expanded_matrix_v1.json")
    dataset = read_json("configs/protocol.json")["dataset"]
    released_idn = read_json("reports/stage3/formal_idn10_v2/comparison.json")
    seeds = protocol["seeds"]
    embedding_info = protocol["instance_noise"]
    records = []

    for shots, kind, rate in conditions():
        for seed in seeds:
            # Reused labels have no new vector artifact; never inherit the previous iteration.
            vectors, provenance, vector_path = None, None, None
            clean_path = Path(f"data/processed/fewshot/seed_{seed}/k_{shots}.jsonl")
            clean_rows = read_jsonl(clean_path)
            if kind == "instance" and shots == 10:
                source = Path(f"data/formal_idn10_v2/seed_{seed}/noise/train.jsonl")
                if not source.is_file():
                    raise FileNotFoundError(source)
                released = next(r for r in released_idn["records"] if r["method"] == "ce" and r["seed"] == seed)
                if file_sha256(source) != released["observed_sha256"]:
                    raise ValueError(f"Released IDN10 labels changed: seed={seed}")
                observed_rows = read_jsonl(source)
                clean_ids = [row["id"] for row in clean_rows]
                if [row["id"] for row in observed_rows] != clean_ids:
                    raise ValueError(f"Existing IDN10 input IDs changed: seed={seed}")
                clean_labels = [int(row["label"]) for row in clean_rows]
                audit_rows = [{"id": row["id"], "clean_label": int(clean_labels[i]),
                               "observed_label": int(row["label"]),
                               "flipped": int(clean_labels[i]) != int(row["label"])}
                              for i, row in enumerate(observed_rows)]
                noise_dir = output / kind / f"k{shots}_r{int(rate*100):02d}" / f"seed_{seed}"
                noise_dir.mkdir(parents=True)
                write_jsonl(noise_dir / "train.jsonl", observed_rows)
                write_jsonl(noise_dir / "audit" / "labels.jsonl", audit_rows)
                diagnostics = {"requested_rate": rate, "changed": sum(r["flipped"] for r in audit_rows),
                               "total": len(audit_rows), "actual_rate": sum(r["flipped"] for r in audit_rows) / len(audit_rows),
                               "source": "released stage3 IDN10 OOF labels"}
                write_json(noise_dir / "audit" / "diagnostics.json", diagnostics)
            else:
                if kind == "instance":
                    if not embeddings_template:
                        raise ValueError("--embeddings-template is required for non-10-shot IDN")
                    vector_path = Path(embeddings_template.format(seed=seed, shots=shots))
                    vectors, provenance = align_frozen_embeddings(vector_path, clean_rows, embedding_info)
                observed_rows, audit_rows, diagnostics = inject(
                    clean_rows, dataset["classes"], rate, seed, kind, vectors,
                    embedding_info["temperature"])
                noise_dir = output / kind / f"k{shots}_r{int(rate*100):02d}" / f"seed_{seed}"
                write_jsonl(noise_dir / "train.jsonl", observed_rows)
                write_jsonl(noise_dir / "audit" / "labels.jsonl", audit_rows)
                write_json(noise_dir / "audit" / "diagnostics.json", diagnostics)

            expected = int(np.floor(len(clean_rows) * rate + .5))
            actual = sum(int(row["clean_label"]) != int(row["observed_label"]) for row in audit_rows)
            if actual != expected:
                raise AssertionError(f"Wrong flip count for {kind}/k{shots}/r{rate}/s{seed}: {actual} != {expected}")
            records.append({"shots_per_class": shots, "noise": kind, "rate": rate,
                            "seed": seed, "clean_path": str(clean_path),
                            "clean_sha256": file_sha256(clean_path),
                            "train_path": str(noise_dir / "train.jsonl"),
                            "train_sha256": file_sha256(noise_dir / "train.jsonl"),
                            "audit_path": str(noise_dir / "audit/labels.jsonl"),
                            "audit_sha256": file_sha256(noise_dir / "audit/labels.jsonl"),
                            "flipped": actual, "total": len(clean_rows),
                            "embedding_path": str(vector_path) if vector_path else None,
                            "embedding_sha256": file_sha256(vector_path) if vector_path else None,
                            "encoder": provenance if kind == "instance" else None,
                            "reuse_source_sha256": file_sha256(source) if kind == "instance" and shots == 10 else None,
                            "reuse_existing_10shot_results": shots == 10 and rate == .3})

    matrix = {"schema_version": 1, "protocol": "configs/expanded_matrix_v1.json",
              "protocol_sha256": file_sha256("configs/expanded_matrix_v1.json"),
              "dataset_sha256": dataset["sha256"], "seeds": seeds,
              "records": records}
    write_json(output / "matrix.json", matrix)
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/stage5_expanded"))
    parser.add_argument("--embeddings-template", help="NPZ path template containing {seed} and {shots}")
    args = parser.parse_args()
    matrix = prepare(args.output, args.embeddings_template)
    print(f"Prepared {len(matrix['records'])} matched noise inputs in {args.output}")
    print("Every input is paired by clean subset and seed; audit labels remain separate from train.jsonl")


if __name__ == "__main__":
    main()
