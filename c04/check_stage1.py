"""Verify official data and materialize the symmetric-noise grid (no model training)."""
from collections import Counter
from datetime import datetime, timezone
import math
from pathlib import Path
import platform
import sys

import numpy as np

from .common import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from .noise import inject
from .prepare import nested_sample, validate_raw


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    protocol = read_json("configs/protocol.json")
    root = Path("data/processed")
    manifest = read_json(root/"manifest.json")
    require(manifest["source_sha256"] == file_sha256("data/raw/data_full.json"), "Raw checksum mismatch")
    require(manifest["protocol_sha256"] == file_sha256("configs/protocol.json"), "Protocol changed; rerun prepare")
    validate_raw(read_json("data/raw/data_full.json"), protocol["dataset"])
    for name, checksum in manifest["files"].items():
        require(file_sha256(root/name) == checksum, f"Prepared file modified: {name}")
    protected = {p.name: file_sha256(p) for p in (root/"clean").glob("*.jsonl")}
    full = read_jsonl(root/"clean/train.jsonl")
    generated = []
    for seed in protocol["seeds"]:
        previous = set()
        for k in protocol["shots"]:
            source = root/f"fewshot/seed_{seed}/k_{k}.jsonl"
            rows = read_jsonl(source)
            require(rows == nested_sample(full, k, seed), "Sampling replay mismatch")
            ids = {r["id"] for r in rows}
            require(previous <= ids, "Few-shot subsets not nested")
            require(set(Counter(r["label"] for r in rows).values()) == {k}, "Unbalanced few-shot sample")
            previous = ids
            previous_flips = set()
            for rate in protocol["noise_rates"]:
                a, audit, diag = inject(rows, 150, rate, seed)
                replay, replay_audit, replay_diag = inject(rows, 150, rate, seed)
                require((a, audit, diag) == (replay, replay_audit, replay_diag), "Noise replay mismatch")
                flips = {r["id"] for r in audit if r["flipped"]}
                require(len(flips) == int(math.floor(len(rows) * rate + 0.5)), "Wrong noise count")
                require(all(set(r) == {"id", "text", "label"} for r in a), "Audit truth leaked into training input")
                require(previous_flips <= flips, "Corruption masks not nested")
                previous_flips = flips
                dest = Path(f"data/noisy/symmetric/seed_{seed}/k_{k}/r_{rate:.1f}")
                write_jsonl(dest/"train.jsonl", a)
                write_jsonl(dest/"audit/labels.jsonl", audit)
                write_json(dest/"audit/diagnostics.json", diag)
                record = {"seed": seed, "shots": k, "rate": rate, "kind": "symmetric",
                          "changed": diag["changed"], "total": len(rows),
                          "train_sha256": file_sha256(dest/"train.jsonl"),
                          "source_sha256": file_sha256(source),
                          "protocol_sha256": file_sha256("configs/protocol.json"),
                          "audit_sha256": file_sha256(dest/"audit/labels.jsonl"),
                          "numpy_version": np.__version__}
                write_json(dest/"manifest.json", record)
                generated.append(record)
    require(protected == {p.name: file_sha256(p) for p in (root/"clean").glob("*.jsonl")},
            "Clean or evaluation files changed during injection")
    report = {"stage": 1, "scope": "real_CLINC150_data_pipeline_only_no_model_results",
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
              "dataset_revision": protocol["dataset"]["revision"],
              "source_sha256": manifest["source_sha256"],
              "counts": manifest["counts"], "normalized_text_overlap": manifest["normalized_text_overlap"],
              "checks": {"official_counts_and_labels": True, "nested_fewshot": True,
                         "deterministic_replay": True, "exact_noise_counts": True,
                         "clean_splits_unchanged": True, "audit_truth_separate": True},
              "symmetric_datasets": generated,
              "not_yet_run": ["real_sentence_encoder_IDN", "BiLSTM_training", "three_seed_model_evaluation"]}
    write_json("reports/stage1_validation.json", report)
    print(f"PASS: official counts, nested subsets, deterministic replay, {len(generated)} noise datasets; no model metrics claimed.")


if __name__ == "__main__":
    main()
