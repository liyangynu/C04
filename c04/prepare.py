"""Download/import pinned CLINC150 and create nested, clean few-shot splits.

Usage: python -m c04.prepare [--source local_data_full.json]
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import random
import urllib.request

from .common import file_sha256, read_json, stable_seed, write_json, write_jsonl


def verify_source_checksum(source, expected):
    checksum = file_sha256(source)
    if checksum != expected["sha256"]:
        raise ValueError(f"Dataset SHA256 mismatch: expected {expected['sha256']}, got {checksum}")
    return checksum


def validate_raw(raw, expected):
    if set(raw) != set(expected["split_counts"]):
        raise ValueError(f"Unexpected split fields: {sorted(raw)}")
    for split, count in expected["split_counts"].items():
        if len(raw[split]) != count:
            raise ValueError(f"{split}: expected {count}, got {len(raw[split])}")
        if any(not isinstance(r, list) or len(r) != 2 or
               not all(isinstance(x, str) and x.strip() for x in r) for r in raw[split]):
            raise ValueError(f"Invalid records in {split}")
    labels = sorted({r[1] for r in raw["train"]})
    if len(labels) != expected["classes"] or "oos" in labels:
        raise ValueError("Wrong in-scope label set")
    for split, per_class in [("train", 100), ("val", 20), ("test", 30)]:
        counts = Counter(r[1] for r in raw[split])
        if set(counts) != set(labels) or set(counts.values()) != {per_class}:
            raise ValueError(f"Nonstandard per-class counts in {split}")
    for split in ["oos_train", "oos_val", "oos_test"]:
        if {r[1] for r in raw[split]} != {"oos"}:
            raise ValueError(f"Invalid OOS labels in {split}")
    return labels


def nested_sample(rows, shots, seed):
    groups = {}
    for row in rows:
        groups.setdefault(row["label"], []).append(row)
    selected = []
    for label in sorted(groups):
        group = sorted(groups[label], key=lambda x: x["id"])
        random.Random(stable_seed(seed, f"sampling:{label}")).shuffle(group)
        if shots > len(group):
            raise ValueError(f"Requested {shots} rows for label {label}, only {len(group)} available")
        selected.extend(group[:shots])
    return sorted(selected, key=lambda x: x["id"])


def prepare(source, output, protocol):
    source_checksum = verify_source_checksum(source, protocol["dataset"])
    raw = read_json(source)
    names = validate_raw(raw, protocol["dataset"])
    label_to_id = {name: i for i, name in enumerate(names)}
    output = Path(output)
    write_json(output / "labels.json", label_to_id)
    canonical = {}
    for split, pairs in raw.items():
        canonical[split] = [
            {"id": f"{split}:{i:05d}", "text": text,
             "label": -1 if name == "oos" else label_to_id[name]}
            for i, (text, name) in enumerate(pairs)
        ]
        write_jsonl(output / "clean" / f"{split}.jsonl", canonical[split])
    for seed in protocol["seeds"]:
        for shots in protocol["shots"]:
            write_jsonl(output / "fewshot" / f"seed_{seed}" / f"k_{shots}.jsonl",
                        nested_sample(canonical["train"], shots, seed))
    # Text duplicates exist in some corpora. Report them; do not silently alter official splits.
    norm = lambda text: " ".join(text.lower().split())
    sets = {s: {norm(r["text"]) for r in canonical[s]} for s in ["train", "val", "test"]}
    overlap = {f"{a}__{b}": len(sets[a] & sets[b])
               for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    manifest = {
        "dataset": protocol["dataset"], "source_sha256": source_checksum,
        "source_path": str(Path(source).resolve()),
        "labels_sha256": file_sha256(output / "labels.json"),
        "protocol_sha256": file_sha256("configs/protocol.json"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {s: len(rows) for s, rows in canonical.items()},
        "normalized_text_overlap": overlap,
        "files": {str(p.relative_to(output)).replace("\\", "/"): file_sha256(p)
                  for p in sorted(output.rglob("*.jsonl"))},
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Import a local official data_full.json without networking")
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    protocol = read_json("configs/protocol.json")
    spec = protocol["dataset"]
    if args.source is None:
        source = Path("data/raw/data_full.json")
        provenance = source.with_suffix(".provenance.json")
        url = f"https://raw.githubusercontent.com/clinc/oos-eval/{spec['revision']}/{spec['path']}"
        if source.exists():
            if not provenance.exists():
                raise ValueError("Cached file has no provenance; import explicitly with --source or remove cache")
            cached = read_json(provenance)
            if cached.get("revision") != spec["revision"] or cached.get("sha256") != file_sha256(source):
                raise ValueError("Cached dataset does not match pinned provenance")
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "C04-reproducibility/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            source.parent.mkdir(parents=True, exist_ok=True)
            temp = source.with_suffix(".download")
            temp.write_bytes(payload)
            verify_source_checksum(temp, spec)
            validate_raw(read_json(temp), spec)
            temp.replace(source)
            write_json(provenance, {"url": url, "revision": spec["revision"],
                "downloaded_utc": datetime.now(timezone.utc).isoformat(), "sha256": file_sha256(source)})
    else:
        source = args.source
    result = prepare(source, args.output, protocol)
    result["source_mode"] = "verified_local_import" if args.source else "pinned_download"
    write_json(args.output / "manifest.json", result)
    print(f"Validated {len(result['files'])} prepared files; SHA256={result['source_sha256']}")


if __name__ == "__main__":
    main()
