"""Run a reproducible CE/GCE matrix on one fixed noise protocol."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .common import file_sha256, read_json, write_json


def run_command(command, log_path):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", env=env)
    with Path(log_path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(result.stdout)
    log = result.stdout
    if result.returncode:
        raise RuntimeError(f"Command failed; see {log_path}\n{log}")
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 2026])
    parser.add_argument("--shots", type=int, nargs="+", default=[10])
    parser.add_argument("--noise", choices=["symmetric", "instance"], required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--method", choices=["ce", "gce"], default="ce")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--embeddings-template", help="Path template with {seed} and {shots} for instance noise")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.rate <= 1 or args.epochs < 1:
        raise ValueError("Noise rate must be in [0,1] and epochs positive")
    protocol = read_json("configs/protocol.json")
    if any(seed not in protocol["seeds"] for seed in args.seeds) or any(k not in protocol["shots"] for k in args.shots):
        raise ValueError("Seeds and shots must come from the fixed protocol")
    if args.noise == "instance" and not args.embeddings_template:
        raise ValueError("Instance noise requires --embeddings-template")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Matrix output is nonempty; use a new path")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = Path("configs/gce.json" if args.method == "gce" else "configs/baseline.json")
    records = []
    for shots in args.shots:
        for seed in args.seeds:
            run_id = f"{args.noise}{int(args.rate*100):02d}_k{shots}_s{seed}_{args.method}"
            run_dir = args.output / run_id
            run_dir.mkdir(parents=True)
            clean_train = Path(f"data/processed/fewshot/seed_{seed}/k_{shots}.jsonl")
            if not clean_train.is_file():
                raise FileNotFoundError(clean_train)
            noise_dir = run_dir / "noise"
            noise_command = [sys.executable, "-m", "c04.noise", "--input", str(clean_train),
                             "--kind", args.noise, "--rate", str(args.rate), "--seed", str(seed),
                             "--output", str(noise_dir)]
            if args.noise == "instance":
                embedding_path = Path(args.embeddings_template.format(seed=seed, shots=shots))
                if not embedding_path.is_file():
                    raise FileNotFoundError(embedding_path)
                noise_command.extend(["--embeddings", str(embedding_path)])
            run_command(noise_command, run_dir / "noise.log")
            train_command = [sys.executable, "-m", "c04.train", "--train", str(noise_dir / "train.jsonl"),
                             "--config", str(config), "--seed", str(seed), "--epochs", str(args.epochs),
                             "--device", args.device, "--output", str(run_dir / "training")]
            run_command(train_command, run_dir / "train.log")
            eval_command = [sys.executable, "-m", "c04.evaluate", "--checkpoint", str(run_dir / "training/best.pt"),
                            "--device", args.device, "--output", str(run_dir / "evaluation")]
            run_command(eval_command, run_dir / "evaluate.log")
            summary = read_json(run_dir / "training/summary.json")
            metrics = read_json(run_dir / "evaluation/metrics.json")
            record = {"seed": seed, "shots_per_class": shots, "method": args.method,
                      "noise": args.noise, "rate": args.rate,
                      "clean_train_sha256": file_sha256(clean_train),
                      "noisy_train_sha256": file_sha256(noise_dir / "train.jsonl"),
                      "best_epoch": summary["best_epoch"], "best_val_macro_f1": summary["best_val_macro_f1"],
                      "test_macro_f1": metrics["macro_f1"], "test_accuracy": metrics["accuracy"],
                      "oos_auroc": metrics["oos_auroc"], "total_epoch_seconds": summary["total_epoch_seconds"],
                      "peak_gpu_allocated_mb": max((entry["peak_gpu_memory_mb"] or 0) for entry in summary["history"]),
                      "run_dir": str(run_dir)}
            records.append(record)
            write_json(args.output / "progress.json", records)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["shots_per_class"]].append(record)
    summaries = {}
    for shots, items in grouped.items():
        summaries[str(shots)] = {}
        for metric in ["test_macro_f1", "test_accuracy", "oos_auroc"]:
            values = np.array([item[metric] for item in items])
            summaries[str(shots)][metric] = {"mean": float(values.mean()),
                "std_sample": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n": len(values), "values": [float(value) for value in values]}
    report = {"stage": 3, "method": args.method, "noise": args.noise, "rate": args.rate,
              "seeds": args.seeds, "shots_per_class": args.shots, "epochs_max": args.epochs,
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "dataset_sha256": protocol["dataset"]["sha256"], "records": records,
              "summary_by_shots": summaries, "wall_seconds": time.perf_counter() - started,
              "selection": "validation macro-F1 only"}
    write_json(args.output / "matrix.json", report)
    print(f"Wrote {len(records)} runs to {args.output / 'matrix.json'}")


if __name__ == "__main__":
    main()
