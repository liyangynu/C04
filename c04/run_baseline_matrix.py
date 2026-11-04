"""Run the clean CE baseline across requested seeds and data sizes."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .common import file_sha256, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 2026])
    parser.add_argument("--shots", type=int, nargs="+", default=[5, 10, 20, 100])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.shots)) != len(args.shots):
        raise ValueError("Duplicate seeds or sample sizes are not independent experiments")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Matrix output is nonempty; use a new path")
    protocol = read_json("configs/protocol.json")
    if any(seed not in protocol["seeds"] for seed in args.seeds):
        raise ValueError("Seeds must come from the fixed experiment protocol")
    if any(k not in protocol["shots"] for k in args.shots):
        raise ValueError("Sample sizes must come from the fixed experiment protocol")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for shots in args.shots:
        for seed in args.seeds:
            run_id = f"k{shots}_s{seed}"
            training = Path(f"data/processed/fewshot/seed_{seed}/k_{shots}.jsonl")
            if not training.is_file():
                raise FileNotFoundError(training)
            run_dir = args.output/run_id
            run_dir.mkdir(parents=True)
            common = [sys.executable, "-m", "c04.train", "--train", str(training),
                      "--seed", str(seed), "--epochs", str(args.epochs),
                      "--device", args.device, "--output", str(run_dir/"training")]
            for label, command in [("train", common),
                                   ("evaluate", [sys.executable, "-m", "c04.evaluate",
                                    "--checkpoint", str(run_dir/"training/best.pt"),
                                    "--device", args.device, "--output", str(run_dir/"evaluation")])]:
                with (run_dir/f"{label}.log").open("w", encoding="utf-8") as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                log = (run_dir/f"{label}.log").read_text(encoding="utf-8")
                print(f"[{run_id}] {label}\n{log}", flush=True)
                if result.returncode:
                    raise RuntimeError(f"{run_id} {label} failed; see {run_dir}/{label}.log")
            train_summary = read_json(run_dir/"training/summary.json")
            metrics = read_json(run_dir/"evaluation/metrics.json")
            record = {"seed": seed, "shots_per_class": shots,
                "train_sha256": file_sha256(training), "best_epoch": train_summary["best_epoch"],
                "best_val_macro_f1": train_summary["best_val_macro_f1"],
                "test_macro_f1": metrics["macro_f1"], "test_accuracy": metrics["accuracy"],
                "oos_auroc": metrics["oos_auroc"],
                "total_epoch_seconds": train_summary["total_epoch_seconds"],
                "peak_gpu_allocated_mb": max((e["peak_gpu_memory_mb"] or 0)
                    for e in train_summary["history"]),
                "run_dir": str(run_dir)}
            records.append(record)
            write_json(args.output/"progress.json", records)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["shots_per_class"]].append(record)
    summaries = {}
    for shots, items in grouped.items():
        summaries[str(shots)] = {}
        for metric in ["best_val_macro_f1", "test_macro_f1", "test_accuracy", "oos_auroc"]:
            values = np.array([x[metric] for x in items])
            summaries[str(shots)][metric] = {"mean": float(values.mean()),
                "std_sample": float(values.std(ddof=1)) if len(values)>1 else 0.0,
                "n": len(values), "values": [float(v) for v in values]}
    report = {"stage": 2, "method": "clean_cross_entropy", "seeds": args.seeds,
        "shots_per_class": args.shots, "epochs_max": args.epochs, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": protocol["dataset"]["sha256"], "glove": read_json("configs/glove.json"),
        "baseline_config_sha256": file_sha256("configs/baseline.json"),
        "records": records, "summary_by_shots": summaries,
        "wall_seconds": time.perf_counter()-started,
        "interpretation": "The CLINC150 paper reports GloVe-CNN accuracy and OOS recall, not this BiLSTM-Attention macro-F1 or MSP-AUROC configuration; results are reported as a project baseline.",
        "model_metrics_used_for_selection": "validation macro-F1 only"}
    write_json(args.output/"matrix.json", report)
    write_json("reports/stage2/baseline_matrix.json", report)
    print(f"Wrote {len(records)} runs to reports/stage2/baseline_matrix.json")


if __name__ == "__main__":
    main()
