"""Run or resume the frozen stage-5 expanded comparison matrix."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

from .common import file_sha256, read_json, read_jsonl, write_json
from .formal_protocol import SEEDS
from .run_noise_matrix import run_command
from .runtime import git_state


def key(record):
    return (record["noise"], record["rate"], record["shots_per_class"], record["seed"], record["method"])


def metric_summary(records):
    result = {}
    for metric in ("test_macro_f1", "test_accuracy", "oos_auroc", "f1_retention"):
        values = [r[metric] for r in records]
        result[metric] = {"mean": float(np.mean(values)),
                          "std_sample": float(np.std(values, ddof=1)),
                          "n": len(values), "values": values}
    return result


def released_records(noise):
    root = Path(f"reports/stage3/formal_{'idn' if noise == 'instance' else 'sym'}10_v2")
    source = read_json(root / "comparison.json")
    result = []
    for item in source["records"]:
        method = "confident_learning" if item["method"] == "cl_ce" else item["method"]
        result.append({"noise": noise, "rate": .3, "shots_per_class": 10,
                       "seed": item["seed"], "method": method,
                       "test_macro_f1": item["test_macro_f1"],
                       "test_accuracy": item["test_accuracy"], "oos_auroc": item["oos_auroc"],
                       "f1_retention": item["f1_retention"],
                       "route_wall_seconds": item["route_wall_seconds"],
                       "oof_wall_seconds": item["oof_wall_seconds"],
                       "reused_from": str(root / "comparison.json"), "reused": True})
    return result


def fit(method, seed, shots, train_path, run_dir, device):
    started = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = Path("configs/formal_gce.json" if method == "gce" else "configs/formal_ce.json")
    oof_seconds, oof_epoch_seconds, kept = 0.0, 0.0, len(read_jsonl(train_path))
    final_train = Path(train_path)
    if method == "confident_learning":
        oof_dir = run_dir / "oof"
        report_path = oof_dir / "cleaning_report.json"
        if report_path.is_file():
            cleaning = read_json(report_path)
            if cleaning["input_sha256"] != file_sha256(train_path) or not cleaning["coverage_exactly_once"]:
                raise ValueError(f"Existing OOF output does not match current input: {oof_dir}")
        else:
            run_command([sys.executable, "-m", "c04.run_confident_learning", "--train", str(train_path),
                         "--config", str(config), "--seed", str(seed), "--folds", "5", "--device", device,
                         "--output", str(oof_dir)], run_dir / "oof.log")
        cleaning = read_json(oof_dir / "cleaning_report.json")
        oof_seconds = cleaning["wall_seconds"]
        oof_epoch_seconds = sum(f["total_epoch_seconds"] for f in cleaning["fold_records"])
        kept = cleaning["rows_kept"]
        final_train = oof_dir / "cleaned_train.jsonl"

    run_command([sys.executable, "-m", "c04.train", "--train", str(final_train), "--config", str(config),
                 "--seed", str(seed), "--device", device, "--output", str(run_dir / "training")],
                run_dir / "train.log")
    run_command([sys.executable, "-m", "c04.evaluate", "--checkpoint", str(run_dir / "training/best.pt"),
                 "--device", device, "--output", str(run_dir / "evaluation")], run_dir / "evaluate.log")
    training = read_json(run_dir / "training/summary.json")
    metrics = read_json(run_dir / "evaluation/metrics.json")
    history = training["history"]
    return {"seed": seed, "shots_per_class": shots, "method": method,
            "train_sha256": file_sha256(final_train), "config_sha256": file_sha256(config),
            "run_dir": str(run_dir), "best_epoch": training["best_epoch"],
            "epochs_completed": training["epochs_completed"],
            "best_val_macro_f1": training["best_val_macro_f1"],
            "test_macro_f1": metrics["macro_f1"], "test_accuracy": metrics["accuracy"],
            "oos_auroc": metrics["oos_auroc"], "rows_kept": kept,
            "oof_wall_seconds": oof_seconds, "oof_epoch_seconds": oof_epoch_seconds,
            "final_epoch_seconds": training["total_epoch_seconds"],
            "route_wall_seconds": time.perf_counter() - started,
            "peak_final_allocated_mb": max((h["peak_gpu_memory_mb"] or 0) for h in history),
            "peak_final_reserved_mb": max((h["peak_gpu_reserved_mb"] or 0) for h in history),
            "reused": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/stage5_expanded/matrix.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage5_expanded_runs"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--pilot-only", action="store_true", help="Run one new 10-shot symmetric-10% CE fit")
    args = parser.parse_args()
    source = read_json(args.source)
    if source["protocol_sha256"] != file_sha256("configs/expanded_matrix_v1.json"):
        raise ValueError("The matrix does not match the frozen expanded protocol")
    if args.output.exists() and (args.output / "progress.json").exists():
        records = read_json(args.output / "progress.json")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        records = released_records("instance") + released_records("symmetric")
        write_json(args.output / "progress.json", records)

    clean_records = []
    existing_clean = {(r["shots_per_class"], r["seed"]) for r in records if r.get("method") == "clean_ce"}
    for shots in (() if args.pilot_only else (5, 20, 100)):
        for seed in SEEDS:
            k = ("clean", 0.0, shots, seed, "clean_ce")
            if (shots, seed) in existing_clean or any(key(r) == k for r in records):
                continue
            clean_path = Path(f"data/processed/fewshot/seed_{seed}/k_{shots}.jsonl")
            dest = args.output / "clean_ce" / f"k{shots}_s{seed}"
            record = fit("clean_ce", seed, shots, clean_path, dest, args.device)
            record.update({"noise": "clean", "rate": 0.0, "f1_retention": 1.0,
                           "clean_train_sha256": file_sha256(clean_path)})
            clean_records.append(record)
            records.append(record)
            write_json(args.output / "progress.json", records)
            print(f"clean_ce k={shots} seed={seed}: F1={record['test_macro_f1']:.4f}", flush=True)

    clean_f1 = {(r["shots_per_class"], r["seed"]): r["test_macro_f1"]
                for r in records if r.get("method") == "clean_ce"}
    # The 10-shot clean references are already part of the released comparisons.
    for noise in ("instance", "symmetric"):
        for condition in sorted({(r["shots_per_class"], r["rate"]) for r in source["records"]
                                 if r["noise"] == noise}):
            shots, rate = condition
            for seed in SEEDS:
                source_row = next(r for r in source["records"] if r["noise"] == noise and
                                  r["shots_per_class"] == shots and r["rate"] == rate and r["seed"] == seed)
                for method in ("ce", "gce", "confident_learning"):
                    ident = (noise, rate, shots, seed, method)
                    if any(key(r) == ident for r in records):
                        continue
                    if args.pilot_only and not (noise == "symmetric" and rate == .1 and shots == 10 and
                                                seed == 17 and method == "ce"):
                        continue
                    if args.pilot_only and any(r.get("pilot") for r in records):
                        continue
                    dest = args.output / f"{noise}_r{int(rate*100):02d}" / method / f"k{shots}_s{seed}"
                    started = time.perf_counter()
                    record = fit(method, seed, shots, Path(source_row["train_path"]), dest, args.device)
                    record["route_wall_seconds"] = time.perf_counter() - started
                    record.update({"noise": noise, "rate": rate,
                                   "observed_sha256": source_row["train_sha256"],
                                   "clean_train_sha256": source_row["clean_sha256"],
                                   "f1_retention": record["test_macro_f1"] / clean_f1[(shots, seed)],
                                   "pilot": bool(args.pilot_only)})
                    records.append(record)
                    write_json(args.output / "progress.json", records)
                    print(f"{noise} r={rate:.1f} k={shots} {method} seed={seed}: "
                          f"F1={record['test_macro_f1']:.4f}; AUROC={record['oos_auroc']:.4f}", flush=True)
                    if args.pilot_only:
                        break
                if args.pilot_only and any(r.get("pilot") for r in records):
                    break
            if args.pilot_only and any(r.get("pilot") for r in records):
                break
        if args.pilot_only and any(r.get("pilot") for r in records):
            break

    summary = {"source_sha256": file_sha256(args.source), "protocol_sha256": source["protocol_sha256"],
               "records": records, "git": git_state(),
               "status": "pilot" if args.pilot_only else "partial_or_complete"}
    write_json(args.output / "comparison.json", summary)
    print(f"Saved {len(records)} progress records to {args.output / 'comparison.json'}")


if __name__ == "__main__":
    main()
