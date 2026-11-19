"""Record local evidence hashes and measured training costs, including interrupted runs."""
import argparse
from pathlib import Path

from .common import file_sha256, read_json, write_json


def inventory(output):
    root = Path("artifacts")
    measured = []
    incomplete = []
    for training in sorted(root.rglob("training")):
        if not training.is_dir():
            continue
        summary = training / "summary.json"
        if not summary.exists():
            histories = training / "history.json"
            if histories.exists():
                history = read_json(histories)
                incomplete.append({"training": training.as_posix(), "completed_epochs_with_history": len(history),
                                   "recorded_epoch_seconds": sum(h["epoch_seconds"] for h in history)})
            continue
        report = read_json(summary)
        if "history" not in report:
            continue
        measured.append({"training": training.as_posix(), "summary_sha256": file_sha256(summary),
                         "completed_epochs": report["epochs_completed"],
                         "epoch_seconds": report["total_epoch_seconds"],
                         "peak_reserved_mb": max(h.get("peak_gpu_reserved_mb") or 0 for h in report["history"]),
                         "interrupted_archive": "interrupted" in training.as_posix()})
    cnn = []
    for metrics in sorted(Path("artifacts/public_cnn_v1").glob("seed_*/metrics.json")):
        row = read_json(metrics)
        cnn.append({"run": metrics.parent.as_posix(), "epoch_seconds": row["total_epoch_seconds"],
                    "peak_reserved_mb": row["peak_reserved_mb"], "wall_seconds": row["wall_seconds"]})
    evidence = []
    for directory in (Path("artifacts/completion_v1"), Path("artifacts/final_acceptance_v1"),
                      Path("artifacts/public_svm_v1"), Path("artifacts/public_svm_validation_v2")):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and (path.suffix in {".json", ".jsonl", ".csv", ".npz", ".pt", ".joblib", ".png", ".log"}):
                evidence.append({"file": path.as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    total = sum(r["epoch_seconds"] for r in measured + cnn) + sum(r["recorded_epoch_seconds"] for r in incomplete)
    peak = max([r["peak_reserved_mb"] for r in measured + cnn], default=0)
    report = {"completed_bilstm_training_directories": len(measured), "training": measured,
              "incomplete_histories": incomplete, "cnn": cnn, "recorded_epoch_seconds_total": total,
              "recorded_epoch_hours_total": total / 3600, "peak_recorded_pytorch_reserved_mb": peak,
              "evidence": evidence, "evidence_files": len(evidence),
              "scope": "All retained local BiLSTM training summaries, incomplete epoch histories and CNN v1; includes exploratory/diagnostic/repeated fits where retained.",
              "limitations": ["Epoch seconds include validation and exclude evaluation, model/asset loading, installation, download and idle time.",
                              "This is not complete elapsed project wall-clock time; unknown pre-history interruption costs cannot be reconstructed.",
                              "GPU numbers are PyTorch reserved peaks, not whole-device VRAM. One RTX3060 Laptop 6GB was used.",
                              "No evidence hashes can substitute for the separately recorded inference replay and retraining checks."],
              "single_device_hardware_gb": 6, "hardware_within_12gb": True,
              "recorded_epoch_sum_within_48h": total <= 48 * 3600}
    write_json(output, report)
    print(f"Inventory: {len(evidence)} evidence files; {len(measured)} completed BiLSTM fits; recorded epochs {total/3600:.2f}h")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiment_inventory.json"))
    args = parser.parse_args()
    inventory(args.output)


if __name__ == "__main__":
    main()
