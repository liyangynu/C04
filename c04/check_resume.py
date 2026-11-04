"""Compare uninterrupted training with an epoch-boundary stop and resume."""
import argparse
from pathlib import Path
import subprocess
import sys

from .runtime import environment, choose_device
import torch

from .common import file_sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--glove", type=Path, default=Path("data/embeddings/glove.6B.100d.txt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new resume-check output directory")
    args.output.mkdir(parents=True)
    common = [sys.executable, "-m", "c04.train", "--train", str(args.train),
              "--glove", str(args.glove), "--seed", "17", "--device", args.device]
    commands = [common+["--epochs", "2", "--output", str(args.output/"continuous")],
                common+["--epochs", "1", "--output", str(args.output/"resumed")],
                common+["--epochs", "2", "--output", str(args.output/"resumed"), "--resume"]]
    for i, command in enumerate(commands):
        with (args.output/f"step_{i+1}.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
        print((args.output/f"step_{i+1}.log").read_text(encoding="utf-8"), flush=True)
        if result.returncode:
            raise RuntimeError(f"Resume check step {i+1} failed")
    a = torch.load(args.output/"continuous/last.pt", map_location="cpu", weights_only=True)
    b = torch.load(args.output/"resumed/last.pt", map_location="cpu", weights_only=True)
    parameters_match = all(torch.equal(t, b["model"][name]) for name, t in a["model"].items())
    keys = ["epoch", "train_loss", "val_loss", "val_macro_f1", "val_accuracy"]
    metrics_match = all(all(x[k] == y[k] for k in keys) for x, y in zip(a["history"], b["history"]))
    report = {"train_sha256": file_sha256(args.train), "seed": 17, "epochs": 2,
              "environment": environment(choose_device(args.device)),
              "parameters_bitwise_equal": parameters_match,
              "epoch_metrics_exactly_equal": metrics_match,
              "continuous_history": [{k: row[k] for k in keys} for row in a["history"]],
              "resumed_history": [{k: row[k] for k in keys} for row in b["history"]]}
    write_json("reports/stage2_resume_check.json", report)
    if not (parameters_match and metrics_match):
        raise RuntimeError("Resume differs from uninterrupted training; see reports/stage2_resume_check.json")
    print("PASS: all model parameters and per-epoch metric values match exactly.")


if __name__ == "__main__":
    main()
