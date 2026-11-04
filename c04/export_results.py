"""Export compact per-seed predictions, metrics and plots without model weights."""
import argparse
from pathlib import Path
import shutil

from .common import file_sha256, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Export output is nonempty")
    matrix = read_json(args.matrix)
    args.output.mkdir(parents=True, exist_ok=True)
    for record in matrix["records"]:
        run = Path(record["run_dir"])
        name = f"k{record['shots_per_class']}_s{record['seed']}"
        dest = args.output/name
        dest.mkdir()
        for file in ["metrics.json", "predictions.jsonl", "per_class.csv", "confusion_matrix.csv",
                     "learning_curves.png", "confusion_matrix.png", "oos_roc.png"]:
            shutil.copyfile(run/"evaluation"/file, dest/file)
        shutil.copyfile(run/"training/summary.json", dest/"training_summary.json")
        shutil.copyfile(run/"training/run_from_epoch_1.json", dest/"run.json")
        for name in ["train", "evaluate"]:
            # Git checks out text as LF; normalize captured Windows console logs before hashing.
            (dest/f"{name}.log").write_text((run/f"{name}.log").read_text(encoding="utf-8"),
                                           encoding="utf-8", newline="\n")
    write_json(args.output/"matrix.json", matrix)
    checksums = {str(p.relative_to(args.output)).replace("\\", "/"): file_sha256(p)
                 for p in sorted(args.output.rglob("*")) if p.is_file()}
    write_json(args.output/"checksums.json", checksums)
    print(f"Exported {len(matrix['records'])} runs, {len(checksums)} files to {args.output}")


if __name__ == "__main__":
    main()
