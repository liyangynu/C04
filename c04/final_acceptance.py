"""Finish queued single-GPU checks after the supplementary training process ends."""
import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time

from .common import file_sha256, read_json, read_jsonl, write_json
from .local_experiment_audit import audit_oof, collect_runs, replay_final
from .runtime import load_official_split


def command(args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", newline="\n") as stream:
        result = subprocess.run([sys.executable, *args], stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {args}; see {log}")


def environment_check(output):
    pinned = {}
    for line in Path("requirements-formal-lock.txt").read_text().splitlines():
        if "==" in line:
            name, version = line.split("==")
            if importlib.metadata.version(name) != version:
                raise ValueError(f"Installed dependency differs: {name}")
            pinned[name] = version
    import torch, numpy, cleanlab
    prefix = Path(sys.prefix).resolve()
    for module in (torch, numpy, cleanlab):
        if not Path(module.__file__).resolve().is_relative_to(prefix):
            raise ValueError("Dependency was imported from outside the new virtual environment")
    write_json(output / "environment.json", {"all_passed": True, "executable": sys.executable,
               "prefix": str(prefix), "base_prefix": sys.base_prefix, "pinned_packages": pinned,
               "cuda_available": torch.cuda.is_available(), "base_interpreter_shared": True,
               "site_packages_shared": False, "lock_sha256": file_sha256("requirements-formal-lock.txt")})


def compare_retraining(output):
    original = read_json("artifacts/stage3_formal_idn10_v2/comparison.json")
    repeated = read_json(output / "idn10_retraining/comparison.json")
    test, _ = load_official_split("data/processed/clean/test.jsonl", "test")
    oos, _ = load_official_split("data/processed/clean/oos_test.jsonl", "oos_test")
    checked = []
    for record in repeated["records"]:
        old = next(r for r in original["records"] if r["method"] == record["method"] and r["seed"] == record["seed"])
        source, target = Path(old["run_dir"]), Path(record["run_dir"])
        # Different run provenance makes checkpoint bytes differ; compare predictions and weights separately.
        if (file_sha256(source / "evaluation/predictions.jsonl") != file_sha256(target / "evaluation/predictions.jsonl") or
                record["train_sha256"] != old["train_sha256"]):
            raise ValueError(f"Fresh-environment retraining differs: {record['method']}/{record['seed']}")
        import torch
        before = torch.load(source / "training/best.pt", map_location="cpu", weights_only=True)["model"]
        after = torch.load(target / "training/best.pt", map_location="cpu", weights_only=True)["model"]
        if set(before) != set(after) or any(not torch.equal(before[k], after[k]) for k in before):
            raise ValueError("Fresh-environment selected model parameters differ")
        replay_final({"run": target, "record": record, "group": "fresh_idn"}, test + oos, "cuda")
        if record["method"] == "cl_ce":
            audit_oof(target / "oof", replay=True)
        checked.append({"method": record["method"], "seed": record["seed"],
                        "prediction_bytes_equal": True, "model_parameters_equal": True})
    write_json(output / "retraining_verified.json", {"all_passed": True, "records": checked,
               "final_fits": 12, "oof_fold_fits": 15, "source_data_shared_by_checksum": True,
               "scope": "Entire fixed IDN30%, 10-shot, three-seed cohort. The other original conditions are audited and replayed, not all retrained."})


def run(output, wait):
    output.mkdir(parents=True, exist_ok=True)
    environment_check(output)
    if wait:
        while not Path("artifacts/completion_v1/completed.json").exists():
            print("Waiting for supplementary single-GPU training to finish", flush=True)
            time.sleep(30)
    elif not Path("artifacts/completion_v1/completed.json").exists():
        raise ValueError("Supplementary training must finish first")
    command(["-m", "c04.completion_results"], output / "analysis.log")
    print("Supplementary analysis finished", flush=True)
    if not (output / "idn10_retraining/comparison.json").exists():
        command(["-m", "c04.run_formal_comparison", "--source", "data/formal_idn10_v2/matrix.json",
                 "--output", str(output / "idn10_retraining"), "--device", "cuda"], output / "retraining.log")
    compare_retraining(output)
    print("Fresh-environment full IDN cohort retraining matched", flush=True)
    command(["-m", "c04.local_experiment_audit", "--weights", "--oof-weights",
             "--output", str(output / "local_models.json")], output / "local_models.log")
    print("All local final-model and OOF weight checks finished", flush=True)
    command(["-m", "c04.replay_encoder", "--output", str(output / "encoder.json")], output / "encoder.log")
    command(["-m", "c04.public_svm_selection", "--verify-only"], output / "svm.log")
    from .public_cnn_results import replay_weights
    write_json(output / "cnn_weights.json", replay_weights(Path("reports/stage6/public_cnn_v1"), Path("artifacts/public_cnn_v1"), "cuda"))
    command(["-m", "unittest", "discover", "-s", "tests", "-v"], output / "tests.log")
    write_json(output / "completed.json", {"all_passed": True,
               "evidence": {name: file_sha256(output / name) for name in
                            ("environment.json", "retraining_verified.json", "local_models.json", "encoder.json", "cnn_weights.json", "tests.log")},
               "tracked_code_only_policy": "New large experimental evidence remains local; source code, configs and notes are committed."})
    print("PASS: all final experimental acceptance checks finished", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/final_acceptance_v1"))
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    run(args.output, args.wait)


if __name__ == "__main__":
    main()
