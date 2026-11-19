"""Run the frozen supplementary noise and q experiments, preserving earlier runs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

from .common import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from .noise import align_embeddings, inject
from .run_expanded_matrix import fit, key
from .run_noise_matrix import run_command
from .runtime import git_state


PLAN = Path("configs/completion_v1.json")


def condition_name(noise, rate, shots):
    return f"{noise}_r{round(rate * 100):02d}_k{shots}"


def q_name(q):
    return "q_" + str(q).replace(".", "")


def prepare(root, plan):
    path = root / "inputs/matrix.json"
    if path.exists():
        matrix = read_json(path)
        if matrix["plan_sha256"] != file_sha256(PLAN):
            raise ValueError("Supplementary input protocol changed")
        for row in matrix["records"]:
            if file_sha256(row["train_path"]) != row["train_sha256"]:
                raise ValueError("Prepared supplementary labels changed")
        return matrix
    records = []
    for condition in plan["additional_noise_conditions"]:
        noise, rate, shots = (condition[k] for k in ("noise", "rate", "shots_per_class"))
        for seed in plan["seeds"]:
            clean = Path(f"data/processed/fewshot/seed_{seed}/k_{shots}.jsonl")
            rows = read_jsonl(clean)
            vectors_path = Path(f"artifacts/stage3_pilot/idn_embeddings_s{seed}_k10.npz")
            vectors, provenance = align_embeddings(vectors_path, rows)
            manifest = read_json(vectors_path.with_suffix(".manifest.json"))
            if (manifest["sha256"] != file_sha256(vectors_path) or
                    provenance["encoder_revision"] != plan["encoder"]["revision"]):
                raise ValueError("Historical encoder artifact/version changed")
            if provenance["encoder_id"] != plan["encoder"]["id"]:
                # Stage 3 honestly recorded its local model path. Verify its pinned weights;
                # retain that provenance instead of relabeling the old NPZ as a new export.
                expected_weights = read_json("reports/stage3_idn_asset_pilot.json")["encoder"]["model_safetensors_sha256"]
                if file_sha256(Path(provenance["encoder_id"]) / "model.safetensors") != expected_weights:
                    raise ValueError("Historical local encoder weights differ")
            old_observed, _, _ = inject(rows, 150, .3, seed, noise, vectors, plan["encoder"]["temperature"])
            encoded = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in old_observed).encode("utf-8")
            released = read_json("reports/stage3/formal_idn10_v2/comparison.json")["records"]
            expected_sha = next(r["observed_sha256"] for r in released if r["method"] == "ce" and r["seed"] == seed)
            if hashlib.sha256(encoded).hexdigest() != expected_sha:
                raise ValueError("Historical vectors cannot reconstruct the released IDN30 labels")
            observed, audit, diagnostics = inject(rows, 150, rate, seed, noise, vectors,
                                                  plan["encoder"]["temperature"])
            dest = root / "inputs" / condition_name(noise, rate, shots) / f"seed_{seed}"
            write_jsonl(dest / "train.jsonl", observed)
            write_jsonl(dest / "audit/labels.jsonl", audit)
            write_json(dest / "audit/diagnostics.json", diagnostics)
            records.append({**condition, "seed": seed, "train_path": str(dest / "train.jsonl"),
                            "train_sha256": file_sha256(dest / "train.jsonl"),
                            "clean_sha256": file_sha256(clean), "embedding_path": str(vectors_path),
                            "embedding_sha256": file_sha256(vectors_path), "encoder": provenance})
    matrix = {"plan_sha256": file_sha256(PLAN), "records": records}
    write_json(path, matrix)
    return matrix


def choose_previous(progress):
    selected = {}
    for row in progress:
        ident = key(row)
        if ident not in selected or row.get("reused", False):
            selected[ident] = row
    return selected


def fit_q(config_path, train_path, dest, seed, shots, device):
    started = time.perf_counter()
    dest.mkdir(parents=True, exist_ok=True)
    run_command([sys.executable, "-m", "c04.train", "--train", str(train_path),
                 "--config", str(config_path), "--seed", str(seed), "--device", device,
                 "--output", str(dest / "training")], dest / "train.log")
    run_command([sys.executable, "-m", "c04.evaluate", "--checkpoint", str(dest / "training/best.pt"),
                 "--device", device, "--output", str(dest / "evaluation")], dest / "evaluate.log")
    summary, metrics = read_json(dest / "training/summary.json"), read_json(dest / "evaluation/metrics.json")
    return {"seed": seed, "shots_per_class": shots, "run_dir": str(dest),
            "train_sha256": file_sha256(train_path), "config_sha256": file_sha256(config_path),
            "best_epoch": summary["best_epoch"], "epochs_completed": summary["epochs_completed"],
            "best_val_macro_f1": summary["best_val_macro_f1"], "test_macro_f1": metrics["macro_f1"],
            "test_accuracy": metrics["accuracy"], "oos_auroc": metrics["oos_auroc"],
            "route_wall_seconds": time.perf_counter() - started, "reused": False}


def run(root, device):
    plan = read_json(PLAN)
    matrix = prepare(root, plan)
    started = time.perf_counter()
    progress_path = root / "progress.json"
    progress = read_json(progress_path) if progress_path.exists() else []
    if (root / "run.json").exists():
        if read_json(root / "run.json")["plan_sha256"] != file_sha256(PLAN):
            raise ValueError("Existing run uses a different plan")
    else:
        write_json(root / "run.json", {"plan_sha256": file_sha256(PLAN), "git": git_state(),
                                      "python_executable": sys.executable, "prefix": sys.prefix})
    earlier = choose_previous(read_json("artifacts/stage5_expanded_runs/progress.json"))
    clean_f1 = {(r["shots_per_class"], r["seed"]): r["test_macro_f1"]
                for r in earlier.values() if r["method"] == "clean_ce"}
    for source in matrix["records"]:
        noise, rate, shots, seed = (source[k] for k in ("noise", "rate", "shots_per_class", "seed"))
        for method in plan["methods"]:
            ident = (noise, rate, shots, seed, method)
            if any(r["kind"] == "noise" and key(r) == ident for r in progress):
                continue
            dest = root / "noise" / condition_name(noise, rate, shots) / method / f"seed_{seed}"
            record = fit(method, seed, shots, Path(source["train_path"]), dest, device)
            record.update({"kind": "noise", "noise": noise, "rate": rate,
                           "observed_sha256": source["train_sha256"],
                           "clean_train_sha256": source["clean_sha256"],
                           "f1_retention": record["test_macro_f1"] / clean_f1[(shots, seed)]})
            progress.append(record)
            write_json(progress_path, progress)
            print(f"noise {ident}: F1={record['test_macro_f1']:.4f}", flush=True)

    available = {**earlier, **{key(r): r for r in progress if r["kind"] == "noise"}}
    ablations = read_json("artifacts/stage4_ablations_idn10_v2/ablation.json")["records"]
    old_inputs = read_json("data/stage5_expanded/matrix.json")["records"]
    for condition in plan["q_conditions"]:
        noise, rate, shots = (condition[k] for k in ("noise", "rate", "shots_per_class"))
        for q in plan["q_values"]:
            config = copy.deepcopy(read_json(plan["base_gce_config"]))
            config["training"]["q"] = q
            config_path = root / "configs" / f"{q_name(q)}.json"
            write_json(config_path, config)
            for seed in plan["seeds"]:
                if any(r["kind"] == "q" and (r["noise"], r["rate"], r["shots_per_class"], r["seed"], r["q"]) ==
                       (noise, rate, shots, seed, q) for r in progress):
                    continue
                if q == .7:
                    source = available[(noise, rate, shots, seed, "gce")]
                    record = dict(source, reused=True)
                elif noise == "instance" and rate == .3 and shots == 10:
                    source = next(r for r in ablations if r["variant"] == q_name(q) and r["seed"] == seed)
                    record = dict(source, reused=True)
                else:
                    inputs = matrix["records"] + old_inputs
                    source = next(r for r in inputs if all(r[k] == v for k, v in
                                  (("noise", noise), ("rate", rate), ("shots_per_class", shots), ("seed", seed))))
                    dest = root / "q" / condition_name(noise, rate, shots) / q_name(q) / f"seed_{seed}"
                    record = fit_q(config_path, Path(source["train_path"]), dest, seed, shots, device)
                record.update({"kind": "q", "method": "gce", "noise": noise, "rate": rate,
                               "shots_per_class": shots, "q": q})
                progress.append(record)
                write_json(progress_path, progress)
                print(f"q {noise}/{rate}/k{shots}/q{q}/s{seed}: F1={record['test_macro_f1']:.4f}; reused={record['reused']}", flush=True)
    if sum(r["kind"] == "noise" for r in progress) != 18 or sum(r["kind"] == "q" for r in progress) != 90:
        raise ValueError("Supplementary matrix incomplete")
    write_json(root / "completed.json", {"all_runs_finished": True, "progress_sha256": file_sha256(progress_path),
               "plan_sha256": file_sha256(PLAN), "records": len(progress), "invocation_seconds": time.perf_counter() - started,
               "new_final_noise_fits": 18, "new_oof_fold_fits": 30, "new_q_fits": 60})
    print("Completed supplementary matrix: 18 noise results and 90 q results (30 reused).", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/completion_v1"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        matrix = prepare(args.output, read_json(PLAN))
        print(f"Prepared {len(matrix['records'])} supplementary inputs")
    else:
        run(args.output, args.device)


if __name__ == "__main__":
    main()
