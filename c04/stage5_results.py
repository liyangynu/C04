"""Audit and analyze the completed stage-5 expanded comparison."""
import argparse
from pathlib import Path

import numpy as np

from .common import file_sha256, read_json, read_jsonl, write_json
from .paired_statistics import holm_adjust, summarize_pair


def expected_conditions():
    result = [("instance", .3, shots) for shots in (5, 10, 20, 100)]
    result += [("symmetric", .3, shots) for shots in (5, 10, 20, 100)]
    result += [("symmetric", rate, 10) for rate in (.1, .2)]
    return result


def normalized_method(record):
    return "confident_learning" if record["method"] == "cl_ce" else record["method"]


def audit_run(record, source_record):
    if record["reused"]:
        return {"reused": True, "audited_by": "stage3 formal result audit"}
    run = Path(record["run_dir"])
    observed = Path(source_record["train_path"])
    if file_sha256(observed) != source_record["train_sha256"]:
        raise ValueError(f"Observed input checksum changed: {observed}")
    if record["observed_sha256"] != source_record["train_sha256"]:
        raise ValueError(f"Model run does not use the frozen observed input: {run}")
    if record["method"] in {"ce", "gce"}:
        if record["train_sha256"] != source_record["train_sha256"]:
            raise ValueError(f"Direct training input differs from frozen noisy input: {run}")
        training = read_json(run / "training/summary.json")
        if training["fingerprints"]["train"] != source_record["train_sha256"]:
            raise ValueError(f"Training checkpoint fingerprint differs: {run}")
        return {"reused": False, "input_sha256": source_record["train_sha256"],
                "checkpoint_sha256": file_sha256(run / "training/best.pt"),
                "evaluation_sha256": file_sha256(run / "evaluation/metrics.json")}
    if record["method"] != "confident_learning":
        raise ValueError(f"Unknown model method: {record['method']}")
    report = read_json(run / "oof/cleaning_report.json")
    if (report["input_sha256"] != source_record["train_sha256"] or
            report["uses_clean_labels"] or report["uses_test_labels"] or
            not report["coverage_exactly_once"]):
        raise ValueError(f"OOF report violates input isolation/coverage: {run}")
    input_ids = {row["id"] for row in read_jsonl(observed)}
    heldout_ids = []
    for fold in report["fold_records"]:
        training_ids = set(fold["training_ids"])
        fold_heldout = set(fold["heldout_ids"])
        if training_ids & fold_heldout:
            raise ValueError(f"OOF fold overlap: {run}, fold={fold['fold']}")
        heldout_ids.extend(fold_heldout)
    if len(heldout_ids) != len(input_ids) or set(heldout_ids) != input_ids:
        raise ValueError(f"OOF fold coverage differs from observed input: {run}")
    final = read_json(run / "training/summary.json")
    return {"reused": False, "input_sha256": report["input_sha256"],
            "oof_sha256": report["oof_sha256"], "oof_folds": len(report["fold_records"]),
            "oof_rows": report["rows"], "oof_rows_kept": report["rows_kept"],
            "final_training_sha256": final["fingerprints"]["train"],
            "checkpoint_sha256": file_sha256(run / "training/best.pt"),
            "evaluation_sha256": file_sha256(run / "evaluation/metrics.json")}


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std_sample": float(values.std(ddof=1)),
            "n": int(len(values)), "values": values.tolist()}


def markdown_report(report):
    lines = ["# 阶段5：扩展实验结果", "", "固定官方测试集、三种子；结果为均值±样本标准差。置信学习额外报告五折OOF训练成本。", "",
             "## 按条件汇总", "",
             "| 噪声 | 每类样本 | 比例 | CE Macro-F1 | GCE Macro-F1 | 置信学习 Macro-F1 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for group in report["condition_summaries"]:
        vals = group["methods"]
        lines.append(f"| {group['noise']} | {group['shots_per_class']} | {group['rate']:.0%} | "
                     f"{vals['ce']['test_macro_f1']['mean']:.4f}±{vals['ce']['test_macro_f1']['std_sample']:.4f} | "
                     f"{vals['gce']['test_macro_f1']['mean']:.4f}±{vals['gce']['test_macro_f1']['std_sample']:.4f} | "
                     f"{vals['confident_learning']['test_macro_f1']['mean']:.4f}±{vals['confident_learning']['test_macro_f1']['std_sample']:.4f} |")
    cost = report["runtime_summary"]
    lines += ["", "指标完整表、逐种子配对差值、区间和检验见同目录下JSON。 Holm校正在Macro-F1主比较族和OOS AUROC次要比较族内分别执行。", "",
              "## 运行成本", "",
              f"唯一新增模型拟合 {cost['unique_new_model_fits']} 次，其中OOF折训练 {cost['unique_new_oof_fold_fits']} 次；复用已发布结果 {cost['reused_model_results']} 项。正式清单墙钟约 {cost['unique_new_route_wall_seconds'] / 3600:.2f} 小时，运行期间的重复核对另增加 {cost['all_executed_new_route_wall_seconds_including_repeats'] - cost['unique_new_route_wall_seconds']:.0f} 秒。最高PyTorch峰值保留显存 {cost['peak_final_reserved_mb']:.1f} MiB。", "",
              "## 解释范围", "", "10-shot/30%条件复用已发布正式结果；其中六个置信学习项曾因旧方法名不统一被重复运行一次。重复运行与已发布测试Macro-F1逐种子完全相同，统计只使用原发布记录。三种子检验能力有限，结果是固定数据与随机种子下的比较，不作跨数据集或普遍噪声结论。", ""]
    return "\n".join(lines)


def analyze(source_path, progress_path, output_stem):
    source = read_json(source_path)
    progress = read_json(progress_path)
    if source["protocol_sha256"] != file_sha256("configs/expanded_matrix_v1.json"):
        raise ValueError("The source does not match the frozen stage-5 protocol")
    source_map = {(r["noise"], r["rate"], r["shots_per_class"], r["seed"]): r
                  for r in source["records"]}
    chosen = {}
    duplicate_checks = []
    for record in progress:
        noise, rate, shots, seed = (record["noise"], record["rate"],
                                    record["shots_per_class"], record["seed"])
        if record["method"] == "clean_ce":
            continue
        method = normalized_method(record)
        ident = (noise, rate, shots, seed, method)
        existing = chosen.get(ident)
        if existing is None or (record.get("reused", False) and not existing.get("reused", False)):
            if existing is not None:
                duplicate_checks.append({"condition": [noise, rate, shots], "seed": seed,
                                         "method": method,
                                         "duplicate_f1_equal": record["test_macro_f1"] == existing["test_macro_f1"],
                                         "duplicate_f1": record["test_macro_f1"],
                                         "released_f1": existing["test_macro_f1"]})
            chosen[ident] = record
        elif not record.get("reused", False):
            duplicate_checks.append({"condition": [noise, rate, shots], "seed": seed,
                                     "method": method,
                                     "duplicate_f1_equal": record["test_macro_f1"] == existing["test_macro_f1"],
                                     "duplicate_f1": record["test_macro_f1"],
                                     "released_f1": existing["test_macro_f1"]})
    if any(not item["duplicate_f1_equal"] for item in duplicate_checks):
        raise ValueError("A repeated formal result differs from its released value")

    source_conditions = {(r["noise"], r["rate"], r["shots_per_class"]) for r in source["records"]}
    if source_conditions != set(expected_conditions()):
        raise ValueError(f"Frozen source has unexpected condition set: {source_conditions}")
    audits, condition_summaries, pairwise = [], [], []
    selected = {}
    for noise, rate, shots in expected_conditions():
        method_records = {}
        for method in ("ce", "gce", "confident_learning"):
            items = {}
            for seed in (17, 42, 2026):
                ident = (noise, rate, shots, seed, method)
                if ident not in chosen:
                    raise ValueError(f"Missing result: {ident}")
                record = chosen[ident]
                source_record = source_map[(noise, rate, shots, seed)]
                audits.append({"condition": [noise, rate, shots], "seed": seed,
                               "method": method, **audit_run(record, source_record)})
                items[seed] = record
                selected[ident] = record
            method_records[method] = items
        summary_methods = {}
        for method, items in method_records.items():
            summary_methods[method] = {metric: summarize([r[metric] for r in items.values()])
                                       for metric in ("test_macro_f1", "test_accuracy", "oos_auroc", "f1_retention")}
        clean_by_seed = {}
        for seed in (17, 42, 2026):
            clean_options = [r for r in progress if r["method"] == "clean_ce" and
                             r["shots_per_class"] == shots and r["seed"] == seed]
            if not clean_options:
                raise ValueError(f"Missing clean reference for shots={shots}, seed={seed}")
            vals = {r["test_macro_f1"] for r in clean_options}
            if len(vals) != 1:
                raise ValueError(f"Clean references disagree for shots={shots}, seed={seed}")
            clean_by_seed[seed] = vals.pop()
        for method, items in method_records.items():
            if method == "ce":
                continue
            for metric in ("test_macro_f1", "oos_auroc"):
                pair = summarize_pair([items[s][metric] for s in (17, 42, 2026)],
                                      [method_records["ce"][s][metric] for s in (17, 42, 2026)])
                pairwise.append({"family": metric, "noise": noise, "rate": rate,
                                 "shots_per_class": shots, "comparison": f"{method} - noisy CE",
                                 "metric": metric, "seeds": [17, 42, 2026], **pair})
        condition_summaries.append({"noise": noise, "rate": rate, "shots_per_class": shots,
                                    "clean_ce": summarize(list(clean_by_seed.values())),
                                    "methods": summary_methods})

    for metric in ("test_macro_f1", "oos_auroc"):
        indices = [i for i, p in enumerate(pairwise) if p["metric"] == metric]
        adjusted = holm_adjust([pairwise[i]["paired_t"]["p_two_sided"] for i in indices])
        for index, value in zip(indices, adjusted):
            pairwise[index]["paired_t"]["p_holm"] = value
        adjusted = holm_adjust([pairwise[i]["wilcoxon_exact"]["p_two_sided"] for i in indices])
        for index, value in zip(indices, adjusted):
            pairwise[index]["wilcoxon_exact"]["p_holm"] = value

    clean_refs = {(r["shots_per_class"], r["seed"]): r for r in progress if r["method"] == "clean_ce"}
    if len(clean_refs) != 12 or sum(not r.get("reused", False) for r in clean_refs.values()) != 9:
        raise ValueError("Expected clean CE references for four sample sizes and three seeds")
    clean_audits = []
    for (shots, seed), record in clean_refs.items():
        source_record = source_map[("instance", .3, shots, seed)]
        clean_path = Path(source_record["clean_path"])
        if (file_sha256(clean_path) != source_record["clean_sha256"] or
                record.get("train_sha256", source_record["clean_sha256"]) != source_record["clean_sha256"]):
            raise ValueError(f"Clean reference data differs: shots={shots}, seed={seed}")
        if record.get("reused", False):
            audit = {"reused": True, "train_sha256": source_record["clean_sha256"]}
        else:
            run = Path(record["run_dir"])
            training = read_json(run / "training/summary.json")
            if training["fingerprints"]["train"] != source_record["clean_sha256"]:
                raise ValueError(f"Clean reference checkpoint input differs: {run}")
            audit = {"reused": False, "train_sha256": source_record["clean_sha256"],
                     "checkpoint_sha256": file_sha256(run / "training/best.pt"),
                     "evaluation_sha256": file_sha256(run / "evaluation/metrics.json")}
        clean_audits.append({"shots_per_class": shots, "seed": seed, **audit})
    report = {"protocol_sha256": source["protocol_sha256"], "source_sha256": file_sha256(source_path),
              "progress_sha256": file_sha256(progress_path), "expected_conditions": [list(x) for x in expected_conditions()],
              "condition_count": len(condition_summaries), "selected_model_count": len(selected),
              "audited_runs": audits, "duplicate_reruns": duplicate_checks,
              "clean_reference_audits": clean_audits,
              "condition_summaries": condition_summaries, "pairwise_statistics": pairwise,
              "runtime_summary": {
                  "unique_new_model_fits": sum(not r.get("reused", False) for r in selected.values()) +
                                           sum(not r.get("reused", False) for r in clean_refs.values()),
                  "reused_model_results": sum(r.get("reused", False) for r in selected.values()),
                  "redundant_exact_repeat_fits": len(duplicate_checks),
                  "unique_new_oof_fold_fits": sum(a.get("oof_folds", 0) for a in audits),
                  "unique_new_route_wall_seconds": sum(r.get("route_wall_seconds", 0) for r in selected.values()
                                                        if not r.get("reused", False)),
                  "all_executed_new_route_wall_seconds_including_repeats": sum(
                      r.get("route_wall_seconds", 0) for r in progress if not r.get("reused", False)),
                  "all_executed_oof_wall_seconds_including_repeats": sum(
                      r.get("oof_wall_seconds", 0) for r in progress if not r.get("reused", False)),
                  "peak_final_allocated_mb": max((r.get("peak_final_allocated_mb", 0) for r in selected.values()), default=0),
                  "peak_final_reserved_mb": max((r.get("peak_final_reserved_mb", 0) for r in selected.values()), default=0)},
              "test_set_used_for_selection": False,
              "limitations": ["Only three fixed seeds are available; paired inferential tests have low resolution.",
                              "10-shot 30-percent results are reused from earlier formal experiments.",
                              "The stage-5 protocol was frozen after earlier exploratory results."]}
    stem = Path(output_stem)
    write_json(stem.with_suffix(".json"), report)
    stem.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8", newline="\n")
    print(f"PASS: {len(condition_summaries)} conditions; {len(selected)} unique condition/method/seed results; "
          f"{len(audits)} input/OOF audits; wrote {stem}.json and {stem}.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/stage5_expanded/matrix.json"))
    parser.add_argument("--progress", type=Path, default=Path("artifacts/stage5_expanded_runs/progress.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/阶段5扩展实验结果"))
    args = parser.parse_args()
    analyze(args.source, args.progress, args.output)


if __name__ == "__main__":
    main()
