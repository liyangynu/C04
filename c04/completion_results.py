"""Analyze completed supplementary experiments with paired statistics and failure cases."""
import argparse
import csv
import os
from pathlib import Path

import numpy as np

from .check_expanded_predictions import select_records
from .common import file_sha256, read_json, read_jsonl, write_json
from .local_experiment_audit import resolve_run
from .paired_statistics import holm_adjust, summarize_pair
from .run_expanded_matrix import key
from .stage5_results import summarize


def target_assessment(pair_f1, pair_auroc):
    return {"f1_gain_at_least_005_point_estimate": bool(pair_f1["mean_difference"] >= .05),
            "auroc_drop_at_most_002_point_estimate": bool(pair_auroc["mean_difference"] >= -.02),
            "f1_t95_lower_at_least_005": bool(pair_f1["t_95_ci"][0] >= .05),
            "auroc_t95_lower_at_least_minus002": bool(pair_auroc["t_95_ci"][0] >= -.02),
            "note": "Point-estimate targets and t intervals are reported separately; n=3 and no general noninferiority claim."}


def adjust_families(pairs):
    for family in sorted({p["family"] for p in pairs}):
        members = [p for p in pairs if p["family"] == family]
        for test in ("paired_t", "wilcoxon_exact"):
            adjusted = holm_adjust([p[test]["p_two_sided"] for p in members])
            for p, value in zip(members, adjusted):
                p[test]["p_holm"] = value


def analyze(root):
    completion = read_json(root / "progress.json")
    done = read_json(root / "completed.json")
    if (done["progress_sha256"] != file_sha256(root / "progress.json") or
            done["plan_sha256"] != file_sha256("configs/completion_v1.json")):
        raise ValueError("Supplementary protocol/progress changed")
    noisy, clean = select_records(read_json("artifacts/stage5_expanded_runs/progress.json"))
    for record in completion:
        if record["kind"] == "noise":
            if key(record) in noisy:
                raise ValueError("Unexpected duplicate supplementary condition")
            noisy[key(record)] = record
    if len(noisy) != 108:
        raise ValueError("Expected 108 unique noisy results")
    conditions = sorted({ident[:3] for ident in noisy})
    groups, pairs, goals = [], [], []
    for noise, rate, shots in conditions:
        methods = {}
        for method in ("ce", "gce", "confident_learning"):
            records = [noisy[(noise, rate, shots, seed, method)] for seed in (17, 42, 2026)]
            methods[method] = {metric: summarize([r[metric] for r in records]) for metric in
                               ("test_macro_f1", "test_accuracy", "oos_auroc", "f1_retention")}
        groups.append({"noise": noise, "rate": rate, "shots_per_class": shots,
                       "clean_ce": summarize([clean[(shots, s)]["test_macro_f1"] for s in (17, 42, 2026)]),
                       "methods": methods})
        for method in ("gce", "confident_learning"):
            selected = {}
            for metric in ("test_macro_f1", "oos_auroc"):
                pair = summarize_pair(methods[method][metric]["values"], methods["ce"][metric]["values"])
                pairs.append({"noise": noise, "rate": rate, "shots_per_class": shots,
                              "comparison": method + " - noisy CE", "metric": metric,
                              "family": "matrix_" + metric, **pair})
                selected[metric] = pair
            goals.append({"noise": noise, "rate": rate, "shots_per_class": shots, "method": method,
                          **target_assessment(selected["test_macro_f1"], selected["oos_auroc"])})
    primary = []
    for noise, left, right in (("symmetric", "gce", "ce"), ("instance", "confident_learning", "gce")):
        for metric in ("test_macro_f1", "oos_auroc"):
            primary.append({"noise": noise, "rate": .3, "shots_per_class": 10,
                            "comparison": f"{left} - {right}", "metric": metric,
                            "family": "proposal_primary_" + metric,
                            **summarize_pair([noisy[(noise, .3, 10, s, left)][metric] for s in (17, 42, 2026)],
                                             [noisy[(noise, .3, 10, s, right)][metric] for s in (17, 42, 2026)])})
    q_rows = [r for r in completion if r["kind"] == "q"]
    q_summaries, q_pairs = [], []
    plan = read_json("configs/completion_v1.json")
    if len(q_rows) != 90:
        raise ValueError("Expected 90 q-sweep results")
    for condition in plan["q_conditions"]:
        group = {}
        for q in plan["q_values"]:
            records = sorted([r for r in q_rows if r["q"] == q and all(r[k] == v for k, v in condition.items())],
                             key=lambda r: r["seed"])
            if [r["seed"] for r in records] != [17, 42, 2026]:
                raise ValueError("Missing or duplicate q seeds")
            # Earlier released summary records omit validation F1; read the original
            # training summary rather than inventing or dropping that provenance.
            records = [dict(r, best_val_macro_f1=read_json(resolve_run(r) / "training/summary.json")["best_val_macro_f1"])
                       for r in records]
            group[str(q)] = {m: summarize([r[m] for r in records]) for m in
                             ("test_macro_f1", "oos_auroc", "best_val_macro_f1")}
        for q in plan["q_values"]:
            if q == .7:
                continue
            for metric in ("test_macro_f1", "oos_auroc"):
                q_pairs.append({**condition, "q": q, "comparison": f"q={q} - q=0.7", "metric": metric,
                                "family": "q_exploratory_" + metric,
                                **summarize_pair(group[str(q)][metric]["values"], group["0.7"][metric]["values"])})
        q_summaries.append({**condition, "values": group})
    adjust_families(pairs)
    adjust_families(primary)
    adjust_families(q_pairs)
    return {"protocol_sha256": file_sha256("configs/completion_v1.json"),
            "progress_sha256": file_sha256(root / "progress.json"),
            "condition_count": len(groups), "unique_noisy_results": len(noisy),
            "condition_summaries": groups, "matrix_pairwise": pairs, "proposal_primary_pairs": primary,
            "target_assessments": goals, "q_summaries": q_summaries, "q_pairwise": q_pairs,
            "new_fit_counts": {"noise_final": 18, "q_final": 60, "oof_folds": 30},
            "limitations": ["Only three seeds; exact two-sided Wilcoxon cannot attain p<.05 with three nonzero differences.",
                            "Supplementary q/rate protocol was fixed after earlier exploratory results; no test-based main-q selection.",
                            "Interrupted archived fits are additional cost, excluded from the planned 108 successful new fits."]}


def failure_examples(root, output):
    raw = read_json("reports/stage3/formal_idn10_v2/comparison.json")["records"]
    official = read_jsonl("data/processed/clean/test.jsonl")
    by_id = {r["id"]: r for r in official}
    examples = []
    for seed in (17, 42, 2026):
        predictions = {}
        for method in ("ce", "gce", "cl_ce"):
            row = next(r for r in raw if r["seed"] == seed and r["method"] == method)
            predictions[method] = {r["id"]: r for r in read_jsonl(resolve_run(dict(row, reused_from="unused")) /
                                                                    "evaluation/predictions.jsonl")}
        candidates = [sid for sid in by_id if predictions["ce"][sid]["predicted_intent"] == by_id[sid]["label"]
                      and predictions["cl_ce"][sid]["predicted_intent"] != by_id[sid]["label"]]
        # Fixed lexicographic selection of examples, not hand-picked scores.
        for sid in sorted(candidates)[:5]:
            examples.append({"seed": seed, **by_id[sid],
                             "ce": predictions["ce"][sid]["predicted_intent"],
                             "gce": predictions["gce"][sid]["predicted_intent"],
                             "cl_ce": predictions["cl_ce"][sid]["predicted_intent"]})
    write_json(output / "failure_examples.json", {"selection": "First five lexicographic ID test IDs per seed where CE is correct and CL+CE is wrong.",
               "examples": examples, "limitations": "Descriptive errors do not establish a single causal mechanism."})


def plot(report, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for j, noise in enumerate(("symmetric", "instance")):
        groups = sorted([g for g in report["condition_summaries"] if g["noise"] == noise and g["shots_per_class"] == 10],
                        key=lambda g: g["rate"])
        for i, metric in enumerate(("test_macro_f1", "oos_auroc")):
            for method in ("ce", "gce", "confident_learning"):
                axes[i, j].errorbar([g["rate"]*100 for g in groups],
                    [g["methods"][method][metric]["mean"] for g in groups],
                    yerr=[g["methods"][method][metric]["std_sample"] for g in groups],
                    marker="o", capsize=3, label=method)
            axes[i, j].set(title=f"{noise} / 10-shot", ylabel=metric, xlabel="Noise (%)", xticks=[10, 20, 30], ylim=(0, 1))
            axes[i, j].grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "noise_curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    qs = [.2, .4, .7, .9, .95]
    for ax, group in zip(axes.flat, report["q_summaries"]):
        for metric in ("test_macro_f1", "best_val_macro_f1", "oos_auroc"):
            ax.errorbar(qs, [group["values"][str(q)][metric]["mean"] for q in qs],
                        yerr=[group["values"][str(q)][metric]["std_sample"] for q in qs],
                        marker="o", capsize=2, label=metric)
        ax.set(title=f"{group['noise']} {group['rate']:.0%} / k={group['shots_per_class']}",
               xlabel="GCE q", ylabel="Score", xticks=qs, ylim=(0, 1))
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "q_sweep.png", dpi=180)
    plt.close(fig)


def write_report(report, output):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "analysis.json", report)
    lines = ["# 补充实验结果", "", "固定主q=0.7；所有值为三种子均值±样本标准差。", "",
             "| 噪声 | 比例 | 每类样本 | CE F1 | GCE F1 | CL+CE F1 |", "| --- | --- | --- | --- | --- | --- |"]
    for group in report["condition_summaries"]:
        cells = [f"{group['methods'][m]['test_macro_f1']['mean']:.4f}±{group['methods'][m]['test_macro_f1']['std_sample']:.4f}"
                 for m in ("ce", "gce", "confident_learning")]
        lines.append(f"| {group['noise']} | {group['rate']:.0%} | {group['shots_per_class']} | " + " | ".join(cells) + " |")
    lines += ["", "## 开题指定的直接比较", "", "| 条件/差值 | 指标 | 平均差 | 95% t区间 | t/Holm p | Wilcoxon/Holm p |",
              "| --- | --- | --- | --- | --- | --- |"]
    for pair in report["proposal_primary_pairs"]:
        lines.append(f"| {pair['noise']}30% / {pair['comparison']} | {pair['metric']} | {pair['mean_difference']:+.4f} | "
                     f"[{pair['t_95_ci'][0]:+.4f}, {pair['t_95_ci'][1]:+.4f}] | {pair['paired_t']['p_holm']:.4g} | {pair['wilcoxon_exact']['p_holm']:.4g} |")
    lines += ["", "各噪声/样本量条件的点估计目标与区间单独记录在analysis.json；不把p>0.05解释为非劣。",
              "三种子精确Wilcoxon分辨率有限，t推断依赖分布假设。q扫描全部值、验证分数和测试分数保留，不按测试排序替换主q。",
              "", "![noise curves](noise_curves.png)", "", "![q sweep](q_sweep.png)", "",
              "本地完整预测、模型、概率、日志和图表不自动上传GitHub；代码、配置及运行说明进入Git。"]
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with (output / "targets.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["target_assessments"][0]))
        writer.writeheader()
        writer.writerows(report["target_assessments"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/completion_v1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/completion_v1/analysis"))
    args = parser.parse_args()
    report = analyze(args.source)
    write_report(report, args.output)
    failure_examples(args.source, args.output)
    plot(report, args.output)
    print("Analyzed 12 noise conditions, 90 q runs and proposal-specific paired contrasts.")


if __name__ == "__main__":
    main()
