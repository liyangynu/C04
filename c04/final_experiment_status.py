"""Write concise completion notes only after all retained local evidence checks pass."""
import argparse
from pathlib import Path

from .common import file_sha256, read_json


def write_status(acceptance, output):
    done = read_json(acceptance / "completed.json")
    if done.get("all_passed") is not True:
        raise ValueError("Final acceptance has not passed")
    for relative, digest in done["evidence"].items():
        if file_sha256(acceptance / relative) != digest:
            raise ValueError("Final acceptance evidence changed")
    analysis = read_json("artifacts/completion_v1/analysis/analysis.json")
    svm = read_json("artifacts/public_svm_validation_v2/verified.json")
    models = read_json(acceptance / "local_models.json")
    retrained = read_json(acceptance / "retraining_verified.json")
    encoder = read_json(acceptance / "encoder.json")
    if not all(r["all_passed"] for r in (svm, models, retrained, encoder)):
        raise ValueError("Missing successful local audit")
    lines = ["# 实验完成记录", "", "本记录在补充训练、新环境重训和本地推理核验实际完成后生成。原始预测、权重、图表和校验清单保存在D盘本地忽略目录，不自动上传。", "",
             "## 已完成范围", "",
             "- 核心矩阵扩展至12个噪声/样本量条件，三条方法路线、三个固定种子，共108组带噪结果；干净参考为四个样本量、三个种子。",
             "- 对称/实例相关噪声10%/20%/30%的10-shot曲线齐全；30%下5/10/20/100-shot比较齐全。",
             "- 六个条件下q=0.2/0.4/0.7/0.9/0.95、三个种子，共90组q结果，包含30组明确复用。主q仍为0.7。",
             "- 原有模块/嵌入消融与删除、重标注、加权、取消OOF的清洗变体保留；新增开题指定的直接配对统计、逐条件目标检查和真实错误样例。",
             f"- {models['final_model_count']}个不重复最终模型，共{models['prediction_rows']}条保存预测的官方真值、指标、逐类表、矩阵、配置、验证选模、训练专用词表和权重重放通过。",
             f"- {len(models['oof'])}条五折OOF路线的分折、训练词表、清洗决策和折模型概率重放通过，共{sum(r['folds'] for r in models['oof'])}个折模型。",
             f"- 新依赖环境重新训练IDN30%、10-shot全部三种子对照，12次最终拟合和15次OOF拟合完成。{len(retrained['records'])}组预测字节及选中模型参数与原运行相同；其他条件做核验/权重重放，未声称全部重训。",
             f"- MiniLM固定权重重新编码{encoder['total_rows_encoded']}条嵌套子集样本，12份历史向量及IDN标签重建通过；不同子集中的同一文本重复编码按实际工作量计。", "",
             "## 公开基线", "",
             f"SVM v2由预先固定的四项候选按验证准确率选出词频LinearSVC，ID准确率{svm['summary']['id_accuracy']['mean']:.2%}，相对论文91.0%差{svm['id_accuracy_difference_pp']:+.2f}个百分点，±2个百分点数值检查{'通过' if svm['within_two_percentage_points'] else '未通过'}。OOS召回{svm['summary']['oos_recall']['mean']:.2%}，论文为14.5%。四个候选的验证预测和最终16500条测试预测均通过模型重放。",
             "", "三个确定性SVM运行结果相同，标准差为0不表示泛化不确定性为0。原文未公开的求解器/表示选择单独披露；这是补充的公开模型重实现，不是原始代码精确复现。CNN v1 87.41%的未达标结果及SVM v1 88.24%的未达标结果保留。", "",
             "## 主要结果与目标", "",
             "| 噪声/每类样本/比例 | CE Macro-F1 | GCE Macro-F1 | CL+CE Macro-F1 |", "| --- | --- | --- | --- |"]
    for group in analysis["condition_summaries"]:
        values = [f"{group['methods'][m]['test_macro_f1']['mean']:.4f}±{group['methods'][m]['test_macro_f1']['std_sample']:.4f}"
                  for m in ("ce", "gce", "confident_learning")]
        lines.append(f"| {group['noise']}/k{group['shots_per_class']}/{group['rate']:.0%} | " + " | ".join(values) + " |")
    lines += ["", "| 开题直接比较（10-shot、30%） | 指标 | 均值差 | 95%配对t区间 | Holm t p | Holm精确Wilcoxon p |",
              "| --- | --- | --- | --- | --- | --- |"]
    for p in analysis["proposal_primary_pairs"]:
        lines.append(f"| {p['noise']}: {p['comparison']} | {p['metric']} | {p['mean_difference']:+.4f} | "
                     f"[{p['t_95_ci'][0]:+.4f}, {p['t_95_ci'][1]:+.4f}] | {p['paired_t']['p_holm']:.4g} | {p['wilcoxon_exact']['p_holm']:.4g} |")
    lines += ["", "研究目标没有统一达成。GCE的收益随噪声和样本量变化；置信学习删除路线存在明显负结果。具体F1提高0.05与AUROC下降不超过0.02的点估计及区间检查见本地targets.csv，不把点估计达标写成统计非劣。三种子精确Wilcoxon检验分辨率有限，配对t推断依赖分布假设。", "",
              "## 本地证据与剩余交付", "",
              "运行方法及路径见[实验运行与本地结果](实验运行与本地结果.md)。本地 `artifacts/final_acceptance_v1/completed.json` 引用环境、重训、权重重放、编码器及测试日志的校验值；`artifacts/experiment_inventory.json` 记录证据和保留的训练耗时。训练轮次计时不等于完整项目墙钟，PyTorch显存不等于整卡占用。", "",
              "本记录完成的是实验执行、分析和本地复现核验。正式技术报告PDF/源文件、答辩PPT与演示材料仍是后续交付，未在本轮冒充完成。开题报告和要求图片继续只保留本地。", ""]
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Wrote verified experiment status: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, default=Path("artifacts/final_acceptance_v1"))
    parser.add_argument("--output", type=Path, default=Path("docs/实验完成记录.md"))
    args = parser.parse_args()
    write_status(args.acceptance, args.output)


if __name__ == "__main__":
    main()
