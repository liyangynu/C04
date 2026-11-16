"""Post-hoc diagnostics; clean training labels are used only after fitting."""
import argparse
from datetime import date
from pathlib import Path

import numpy as np

from .common import read_json, read_jsonl, write_json
from .formal_protocol import comparison_noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    comparison = read_json(args.results/'comparison.json')
    noise, rate = comparison_noise(comparison)
    noise_name = 'IDN' if noise == 'instance' else '对称噪声'
    scope = 'IDN30' if noise == 'instance' else 'symmetric30'
    records = comparison['records']
    if len(records) != 12:
        raise ValueError('Diagnostics require all 12 completed fits')
    diagnostics = []
    for record in records:
        if record['method'] != 'cl_ce':
            continue
        seed = record['seed']
        clean = read_jsonl(f'data/processed/fewshot/seed_{seed}/k_10.jsonl')
        dest = args.results/'oof'/f'seed_{seed}'
        with np.load(dest/'oof_probabilities.npz', allow_pickle=False) as data:
            if data['ids'].tolist() != [r['id'] for r in clean]:
                raise ValueError('Posthoc clean labels are not aligned')
            wrong = data['observed_labels'] != np.array([r['label'] for r in clean])
        kept = set(read_json(dest/'kept_ids.json'))
        removed = np.array([row['id'] not in kept for row in clean])
        true_positive = int((removed & wrong).sum())
        diagnostics.append({'seed': seed, 'actually_flipped': int(wrong.sum()),
            'flagged': int(removed.sum()), 'removed_flipped': true_positive,
            'removed_originally_clean': int((removed & ~wrong).sum()),
            'kept_flipped': int((~removed & wrong).sum()),
            'detection_precision': true_positive/int(removed.sum()) if removed.any() else None,
            'detection_recall': true_positive/int(wrong.sum()) if wrong.any() else None,
            'final_optimizer_steps': int(np.ceil(record['rows_kept']/64))*record['epochs_completed']})
    paired = {}
    for method in ['gce','cl_ce']:
        paired[method] = {}
        for metric in ['test_macro_f1','oos_auroc']:
            differences = [next(r[metric] for r in records if r['method']==method and r['seed']==seed)
                - next(r[metric] for r in records if r['method']=='ce' and r['seed']==seed)
                for seed in [17,42,2026]]
            paired[method][metric] = {'differences': differences, 'mean':float(np.mean(differences)),
                                      'std_sample':float(np.std(differences,ddof=1))}
    analysis = {'scope':f'{scope}/10-shot/three-seed/v2', 'noise':noise, 'rate':rate,
        'diagnostics_use_clean_training_labels':True,
        'used_for_training_or_threshold_selection':False, 'posthoc_cleaning':diagnostics,
        'paired_against_noisy_ce':paired, 'formal_significance_and_noninferiority_claimed':False}
    write_json(args.output.with_suffix('.json'), analysis)
    text = [f'# {noise_name}30%、10-shot 三方法对照 v2', '',
        f'{date.today().isoformat()}。三个种子：17、42、2026。最多30轮，早停耐心5；验证宏F1选模。保留旧5轮试跑，正式设置及限制见[协议](正式对照协议.md)。', '',
        '## 结果', '', '| 路线 | 测试Macro-F1 | Accuracy | OOS AUROC | 平均F1保持率 |',
        '| --- | --- | --- | --- | --- |']
    names = {'clean_ce':'干净CE（参考）','ce':'带噪CE','gce':'GCE(q=0.7)','cl_ce':'置信学习清洗后CE'}
    for method in names:
        entries = [r for r in records if r['method']==method]
        cells = [f"{np.mean([r[k] for r in entries]):.4f}±{np.std([r[k] for r in entries],ddof=1):.4f}"
                 for k in ['test_macro_f1','test_accuracy','oos_auroc']]
        text.append(f"| {names[method]} | {' | '.join(cells)} | {np.mean([r['f1_retention'] for r in entries]):.4f} |")
    text += ['', '均值±样本标准差；保持率先按种子除以对应干净CE，再求平均。GCE相对带噪CE的平均F1差值为'
        f" {paired['gce']['test_macro_f1']['mean']:+.4f}，清洗路线为 {paired['cl_ce']['test_macro_f1']['mean']:+.4f}。"
        '三种子仅作描述性比较，不据此宣称显著性或OOS非劣；本轮不覆盖其他样本量和噪声比例。', '',
        '## 清洗结果的事后诊断', '',
        '下表仅在全部模型完成后读取原始干净训练标签进行诊断，未输入清洗器或用于调参。错标检测统计针对人工翻转，不代表人工语义判定。', '',
        '| seed | 标记 | 删掉的翻转样本 | 删掉的原始干净样本 | 保留的翻转样本 | 检测精确率 | 检测召回率 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for d in diagnostics:
        text.append(f"| {d['seed']} | {d['flagged']} | {d['removed_flipped']} | {d['removed_originally_clean']} | {d['kept_flipped']} | {d['detection_precision']:.4f} | {d['detection_recall']:.4f} |")
    text += ['', '每个种子实际翻转450条。清洗会删掉原始干净样本，缩小训练集和词表，并减少每轮优化器更新；这几项同时变化，当前结果无法单独识别各因素的因果贡献。结果完整保留，不根据真噪声率或测试分数重新调删样阈值。', '',
        '## 成本', '', '| 路线（三种子合计） | 最终训练含验证/秒 | OOF训练含验证/秒 | 路线墙钟/秒 |',
        '| --- | --- | --- | --- |']
    for method in names:
        entries = [r for r in records if r['method']==method]
        text.append(f"| {names[method]} | {sum(r['final_epoch_seconds'] for r in entries):.2f} | {sum(r['oof_epoch_seconds'] for r in entries):.2f} | {sum(r['route_wall_seconds'] for r in entries):.2f} |")
    text += ['', f"整组27次训练及评估顺序运行的墙钟时间为 {comparison['wall_seconds']:.2f} 秒。路线墙钟包含进程启动、输入扫描、训练、清洗和独立评估；不包含环境安装和下载。",
        '显存的逐轮已分配/保留记录随原始训练报告保存；OOF显存另见每种子的resource_summary.json，不能把这些数值当作整卡占用。', '',
        '## 验收与剩余范围', '',
        f'逐样本预测、150类混淆矩阵、逐类F1、训练日志和五折OOF资产在 `{args.results.as_posix()}/`。独立验收重算12组指标，核对三组OOF的覆盖、隔离、词表及清洗决策。重跑命令见[运行说明](正式对照运行说明.md)。', '',
        '尚未完成：其他样本量/比例的v2配对矩阵、正式消融、完整统计检验、公开论文基线数值验收、最终报告/PPT。旧阈值启发式试跑与本轮cleanlab算法版本不同，不能直接合并解释。']
    args.output.with_suffix('.md').write_text('\n'.join(text)+'\n',encoding='utf-8',newline='\n')


if __name__ == '__main__':
    main()
