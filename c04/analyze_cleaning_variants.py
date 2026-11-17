"""Post-hoc diagnostics of fixed cleaning variants; no tuning from truth."""
import argparse
import os
from pathlib import Path

import numpy as np

from .common import file_sha256, read_json, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('reports/stage4/cleaning_variants_idn10_v2'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    audit = read_json(root.parent/f'{root.name}_audit.json')
    if not audit['all_passed'] or audit['export_sha256'] != file_sha256(root/'checksums.json'):
        raise ValueError('Independent audit must pass first')
    for name, digest in read_json(root/'checksums.json').items():
        if file_sha256(root/name) != digest:
            raise ValueError('Published evidence changed')
    report = read_json(root/'comparison.json')
    plan = report['plan']
    reference = Path(plan['reference_results'])
    if file_sha256(reference/'checksums.json') != report['reference_checksums_sha256']:
        raise ValueError('Reference result version changed')
    for name, digest in read_json(reference/'checksums.json').items():
        if file_sha256(reference/name) != digest:
            raise ValueError('Reference evidence changed')
    previous = read_json(reference/'comparison.json')['records']
    groups = {'ce_reference': [r for r in previous if r['method'] == 'ce'],
              'oof_delete_reference': [r for r in previous if r['method'] == 'cl_ce']}
    for variant in plan['variants']:
        groups[variant] = [r for r in report['records'] if r['variant'] == variant]
    summaries, paired = {}, {}
    for name, rows in groups.items():
        summaries[name] = {metric: {'mean': float(np.mean([r[metric] for r in rows])),
            'std_sample': float(np.std([r[metric] for r in rows], ddof=1))}
            for metric in ['test_macro_f1', 'test_accuracy', 'oos_auroc']}
        paired[name] = {}
        for metric in ['test_macro_f1', 'oos_auroc']:
            delta = [next(r[metric] for r in rows if r['seed'] == seed)
                - next(r[metric] for r in groups['ce_reference'] if r['seed'] == seed) for seed in plan['seeds']]
            paired[name][metric] = {'seeds': plan['seeds'], 'differences': delta,
                'mean': float(np.mean(delta)), 'std_sample': float(np.std(delta, ddof=1))}
    diagnostics = []
    for seed in plan['seeds']:
        clean = read_jsonl(f'data/processed/fewshot/seed_{seed}/k_10.jsonl')
        with np.load(reference/'oof'/f'seed_{seed}'/'oof_probabilities.npz', allow_pickle=False) as data:
            ids, observed = data['ids'].tolist(), data['observed_labels']
            predicted = data['probabilities'].argmax(axis=1)
        if ids != [r['id'] for r in clean]:
            raise ValueError('Posthoc truth alignment differs')
        truth = np.array([r['label'] for r in clean])
        wrong = observed != truth
        for variant in ['relabel', 'weighted', 'insample_delete', 'oof_delete_reference']:
            d = {'seed': seed, 'variant': variant, 'originally_flipped': int(wrong.sum())}
            if variant == 'weighted':
                weight_map = read_json(root/variant/f'k10_s{seed}'/'weights.json')
                weights = np.array([weight_map[sid] for sid in ids])
                d.update({'mean_weight_flipped': float(weights[wrong].mean()),
                    'mean_weight_originally_clean': float(weights[~wrong].mean()),
                    'effective_sample_size': float(weights.sum()**2/(weights**2).sum()),
                    'weighted_flipped_fraction': float(weights[wrong].sum()/weights.sum()),
                    'min_weight': float(weights.min()), 'max_weight': float(weights.max())})
            elif variant == 'relabel':
                decisions = read_json(root/variant/f'k10_s{seed}'/'decisions.json')
                flagged = set(decisions['flagged_ids'])
                selected = np.array([sid in flagged for sid in ids])
                new = np.where(selected, predicted, observed)
                d.update({'relabels': int((new != observed).sum()),
                    'corrected_flipped': int((wrong & (new == truth)).sum()),
                    'damaged_originally_clean': int((~wrong & (new != truth)).sum()),
                    'remaining_wrong': int((new != truth).sum()), 'rows_retained': len(ids),
                    'absent_training_classes': decisions['absent_training_classes']})
            else:
                kept = set(read_json(reference/'oof'/f'seed_{seed}'/'kept_ids.json')) if variant == 'oof_delete_reference' else set(
                    read_json(root/variant/f'k10_s{seed}'/'decisions.json')['kept_ids'])
                removed = np.array([sid not in kept for sid in ids])
                d.update({'removed': int(removed.sum()), 'removed_flipped': int((removed & wrong).sum()),
                    'removed_originally_clean': int((removed & ~wrong).sum()),
                    'kept_flipped': int((~removed & wrong).sum()), 'rows_retained': len(kept)})
            diagnostics.append(d)
    costs = []
    for record in report['records']:
        seed, variant = record['seed'], record['variant']
        summary = read_json(root/variant/f'k10_s{seed}'/'run.json')
        scorer = next(r for r in report['records'] if r['variant'] == 'ce_repeat' and r['seed'] == seed)
        oof_reference = next(r for r in groups['oof_delete_reference'] if r['seed'] == seed)
        costs.append({'variant': variant, 'seed': seed, 'rows': record['rows_kept'],
            'best_epoch': record['best_epoch'], 'epochs_completed': record['epochs_completed'],
            'optimizer_steps': int(np.ceil(record['rows_kept']/summary['config']['training']['batch_size']))*record['epochs_completed'],
            'final_epoch_seconds': record['total_epoch_seconds'], 'route_wall_seconds_measured': record['route_wall_seconds'],
            'reused_oof_training_epoch_seconds': oof_reference['oof_epoch_seconds'] if variant in {'relabel', 'weighted'} else 0.,
            'shared_insample_scorer_training_epoch_seconds': scorer['total_epoch_seconds'] if variant == 'insample_delete' else 0.,
            'shared_insample_score_wall_seconds': scorer['score_wall_seconds'] if variant == 'insample_delete' else 0.,
            'peak_allocated_mb': record['peak_gpu_allocated_mb'], 'peak_reserved_mb': record['peak_gpu_reserved_mb']})
    write_json(args.output.with_suffix('.json'), {'scope': 'IDN30/10-shot/three-seed/cleaning-variants-v2',
        'summaries': summaries, 'paired_against_ce': paired, 'posthoc_diagnostics': diagnostics, 'costs': costs,
        'diagnostics_use_clean_training_labels': True, 'truth_used_for_training_or_tuning': False,
        'formal_significance_claimed': False, 'source_checksums_sha256': file_sha256(root/'checksums.json')})
    names = {'ce_reference': '带噪CE（已有参考）', 'oof_delete_reference': 'OOF删除（已有参考）',
        'ce_repeat': 'CE重跑', 'relabel': 'OOF硬重标注', 'weighted': 'OOF概率加权', 'insample_delete': '训练内打分后删除'}
    lines = ['# IDN30%、10-shot 清洗变体结果', '',
        '2026-11-17。三个种子17/42/2026，30轮上限、耐心5、验证宏F1选模。规则见[协议](清洗变体实验协议.md)，命令见[运行说明](清洗变体运行说明.md)。', '',
        '| 路线 | 测试Macro-F1 | Accuracy | OOS AUROC | 相对CE的F1均值差 |',
        '| --- | --- | --- | --- | --- |']
    for name in groups:
        cells = [f"{summaries[name][m]['mean']:.4f}±{summaries[name][m]['std_sample']:.4f}" for m in ['test_macro_f1', 'test_accuracy', 'oos_auroc']]
        lines.append(f"| {names[name]} | {' | '.join(cells)} | {paired[name]['test_macro_f1']['mean']:+.4f} |")
    lines += ['', '均值±样本标准差；差值逐种子配对后求均值。三种子只作描述性比较，不宣布显著性或OOS非劣。三个CE重跑的预测文件与已有参考逐字节一致。', '',
        '## 重标注的事后诊断', '',
        '以下只在全部训练完成后使用原始干净训练标签，不参与标记、权重或阈值选择。错标是人工注入的标签翻转，不是人工语义判定。', '',
        '| seed | 改标数 | 修正的翻转样本 | 被改坏的原始干净样本 | 改标后仍错 | 缺失训练类数 |',
        '| --- | --- | --- | --- | --- | --- |']
    for d in [r for r in diagnostics if r['variant'] == 'relabel']:
        lines.append(f"| {d['seed']} | {d['relabels']} | {d['corrected_flipped']} | {d['damaged_originally_clean']} | {d['remaining_wrong']} | {len(d['absent_training_classes'])} |")
    lines += ['', '## 删除方式与加权', '',
        '| 来源 | seed | 保留样本 | 删掉翻转样本 | 删掉原始干净样本 | 保留翻转样本 |',
        '| --- | --- | --- | --- | --- | --- |']
    for d in [r for r in diagnostics if r['variant'] in {'oof_delete_reference', 'insample_delete'}]:
        lines.append(f"| {names[d['variant']]} | {d['seed']} | {d['rows_retained']} | {d['removed_flipped']} | {d['removed_originally_clean']} | {d['kept_flipped']} |")
    lines += ['', '| seed | 翻转样本平均权重 | 原始干净样本平均权重 | 加权后的翻转质量占比 | 有效样本量 |',
        '| --- | --- | --- | --- | --- |']
    for d in [r for r in diagnostics if r['variant'] == 'weighted']:
        lines.append(f"| {d['seed']} | {d['mean_weight_flipped']:.4f} | {d['mean_weight_originally_clean']:.4f} | {d['weighted_flipped_fraction']:.4f} | {d['effective_sample_size']:.1f} |")
    lines += ['', '有效样本量为(Σw)²/Σw²，仅描述权重集中程度，不等于独立观测数量。加权后的翻转质量占比为翻转样本权重和除以全部权重和。', '',
        '训练内打分没有折外隔离，模型见过每个被打分样本；即使其后分类分数较高，也不能据此认定它比OOF更能无偏识别错标。其打分模型使用全部1500条训练，而OOF每折只用1200条，额外算力和训练样本量也不同。', '',
        '删除会缩小训练集和词表并减少每轮更新；重标注/加权保留全部文本。处理方式、样本数量、词表及更新量同时变化，当前实验不能单独归因于某一个因素。', '',
        f'![清洗变体汇总]({args.output.with_suffix(".png").name})', '',
        '## 成本', '',
        '| 路线 | seed | 样本数 | 最佳/完成轮数 | 更新数 | 最终训练含验证/秒 | 复用OOF训练/秒 | 共享训练内打分模型训练/秒 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for c in costs:
        lines.append(f"| {names[c['variant']]} | {c['seed']} | {c['rows']} | {c['best_epoch']}/{c['epochs_completed']} | {c['optimizer_steps']} | {c['final_epoch_seconds']:.2f} | {c['reused_oof_training_epoch_seconds']:.2f} | {c['shared_insample_scorer_training_epoch_seconds']:.2f} |")
    lines += ['', f"本轮12次新训练和评估墙钟 {report['wall_seconds']:.2f} 秒。OOF概率复用上一阶段15次训练，表中成本为历史实测归属，不是本轮重新运行；重标注和加权各自完整执行都需这份OOF成本，但本轮共享资产不能重复累计为新增成本。",
        '取消OOF路线需CE打分模型和最终重训，打分模型在CE重跑中共享。其训练内推理耗时、各路线墙钟及显存详见JSON；CE重跑墙钟还含独立测试/绘图，不能当作纯清洗成本。', '',
        '## 验收与后续', '',
        f'12组共66000条预测、处理决策、词表、权重及训练内概率保存在 `{root.as_posix()}/`。核验独立重算指标，重放处理规则，重建训练哈希并核对权重对齐；概率来源的模型/训练哈希同时保存。', '',
        '完整模型权重保留本地，结果核验不等于从模型重新生成概率或全新环境重训。后续仍需其他样本量/噪声比例、正式配对统计、公开基线验收及报告/PPT。']
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    cache = Path('tmp/matplotlib').resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_names = ['ce_reference', 'oof_delete_reference', 'relabel', 'weighted', 'insample_delete']
    ticklabels = ['Noisy CE', 'OOF\ndelete', 'OOF\nrelabel', 'OOF\nweighted', 'In-sample\ndelete']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, label in zip(axes, ['test_macro_f1', 'oos_auroc'], ['Test Macro-F1', 'OOS AUROC']):
        for x, name in enumerate(plot_names):
            values = [r[metric] for r in groups[name]]
            ax.scatter(np.full(3, x), values, alpha=.65, color='#236999')
            ax.plot([x-.2, x+.2], [np.mean(values)]*2, color='#172c3b', linewidth=3)
        ax.set(xticks=range(5), xticklabels=ticklabels, ylabel=label, ylim=(0, 1))
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('IDN30% / 10-shot / seeds 17, 42, 2026')
    fig.tight_layout()
    fig.savefig(args.output.with_suffix('.png'), dpi=180)
    plt.close(fig)
    print('Wrote cleaning-variant outcomes, post-hoc diagnostics and cost attribution.')


if __name__ == '__main__':
    main()
