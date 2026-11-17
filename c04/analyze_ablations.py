"""Describe the frozen ablations without selecting a new main configuration."""
import argparse
import os
from pathlib import Path

import numpy as np

from .common import file_sha256, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('reports/stage4/ablations_idn10_v2'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    audit = read_json(root.parent/f'{root.name}_audit.json')
    if not audit['all_passed'] or audit['export_sha256'] != file_sha256(root/'checksums.json'):
        raise ValueError('Independent audit must pass before analysis')
    for name, digest in read_json(root/'checksums.json').items():
        if file_sha256(root/name) != digest:
            raise ValueError('Ablation evidence changed')
    report = read_json(root/'ablation.json')
    plan, records = report['plan'], report['records']
    reference = Path(plan['reference_results'])
    if file_sha256(reference/'checksums.json') != report['reference_checksums_sha256']:
        raise ValueError('Reference result version changed')
    baseline = read_json(reference/'gce/matrix.json')['records']
    groups = {'reference': baseline}
    for variant in plan['variants']:
        groups[variant['name']] = [r for r in records if r['variant'] == variant['name']]
    metrics = ['best_val_macro_f1', 'test_macro_f1', 'test_accuracy', 'oos_auroc']
    summaries, paired = {}, {}
    for name, rows in groups.items():
        summaries[name] = {metric: {'mean': float(np.mean([r[metric] for r in rows])),
            'std_sample': float(np.std([r[metric] for r in rows], ddof=1))} for metric in metrics}
        if name == 'reference':
            continue
        paired[name] = {}
        for metric in metrics:
            values = [next(r[metric] for r in rows if r['seed'] == seed)
                - next(r[metric] for r in baseline if r['seed'] == seed) for seed in plan['seeds']]
            paired[name][metric] = {'seeds': plan['seeds'], 'differences': values,
                'mean': float(np.mean(values)), 'std_sample': float(np.std(values, ddof=1))}
    cost = []
    for name, rows in groups.items():
        if name == 'reference':
            continue
        for row in rows:
            path = root/name/f"k10_s{row['seed']}"
            run = read_json(path/'run.json')
            cost.append({'variant': name, 'seed': row['seed'], 'parameters': run['parameters'],
                'best_epoch': row['best_epoch'], 'epochs_completed': row['epochs_completed'],
                'optimizer_steps': int(np.ceil(run['train_rows']/run['config']['training']['batch_size']))*row['epochs_completed'],
                'epoch_seconds': row['total_epoch_seconds'], 'route_wall_seconds': row['route_wall_seconds'],
                'peak_allocated_mb': row['peak_gpu_allocated_mb'], 'peak_reserved_mb': row['peak_gpu_reserved_mb']})
    analysis = {'scope': 'IDN30/10-shot/three-seed/v2', 'summaries': summaries,
        'paired_against_gce_q07_attention': paired, 'new_run_costs': cost,
        'reference_reused': True, 'new_training_fits': 18, 'main_q_unchanged': .7,
        'statistical_significance_claimed': False, 'test_based_configuration_selection': False,
        'source_checksums_sha256': file_sha256(root/'checksums.json')}
    write_json(args.output.with_suffix('.json'), analysis)
    labels = {'reference': '注意力+可微调嵌入，q=0.7（已有参考）', 'last_state': '末状态汇聚，q=0.7',
        'frozen_embeddings': '冻结嵌入，q=0.7', 'q_02': 'q=0.2', 'q_04': 'q=0.4', 'q_09': 'q=0.9', 'q_095': 'q=0.95'}
    lines = ['# IDN30%、10-shot 三组消融结果', '',
        '2026-11-17。种子17/42/2026，30轮上限、耐心5、验证宏F1选模。设置见[消融协议](消融实验协议.md)，命令见[运行说明](消融实验运行说明.md)。', '',
        '新增18次训练；完整q=0.7参考复用上一阶段三种子，不计为本轮新训练。', '',
        '| 设置 | 最佳验证Macro-F1 | 测试Macro-F1 | Accuracy | OOS AUROC | 相对参考F1差值 |',
        '| --- | --- | --- | --- | --- | --- |']
    for name in groups:
        cells = [f"{summaries[name][m]['mean']:.4f}±{summaries[name][m]['std_sample']:.4f}" for m in metrics]
        delta = '—' if name == 'reference' else f"{paired[name]['test_macro_f1']['mean']:+.4f}"
        lines.append(f"| {labels[name]} | {' | '.join(cells)} | {delta} |")
    lines += ['', '表中为三种子均值±样本标准差。差值先按种子配对，再取均值；不是显著性检验。q=0.7仍为固定主参数，其他q仅作敏感性检查，不根据测试排名改写主配置。', '',
        '## 模块与嵌入', '',
        f"末状态汇聚相对注意力参考的平均测试F1差值为 {paired['last_state']['test_macro_f1']['mean']:+.4f}；冻结嵌入的差值为 {paired['frozen_embeddings']['test_macro_f1']['mean']:+.4f}。",
        '移除注意力会改变参数量及初始化随机数消耗，同一种子不代表所有共享层初始参数逐项相同。实际训练轮数也可能因早停不同，当前结果只适用于本组数据、模型和训练规则，不能概括为注意力或微调在所有条件下无效。', '',
        '三个冻结嵌入的最佳模型都与初始嵌入矩阵逐项一致；导出初始/最佳矩阵供独立检查。其他分类参数仍参与学习。', '',
        '## q敏感性', '',
        '图中同时保留验证与测试表现。q较大时GCE对低目标类概率样本的梯度较小；固定有限预算下的差异可能同时反映收敛速度与噪声鲁棒性，当前实验不能单独分离二者。', '',
        f'![消融汇总]({args.output.with_suffix(".png").name})', '',
        '## 成本与训练过程', '',
        '| 设置 | seed | 最佳/完成轮数 | 参数量 | 更新次数 | 训练含验证/秒 | 路线墙钟/秒 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in cost:
        lines.append(f"| {labels[r['variant']]} | {r['seed']} | {r['best_epoch']}/{r['epochs_completed']} | {r['parameters']} | {r['optimizer_steps']} | {r['epoch_seconds']:.2f} | {r['route_wall_seconds']:.2f} |")
    lines += ['', f"本轮顺序训练和评估墙钟共 {report['wall_seconds']:.2f} 秒，不含环境安装与下载。逐轮PyTorch已分配/保留显存随训练摘要保存，不是整卡占用。", '',
        '## 验收范围', '',
        f'原始18组共99000条预测、配置快照、词表、日志、图表及冻结矩阵在 `{root.as_posix()}/`。独立核验重算指标，检查每项变更字段、同种子输入/验证指纹、运行环境和词表来源。', '',
        '本轮完成模块、嵌入、q三个消融方向在一个条件下的检查；尚未完成清洗变体/取消OOF消融、其他样本量和噪声比例、完整配对统计、公开基线验收、最终报告与PPT。']
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    cache = Path('tmp/matplotlib').resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, x in zip(['reference', 'last_state', 'frozen_embeddings'], range(3)):
        values = [r['test_macro_f1'] for r in groups[name]]
        axes[0].scatter(np.full(3, x), values, alpha=.6, color='#236999')
        axes[0].plot([x-.15, x+.15], [np.mean(values)]*2, color='#172c3b', linewidth=3)
    axes[0].set(xticks=[0, 1, 2], xticklabels=['Attention\n(reference)', 'Final states', 'Frozen\nembeddings'],
                ylabel='Test Macro-F1', ylim=(0, 1), title='Module and embedding ablations')
    qs, order = [.2, .4, .7, .9, .95], ['q_02', 'q_04', 'reference', 'q_09', 'q_095']
    for metric, label, color in [('best_val_macro_f1', 'Validation', '#236999'), ('test_macro_f1', 'Test', '#b56512')]:
        axes[1].errorbar(qs, [summaries[n][metric]['mean'] for n in order],
            yerr=[summaries[n][metric]['std_sample'] for n in order], label=label,
            color=color, marker='o', capsize=3)
    axes[1].set(xlabel='GCE q', ylabel='Macro-F1', ylim=(0, 1), title='q sensitivity (mean +/- sample SD)')
    axes[1].legend()
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('IDN30% / 10-shot / seeds 17, 42, 2026')
    fig.tight_layout()
    fig.savefig(args.output.with_suffix('.png'), dpi=180)
    plt.close(fig)
    print('Wrote descriptive ablation tables, paired differences and figure.')


if __name__ == '__main__':
    main()
