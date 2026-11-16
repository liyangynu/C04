"""Describe paired outcomes across the two frozen 30% noise conditions."""
import argparse
import os
from pathlib import Path

import numpy as np

from .common import file_sha256, read_json, write_json
from .formal_protocol import comparison_noise, METHODS, SEEDS


def checked_comparison(root, expected_noise):
    for name, digest in read_json(root/'checksums.json').items():
        if file_sha256(root/name) != digest:
            raise ValueError(f'Published result changed: {root / name}')
    audit = read_json(root.parent/f'{root.name}_audit.json')
    if not audit['all_passed'] or audit['export_sha256'] != file_sha256(root/'checksums.json'):
        raise ValueError('Run independent result audit before cross-condition analysis')
    comparison = read_json(root/'comparison.json')
    if comparison_noise(comparison) != (expected_noise, .3):
        raise ValueError('Unexpected noise condition')
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--idn', type=Path, default=Path('reports/stage3/formal_idn10_v2'))
    parser.add_argument('--symmetric', type=Path, default=Path('reports/stage3/formal_sym10_v2'))
    parser.add_argument('--output', type=Path, required=True, help='Output file stem')
    args = parser.parse_args()
    idn = checked_comparison(args.idn, 'instance')
    sym = checked_comparison(args.symmetric, 'symmetric')
    indexed = [{(r['method'], r['seed']): r for r in result['records']} for result in [idn, sym]]
    clean_checks = []
    for method in METHODS:
        for seed in SEEDS:
            key = (method, seed)
            left, right = [index[key] for index in indexed]
            run_name = f'k10_s{seed}'
            configs = [read_json(root/method/run_name/'run.json')['config']
                       for root in [args.idn, args.symmetric]]
            if configs[0] != configs[1]:
                raise ValueError('Noise conditions use different model/training settings')
            if method == 'clean_ce':
                if left['train_sha256'] != right['train_sha256']:
                    raise ValueError('Different clean fewshot reference')
                hashes = [file_sha256(root/method/run_name/'predictions.jsonl')
                          for root in [args.idn, args.symmetric]]
                if hashes[0] != hashes[1]:
                    raise ValueError('Repeated clean-reference predictions differ')
                clean_checks.append({'seed': seed, 'predictions_sha256': hashes[0], 'byte_identical': True})
    differences = {}
    for method in METHODS:
        differences[method] = {}
        for metric in ['test_macro_f1', 'test_accuracy', 'oos_auroc']:
            values = [indexed[1][method, seed][metric]-indexed[0][method, seed][metric] for seed in SEEDS]
            differences[method][metric] = {'seeds': SEEDS, 'symmetric_minus_idn': values,
                'mean': float(np.mean(values)), 'std_sample': float(np.std(values, ddof=1))}
    write_json(args.output.with_suffix('.json'), {
        'scope': '10-shot, 30% noise, three seeds, v2',
        'sources': {str(root): file_sha256(root/'checksums.json') for root in [args.idn, args.symmetric]},
        'clean_reference_repeated_predictions': clean_checks,
        'differences': differences, 'significance_claimed': False,
        'noise_selection_and_target_namespaces_differ_by_type': True})
    names = {'clean_ce': '干净CE', 'ce': '带噪CE', 'gce': 'GCE(q=0.7)', 'cl_ce': '清洗后CE'}
    lines = ['# 10-shot、30%噪声的两类条件比较', '',
        '沿用v2配置，每类条件各三个种子。本表为描述性比较，不替代正式统计检验。', '',
        '| 方法 | IDN Macro-F1 | 对称噪声 Macro-F1 | 对称−IDN配对均值 |',
        '| --- | --- | --- | --- |']
    for method in METHODS:
        cells = []
        for index in indexed:
            values = [index[method, seed]['test_macro_f1'] for seed in SEEDS]
            cells.append(f'{np.mean(values):.4f}±{np.std(values, ddof=1):.4f}')
        lines.append(f"| {names[method]} | {' | '.join(cells)} | {differences[method]['test_macro_f1']['mean']:+.4f} |")
    lines += ['', '误差项为种子间样本标准差，非置信区间。两类噪声使用相同干净子集、模型与预算，但噪声选择和目标标签采用不同随机命名空间，翻转集合不相同；不能将分数差单独归因于语义相似性。', '',
        '三个干净CE参考在本轮重新训练，逐样本预测与上一轮逐字节一致。该检查只覆盖干净参考，不代表所有带噪模型重复训练均已验证。', '',
        '![两类噪声下的逐种子结果]('+args.output.with_suffix('.png').name+')', '',
        '图中每条细线表示同一方法、同一种子在两种条件下的结果，粗线表示三种子均值。完整逐种子差值和OOS AUROC比较见同名JSON。清洗诊断和各自成本见两组正式结果报告。']
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    cache = Path('tmp/matplotlib').resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {'ce': '#1764a0', 'gce': '#b56512', 'cl_ce': '#387a45'}
    for ax, metric, label in zip(axes, ['test_macro_f1', 'oos_auroc'], ['Test Macro-F1', 'OOS AUROC']):
        for method, color in colors.items():
            points = np.array([[index[method, seed][metric] for index in indexed] for seed in SEEDS])
            for row in points:
                ax.plot([0, 1], row, color=color, alpha=.3, linewidth=1, marker='o', markersize=3)
            ax.plot([0, 1], points.mean(axis=0), color=color, linewidth=2, marker='s',
                    label={'ce': 'CE', 'gce': 'GCE (q=0.7)', 'cl_ce': 'CL + CE'}[method])
        ax.set(xticks=[0, 1], xticklabels=['IDN30%', 'Symmetric30%'], xlim=(-.15, 1.15),
               ylim=(0, 1), ylabel=label, title=label)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(loc='best')
    fig.suptitle('10-shot / seeds 17, 42, 2026 / maximum 30 epochs')
    fig.tight_layout()
    fig.savefig(args.output.with_suffix('.png'), dpi=180)
    plt.close(fig)
    print('PASS: matched configurations; 3 repeated clean references have identical predictions.')


if __name__ == '__main__':
    main()
