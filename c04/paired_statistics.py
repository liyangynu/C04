"""Paired seed-level inference for the fixed formal comparison results."""
import argparse
from pathlib import Path

import numpy as np
from scipy import stats

from .common import file_sha256, read_json, write_json


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count-rank)*values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def summarize_pair(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError('Paired samples must be same-length one-dimensional arrays')
    delta = left-right
    n = len(delta)
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1))
    se = sd/np.sqrt(n)
    critical = float(stats.t.ppf(0.975, n-1))
    if np.all(delta == 0):
        t_stat, t_p = 0.0, 1.0
        w_stat, w_p = 0.0, 1.0
    else:
        t = stats.ttest_rel(left, right, alternative='two-sided')
        t_stat, t_p = float(t.statistic), float(t.pvalue)
        w = stats.wilcoxon(delta, alternative='two-sided', method='exact', zero_method='wilcox')
        w_stat, w_p = float(w.statistic), float(w.pvalue)
    return {
        'n_pairs': n,
        'left_values': left.tolist(),
        'right_values': right.tolist(),
        'paired_differences': delta.tolist(),
        'mean_difference': mean,
        'sd_difference': sd,
        't_95_ci': [mean-critical*se, mean+critical*se],
        'paired_t': {'statistic': t_stat, 'p_two_sided': t_p},
        'wilcoxon_exact': {'statistic': w_stat, 'p_two_sided': w_p},
    }


def _records(path, key):
    records = read_json(path)['records']
    result = {}
    for record in records:
        name = record[key]
        seed = int(record['seed'])
        if seed in result.setdefault(name, {}):
            raise ValueError(f'Duplicate seed {seed} for {name} in {path}')
        result[name][seed] = record
    return result


def _compare(left_records, right_records, metric, label, family, require_same_observed=True):
    seeds = sorted(left_records)
    if seeds != sorted(right_records) or len(seeds) != 3:
        raise ValueError(f'{label}: expected the same three seed IDs')
    for seed in seeds:
        left, right = left_records[seed], right_records[seed]
        if require_same_observed and 'observed_sha256' in left and 'observed_sha256' in right:
            if left['observed_sha256'] != right['observed_sha256']:
                raise ValueError(f'{label}: paired observed input differs for seed {seed}')
    return {
        'family': family,
        'comparison': label,
        'metric': metric,
        'seeds': seeds,
        **summarize_pair([left_records[s][metric] for s in seeds],
                         [right_records[s][metric] for s in seeds]),
    }


def verify_bundle(root, report_name, audit_path):
    audit = read_json(audit_path)
    if audit.get('all_passed') is not True:
        raise ValueError(f'Prior result audit did not pass: {audit_path}')
    checksums = read_json(root/'checksums.json')
    for relative, digest in checksums.items():
        if file_sha256(root/relative) != digest:
            raise ValueError(f'Result evidence checksum differs: {root/relative}')
    if report_name not in checksums or file_sha256(root/report_name) != checksums[report_name]:
        raise ValueError(f'Result summary is not covered by checksums: {root/report_name}')
    return file_sha256(root/'checksums.json')


def analyze(idn_path, symmetric_path, ablation_path, cleaning_path):
    idn_root, symmetric_root = idn_path.parent, symmetric_path.parent
    ablation_root, cleaning_root = ablation_path.parent, cleaning_path.parent
    sources = {
        'idn': verify_bundle(idn_root, 'comparison.json', idn_root.parent/'formal_idn10_v2_audit.json'),
        'symmetric': verify_bundle(symmetric_root, 'comparison.json', symmetric_root.parent/'formal_sym10_v2_audit.json'),
        'ablations': verify_bundle(ablation_root, 'ablation.json', ablation_root.parent/'ablations_idn10_v2_audit.json'),
        'cleaning': verify_bundle(cleaning_root, 'comparison.json', cleaning_root.parent/'cleaning_variants_idn10_v2_audit.json'),
    }
    idn, symmetric = _records(idn_path, 'method'), _records(symmetric_path, 'method')
    ablation, cleaning = _records(ablation_path, 'variant'), _records(cleaning_path, 'variant')
    comparisons = []
    for condition, records in [('IDN30%', idn), ('Symmetric30%', symmetric)]:
        for method in ['gce', 'cl_ce']:
            comparisons.append(_compare(records[method], records['ce'], 'test_macro_f1',
                f'{condition}: {method} - noisy CE', 'primary_macro_f1'))
            comparisons.append(_compare(records[method], records['ce'], 'oos_auroc',
                f'{condition}: {method} - noisy CE', 'primary_oos_auroc'))
    reference = idn['gce']
    for variant in ['last_state', 'frozen_embeddings', 'q_02', 'q_04', 'q_09', 'q_095']:
        comparisons.append(_compare(ablation[variant], reference, 'test_macro_f1',
            f'IDN30%: {variant} - GCE q=0.7', 'exploratory_ablations_macro_f1'))
        comparisons.append(_compare(ablation[variant], reference, 'oos_auroc',
            f'IDN30%: {variant} - GCE q=0.7', 'exploratory_ablations_oos_auroc'))
    for variant in ['relabel', 'weighted', 'insample_delete']:
        comparisons.append(_compare(cleaning[variant], idn['ce'], 'test_macro_f1',
            f'IDN30%: {variant} - noisy CE', 'exploratory_cleaning_macro_f1'))
        comparisons.append(_compare(cleaning[variant], idn['ce'], 'oos_auroc',
            f'IDN30%: {variant} - noisy CE', 'exploratory_cleaning_oos_auroc'))
    for family in ['primary_macro_f1', 'exploratory_ablations_macro_f1', 'exploratory_cleaning_macro_f1']:
        indices = [i for i, row in enumerate(comparisons) if row['family'] == family]
        adjusted = holm_adjust([comparisons[i]['paired_t']['p_two_sided'] for i in indices])
        for index, p_value in zip(indices, adjusted):
            comparisons[index]['paired_t']['p_holm_within_family'] = p_value
        wilcoxon_adjusted = holm_adjust([comparisons[i]['wilcoxon_exact']['p_two_sided'] for i in indices])
        for index, p_value in zip(indices, wilcoxon_adjusted):
            comparisons[index]['wilcoxon_exact']['p_holm_within_family'] = p_value
    return {
        'title': 'Paired seed-level statistics for fixed 10-shot, 30% noise comparisons',
        'seeds': [17, 42, 2026],
        'source_checksums_sha256': sources,
        'interpretation': {
            'primary_family': 'Four Macro-F1 comparisons: GCE and confident-learning CE vs noisy CE, in IDN and symmetric noise; Holm adjustment controls this family.',
            'exploratory_families': 'Ablations and cleaning variants are separate exploratory families; Holm adjustment is within each family.',
            'confidence_interval': '95%配对t区间针对种子层面平均差；假设种子差值相互独立且总体近似正态。',
            'tests': '双侧配对t检验与双侧精确Wilcoxon符号秩敏感性检验。',
            'limitations': '只有三个配对种子，t区间和t检验易受分布假设影响；三个非零差值时精确Wilcoxon双侧p值最低为0.25。种子层面的检验不能证明跨数据集或数据划分的总体性能。',
            'protocol_status': '这是对已固定并发布结果的事后统计分析，不是预注册研究；本次分析未根据测试结果调参。'
        },
        'comparisons': comparisons,
    }


def render_markdown(report):
    labels = {
        'primary_macro_f1': '主要比较：Macro-F1（Holm校正）',
        'exploratory_ablations_macro_f1': '探索性消融：Macro-F1（Holm校正）',
        'exploratory_cleaning_macro_f1': '探索性清洗：Macro-F1（Holm校正）',
    }
    lines = ['# 固定实验的配对统计分析', '',
        '10-shot、30%噪声，配对种子17/42/2026。差值均为左侧方法减右侧参照；区间针对种子层面的平均差。']
    selected = [r for r in report['comparisons'] if r['metric'] == 'test_macro_f1']
    for family in labels:
        family_rows = [r for r in selected if r['family'] == family]
        if not family_rows:
            continue
        lines += ['', f'## {labels[family]}', '',
            '| 比较 | 差值（seed 17 / 42 / 2026） | 平均差 [95% t区间] | t检验 p / Holm | 精确Wilcoxon p / Holm |',
            '| --- | --- | --- | --- | --- |']
        for row in family_rows:
            d = ', '.join(f"{x:+.4f}" for x in row['paired_differences'])
            lo, hi = row['t_95_ci']
            t = row['paired_t']
            w = row['wilcoxon_exact']
            lines.append(f"| {row['comparison']} | {d} | {row['mean_difference']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {t['p_two_sided']:.4f} / {t['p_holm_within_family']:.4f} | {w['p_two_sided']:.4f} / {w['p_holm_within_family']:.4f} |")
    info = report['interpretation']
    lines += ['', '## 口径与局限', '', info['confidence_interval'], '', info['tests'], '',
        info['limitations'], '', info['protocol_status'], '',
        'AUROC使用同样的配对统计量，保存在JSON中，作为次要指标单独解释；不与主要Macro-F1假设混合。探索性结果只作后续假设生成，不推导方法普遍优越性。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--idn', type=Path, default=Path('reports/stage3/formal_idn10_v2/comparison.json'))
    parser.add_argument('--symmetric', type=Path, default=Path('reports/stage3/formal_sym10_v2/comparison.json'))
    parser.add_argument('--ablations', type=Path, default=Path('reports/stage4/ablations_idn10_v2/ablation.json'))
    parser.add_argument('--cleaning', type=Path, default=Path('reports/stage4/cleaning_variants_idn10_v2/comparison.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/stage4/paired_statistics.json'))
    parser.add_argument('--markdown', type=Path, default=Path('docs/配对统计结果.md'))
    args = parser.parse_args()
    report = analyze(args.idn, args.symmetric, args.ablations, args.cleaning)
    write_json(args.output, report)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding='utf-8', newline='\n')
    print(f'Wrote paired statistics for {len(read_json(args.output)["comparisons"])} comparisons.')


if __name__ == '__main__':
    main()
