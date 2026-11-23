"""Export the final core matrix and verify its published seed-level tables."""
import argparse
import csv
import io
import math
from pathlib import Path
import statistics

from .common import file_sha256, read_json, write_json

SEEDS = [17, 42, 2026]
METHODS = ['ce', 'gce', 'confident_learning']
METRICS = ['test_macro_f1', 'test_accuracy', 'oos_auroc', 'f1_retention']
FIELDS = ['noise', 'rate', 'shots_per_class', 'method', 'seed', 'config',
          'best_epoch', 'best_val_macro_f1', 'epochs_completed', *METRICS,
          'prediction_rows', 'predictions_sha256', 'train_sha256']


def conditions():
    return {(noise, rate, shots) for noise in ('instance', 'symmetric')
            for rate, shots in [(.1, 10), (.2, 10), (.3, 5), (.3, 10), (.3, 20), (.3, 100)]}


def identity(row):
    return row['noise'], row['rate'], row['shots_per_class'], row['method'], row['seed']


def close(actual, expected, description):
    if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual - expected) > 1e-12:
        raise ValueError(f'{description} differs')


def summaries(records):
    groups = {}
    for row in records:
        groups.setdefault(identity(row)[:4], []).append(row)
    result = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row['seed'])
        if [row['seed'] for row in rows] != SEEDS:
            raise ValueError('Missing or duplicate seeds in a condition/method')
        metrics = {}
        for field in METRICS:
            values = [row[field] for row in rows]
            metrics[field] = {'values': values, 'mean': statistics.mean(values),
                              'std_sample': statistics.stdev(values)}
        result.append(dict(zip(['noise', 'rate', 'shots_per_class', 'method'], key),
                           seeds=SEEDS, n=3, metrics=metrics))
    return result


def csv_text(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def summary_rows(groups):
    rows = []
    fields = ['noise', 'rate', 'shots_per_class', 'method', 'n']
    fields += [f'{metric}_{stat}' for metric in METRICS for stat in ('mean', 'std_sample')]
    for group in groups:
        row = {field: group[field] for field in fields[:5]}
        row.update({f'{metric}_{stat}': group['metrics'][metric][stat]
                    for metric in METRICS for stat in ('mean', 'std_sample')})
        rows.append(row)
    return rows, fields


def verify(output):
    """Check coverage, seed/config associations and aggregates without local models."""
    checksums = read_json(output / 'checksums.json')
    required = {'results.json', 'core_results.csv', 'clean_reference.csv', 'summary.csv',
                'provenance.json', 'validation.json', 'README.md'}
    if set(checksums) != required:
        raise ValueError('Result package file manifest differs')
    for name, digest in checksums.items():
        if file_sha256(output / name) != digest:
            raise ValueError(f'Checksum differs: {name}')
    package = read_json(output / 'results.json')
    if package['seeds'] != SEEDS or package['condition_count'] != 12:
        raise ValueError('Protocol seeds/condition count differs')
    noisy, clean = package['noisy_results'], package['clean_references']
    expected = {(n, r, k, m, s) for n, r, k in conditions() for m in METHODS for s in SEEDS}
    if len(noisy) != 108 or {identity(row) for row in noisy} != expected:
        raise ValueError('Expected exactly 108 distinct condition/method/seed results')
    if len(clean) != 12 or {identity(row) for row in clean} != {
            ('clean', 0.0, k, 'clean_ce', s) for k in (5, 10, 20, 100) for s in SEEDS}:
        raise ValueError('Expected exactly 12 distinct clean references')
    provenance = read_json(output / 'provenance.json')
    for path, digest in provenance['config_sha256'].items():
        if file_sha256(path) != digest:
            raise ValueError(f'Configuration changed: {path}')
    clean_map = {(row['shots_per_class'], row['seed']): row for row in clean}
    for row in noisy + clean:
        for field in METRICS[:3]:
            if not math.isfinite(row[field]) or not 0 <= row[field] <= 1:
                raise ValueError(f'Invalid metric: {field}')
        expected_config = 'configs/formal_gce.json' if row['method'] == 'gce' else 'configs/formal_ce.json'
        if row['config'] != expected_config or row['config_sha256'] != file_sha256(expected_config):
            raise ValueError('Method/config association differs')
        config = read_json(expected_config)
        history = row['validation_history']
        if ([item['epoch'] for item in history] != list(range(1, row['epochs_completed'] + 1))
                or not history or len(history) > config['training']['max_epochs']
                or any(not math.isfinite(item['val_macro_f1']) for item in history)):
            raise ValueError('Incomplete validation history')
        best = max(history, key=lambda item: item['val_macro_f1'])
        if row['best_epoch'] != best['epoch']:
            raise ValueError('Selected epoch differs from validation selection')
        close(row['best_val_macro_f1'], best['val_macro_f1'], 'Best validation F1')
        reference = clean_map[(row['shots_per_class'], row['seed'])]['test_macro_f1']
        close(row['f1_retention'], row['test_macro_f1'] / reference, 'F1 retention')
        if row['prediction_rows'] != 5500:
            raise ValueError('Expected 4500 ID plus 1000 OOS predictions')
    groups = summaries(noisy + clean)
    if package['summaries'] != groups:
        raise ValueError('Published means or sample standard deviations differ')
    aggregate, aggregate_fields = summary_rows(groups)
    for name, rows, fields in [('core_results.csv', noisy, FIELDS),
                                ('clean_reference.csv', clean, FIELDS),
                                ('summary.csv', aggregate, aggregate_fields)]:
        if (output / name).read_text(encoding='utf-8') != csv_text(rows, fields):
            raise ValueError(f'CSV differs from seed-level JSON: {name}')
    validation = read_json(output / 'validation.json')
    if (not validation['all_passed'] or validation['prediction_runs'] != 120
            or validation['prediction_rows'] != 660000):
        raise ValueError('Incomplete local prediction audit')
    print('PASS: 108 noisy runs, 12 clean references, 3 seeds, configs and 40 summary rows')


def export(output):
    """Read completed local runs; recompute saved prediction metrics before exporting."""
    from .check_expanded_predictions import (select_records, run_directory,
                                             verify_predictions, verify_training)
    from .runtime import load_official_split
    if output.exists():
        raise ValueError('Use a new output directory; existing releases are not overwritten')
    base = Path('artifacts/stage5_expanded_runs/progress.json')
    extra = Path('artifacts/completion_v1/progress.json')
    completion = read_json('artifacts/completion_v1/completed.json')
    if (completion['progress_sha256'] != file_sha256(extra)
            or completion['plan_sha256'] != file_sha256('configs/completion_v1.json')):
        raise ValueError('Supplementary protocol/progress changed')
    noisy, clean = select_records(read_json(base))
    for row in read_json(extra):
        if row['kind'] == 'noise':
            key = (row['noise'], row['rate'], row['shots_per_class'], row['seed'], row['method'])
            if key in noisy:
                raise ValueError('Duplicate supplementary run')
            noisy[key] = row
    if len(noisy) != 108:
        raise ValueError('Expected 108 core noisy results')
    test, labels = load_official_split('data/processed/clean/test.jsonl', 'test')
    oos, _ = load_official_split('data/processed/clean/oos_test.jsonl', 'oos_test')
    truth = [(row['id'], row['label']) for row in test + oos]
    exported = []
    for record in [*noisy.values(), *clean.values()]:
        directory, run_path, summary_path = run_directory(record)
        values = verify_predictions(directory, truth, labels, record)
        verify_training(directory, run_path, summary_path, record)
        run, training = read_json(run_path), read_json(summary_path)
        is_clean = record['method'] == 'clean_ce'
        config = 'configs/formal_gce.json' if record['method'] == 'gce' else 'configs/formal_ce.json'
        exported.append({
            'noise': 'clean' if is_clean else record['noise'],
            'rate': 0.0 if is_clean else record['rate'],
            'shots_per_class': record['shots_per_class'],
            'method': 'confident_learning' if record['method'] == 'cl_ce' else record['method'],
            'seed': record['seed'], 'config': config, 'config_sha256': file_sha256(config),
            'best_epoch': training['best_epoch'], 'best_val_macro_f1': training['best_val_macro_f1'],
            'epochs_completed': training['epochs_completed'],
            'validation_history': [{'epoch': h['epoch'], 'val_macro_f1': h['val_macro_f1']}
                                   for h in training['history']],
            **values, 'f1_retention': record['f1_retention'],
            'train_sha256': run['fingerprints']['train'],
            'evaluation_sha256': file_sha256(directory / 'metrics.json'),
            'training_summary_sha256': file_sha256(summary_path),
            'input_fingerprints': run['fingerprints'],
        })
        if len(exported) % 20 == 0:
            print(f'Checked {len(exported)}/120 saved prediction runs', flush=True)
    noisy_rows = sorted([r for r in exported if r['method'] != 'clean_ce'], key=identity)
    clean_rows = sorted([r for r in exported if r['method'] == 'clean_ce'], key=identity)
    groups = summaries(noisy_rows + clean_rows)
    analysis_path = Path('artifacts/completion_v1/analysis/analysis.json')
    analysis = read_json(analysis_path)
    if (analysis['progress_sha256'] != file_sha256(extra) or analysis['condition_count'] != 12
            or analysis['unique_noisy_results'] != 108):
        raise ValueError('Final analysis is stale')
    for group in groups:
        if group['noise'] == 'clean':
            continue
        original = next(item for item in analysis['condition_summaries']
                        if (item['noise'], item['rate'], item['shots_per_class']) ==
                        (group['noise'], group['rate'], group['shots_per_class']))
        for field in METRICS:
            saved = original['methods'][group['method']][field]
            for value, expected in zip(group['metrics'][field]['values'], saved['values']):
                close(value, expected, 'Final analysis seed value')
            for stat in ('mean', 'std_sample'):
                close(group['metrics'][field][stat], saved[stat], 'Final analysis aggregate')
    output.mkdir(parents=True)
    write_json(output / 'results.json', {'schema_version': 1, 'seeds': SEEDS,
               'condition_count': 12, 'noisy_results': noisy_rows,
               'clean_references': clean_rows, 'summaries': groups})
    for name, rows in [('core_results.csv', noisy_rows), ('clean_reference.csv', clean_rows)]:
        (output / name).write_text(csv_text(rows, FIELDS), encoding='utf-8', newline='\n')
    aggregate, fields = summary_rows(groups)
    (output / 'summary.csv').write_text(csv_text(aggregate, fields), encoding='utf-8', newline='\n')
    config_paths = ['configs/formal_ce.json', 'configs/formal_gce.json',
                    'configs/expanded_matrix_v1.json', 'configs/completion_v1.json', 'configs/protocol.json']
    write_json(output / 'provenance.json', {
        'config_sha256': {path: file_sha256(path) for path in config_paths},
        'local_source_sha256': {str(path): file_sha256(path) for path in [base, extra, analysis_path]},
        'dataset_sha256': read_json('configs/protocol.json')['dataset']['sha256'],
        'exporter_sha256': file_sha256(Path(__file__)),
        'seed_order': SEEDS, 'deduplication': '12 clean references and 108 distinct noisy condition/method/seed keys',
    })
    write_json(output / 'validation.json', {
        'all_passed': True, 'prediction_runs': len(exported),
        'prediction_rows': sum(row['prediction_rows'] for row in exported),
        'scope': 'Recomputed saved predictions against official IDs/truth, per-class metrics, confusion matrices, training configs, validation selection and final aggregates.',
        'limitations': ['Raw predictions and weights remain local; only their hashes are exported.',
                        'Export does not retrain models or replay weights/OOF/encoder.',
                        'Verify-only checks the published tables and configs, not absent local predictions.'],
    })
    (output / 'README.md').write_text('''# 最终核心实验结果

固定种子：17、42、2026。两类噪声各含10-shot下10%/20%/30%，以及30%下5/20/100-shot，共12个条件；每个条件包含CE、GCE、置信学习清洗后CE，共108组带噪结果。另列5/10/20/100-shot各三个种子的12组干净CE参考，避免重复计数。

| 文件 | 内容 |
| --- | --- |
| core_results.csv | 108组逐种子指标、配置路径、验证选模和预测/输入校验值 |
| clean_reference.csv | 12组同预算干净参考 |
| summary.csv | 36组方法/条件及4组干净参考的均值和样本标准差 |
| results.json | 完整精度逐种子结果、验证历史、输入指纹和汇总 |
| provenance.json | 数据、配置、来源进度和导出代码的SHA256 |
| validation.json | 导出时针对本地120组预测执行的核验范围和限制 |
| checksums.json | 本目录交付文件SHA256 |

指标使用0—1尺度。F1为150类宏F1；OOS为正类，分数为1−最大softmax概率。F1保持率逐种子除以同样本量的干净CE，再计算均值与样本标准差（ddof=1）。每次评估含4500条ID和1000条OOS记录。置信学习最终分类器使用CE，其五折OOF过程另有预算。

在仓库根目录运行 `python -m c04.final_core_results --verify-only`，仅使用Python标准库即可检查三个种子覆盖、配置对应关系、验证选模、保持率、汇总、CSV/JSON一致性和文件校验值；此命令无需下载数据或模型。

重新导出需先完成全部本地训练及分析，再运行 `python -m c04.final_core_results --output reports/final_core_reexport`。导出会读取官方测试真值和120组保存预测，重算指标，并核对逐类表、混淆矩阵、配置、验证选模及最终分析。输出目录须不存在。

本目录不含原始逐样本预测和模型权重；其哈希不等同于文件本身。仅核对本目录不能替代原始预测核验、模型重放或新环境重训。两组10-shot、30%正式对照的预测和OOF证据另见 `reports/stage3/`。训练方法及依赖顺序见仓库README和阶段运行说明。

GCE固定q=0.7，按验证宏F1选模。保留全部负结果，不以测试分数重新选q。三个种子的标准差不代表跨数据集的不确定性，置信学习未取得一致收益。
''', encoding='utf-8', newline='\n')
    write_json(output / 'checksums.json', {path.name: file_sha256(path)
               for path in sorted(output.iterdir()) if path.is_file()})
    verify(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/final_core'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    verify(args.output) if args.verify_only else export(args.output)


if __name__ == '__main__':
    main()
