"""Run the fixed v2 IDN10-shot cohort: clean CE, noisy CE/GCE and CL+CE.

Inputs are the saved observed files from the earlier CE matrix. Neither this
runner nor the cleaner reads audit labels. Every fit runs sequentially on one GPU.
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

from .common import file_sha256, read_json, write_json
from .run_noise_matrix import run_command
from .runtime import git_state


def summarize(records):
    return {'10': {metric: {'mean': float(np.mean([r[metric] for r in records])),
        'std_sample': float(np.std([r[metric] for r in records], ddof=1)),
        'n': len(records), 'values': [r[metric] for r in records]}
        for metric in ['best_val_macro_f1', 'test_macro_f1', 'test_accuracy', 'oos_auroc']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('artifacts/stage3_idn_matrix_ce_k10/matrix.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    source = read_json(args.source)
    if (source['noise'], source['rate'], source['shots_per_class'], source['seeds']) != ('instance', .3, [10], [17,42,2026]):
        raise ValueError('Expected the fixed three-seed IDN30% 10-shot cohort')
    if args.output.exists():
        raise ValueError('Use a new output directory')
    for r in source['records']:
        observed = Path(r['run_dir'])/'noise/train.jsonl'
        if file_sha256(observed) != r['noisy_train_sha256']:
            raise ValueError('Source observed data changed')
        if file_sha256(f"data/processed/fewshot/seed_{r['seed']}/k_10.jsonl") != r['clean_train_sha256']:
            raise ValueError('Clean reference changed')
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    all_records = []
    for method in ['clean_ce', 'ce', 'gce', 'cl_ce']:
        records = []
        for r in source['records']:
            seed = r['seed']
            run = args.output/method/f'k10_s{seed}'
            run.mkdir(parents=True)
            observed = Path(r['run_dir'])/'noise/train.jsonl'
            train = Path(f'data/processed/fewshot/seed_{seed}/k_10.jsonl') if method == 'clean_ce' else observed
            oof_seconds, oof_epoch_seconds, kept = 0.0, 0.0, 1500
            config = Path('configs/formal_gce.json' if method == 'gce' else 'configs/formal_ce.json')
            clock = time.perf_counter()
            if method == 'cl_ce':
                run_command([sys.executable, '-m', 'c04.run_confident_learning', '--train', str(observed),
                    '--config', str(config), '--seed', str(seed), '--device', args.device,
                    '--output', str(run/'oof')], run/'oof.log')
                cleaning = read_json(run/'oof/cleaning_report.json')
                oof_seconds = cleaning['wall_seconds']
                oof_epoch_seconds = sum(f['total_epoch_seconds'] for f in cleaning['fold_records'])
                kept = cleaning['rows_kept']
                train = run/'oof/cleaned_train.jsonl'
            run_command([sys.executable, '-m', 'c04.train', '--train', str(train), '--config', str(config),
                '--seed', str(seed), '--device', args.device, '--output', str(run/'training')], run/'train.log')
            run_command([sys.executable, '-m', 'c04.evaluate', '--checkpoint', str(run/'training/best.pt'),
                '--device', args.device, '--output', str(run/'evaluation')], run/'evaluate.log')
            train_report = read_json(run/'training/summary.json')
            metrics = read_json(run/'evaluation/metrics.json')
            history = train_report['history']
            record = {'seed': seed, 'shots_per_class': 10, 'method': method,
                'observed_sha256': r['noisy_train_sha256'], 'train_sha256': file_sha256(train),
                'config_sha256': file_sha256(config), 'run_dir': str(run),
                'best_epoch': train_report['best_epoch'], 'epochs_completed': train_report['epochs_completed'],
                'best_val_macro_f1': train_report['best_val_macro_f1'],
                'test_macro_f1': metrics['macro_f1'], 'test_accuracy': metrics['accuracy'], 'oos_auroc': metrics['oos_auroc'],
                'rows_kept': kept, 'sample_retention': kept/1500,
                'oof_wall_seconds': oof_seconds, 'oof_epoch_seconds': oof_epoch_seconds,
                'final_epoch_seconds': train_report['total_epoch_seconds'],
                'route_wall_seconds': time.perf_counter()-clock,
                'peak_final_allocated_mb': max(h['peak_gpu_memory_mb'] or 0 for h in history),
                'peak_final_reserved_mb': max(h['peak_gpu_reserved_mb'] or 0 for h in history)}
            records.append(record)
            all_records.append(record)
            write_json(args.output/'progress.json', all_records)
            print(f"{method} seed={seed}: F1={metrics['macro_f1']:.4f}; AUROC={metrics['oos_auroc']:.4f}; kept={kept}", flush=True)
        write_json(args.output/method/'matrix.json', {'records': records, 'shots_per_class': [10],
            'summary_by_shots': summarize(records), 'method': method, 'epochs_max': 30, 'patience': 5,
            'seeds': source['seeds'], 'source_matrix_sha256': file_sha256(args.source), 'git': git_state()})
    reference = {r['seed']: r['test_macro_f1'] for r in all_records if r['method'] == 'clean_ce'}
    for record in all_records:
        record['f1_retention'] = record['test_macro_f1']/reference[record['seed']]
    write_json(args.output/'comparison.json', {'records': all_records, 'git': git_state(),
        'wall_seconds': time.perf_counter()-started, 'protocol': 'docs/正式对照协议.md',
        'summary_by_method': {m: summarize([r for r in all_records if r['method'] == m])['10']
            for m in ['clean_ce','ce','gce','cl_ce']}})


if __name__ == '__main__':
    main()
