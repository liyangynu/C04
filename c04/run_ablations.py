"""Run the frozen IDN10-shot ablations, one fit at a time."""
import argparse
import copy
from pathlib import Path
import sys
import time

from .common import file_sha256, read_json, write_json
from .formal_protocol import validate_source
from .run_formal_comparison import summarize
from .run_noise_matrix import run_command
from .runtime import git_state


def variant_config(base, variant):
    result = copy.deepcopy(base)
    for key, value in variant['changes'].items():
        section, field = key.split('.')
        if key not in {'model.pooling', 'model.embedding_trainable', 'training.q'}:
            raise ValueError('Unplanned configuration field')
        result[section][field] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=Path('configs/ablation_idn10_v2.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    plan, source = read_json(args.plan), read_json(args.source)
    validate_source(source)
    for field in ['noise', 'rate', 'shots_per_class', 'seeds']:
        if plan[field] != source[field]:
            raise ValueError('Source does not match frozen ablation plan')
    base = read_json(plan['base_config'])
    reference = Path(plan['reference_results'])
    for path, digest in read_json(reference/'checksums.json').items():
        if file_sha256(reference/path) != digest:
            raise ValueError('Reference evidence changed')
    baseline = read_json(reference/'gce/matrix.json')
    for row in source['records']:
        train = Path(row['run_dir'])/'noise/train.jsonl'
        previous = next(r for r in baseline['records'] if r['seed'] == row['seed'])
        if file_sha256(train) != row['noisy_train_sha256'] or row['noisy_train_sha256'] != previous['train_sha256']:
            raise ValueError('Ablation and reference training inputs differ')
        if read_json(reference/'gce'/f"k10_s{row['seed']}"/'run.json')['config'] != base:
            raise ValueError('Reference configuration differs from plan')
    if args.output.exists():
        raise ValueError('Use a new ablation output directory')
    args.output.mkdir(parents=True)
    write_json(args.output/'plan.json', plan)
    write_json(args.output/'base_config.json', base)
    started = time.perf_counter()
    records = []
    for variant in plan['variants']:
        config = variant_config(base, variant)
        config_path = args.output/'configs'/f"{variant['name']}.json"
        write_json(config_path, config)
        group = []
        for source_row in source['records']:
            seed = source_row['seed']
            run = args.output/variant['name']/f'k10_s{seed}'
            run.mkdir(parents=True)
            train = Path(source_row['run_dir'])/'noise/train.jsonl'
            clock = time.perf_counter()
            run_command([sys.executable, '-m', 'c04.train', '--train', str(train),
                '--config', str(config_path), '--seed', str(seed), '--device', args.device,
                '--output', str(run/'training')], run/'train.log')
            run_command([sys.executable, '-m', 'c04.evaluate', '--checkpoint', str(run/'training/best.pt'),
                '--device', args.device, '--output', str(run/'evaluation')], run/'evaluate.log')
            summary, metrics = read_json(run/'training/summary.json'), read_json(run/'evaluation/metrics.json')
            record = {'variant': variant['name'], 'seed': seed, 'shots_per_class': 10,
                'train_sha256': file_sha256(train), 'config_sha256': file_sha256(config_path),
                'run_dir': str(run), 'best_epoch': summary['best_epoch'],
                'epochs_completed': summary['epochs_completed'], 'best_val_macro_f1': summary['best_val_macro_f1'],
                'test_macro_f1': metrics['macro_f1'], 'test_accuracy': metrics['accuracy'], 'oos_auroc': metrics['oos_auroc'],
                'total_epoch_seconds': summary['total_epoch_seconds'], 'route_wall_seconds': time.perf_counter()-clock,
                'peak_gpu_allocated_mb': max(h['peak_gpu_memory_mb'] or 0 for h in summary['history']),
                'peak_gpu_reserved_mb': max(h['peak_gpu_reserved_mb'] or 0 for h in summary['history'])}
            if variant['name'] == 'frozen_embeddings':
                import numpy as np
                import torch
                checkpoint = torch.load(run/'training/best.pt', map_location='cpu', weights_only=True)
                initial = np.load(run/'training/assets/embeddings.npy', allow_pickle=False)
                unchanged = np.array_equal(initial, checkpoint['model']['embedding.weight'].numpy())
                if not unchanged:
                    raise ValueError('Frozen embedding weights changed')
                record['embedding_weights_unchanged'] = True
            records.append(record)
            group.append(record)
            write_json(args.output/'progress.json', records)
            print(f"{variant['name']} seed={seed}: F1={metrics['macro_f1']:.4f}; AUROC={metrics['oos_auroc']:.4f}", flush=True)
        write_json(args.output/variant['name']/'matrix.json', {'records': group, 'shots_per_class': [10],
            'summary_by_shots': summarize(group), 'seeds': plan['seeds'], 'variant': variant,
            'noise': plan['noise'], 'rate': plan['rate']})
    write_json(args.output/'ablation.json', {'records': records, 'plan': plan,
        'plan_sha256': file_sha256(args.plan), 'reference_checksums_sha256': file_sha256(reference/'checksums.json'),
        'source_matrix_sha256': file_sha256(args.source), 'git': git_state(),
        'wall_seconds': time.perf_counter()-started, 'new_fit_count': len(records),
        'reference_fits_reused': 3})


if __name__ == '__main__':
    main()
