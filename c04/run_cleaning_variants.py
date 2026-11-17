"""Run fixed cleaning variants and an explicit in-sample scoring ablation."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

from .cleaning_variants import transform
from .common import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from .formal_protocol import validate_source
from .run_formal_comparison import summarize
from .run_noise_matrix import run_command
from .runtime import git_state


def training_scores(checkpoint_path, train, output, device_name):
    import torch
    from .model import BiLSTMClassifier
    from .runtime import choose_device, configure_runtime, make_loader, predict
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    configure_runtime(checkpoint['seed'], config['training'])
    device = choose_device(device_name)
    rows = read_jsonl(train)
    model = BiLSTMClassifier(checkpoint['model']['embedding.weight'],
        **{k: v for k, v in config['model'].items() if k != 'embedding_dim'}).to(device)
    model.load_state_dict(checkpoint['model'])
    loader = make_loader(rows, checkpoint['vocab'], config['text']['max_length'], config['training']['batch_size'])
    logits, _ = predict(model, loader, device)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy().astype(np.float64)
    np.savez_compressed(output, ids=np.array([r['id'] for r in rows]),
        observed_labels=np.array([r['label'] for r in rows]), probabilities=probabilities,
        checkpoint_sha256=np.array(file_sha256(checkpoint_path)), train_sha256=np.array(file_sha256(train)),
        scoring_mode=np.array('in_sample_not_oof'))
    return probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    plan_path = Path('configs/cleaning_variants_idn10_v2.json')
    plan, source = read_json(plan_path), read_json(args.source)
    validate_source(source)
    if any(source[f] != plan[f] for f in ['noise', 'rate', 'shots_per_class', 'seeds']):
        raise ValueError('Source differs from fixed cleaning-variant plan')
    reference = Path(plan['reference_results'])
    for name, digest in read_json(reference/'checksums.json').items():
        if file_sha256(reference/name) != digest:
            raise ValueError('Reference evidence changed')
    comparison = read_json(reference/'comparison.json')
    rows_by_seed, oof_by_seed = {}, {}
    for r in source['records']:
        seed = r['seed']
        path = Path(r['run_dir'])/'noise/train.jsonl'
        baseline = next(x for x in comparison['records'] if x['method'] == 'ce' and x['seed'] == seed)
        if file_sha256(path) != baseline['train_sha256'] or file_sha256(path) != r['noisy_train_sha256']:
            raise ValueError('Observed training input changed')
        rows = read_jsonl(path)
        with np.load(reference/'oof'/f'seed_{seed}'/'oof_probabilities.npz', allow_pickle=False) as data:
            if data['ids'].tolist() != [r['id'] for r in rows] or data['observed_labels'].tolist() != [r['label'] for r in rows]:
                raise ValueError('OOF and observed labels do not align')
            oof_by_seed[seed] = data['probabilities']
        deleted, _, _ = transform(rows, oof_by_seed[seed], 'delete')
        if [r['id'] for r in deleted] != read_json(reference/'oof'/f'seed_{seed}'/'kept_ids.json'):
            raise ValueError('Published OOF cleaning cannot be replayed')
        rows_by_seed[seed] = rows
    if args.output.exists():
        raise ValueError('Use a new experiment directory')
    args.output.mkdir(parents=True)
    write_json(args.output/'plan.json', plan)
    started = time.perf_counter()
    records, insample = [], {}
    for variant in plan['variants']:
        group = []
        for seed in plan['seeds']:
            run = args.output/variant/f'k10_s{seed}'
            run.mkdir(parents=True)
            clock = time.perf_counter()
            rows = rows_by_seed[seed]
            probabilities = insample[seed] if variant == 'insample_delete' else oof_by_seed[seed]
            weights = None
            if variant == 'ce_repeat':
                train_rows = rows
                decisions = {'variant': variant, 'rows': len(rows), 'rows_kept': len(rows), 'uses_clean_labels': False}
            else:
                train_rows, weights, decisions = transform(rows, probabilities, variant, plan['weight_floor'])
            train = run/'input/train.jsonl'
            write_jsonl(train, train_rows)
            write_json(run/'input/decisions.json', decisions)
            command = [sys.executable, '-m', 'c04.train', '--train', str(train), '--config', plan['config'],
                '--seed', str(seed), '--device', args.device, '--output', str(run/'training')]
            if weights is not None:
                weight_path = run/'input/weights.json'
                write_json(weight_path, weights)
                command += ['--sample-weights', str(weight_path)]
            run_command(command, run/'train.log')
            run_command([sys.executable, '-m', 'c04.evaluate', '--checkpoint', str(run/'training/best.pt'),
                '--device', args.device, '--output', str(run/'evaluation')], run/'evaluate.log')
            summary, metrics = read_json(run/'training/summary.json'), read_json(run/'evaluation/metrics.json')
            score_seconds = 0.
            if variant == 'ce_repeat':
                score_start = time.perf_counter()
                insample[seed] = training_scores(run/'training/best.pt', train, run/'insample_probabilities.npz', args.device)
                score_seconds = time.perf_counter()-score_start
            record = {'variant': variant, 'seed': seed, 'shots_per_class': 10, 'run_dir': str(run),
                'train_sha256': file_sha256(train), 'config_sha256': file_sha256(plan['config']),
                'weights_sha256': file_sha256(run/'input/weights.json') if weights is not None else None,
                'rows_kept': len(train_rows), 'best_epoch': summary['best_epoch'],
                'epochs_completed': summary['epochs_completed'], 'best_val_macro_f1': summary['best_val_macro_f1'],
                'test_macro_f1': metrics['macro_f1'], 'test_accuracy': metrics['accuracy'], 'oos_auroc': metrics['oos_auroc'],
                'total_epoch_seconds': summary['total_epoch_seconds'], 'route_wall_seconds': time.perf_counter()-clock,
                'score_wall_seconds': score_seconds,
                'peak_gpu_allocated_mb': max(h['peak_gpu_memory_mb'] or 0 for h in summary['history']),
                'peak_gpu_reserved_mb': max(h['peak_gpu_reserved_mb'] or 0 for h in summary['history'])}
            if variant == 'ce_repeat':
                old = reference/'ce'/f'k10_s{seed}'/'predictions.jsonl'
                record['predictions_match_previous_ce'] = file_sha256(old) == file_sha256(run/'evaluation/predictions.jsonl')
                if not record['predictions_match_previous_ce']:
                    raise ValueError('Unweighted CE changed; stop before other variants')
            records.append(record)
            group.append(record)
            write_json(args.output/'progress.json', records)
            print(f"{variant} seed={seed}: F1={metrics['macro_f1']:.4f}; AUROC={metrics['oos_auroc']:.4f}; rows={len(train_rows)}", flush=True)
        write_json(args.output/variant/'matrix.json', {'records': group, 'shots_per_class': [10],
            'summary_by_shots': summarize(group), 'seeds': plan['seeds'], 'variant': variant})
    write_json(args.output/'comparison.json', {'records': records, 'plan': plan, 'plan_sha256': file_sha256(plan_path),
        'reference_checksums_sha256': file_sha256(reference/'checksums.json'),
        'config_sha256': file_sha256(plan['config']), 'new_fits': 12, 'oof_fits_reused': 15,
        'wall_seconds': time.perf_counter()-started, 'git': git_state()})


if __name__ == '__main__':
    main()
