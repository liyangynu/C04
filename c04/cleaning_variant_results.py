"""Publish and independently replay cleaning-variant inputs and metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .cleaning_variants import transform
from .common import file_sha256, read_json, write_json
from .formal_results import copy_text
from .runtime import load_official_split
from .sample_weights import validate_weights
from .text import build_vocab


def rows_sha256(rows):
    content = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def export(source, output):
    if output.exists():
        raise ValueError('Use a new export directory')
    comparison = read_json(source/'comparison.json')
    output.mkdir(parents=True)
    for name in ['comparison.json', 'plan.json']:
        copy_text(source/name, output/name)
    for variant in comparison['plan']['variants']:
        subprocess.run([sys.executable, '-m', 'c04.export_results', '--matrix', str(source/variant/'matrix.json'),
                        '--output', str(output/variant)], check=True)
    for record in comparison['records']:
        run = Path(record['run_dir'])
        dest = output/record['variant']/f"k10_s{record['seed']}"
        copy_text(run/'input/decisions.json', dest/'decisions.json')
        copy_text(run/'training/assets/vocab.json', dest/'vocab.json')
        if record['variant'] == 'weighted':
            copy_text(run/'input/weights.json', dest/'weights.json')
        if record['variant'] == 'ce_repeat':
            copy_text(run/'insample_probabilities.npz', dest/'insample_probabilities.npz')
    write_json(output/'checksums.json', {p.relative_to(output).as_posix(): file_sha256(p)
        for p in sorted(output.rglob('*')) if p.is_file()})


def verify(output):
    for name, digest in read_json(output/'checksums.json').items():
        if file_sha256(output/name) != digest:
            raise ValueError(f'Changed result: {name}')
    report, plan = read_json(output/'comparison.json'), read_json(output/'plan.json')
    plan_path = Path('configs/cleaning_variants_idn10_v2.json')
    if plan != read_json(plan_path) or report['plan'] != plan or report['plan_sha256'] != file_sha256(plan_path):
        raise ValueError('Plan changed')
    expected = {(v, s) for v in plan['variants'] for s in plan['seeds']}
    keys = [(r['variant'], r['seed']) for r in report['records']]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Missing or duplicate runs')
    reference = Path(plan['reference_results'])
    if report['reference_checksums_sha256'] != file_sha256(reference/'checksums.json'):
        raise ValueError('Reference version changed')
    for name, digest in read_json(reference/'checksums.json').items():
        if file_sha256(reference/name) != digest:
            raise ValueError('Reference evidence changed')
    config = read_json(plan['config'])
    if report['config_sha256'] != file_sha256(plan['config']):
        raise ValueError('Configuration changed')
    official, _ = load_official_split('data/processed/clean/train.jsonl', 'train')
    by_id = {r['id']: r for r in official}
    checked = []
    for variant in plan['variants']:
        subprocess.run([sys.executable, '-m', 'c04.check_results', '--results', str(output/variant),
            '--report', str(output.parent/f'{output.name}_{variant}_check.json')], check=True)
        matrix = read_json(output/variant/'matrix.json')
        if sorted(r['seed'] for r in matrix['records']) != sorted(plan['seeds']):
            raise ValueError('Matrix seed pairs differ')
        for record in [r for r in report['records'] if r['variant'] == variant]:
            seed = record['seed']
            dest = output/variant/f'k10_s{seed}'
            with np.load(reference/'oof'/f'seed_{seed}'/'oof_probabilities.npz', allow_pickle=False) as data:
                ids, labels, probabilities = data['ids'].tolist(), data['observed_labels'].tolist(), data['probabilities']
            rows = [dict(by_id[sid], label=int(label)) for sid, label in zip(ids, labels)]
            baseline = read_json(reference/'ce'/f'k10_s{seed}'/'run.json')
            if rows_sha256(rows) != baseline['fingerprints']['train']:
                raise ValueError('Observed rows differ from paired source')
            deleted, _, _ = transform(rows, probabilities, 'delete')
            if [r['id'] for r in deleted] != read_json(reference/'oof'/f'seed_{seed}'/'kept_ids.json'):
                raise ValueError('OOF deletion reference cannot be replayed')
            if variant == 'insample_delete':
                score_path = output/'ce_repeat'/f'k10_s{seed}'/'insample_probabilities.npz'
                scorer_metrics = read_json(output/'ce_repeat'/f'k10_s{seed}'/'metrics.json')
                with np.load(score_path, allow_pickle=False) as data:
                    if data['ids'].tolist() != ids or data['observed_labels'].tolist() != labels:
                        raise ValueError('In-sample score IDs or observed labels differ')
                    if str(data['scoring_mode'].item()) != 'in_sample_not_oof':
                        raise ValueError('In-sample evidence mislabeled as OOF')
                    if str(data['checkpoint_sha256'].item()) != scorer_metrics['checkpoint_sha256']:
                        raise ValueError('In-sample score checkpoint differs')
                    if str(data['train_sha256'].item()) != rows_sha256(rows):
                        raise ValueError('In-sample scorer training input differs')
                    probabilities = data['probabilities']
            weights = None
            if variant == 'ce_repeat':
                training = rows
                decisions = {'variant': variant, 'rows': len(rows), 'rows_kept': len(rows), 'uses_clean_labels': False}
                if file_sha256(dest/'predictions.jsonl') != file_sha256(reference/'ce'/f'k10_s{seed}'/'predictions.jsonl'):
                    raise ValueError('Unweighted CE regression check failed')
            else:
                training, weights, decisions = transform(rows, probabilities, variant, plan['weight_floor'])
            if decisions != read_json(dest/'decisions.json'):
                raise ValueError('Decisions cannot be replayed')
            if rows_sha256(training) != record['train_sha256'] or len(training) != record['rows_kept']:
                raise ValueError('Transformed training input differs')
            run, metrics, summary = [read_json(dest/p) for p in ['run.json', 'metrics.json', 'training_summary.json']]
            fingerprints = dict(baseline['fingerprints'], train=record['train_sha256'])
            if weights is not None:
                if read_json(dest/'weights.json') != weights or record['weights_sha256'] != file_sha256(dest/'weights.json'):
                    raise ValueError('Weight replay differs')
                fingerprints['sample_weights'] = record['weights_sha256']
                if run['sample_weighting'] != validate_weights(training, weights):
                    raise ValueError('Weight reduction or distribution changed')
            elif record['weights_sha256'] is not None:
                raise ValueError('Unexpected weights on unweighted route')
            if run['config'] != config or run['seed'] != seed or run['environment'] != baseline['environment']:
                raise ValueError('Training configuration, seed or environment differs')
            if record['config_sha256'] != report['config_sha256'] or run['fingerprints'] != fingerprints:
                raise ValueError('Training fingerprint mismatch')
            if run['train_rows'] != len(training) or read_json(dest/'vocab.json') != build_vocab(training):
                raise ValueError('Training size or vocabulary differs')
            if next(r for r in matrix['records'] if r['seed'] == seed) != record:
                raise ValueError('Root comparison and matrix differ')
            for field, key in [('test_macro_f1', 'macro_f1'), ('test_accuracy', 'accuracy'), ('oos_auroc', 'oos_auroc')]:
                if abs(record[field]-metrics[key]) > 1e-12:
                    raise ValueError('Root metrics differ from verified predictions')
            if record['epochs_completed'] != summary['epochs_completed'] or record['best_epoch'] != metrics['selected_epoch']:
                raise ValueError('Selected epoch mismatch')
            if metrics['train_sha256'] != record['train_sha256']:
                raise ValueError('Evaluation used a different training input')
            checked.append({'variant': variant, 'seed': seed, 'rows': len(training), 'passed': True})
    write_json(output.parent/f'{output.name}_audit.json', {'all_passed': True, 'runs': checked,
        'prediction_runs_verified': 12, 'unweighted_ce_repeats_verified': 3,
        'export_sha256': file_sha256(output/'checksums.json')})
    print('PASS: 12 cleaning-variant runs, input decisions, weights and CE repeats verified.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    if args.source:
        export(args.source, args.results)
    verify(args.results)


if __name__ == '__main__':
    main()
