"""Export ablation evidence and audit configuration, inputs and predictions."""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

from .common import file_sha256, read_json, read_jsonl, write_json
from .formal_results import copy_text
from .run_ablations import variant_config
from .runtime import load_official_split
from .text import build_vocab


def export(source, output):
    if output.exists():
        raise ValueError('Use a new export directory')
    report = read_json(source/'ablation.json')
    output.mkdir(parents=True)
    for name in ['ablation.json', 'plan.json', 'base_config.json']:
        copy_text(source/name, output/name)
    for variant in report['plan']['variants']:
        name = variant['name']
        copy_text(source/'configs'/f'{name}.json', output/'configs'/f'{name}.json')
        subprocess.run([sys.executable, '-m', 'c04.export_results', '--matrix', str(source/name/'matrix.json'),
                        '--output', str(output/name)], check=True)
    for record in report['records']:
        original = Path(record['run_dir'])/'training'
        dest = output/record['variant']/f"k10_s{record['seed']}"
        copy_text(original/'assets/vocab.json', dest/'vocab.json')
        if record['variant'] == 'frozen_embeddings':
            import torch
            if file_sha256(original/'best.pt') != read_json(dest/'metrics.json')['checkpoint_sha256']:
                raise ValueError('Embedding evidence is not from the evaluated checkpoint')
            checkpoint = torch.load(original/'best.pt', map_location='cpu', weights_only=True)
            initial = np.load(original/'assets/embeddings.npy', allow_pickle=False)
            np.savez_compressed(dest/'embedding_check.npz', initial=initial,
                                selected=checkpoint['model']['embedding.weight'].numpy())
    write_json(output/'checksums.json', {p.relative_to(output).as_posix(): file_sha256(p)
        for p in sorted(output.rglob('*')) if p.is_file()})


def verify(output):
    for name, digest in read_json(output/'checksums.json').items():
        if file_sha256(output/name) != digest:
            raise ValueError(f'Ablation evidence changed: {name}')
    report, plan = read_json(output/'ablation.json'), read_json(output/'plan.json')
    if plan != read_json('configs/ablation_idn10_v2.json') or report['plan'] != plan:
        raise ValueError('Frozen plan changed')
    if report['plan_sha256'] != file_sha256('configs/ablation_idn10_v2.json'):
        raise ValueError('Plan checksum differs')
    base = read_json(output/'base_config.json')
    if base != read_json(plan['base_config']):
        raise ValueError('Reference configuration changed')
    expected = {(v['name'], seed) for v in plan['variants'] for seed in plan['seeds']}
    actual = [(r['variant'], r['seed']) for r in report['records']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Missing or duplicate ablation pairs')
    reference = Path(plan['reference_results'])
    if file_sha256(reference/'checksums.json') != report['reference_checksums_sha256']:
        raise ValueError('Reference result version changed')
    for name, digest in read_json(reference/'checksums.json').items():
        if file_sha256(reference/name) != digest:
            raise ValueError('Reference evidence changed')
    official, _ = load_official_split('data/processed/clean/train.jsonl', 'train')
    by_id = {r['id']: r for r in official}
    rows_by_seed = {}
    for seed in plan['seeds']:
        path = reference/'oof'/f'seed_{seed}'/'oof_probabilities.npz'
        with np.load(path, allow_pickle=False) as data:
            ids, labels = data['ids'].tolist(), data['observed_labels'].tolist()
        rows_by_seed[seed] = [dict(by_id[sid], label=int(label)) for sid, label in zip(ids, labels)]
    checked = []
    for variant in plan['variants']:
        name = variant['name']
        subprocess.run([sys.executable, '-m', 'c04.check_results', '--results', str(output/name),
                        '--report', str(output.parent/f'{output.name}_{name}_check.json')], check=True)
        expected_config = variant_config(base, variant)
        config_path = output/'configs'/f'{name}.json'
        if read_json(config_path) != expected_config:
            raise ValueError('Variant modifies unplanned settings')
        matrix = read_json(output/name/'matrix.json')
        if sorted(r['seed'] for r in matrix['records']) != sorted(plan['seeds']):
            raise ValueError('Variant matrix has missing or duplicate seeds')
        for record in [r for r in report['records'] if r['variant'] == name]:
            seed = record['seed']
            dest = output/name/f'k10_s{seed}'
            run, training, metrics = [read_json(dest/p) for p in ['run.json', 'training_summary.json', 'metrics.json']]
            baseline = read_json(reference/'gce'/f'k10_s{seed}'/'run.json')
            if run['config'] != expected_config or run['seed'] != seed:
                raise ValueError('Run configuration or seed differs')
            if record['config_sha256'] != file_sha256(config_path):
                raise ValueError('Recorded config checksum differs')
            if run['environment'] != baseline['environment']:
                raise ValueError('Ablation and reference runtimes differ')
            if run['fingerprints'] != baseline['fingerprints'] or metrics['train_sha256'] != record['train_sha256']:
                raise ValueError('Paired training or validation inputs differ')
            if record['train_sha256'] != baseline['fingerprints']['train']:
                raise ValueError('Recorded training checksum differs')
            if read_json(dest/'vocab.json') != build_vocab(rows_by_seed[seed]):
                raise ValueError('Ablation vocabulary is not built from paired training text')
            peer = next(r for r in matrix['records'] if r['seed'] == seed)
            if peer != record:
                raise ValueError('Root report and per-variant matrix differ')
            for field, key in [('test_macro_f1', 'macro_f1'), ('test_accuracy', 'accuracy'), ('oos_auroc', 'oos_auroc')]:
                if abs(record[field]-metrics[key]) > 1e-12:
                    raise ValueError('Root metric differs from checked predictions')
            if record['epochs_completed'] != training['epochs_completed'] or record['best_epoch'] != metrics['selected_epoch']:
                raise ValueError('Training/selected epoch differs')
            if name == 'frozen_embeddings':
                with np.load(dest/'embedding_check.npz', allow_pickle=False) as data:
                    if data['initial'].shape != (len(read_json(dest/'vocab.json')), base['model']['embedding_dim']):
                        raise ValueError('Frozen embedding evidence shape differs')
                    if not np.array_equal(data['initial'], data['selected']):
                        raise ValueError('Frozen embedding weights changed')
                if record.get('embedding_weights_unchanged') is not True:
                    raise ValueError('Missing frozen weight check')
            checked.append({'variant': name, 'seed': seed, 'passed': True})
    write_json(output.parent/f'{output.name}_audit.json', {'all_passed': True, 'runs': checked,
        'prediction_runs_verified': len(checked), 'frozen_embedding_checks': 3,
        'export_sha256': file_sha256(output/'checksums.json')})
    print(f'PASS: {len(checked)} ablation runs match paired data, fixed changes and predictions.')


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
