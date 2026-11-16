"""Rebuild released observed labels for an exact training replay."""
import argparse
from pathlib import Path

import numpy as np

from .common import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from .formal_protocol import comparison_noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('reports/stage3/formal_idn10_v2'))
    parser.add_argument('--output', type=Path, default=Path('data/formal_idn10_v2'))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new data output directory')
    comparison = read_json(args.results/'comparison.json')
    noise, rate = comparison_noise(comparison)
    checksums = read_json(args.results/'checksums.json')
    records = []
    for seed in [17,42,2026]:
        relative = f'oof/seed_{seed}/oof_probabilities.npz'
        path = args.results/relative
        if file_sha256(path) != checksums[relative]:
            raise ValueError('Released OOF artifact changed')
        clean_path = Path(f'data/processed/fewshot/seed_{seed}/k_10.jsonl')
        rows = read_jsonl(clean_path)
        with np.load(path, allow_pickle=False) as data:
            if data['ids'].tolist() != [r['id'] for r in rows]:
                raise ValueError('Fewshot IDs changed')
            observed = data['observed_labels'].tolist()
        expected = next(r for r in comparison['records'] if r['method'] == 'ce' and r['seed'] == seed)
        if file_sha256(clean_path) != next(r for r in comparison['records'] if r['method'] == 'clean_ce' and r['seed'] == seed)['train_sha256']:
            raise ValueError('Clean fewshot data changed')
        dest = args.output/f'seed_{seed}/noise/train.jsonl'
        write_jsonl(dest, [dict(row, label=int(observed[i])) for i,row in enumerate(rows)])
        if file_sha256(dest) != expected['observed_sha256']:
            raise ValueError('Rebuilt noise file checksum mismatch')
        records.append({'seed':seed, 'run_dir':str(dest.parent.parent),
            'clean_train_sha256':file_sha256(clean_path), 'noisy_train_sha256':file_sha256(dest)})
    write_json(args.output/'matrix.json', {'noise':noise, 'rate':rate, 'shots_per_class':[10],
        'seeds':[17,42,2026], 'records':records, 'replay_source':str(args.results)})
    print(f'PASS: rebuilt 3 observed training files; source={args.output / "matrix.json"}')


if __name__ == '__main__':
    main()
