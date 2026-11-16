"""Build the fixed symmetric30%, 10-shot cohort before training."""
import argparse
from pathlib import Path

from .common import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from .formal_protocol import SEEDS
from .noise import inject
from .runtime import git_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new source output directory')
    manifest = read_json('data/processed/manifest.json')
    records = []
    for seed in SEEDS:
        relative = f'fewshot/seed_{seed}/k_10.jsonl'
        clean = Path('data/processed')/relative
        if file_sha256(clean) != manifest['files'][relative]:
            raise ValueError('Prepared fewshot data changed')
        training, audit, diagnostics = inject(read_jsonl(clean), 150, .3, seed, 'symmetric')
        run = args.output/f'seed_{seed}'
        write_jsonl(run/'noise/train.jsonl', training)
        write_jsonl(run/'noise/audit/labels.jsonl', audit)
        write_json(run/'noise/audit/diagnostics.json', diagnostics)
        records.append({'seed': seed, 'run_dir': str(run),
            'clean_train_sha256': file_sha256(clean),
            'noisy_train_sha256': file_sha256(run/'noise/train.jsonl')})
    write_json(args.output/'matrix.json', {'noise': 'symmetric', 'rate': .3,
        'shots_per_class': [10], 'seeds': SEEDS, 'records': records, 'git': git_state()})
    print(f'Prepared 3 symmetric30% inputs, 450 flips each: {args.output / "matrix.json"}')


if __name__ == '__main__':
    main()
