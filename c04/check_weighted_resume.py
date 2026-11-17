"""Check two-epoch weighted training against one epoch plus exact resume."""
import argparse
from pathlib import Path
import subprocess
import sys

from .common import file_sha256, read_json, write_json
from .run_noise_matrix import run_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new diagnostic directory')
    args.output.mkdir(parents=True)
    common = [sys.executable, '-m', 'c04.train', '--train', str(args.train),
        '--config', 'configs/formal_ce.json', '--seed', '17', '--device', 'cuda']
    weighted = common + ['--sample-weights', str(args.weights)]
    for name, epochs, resume in [('continuous', '2', False), ('resumed', '1', False), ('resumed', '2', True)]:
        command = weighted + ['--output', str(args.output/name), '--epochs', epochs]
        if resume:
            command += ['--resume']
        run_command(command, args.output/f'{name}_{epochs}.log')
    import torch
    left = torch.load(args.output/'continuous/last.pt', map_location='cpu', weights_only=True)
    right = torch.load(args.output/'resumed/last.pt', map_location='cpu', weights_only=True)
    if not all(torch.equal(value, right['model'][key]) for key, value in left['model'].items()):
        raise ValueError('Weighted resumed model parameters differ')
    for a, b in zip(left['history'], right['history']):
        for key in ['epoch', 'train_loss', 'val_loss', 'val_macro_f1', 'val_accuracy']:
            if a[key] != b[key]:
                raise ValueError('Weighted resumed training trajectory differs')
    weights = read_json(args.weights)
    # Reverse values to preserve their mean while changing the ID association.
    altered = dict(zip(weights, reversed(list(weights.values()))))
    if altered == weights:
        raise ValueError('Diagnostic requires unequal weights')
    altered_path = args.output/'changed_weights.json'
    write_json(altered_path, altered)
    for name, options in [('missing_weights', []), ('changed_weights', ['--sample-weights', str(altered_path)])]:
        command = common + options + ['--output', str(args.output/'resumed'), '--epochs', '3', '--resume']
        result = subprocess.run(command, capture_output=True, text=True)
        (args.output/f'{name}.log').write_text(result.stdout+result.stderr, encoding='utf-8', newline='\n')
        if result.returncode == 0 or 'Resume seed or data changed' not in result.stderr:
            raise ValueError('Resume did not reject changed weight fingerprints')
    write_json(args.report, {'scope': 'two-epoch weighted diagnostic; separate from formal 30-epoch comparison',
        'train_sha256': file_sha256(args.train), 'weights_sha256': file_sha256(args.weights),
        'model_parameters_byte_identical': True, 'loss_and_validation_trajectory_exact': True,
        'missing_weights_rejected': True, 'changed_weights_rejected': True,
        'new_training_invocations': 3, 'all_passed': True})
    print('PASS: weighted resume is exact; missing or changed weight files are rejected.')


if __name__ == '__main__':
    main()
