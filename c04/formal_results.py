"""Export and independently audit the v2 comparison, including OOF evidence."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .common import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from .confident_learning import estimate_label_issues
from .runtime import load_official_split
from .text import build_vocab


def copy_text(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix in {'.npz', '.png'}:
        shutil.copyfile(source, dest)
    else:
        dest.write_text(source.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')


def export(source, output):
    if output.exists():
        raise ValueError('Use a new export directory')
    comparison = read_json(source/'comparison.json')
    output.mkdir(parents=True)
    write_json(output/'comparison.json', comparison)
    for method in ['clean_ce','ce','gce','cl_ce']:
        subprocess.run([sys.executable, '-m', 'c04.export_results', '--matrix', str(source/method/'matrix.json'),
                        '--output', str(output/method)], check=True)
    for record in comparison['records']:
        if record['method'] != 'cl_ce':
            continue
        seed = record['seed']
        original = Path(record['run_dir'])/'oof'
        dest = output/'oof'/f'seed_{seed}'
        copy_text(original/'oof_probabilities.npz', dest/'oof_probabilities.npz')
        copy_text(original/'cleaning_report.json', dest/'cleaning_report.json')
        cleaned = read_jsonl(original/'cleaned_train.jsonl')
        write_json(dest/'kept_ids.json', [row['id'] for row in cleaned])
        report = read_json(original/'cleaning_report.json')
        fold_histories = []
        for fold in report['fold_records']:
            fold_path = original/f"fold_{fold['fold']}"/'training'
            copy_text(fold_path/'assets/vocab.json', dest/f"fold_{fold['fold']}_vocab.json")
            summary = read_json(fold_path/'summary.json')
            fold_histories.append(summary)
        write_json(dest/'fold_training_summaries.json', fold_histories)
        write_json(dest/'resource_summary.json', {
            'oof_epoch_seconds': sum(s['total_epoch_seconds'] for s in fold_histories),
            'peak_oof_allocated_mb': max(h['peak_gpu_memory_mb'] or 0 for s in fold_histories for h in s['history']),
            'peak_oof_reserved_mb': max(h['peak_gpu_reserved_mb'] or 0 for s in fold_histories for h in s['history'])})
    write_json(output/'checksums.json', {p.relative_to(output).as_posix(): file_sha256(p)
        for p in sorted(output.rglob('*')) if p.is_file()})


def verify(output):
    for relative, digest in read_json(output/'checksums.json').items():
        if file_sha256(output/relative) != digest:
            raise ValueError(f'Changed export: {relative}')
    for method in ['clean_ce','ce','gce','cl_ce']:
        subprocess.run([sys.executable,'-m','c04.check_results','--results',str(output/method),
                        '--report',str(output.parent/f'{output.name}_{method}_check.json')], check=True)
    official, _ = load_official_split('data/processed/clean/train.jsonl', 'train')
    by_id = {r['id']: r for r in official}
    comparison = read_json(output/'comparison.json')
    reports = []
    for record in comparison['records']:
        method, seed = record['method'], record['seed']
        run = read_json(output/method/f'k10_s{seed}'/'run.json')
        matrix_record = next(r for r in read_json(output/method/'matrix.json')['records'] if r['seed'] == seed)
        for metric in ['test_macro_f1','test_accuracy','oos_auroc']:
            if abs(matrix_record[metric]-record[metric]) > 1e-12:
                raise ValueError('Comparison differs from per-method matrix')
        if run['fingerprints']['train'] != record['train_sha256']:
            raise ValueError('Training fingerprints differ')
        if run['config']['training']['max_epochs'] != 30 or run['config']['training']['patience'] != 5:
            raise ValueError('Budget differs')
        peers = [r for r in comparison['records'] if r['seed'] == seed]
        if len({r['observed_sha256'] for r in peers}) != 1:
            raise ValueError('Methods did not share observed data')
        clean_f1 = next(r['test_macro_f1'] for r in peers if r['method'] == 'clean_ce')
        if not np.isclose(record['f1_retention'], record['test_macro_f1']/clean_f1, atol=1e-12):
            raise ValueError('F1 retention differs')
        if method != 'cl_ce':
            continue
        dest = output/'oof'/f'seed_{seed}'
        report = read_json(dest/'cleaning_report.json')
        if file_sha256(dest/'oof_probabilities.npz') != report['oof_sha256']:
            raise ValueError('OOF hash differs')
        with np.load(dest/'oof_probabilities.npz', allow_pickle=False) as artifact:
            ids = artifact['ids'].tolist()
            probabilities = artifact['probabilities']
            observed = artifact['observed_labels']
            fold_ids = artifact['fold']
        rows = [dict(by_id[sid], label=int(observed[i])) for i, sid in enumerate(ids)]
        # Rebuild observed training JSON in memory with the same canonical writer.
        import hashlib, json
        encoded = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows).encode('utf-8')
        if hashlib.sha256(encoded).hexdigest() != record['observed_sha256']:
            raise ValueError('Observed label export differs from paired input')
        subset = read_jsonl(f'data/processed/fewshot/seed_{seed}/k_10.jsonl')
        if [r['id'] for r in subset] != ids:
            raise ValueError('OOF rows differ from fixed fewshot IDs')
        coverage = []
        for fold in report['fold_records']:
            heldout, train = set(fold['heldout_ids']), set(fold['training_ids'])
            if heldout & train or heldout | train != set(ids):
                raise ValueError('OOF fold leakage or missing rows')
            expected = {sid for i, sid in enumerate(ids) if fold_ids[i] == fold['fold']-1}
            if heldout != expected:
                raise ValueError('OOF assignment differs')
            coverage.extend(fold['heldout_ids'])
            expected_vocab = build_vocab([r for r in rows if r['id'] in train])
            if expected_vocab != read_json(dest/f"fold_{fold['fold']}_vocab.json"):
                raise ValueError('Fold vocabulary is not training-only')
        if sorted(coverage) != sorted(ids) or len(set(ids)) != len(ids):
            raise ValueError('OOF coverage is not exactly once')
        result = estimate_label_issues(rows, probabilities, 150)
        kept = [r['id'] for i, r in enumerate(rows) if result['keep_mask'][i]]
        if kept != read_json(dest/'kept_ids.json') or len(kept) != record['rows_kept']:
            raise ValueError('Cleaning cannot be reproduced from OOF scores')
        cleaned = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n'
                          for i, r in enumerate(rows) if result['keep_mask'][i]).encode('utf-8')
        if hashlib.sha256(cleaned).hexdigest() != record['train_sha256']:
            raise ValueError('Cleaned training data checksum differs')
        if not np.array_equal(result['confident_joint'], report['confident_joint']):
            raise ValueError('Calibrated joint differs')
        reports.append({'seed': seed, 'oof_rows': len(ids), 'kept': len(kept),
                        'coverage_disjoint_vocabulary_and_cleaning_verified': True})
    write_json(output.parent/f'{output.name}_audit.json', {'all_passed': True, 'oof': reports,
        'export_sha256': file_sha256(output/'checksums.json'), 'prediction_runs_verified': 12})
    print('PASS: 12 predictions and 3 five-fold cleaning pipelines independently verified.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Export before verifying; omit to check a checkout')
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    if args.source:
        export(args.source, args.results)
    verify(args.results)


if __name__ == '__main__':
    main()
