"""Observed-label-only cleaning transformations; no audit truth is accepted."""
import numpy as np

from .confident_learning import estimate_label_issues


def transform(rows, probabilities, variant, weight_floor=.05):
    result = estimate_label_issues(rows, probabilities, 150)
    issue = result['issue_mask']
    if variant == 'relabel':
        training = [dict(row, label=int(result['predicted'][i]) if issue[i] else row['label'])
                    for i, row in enumerate(rows)]
        weights = None
    elif variant == 'weighted':
        training = [dict(row) for row in rows]
        observed = np.array([row['label'] for row in rows])
        raw = np.clip(np.asarray(probabilities)[np.arange(len(rows)), observed], weight_floor, 1.)
        normalized = raw/raw.mean()
        weights = {r['id']: float(normalized[i]) for i, r in enumerate(rows)}
    elif variant in {'delete', 'insample_delete'}:
        training = [dict(row) for i, row in enumerate(rows) if result['keep_mask'][i]]
        weights = None
    else:
        raise ValueError('Unknown cleaning transformation')
    report = {'variant': variant, 'flagged_ids': [row['id'] for i, row in enumerate(rows) if issue[i]],
        'kept_ids': [r['id'] for r in training], 'rows': len(rows), 'rows_kept': len(training),
        'labels_changed': sum(r['label'] != rows[i]['label'] for i, r in enumerate(training)) if len(training) == len(rows) else 0,
        'absent_training_classes': sorted(set(range(150)) - {r['label'] for r in training}),
        'confident_joint': result['confident_joint'].tolist(), 'implementation': result['implementation'],
        'uses_clean_labels': False, 'uses_test_labels': False}
    return training, weights, report
