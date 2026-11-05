"""Small, auditable confident-learning primitives."""

import numpy as np

from .common import stable_seed


def make_folds(rows, classes, folds=5, seed=17):
    """Build deterministic OOF folds from observed training labels only."""
    if folds < 2 or not rows:
        raise ValueError("At least two folds and one row are required")
    labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
    if np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError("Observed labels outside the configured classes")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate row IDs")
    counts = np.bincount(labels, minlength=classes)
    underfull = np.flatnonzero(counts < folds).tolist()
    rng = np.random.default_rng(stable_seed(seed, "confident-learning-folds"))
    buckets = [[] for _ in range(folds)]
    if underfull:
        order = rng.permutation(len(rows))
        for position, index in enumerate(order):
            buckets[position % folds].append(int(index))
        mode = "balanced_non_stratified"
    else:
        for label in range(classes):
            indices = np.flatnonzero(labels == label)
            indices = indices[rng.permutation(len(indices))]
            for position, index in enumerate(indices):
                buckets[position % folds].append(int(index))
        mode = "observed_label_stratified"
    result = [np.asarray(sorted(bucket), dtype=np.int64) for bucket in buckets]
    coverage = np.concatenate(result)
    if len(coverage) != len(rows) or len(np.unique(coverage)) != len(rows):
        raise AssertionError("OOF folds must cover each row exactly once")
    return result, {"folds": folds, "mode": mode, "underfull_observed_classes": underfull,
                    "fold_sizes": [int(len(x)) for x in result], "seed": seed}


def validate_oof_probabilities(probabilities, rows, classes):
    """Validate a fixed-shape OOF probability matrix."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(rows), classes):
        raise ValueError("OOF probabilities must have shape (rows, classes)")
    if not np.isfinite(probabilities).all() or np.any(probabilities < -1e-10):
        raise ValueError("OOF probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("OOF probability rows must sum to one")
    return probabilities


def estimate_label_issues(rows, probabilities, classes):
    """Estimate removable issues from observed labels and OOF probabilities."""
    probabilities = validate_oof_probabilities(probabilities, rows, classes)
    observed = np.asarray([r["label"] for r in rows], dtype=np.int64)
    thresholds = np.full(classes, np.nan, dtype=np.float64)
    for label in range(classes):
        selected = probabilities[observed == label, label]
        if len(selected):
            thresholds[label] = float(np.mean(selected))
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(rows)), predicted]
    candidate_threshold = thresholds[predicted]
    issue_mask = (predicted != observed) & np.isfinite(candidate_threshold) & (confidence >= candidate_threshold)
    keep_mask = ~issue_mask
    protected = []
    for label in np.unique(observed):
        members = np.flatnonzero(observed == label)
        if not np.any(keep_mask[members]):
            selected = members[np.argmax(probabilities[members, label])]
            keep_mask[selected] = True
            issue_mask[selected] = False
            protected.append(int(label))
    confident = confidence >= np.where(np.isfinite(candidate_threshold), candidate_threshold, np.inf)
    confident_joint = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confident_joint, (observed[confident], predicted[confident]), 1)
    return {"thresholds": thresholds, "predicted": predicted,
            "issue_mask": issue_mask, "keep_mask": keep_mask,
            "protected_observed_classes": protected,
            "confident_joint": confident_joint}
