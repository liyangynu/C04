"""Small, auditable confident-learning primitives."""

import numpy as np

from .common import stable_seed


def make_folds(rows, classes, folds=5, seed=17):
    """Build deterministic OOF folds from observed training labels only."""
    if folds < 2 or len(rows) < folds:
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
        offset = 0
        for label in range(classes):
            indices = np.flatnonzero(labels == label)
            indices = indices[rng.permutation(len(indices))]
            for position, index in enumerate(indices):
                buckets[(offset + position) % folds].append(int(index))
            offset = (offset + len(indices)) % folds
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
    """Calibrated confident joint and prune-by-noise-rate, cleanlab 2.9.0.

    Northcutt et al. (2021); implementation supplied by cleanlab (AGPL-3.0).
    No third-party source is copied. Fixed 150-way scores include absent classes.
    """
    import cleanlab
    from cleanlab.count import compute_confident_joint, get_confident_thresholds
    from cleanlab.filter import find_label_issues
    if cleanlab.__version__ != "2.9.0":
        raise RuntimeError("Expected cleanlab==2.9.0; install requirements-cleaning.txt")
    probabilities = validate_oof_probabilities(probabilities, rows, classes)
    observed = np.asarray([r["label"] for r in rows], dtype=np.int64)
    if np.any(observed < 0) or np.any(observed >= classes):
        raise ValueError("Invalid observed labels")
    thresholds = get_confident_thresholds(observed, probabilities)
    confident_joint = compute_confident_joint(observed, probabilities, calibrate=True)
    issue_mask = find_label_issues(observed, probabilities, confident_joint=confident_joint,
                                 filter_by="prune_by_noise_rate", min_examples_per_class=1,
                                 n_jobs=1)
    predicted = probabilities.argmax(axis=1)
    keep_mask = ~issue_mask
    protected = []
    for label in np.unique(observed):
        members = np.flatnonzero(observed == label)
        if not np.any(keep_mask[members]):
            selected = members[np.argmax(probabilities[members, label])]
            keep_mask[selected] = True
            issue_mask[selected] = False
            protected.append(int(label))
    return {"thresholds": thresholds, "predicted": predicted,
            "issue_mask": issue_mask, "keep_mask": keep_mask,
            "protected_observed_classes": protected,
            "confident_joint": confident_joint,
            "implementation": "cleanlab-2.9.0/prune_by_noise_rate",
            "absent_observed_classes": np.flatnonzero(np.bincount(observed, minlength=classes) == 0).tolist()}
