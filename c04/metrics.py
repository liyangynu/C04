"""Closed-set classification and OOS scoring, with OOS as the positive class."""
import numpy as np


def classification_metrics(truth, predicted, classes):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if truth.shape != predicted.shape or truth.ndim != 1 or not len(truth):
        raise ValueError("Expected equally sized nonempty label vectors")
    if not np.issubdtype(truth.dtype, np.integer) or not np.issubdtype(predicted.dtype, np.integer):
        raise ValueError("Labels must be integers")
    if np.any(truth < 0) or np.any(truth >= classes) or np.any(predicted < 0) or np.any(predicted >= classes):
        raise ValueError("Closed-set labels out of range")
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    tp = np.diag(matrix)
    support, selected = matrix.sum(axis=1), matrix.sum(axis=0)
    precision = np.divide(tp, selected, out=np.zeros(classes), where=selected>0)
    recall = np.divide(tp, support, out=np.zeros(classes), where=support>0)
    f1 = np.divide(2*tp, support+selected, out=np.zeros(classes), where=(support+selected)>0)
    return {"accuracy": float(tp.sum()/len(truth)), "macro_f1": float(f1.mean()),
            "confusion_matrix": matrix, "precision": precision, "recall": recall,
            "f1": f1, "support": support}


def binary_auroc(truth, scores):
    truth, scores = np.asarray(truth), np.asarray(scores, dtype=np.float64)
    if truth.ndim != 1 or truth.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid binary scores")
    if not np.isin(truth, [0, 1]).all() or len(np.unique(truth)) != 2:
        raise ValueError("AUROC requires both binary classes")
    # Mann-Whitney U with average ranks, including tied MSP scores.
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.repeat((starts+1+ends)/2, ends-starts)
    positives = truth == 1
    n_positive = int(positives.sum())
    n_negative = len(truth)-n_positive
    return float((ranks[positives].sum()-n_positive*(n_positive+1)/2)/(n_positive*n_negative))


def roc_points(truth, scores):
    truth, scores = np.asarray(truth), np.asarray(scores, dtype=np.float64)
    binary_auroc(truth, scores)
    order = np.argsort(-scores, kind="stable")
    labels, values = truth[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(values)), len(values)-1]
    tp = np.cumsum(labels)[ends]
    fp = (ends+1)-tp
    return np.r_[0, fp/(truth==0).sum()], np.r_[0, tp/(truth==1).sum()]
