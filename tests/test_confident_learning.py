"""Tests for OOF fold coverage and observed-label-only cleaning."""

import unittest

import numpy as np

from c04.confident_learning import estimate_label_issues, make_folds


def rows(classes=3, per_class=5):
    return [{"id": f"train:{label:02d}-{i:02d}", "text": "x", "label": label}
            for label in range(classes) for i in range(per_class)]


class ConfidentLearningTests(unittest.TestCase):
    def test_stratified_folds_cover_each_row_once(self):
        data = rows()
        folds, report = make_folds(data, classes=3, folds=5, seed=17)
        self.assertEqual(report["mode"], "observed_label_stratified")
        self.assertEqual(sorted(np.concatenate(folds).tolist()), list(range(len(data))))
        self.assertEqual([len(fold) for fold in folds], [3, 3, 3, 3, 3])

    def test_underfull_class_uses_balanced_fallback(self):
        data = rows(per_class=2)
        folds, report = make_folds(data, classes=3, folds=5, seed=17)
        self.assertEqual(report["mode"], "balanced_non_stratified")
        self.assertEqual(sorted(np.concatenate(folds).tolist()), list(range(len(data))))
        self.assertEqual(len(set(report["underfull_observed_classes"])), 3)

    def test_issue_estimate_uses_oof_scores_and_protects_class(self):
        data = rows(per_class=2)
        probabilities = np.full((len(data), 3), 0.05)
        for index, row in enumerate(data):
            probabilities[index, row["label"]] = 0.9
        probabilities[0] = [0.05, 0.9, 0.05]
        result = estimate_label_issues(data, probabilities, classes=3)
        self.assertTrue(result["issue_mask"][0])
        self.assertEqual(int(result["issue_mask"].sum()), 1)
        self.assertEqual(result["confident_joint"].shape, (3, 3))

    def test_bad_probability_matrix_is_rejected(self):
        data = rows()
        with self.assertRaises(ValueError):
            estimate_label_issues(data, np.ones((len(data), 2)), classes=3)


if __name__ == "__main__":
    unittest.main()
