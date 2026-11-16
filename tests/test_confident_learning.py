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

    def test_calibrated_joint_and_missing_class(self):
        data = rows(classes=2, per_class=20)
        p = np.full((len(data), 3), 0.01)
        for i, row in enumerate(data):
            p[i, row['label']] = 0.98
        p[0] = [0.01, 0.98, 0.01]
        result = estimate_label_issues(data, p, 3)
        self.assertEqual(result['absent_observed_classes'], [2])
        np.testing.assert_array_equal(result['confident_joint'].sum(axis=1), [20, 20, 0])
        self.assertEqual(result['confident_joint'].shape, (3, 3))
        self.assertTrue(result['issue_mask'][0])
        for label in (0, 1):
            self.assertTrue(any(result['keep_mask'][i] for i, r in enumerate(data) if r['label'] == label))

    def test_uneven_class_counts_produce_balanced_disjoint_folds(self):
        data = rows(classes=3, per_class=7)
        folds, report = make_folds(data, 3)
        self.assertLessEqual(max(report['fold_sizes'])-min(report['fold_sizes']), 1)
        for fold in folds:
            self.assertTrue(set(fold).isdisjoint(set(range(len(data)))-set(fold)))
        with self.assertRaises(ValueError):
            make_folds(data[:3], 3)


if __name__ == "__main__":
    unittest.main()
