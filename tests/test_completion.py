"""Selection and target checks; synthetic values are not experiment results."""
import unittest

from c04.completion_results import target_assessment
from c04.public_svm_selection import select_candidate


class CompletionTests(unittest.TestCase):
    def test_validation_tie_uses_declared_order_and_never_test_score(self):
        rows = [{"candidate": "first", "validation_accuracy": .8, "test_accuracy": .1},
                {"candidate": "second", "validation_accuracy": .8, "test_accuracy": .99}]
        self.assertEqual(select_candidate(rows), "first")
        rows[1]["validation_accuracy"] = .81
        self.assertEqual(select_candidate(rows), "second")

    def test_invalid_validation_score_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Finite validation"):
            select_candidate([{"candidate": "bad", "validation_accuracy": float("nan")}])

    def test_point_estimate_success_does_not_imply_interval_support(self):
        result = target_assessment({"mean_difference": .06, "t_95_ci": [-.01, .13]},
                                   {"mean_difference": -.01, "t_95_ci": [-.04, .02]})
        self.assertTrue(result["f1_gain_at_least_005_point_estimate"])
        self.assertTrue(result["auroc_drop_at_most_002_point_estimate"])
        self.assertFalse(result["f1_t95_lower_at_least_005"])
        self.assertFalse(result["auroc_t95_lower_at_least_minus002"])


if __name__ == "__main__":
    unittest.main()
