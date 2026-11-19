"""Selection and target checks; synthetic values are not experiment results."""
import unittest
import tempfile
from pathlib import Path
import shutil

from c04.common import read_json, write_json
from c04.completion_results import target_assessment
from c04.local_experiment_audit import audit_oof
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


@unittest.skipUnless(Path("artifacts/stage5_expanded_runs/instance_r30/confident_learning/k5_s17/oof/cleaning_report.json").exists(),
                     "Requires local OOF evidence")
class OOFIntegrityTests(unittest.TestCase):
    def test_overlap_is_rejected_even_when_report_claims_exact_coverage(self):
        source = Path("artifacts/stage5_expanded_runs/instance_r30/confident_learning/k5_s17/oof")
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)
            shutil.copyfile(source / "oof_probabilities.npz", dest / "oof_probabilities.npz")
            report = read_json(source / "cleaning_report.json")
            fold = report["fold_records"][0]
            fold["heldout_ids"][0] = fold["training_ids"][0]
            self.assertTrue(report["coverage_exactly_once"])
            write_json(dest / "cleaning_report.json", report)
            with self.assertRaisesRegex(ValueError, "OOF fold overlap/assignment"):
                audit_oof(dest)


if __name__ == "__main__":
    unittest.main()
