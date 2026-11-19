"""Corruption checks using a small synthetic prediction set, not experiment data."""
import csv
from pathlib import Path
import tempfile
import unittest

from c04.check_expanded_predictions import verify_predictions
from c04.common import read_jsonl, write_json, write_jsonl


class ExpandedPredictionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.labels = {"a": 0, "b": 1, "c": 2}
        self.expected = [("test:0", 0), ("test:1", 1), ("test:2", 2), ("oos_test:0", -1)]
        write_jsonl(self.root / "predictions.jsonl", [
            {"id": sid, "true_label": truth, "predicted_intent": max(0, truth),
             "oos_score": .9 if truth == -1 else .1} for sid, truth in self.expected])
        write_json(self.root / "metrics.json", {"macro_f1": 1., "accuracy": 1., "oos_auroc": 1.})
        self.record = {"test_macro_f1": 1., "test_accuracy": 1., "oos_auroc": 1.}
        with (self.root / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerows([["true_id/predicted_id", 0, 1, 2],
                              [0, 1, 0, 0], [1, 0, 1, 0], [2, 0, 0, 1]])
        with (self.root / "per_class.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["class_id", "intent", "precision", "recall", "f1", "support"])
            writer.writerows([[i, name, 1., 1., 1., 1] for name, i in self.labels.items()])

    def verify(self):
        return verify_predictions(self.root, self.expected, self.labels, self.record)

    def test_valid_set_and_changed_truth(self):
        self.assertEqual(self.verify()["prediction_rows"], 4)
        path = self.root / "predictions.jsonl"
        rows = read_jsonl(path)
        rows[0]["true_label"] = 1
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "IDs/truth"):
            self.verify()

    def test_wrong_per_class_value_is_rejected_even_when_overall_scores_match(self):
        path = self.root / "per_class.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("0,a,1.0,1.0,1.0,1", "0,a,1.0,1.0,0.5,1"),
                        encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Per-class f1"):
            self.verify()

    def test_swapped_confusion_axes_are_rejected(self):
        path = self.root / "confusion_matrix.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("true_id/predicted_id,0,1,2",
                                                               "true_id/predicted_id,1,0,2"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Confusion matrix"):
            self.verify()

    def test_nonfinite_oos_score_is_rejected(self):
        path = self.root / "predictions.jsonl"
        rows = read_jsonl(path)
        rows[-1]["oos_score"] = float("nan")
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "Invalid predicted intents or OOS scores"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
