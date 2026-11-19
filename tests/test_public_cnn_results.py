import shutil
import tempfile
import unittest
from pathlib import Path

from c04.common import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from c04.public_cnn_results import verify


@unittest.skipUnless(Path("reports/stage6/public_cnn_v1/checksums.json").is_file() and
                     Path("data/processed/clean/test.jsonl").is_file(),
                     "Requires released CNN evidence and prepared official data")
class CNNResultIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "results"
        shutil.copytree("reports/stage6/public_cnn_v1", self.root)

    def update_checksums(self):
        write_json(self.root / "checksums.json", {p.relative_to(self.root).as_posix(): file_sha256(p)
                   for p in sorted(self.root.rglob("*")) if p.is_file() and p.name != "checksums.json"})

    def test_modified_truth_rejected_even_with_updated_hashes(self):
        path = self.root / "seed_17/predictions.jsonl"
        rows = read_jsonl(path)
        rows[0]["true_label"] = (rows[0]["true_label"] + 1) % 150
        write_jsonl(path, rows)
        report_path = self.root / "seed_17/metrics.json"
        report = read_json(report_path)
        report["predictions_sha256"] = file_sha256(path)
        write_json(report_path, report)
        self.update_checksums()
        with self.assertRaisesRegex(ValueError, "IDs/truth"):
            verify(self.root)

    def test_validation_selection_cannot_be_replaced_by_another_epoch(self):
        path = self.root / "seed_17/metrics.json"
        report = read_json(path)
        report["best_epoch"] = 1
        write_json(path, report)
        self.update_checksums()
        with self.assertRaisesRegex(ValueError, "validation criterion"):
            verify(self.root)

    def test_extra_vocabulary_token_rejected_even_with_updated_hashes(self):
        path = self.root / "seed_17/vocab.json"
        vocab = read_json(path)
        vocab["unused-test-token"] = len(vocab)
        write_json(path, vocab)
        self.update_checksums()
        with self.assertRaisesRegex(ValueError, "training-only vocabulary"):
            verify(self.root)
