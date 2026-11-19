"""Regression checks for resuming old progress and freezing new noise inputs."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from c04.common import file_sha256, read_json, write_json, write_jsonl
from c04.prepare_expanded_matrix import align_frozen_embeddings, prepare
from c04.run_expanded_matrix import key


class ExpandedMatrixTests(unittest.TestCase):
    def test_old_cl_name_is_skipped_when_resuming(self):
        record = {"noise": "instance", "rate": .3, "shots_per_class": 10,
                  "seed": 17, "method": "cl_ce"}
        wanted = ("instance", .3, 10, 17, "confident_learning")
        self.assertTrue(any(key(r) == wanted for r in [record]))
        self.assertNotEqual(key(record), key(dict(record, seed=42)))
        self.assertNotEqual(key(record), key(dict(record, method="ce")))
        self.assertEqual(record["method"], "cl_ce")  # Keep the historical record unchanged.

    def test_embedding_encoder_and_revision_must_both_match(self):
        protocol = {"encoder_id": "frozen-model", "encoder_revision": "fixed-revision"}
        rows = [{"id": "train:1"}, {"id": "train:2"}]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "vectors.npz"
            for model, revision, accepted in (("frozen-model", "fixed-revision", True),
                                              ("other-model", "fixed-revision", False),
                                              ("frozen-model", "other-revision", False)):
                np.savez(path, ids=[r["id"] for r in reversed(rows)],
                         embeddings=np.array([[0., 1.], [1., 0.]]),
                         encoder_id=model, encoder_revision=revision)
                if accepted:
                    vectors, provenance = align_frozen_embeddings(path, rows, protocol)
                    np.testing.assert_array_equal(vectors, np.eye(2))
                    self.assertEqual(provenance, protocol)
                else:
                    with self.assertRaisesRegex(ValueError, "frozen stage-5 protocol"):
                        align_frozen_embeddings(path, rows, protocol)

    def test_reused_idn_can_be_prepared_first_without_encoder_carryover(self):
        # Synthetic fixture: deliberately run only the reused branch, with no previous loop.
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            try:
                os.chdir(temp)
                write_json("configs/expanded_matrix_v1.json", {"seeds": [17],
                    "instance_noise": {"temperature": .2}})
                write_json("configs/protocol.json", {"dataset": {"classes": 3, "sha256": "fixture"}})
                clean = [{"id": f"train:{i}", "text": f"text {i}", "label": i % 3}
                         for i in range(30)]
                observed = [dict(row, label=(row["label"] + 1) % 3) if i < 9 else row
                            for i, row in enumerate(clean)]
                write_jsonl("data/processed/fewshot/seed_17/k_10.jsonl", clean)
                source = Path("data/formal_idn10_v2/seed_17/noise/train.jsonl")
                write_jsonl(source, observed)
                write_json("reports/stage3/formal_idn10_v2/comparison.json", {"records": [
                    {"method": "ce", "seed": 17, "observed_sha256": file_sha256(source)}]})
                with patch("c04.prepare_expanded_matrix.conditions", return_value=[(10, "instance", .3)]):
                    matrix = prepare(Path("output"), None)
                record = matrix["records"][0]
                self.assertIsNone(record["encoder"])
                self.assertIsNone(record["embedding_path"])
                self.assertEqual(record["train_sha256"], file_sha256(source))
                self.assertEqual(record["flipped"], 9)
                self.assertEqual(read_json("output/matrix.json"), matrix)
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
