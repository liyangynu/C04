"""Tests for the frozen sentence-embedding artifact contract."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from c04.sentence_embeddings import write_embedding_artifact


class SentenceEmbeddingArtifactTests(unittest.TestCase):
    def test_artifact_records_provenance_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.npz"
            manifest = write_embedding_artifact(path, ["train:1", "train:2"],
                                                [[1.0, 0.0], [0.0, 1.0]],
                                                "test-encoder", "commit-abc", 8, "cpu")
            self.assertEqual(manifest["rows"], 2)
            self.assertEqual(manifest["dimension"], 2)
            self.assertEqual(manifest["encoder_revision"], "commit-abc")
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(set(data.files), {"ids", "embeddings", "encoder_id", "encoder_revision"})
            self.assertTrue(path.with_suffix(".manifest.json").exists())

    def test_rejects_duplicate_ids_and_zero_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.npz"
            with self.assertRaises(ValueError):
                write_embedding_artifact(path, ["train:1", "train:1"], [[1, 0], [0, 1]], "e", "r", 1, "cpu")
            with self.assertRaises(ValueError):
                write_embedding_artifact(path, ["train:1"], [[0, 0]], "e", "r", 1, "cpu")


if __name__ == "__main__":
    unittest.main()
