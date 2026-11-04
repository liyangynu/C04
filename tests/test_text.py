"""Vocabulary isolation and pretrained embedding alignment."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from c04.embeddings import load_glove
from c04.text import build_vocab, encode, text_stats, tokenize


class TextTests(unittest.TestCase):
    def test_vocab_rejects_validation_and_is_order_independent(self):
        rows = [{"id": "train:1", "text": "Book a flight"}, {"id": "train:2", "text": "Book a table"}]
        vocab = build_vocab(rows)
        self.assertEqual(vocab, build_vocab(list(reversed(rows))))
        self.assertEqual(vocab["<pad>"], 0)
        self.assertEqual(vocab["<unk>"], 1)
        self.assertNotIn("validation-only", vocab)
        with self.assertRaises(ValueError):
            build_vocab([{"id": "val:1", "text": "validation-only"}])

    def test_empty_unknown_and_truncation(self):
        vocab = {"<pad>": 0, "<unk>": 1, "book": 2}
        self.assertEqual(encode("", vocab, 4), [1])
        self.assertEqual(encode("BOOK unknown third", vocab, 2), [2, 1])
        self.assertEqual(tokenize("Don't book $20!"), ["don't", "book", "$", "20", "!"])
        self.assertEqual(text_stats([{"text": "book unknown third"}], vocab, 2)["truncated_rows"], 1)

    def test_glove_mapping_padding_and_seed(self):
        vocab = {"<pad>": 0, "<unk>": 1, "book": 2, "missing": 3}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"glove.txt"
            path.write_text("unused 3 4\nbook 1 2\n", encoding="utf-8")
            matrix, found = load_glove(path, vocab, 2, 17)
            replay, _ = load_glove(path, vocab, 2, 17)
            other, _ = load_glove(path, vocab, 2, 42)
            np.testing.assert_array_equal(matrix[0], [0, 0])
            np.testing.assert_array_equal(matrix[2], [1, 2])
            np.testing.assert_array_equal(matrix, replay)
            self.assertFalse(np.array_equal(matrix[3], other[3]))
            self.assertEqual(found, {"book"})


if __name__ == "__main__":
    unittest.main()
