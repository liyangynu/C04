"""Padding invariance, trainable embeddings and metric edge cases."""
import unittest

from c04.runtime import configure_runtime
import numpy as np
import torch

from c04.metrics import binary_auroc, classification_metrics, roc_points
from c04.model import BiLSTMClassifier


class MetricTests(unittest.TestCase):
    def test_macro_f1_includes_absent_classes(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 3)
        self.assertAlmostEqual(result["accuracy"], .75)
        self.assertAlmostEqual(result["macro_f1"], (2/3+.8)/3)
        np.testing.assert_array_equal(result["support"], [2, 2, 0])

    def test_auroc_direction_and_ties(self):
        self.assertEqual(binary_auroc([0, 0, 1, 1], [.1, .2, .8, .9]), 1)
        self.assertEqual(binary_auroc([0, 0, 1, 1], [.9, .8, .2, .1]), 0)
        self.assertEqual(binary_auroc([0, 0, 1, 1], [.5]*4), .5)
        with self.assertRaises(ValueError):
            binary_auroc([1, 1], [.1, .2])

    def test_auroc_agrees_with_pairwise_definition(self):
        rng = np.random.default_rng(17)
        truth = np.r_[np.zeros(19), np.ones(11)]
        scores = rng.integers(0, 5, len(truth))
        positive, negative = scores[truth==1], scores[truth==0]
        expected = np.mean((positive[:, None]>negative) + .5*(positive[:, None]==negative))
        self.assertAlmostEqual(binary_auroc(truth, scores), expected)
        fpr, tpr = roc_points(truth, scores)
        self.assertAlmostEqual(float(np.trapezoid(tpr, fpr)), expected)


class ModelTests(unittest.TestCase):
    def setUp(self):
        configure_runtime(17, {"cpu_threads": 1, "deterministic": True})
        self.embedding = torch.randn(12, 4)
        self.embedding[0] = 0

    def test_padding_and_mixed_length_batch_invariance(self):
        for pooling in ("attention", "last"):
            model = BiLSTMClassifier(self.embedding, hidden_dim=3, dropout=0, classes=3, pooling=pooling).eval()
            with torch.inference_mode():
                alone = model(torch.tensor([[2, 3]]), torch.tensor([2]))
                padded = model(torch.tensor([[2, 3, 0, 0], [4, 5, 6, 7]]), torch.tensor([2, 4]))[0:1]
            torch.testing.assert_close(alone, padded)

    def test_gradient_reaches_embeddings_and_padding_stays_zero(self):
        model = BiLSTMClassifier(self.embedding, hidden_dim=3, dropout=0, classes=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        before = model.embedding.weight.detach().clone()
        loss = torch.nn.functional.cross_entropy(
            model(torch.tensor([[2, 3, 0], [3, 4, 5]]), torch.tensor([2, 3])), torch.tensor([0, 1]))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(model.embedding.weight[2], before[2]))
        torch.testing.assert_close(model.embedding.weight[0], torch.zeros(4))
        frozen = BiLSTMClassifier(self.embedding, embedding_trainable=False)
        self.assertFalse(frozen.embedding.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
