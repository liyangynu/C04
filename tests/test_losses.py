"""Tests for CE/GCE loss definitions and validation."""

import unittest

import torch

from c04.losses import generalized_cross_entropy, training_loss


class LossTests(unittest.TestCase):
    def test_gce_is_zero_for_confident_correct_prediction(self):
        logits = torch.tensor([[30.0, 0.0], [0.0, 30.0]])
        targets = torch.tensor([0, 1])
        loss = generalized_cross_entropy(logits, targets, q=0.7)
        self.assertTrue(torch.all(loss < 1e-8))

    def test_gce_is_bounded_and_less_sensitive_than_ce(self):
        logits = torch.tensor([[0.0, 1.0], [-20.0, 0.0]])
        targets = torch.tensor([0, 0])
        gce = generalized_cross_entropy(logits, targets, q=0.7)
        ce = training_loss(logits, targets, {"loss": "ce"})
        self.assertTrue(torch.all(gce >= 0))
        self.assertTrue(torch.all(gce <= 1 / 0.7 + 1e-6))
        self.assertLess(float(gce[1]), float(ce[1]))

    def test_invalid_q_and_loss_are_rejected(self):
        logits = torch.zeros(2, 3)
        targets = torch.tensor([0, 1])
        for q in (0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                generalized_cross_entropy(logits, targets, q)
        with self.assertRaises(ValueError):
            training_loss(logits, targets, {"loss": "unknown"})


if __name__ == "__main__":
    unittest.main()
