"""Weight alignment, gradient scaling and validation for the weighted route."""
import unittest
import torch

from c04.runtime import make_loader
from c04.sample_weights import reduce_loss, validate_weights


class SampleWeightTests(unittest.TestCase):
    def test_id_alignment_survives_shuffle_and_partial_batch(self):
        rows = [{'id': f'train:{i}', 'text': 'word', 'label': i} for i in range(5)]
        weights = {r['id']: .5+.25*i for i, r in enumerate(rows)}
        validate_weights(rows, weights)
        loader = make_loader(rows, {'<pad>': 0, '<unk>': 1, 'word': 2}, 4, 2, True,
                             torch.Generator().manual_seed(17), sample_weights=weights)
        seen = []
        for _, _, labels, batch_weights in loader:
            for label, weight in zip(labels.tolist(), batch_weights.tolist()):
                self.assertEqual(weight, weights[f'train:{label}'])
                seen.append(label)
        self.assertEqual(sorted(seen), list(range(5)))

    def test_global_weighting_keeps_gradient_scale_and_unit_weight_equivalence(self):
        x = torch.tensor([2., 4.], requires_grad=True)
        loss = reduce_loss(x, torch.tensor([.25, .75]))
        self.assertEqual(loss.item(), 1.75)
        loss.backward()
        torch.testing.assert_close(x.grad, torch.tensor([.125, .375]))
        self.assertTrue(torch.equal(reduce_loss(x), reduce_loss(x, torch.ones(2))))

    def test_missing_extra_invalid_and_unnormalized_weights_rejected(self):
        rows = [{'id': 'train:1'}, {'id': 'train:2'}]
        for weights in [{}, {'train:1': 1, 'train:2': 1, 'test:3': 1},
                        {'train:1': 0, 'train:2': 2}, {'train:1': float('nan'), 'train:2': 1},
                        {'train:1': .2, 'train:2': .2}]:
            with self.assertRaises(ValueError):
                validate_weights(rows, weights)


if __name__ == '__main__':
    unittest.main()
