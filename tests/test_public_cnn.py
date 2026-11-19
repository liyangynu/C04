import unittest
import torch

from c04.public_cnn import PublicCNN, prediction_metrics


class PublicCNNTests(unittest.TestCase):
    def test_short_sentence_and_mixed_padding_do_not_change_prediction(self):
        torch.manual_seed(17)
        model = PublicCNN(torch.randn(12, 5), filters=7, dense_dim=9, dropout=0).eval()
        model.embedding.weight.data[0] = 0
        short = model(torch.tensor([[2]]), torch.tensor([1]))
        mixed = model(torch.tensor([[2, 0, 0, 0, 0], [3, 4, 5, 6, 7]]), torch.tensor([1, 5]))
        torch.testing.assert_close(short[0], mixed[0])
        short.sum().backward()
        self.assertGreater(float(model.embedding.weight.grad[2].abs().sum()), 0)

    def test_oos_rejection_counts_as_an_id_error(self):
        rows = [{"true_label": 0, "predicted_class": 150, "p_oos": .6},
                {"true_label": 1, "predicted_class": 1, "p_oos": .1},
                {"true_label": 150, "predicted_class": 150, "p_oos": .8},
                {"true_label": 150, "predicted_class": 1, "p_oos": .2}]
        values = prediction_metrics(rows)
        self.assertEqual(values["id_accuracy"], .5)
        self.assertEqual(values["oos_recall"], .5)
        self.assertEqual(values["id_macro_f1"], 1/150)
        self.assertEqual(values["oos_auroc_p_oos"], .75)
