import unittest

from c04.paired_statistics import holm_adjust, summarize_pair


class PairedStatisticsTests(unittest.TestCase):
    def test_three_seed_paired_summary_and_exact_wilcoxon_limit(self):
        result = summarize_pair([0.6, 0.5, 0.7], [0.5, 0.5, 0.6])
        for actual, expected in zip(result['paired_differences'], [0.1, 0.0, 0.1]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(result['mean_difference'], 0.2/3)
        self.assertEqual(result['wilcoxon_exact']['p_two_sided'], 0.5)
        self.assertLess(result['t_95_ci'][0], result['mean_difference'])
        self.assertGreater(result['t_95_ci'][1], result['mean_difference'])

    def test_holm_adjustment_is_monotonic_in_sorted_p_values(self):
        self.assertEqual(holm_adjust([0.04, 0.01, 0.03, 0.2]), [0.09, 0.04, 0.09, 0.2])

    def test_identical_pairs_have_zero_interval_and_unit_p(self):
        result = summarize_pair([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(result['mean_difference'], 0.0)
        self.assertEqual(result['t_95_ci'], [0.0, 0.0])
        self.assertEqual(result['paired_t']['p_two_sided'], 1.0)
        self.assertEqual(result['wilcoxon_exact']['p_two_sided'], 1.0)


if __name__ == '__main__':
    unittest.main()
