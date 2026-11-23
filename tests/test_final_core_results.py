"""Check that released tables reject incomplete or inconsistent seed results."""
import shutil
import tempfile
import unittest
from pathlib import Path

from c04.common import file_sha256, read_json, write_json
from c04.final_core_results import verify


class FinalCoreResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'final_core'
        shutil.copytree('reports/final_core', self.root)

    def rewrite_results_and_checksum(self, package):
        write_json(self.root / 'results.json', package)
        checksums = read_json(self.root / 'checksums.json')
        checksums['results.json'] = file_sha256(self.root / 'results.json')
        write_json(self.root / 'checksums.json', checksums)

    def test_released_package_passes_without_local_training_assets(self):
        verify(self.root)

    def test_duplicate_seed_is_rejected_even_with_updated_checksum(self):
        package = read_json(self.root / 'results.json')
        package['noisy_results'][1] = dict(package['noisy_results'][0])
        self.rewrite_results_and_checksum(package)
        with self.assertRaisesRegex(ValueError, '108 distinct'):
            verify(self.root)

    def test_incorrect_retention_is_rejected_even_with_updated_checksum(self):
        package = read_json(self.root / 'results.json')
        package['noisy_results'][0]['f1_retention'] += .01
        self.rewrite_results_and_checksum(package)
        with self.assertRaisesRegex(ValueError, 'F1 retention'):
            verify(self.root)

    def test_test_epoch_cannot_replace_validation_selected_epoch(self):
        package = read_json(self.root / 'results.json')
        row = package['noisy_results'][0]
        row['best_epoch'] = 1 if row['best_epoch'] != 1 else 2
        self.rewrite_results_and_checksum(package)
        with self.assertRaisesRegex(ValueError, 'validation selection'):
            verify(self.root)


if __name__ == '__main__':
    unittest.main()
