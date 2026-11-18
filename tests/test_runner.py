import sys
import tempfile
import unittest
from pathlib import Path

from c04.run_noise_matrix import run_command


class RunnerTests(unittest.TestCase):
    def test_command_log_is_normalized_to_utf8(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "child.log"
            output = run_command([sys.executable, "-c", "print(chr(177))"], path)
            self.assertIn("±", output)
            self.assertIn("±", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
