"""Range-download recovery and checksum validation without network access."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from c04.download import download


class Response(io.BytesIO):
    def __init__(self, payload, start, end, total):
        super().__init__(payload)
        self.status = 206
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}


class DownloadTests(unittest.TestCase):
    def test_resume_existing_partial_and_verify_content(self):
        content = b"abcdef0123456789"
        checksum = hashlib.sha256(content).hexdigest()
        seen = []
        def fetch(request, timeout):
            span = request.get_header("Range").split("=")[1]
            start, end = map(int, span.split("-"))
            end = min(end, len(content)-1)
            seen.append(start)
            return Response(content[start:end+1], start, end, len(content))
        with tempfile.TemporaryDirectory() as d, patch("c04.download.urllib.request.urlopen", side_effect=fetch):
            output = Path(d)/"asset.bin"
            output.with_name("asset.bin.download").write_bytes(content[:3])
            self.assertEqual(download("https://example.test/asset", output, checksum, chunk_bytes=4), checksum)
            self.assertEqual(output.read_bytes(), content)
            self.assertEqual(seen, [3, 7, 11, 15])

    def test_incorrect_checksum_is_not_promoted_to_final_file(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/"asset.bin"
            with patch("c04.download.urllib.request.urlopen", return_value=Response(b"abc", 0, 2, 3)):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    download("https://example.test/asset", output, "0"*64)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
