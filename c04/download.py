"""Resume large official downloads with bounded Range requests and SHA256 checks."""
import argparse
from pathlib import Path
import re
import time
import urllib.request

from .common import file_sha256


def download(url, output, expected_sha256=None, chunk_bytes=32*1024**2):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        checksum = file_sha256(output)
        if expected_sha256 and checksum != expected_sha256:
            raise ValueError(f"Existing file checksum mismatch: {output}")
        return checksum
    partial = output.with_name(output.name+".download")
    offset = partial.stat().st_size if partial.exists() else 0
    total = None
    failures = 0
    while total is None or offset < total:
        end = offset+chunk_bytes-1
        request = urllib.request.Request(url, headers={"User-Agent": "C04/0.2",
            "Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if response.status != 206 or not match or int(match[1]) != offset:
                    raise ValueError("Server did not honor the requested byte range")
                stop, total = int(match[2])+1, int(match[3])
                if stop > end+1 or stop <= offset or total < stop:
                    raise ValueError("Invalid Content-Range response")
                with partial.open("ab") as stream:
                    while offset < stop:
                        block = response.read(min(1024**2, stop-offset))
                        if not block:
                            raise OSError("Download ended before the requested range was complete")
                        stream.write(block)
                        offset += len(block)
            failures = 0
            print(f"{output.name}: {offset/1024**2:.1f}/{total/1024**2:.1f} MiB", flush=True)
        except (OSError, ValueError) as error:
            # Completed bytes stay on disk; retry starts at the current exact offset.
            failures += 1
            if failures >= 4:
                raise RuntimeError(f"Download paused at {offset} bytes; rerun to resume") from error
            print(f"retry {failures}: {error}", flush=True)
            time.sleep(min(failures, 3))
    checksum = file_sha256(partial)
    if expected_sha256 and checksum != expected_sha256:
        raise ValueError(f"Downloaded checksum mismatch: {checksum}; expected {expected_sha256}")
    partial.replace(output)
    print(f"{output.name}: SHA256={checksum}", flush=True)
    return checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    download(args.url, args.output, args.sha256)


if __name__ == "__main__":
    main()
