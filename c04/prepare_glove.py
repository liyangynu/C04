"""Download and verify Stanford GloVe 6B, extracting only the 100d member."""
import argparse
from pathlib import Path
import shutil
import zipfile

from .common import file_sha256, read_json, write_json
from .download import download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/embeddings"))
    args = parser.parse_args()
    spec = read_json("configs/glove.json")
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output/"glove.6B.zip"
    download(spec["source"], archive, spec["archive_sha256"])
    member = args.output/spec["member"]
    if not member.exists():
        partial = member.with_suffix(".txt.download")
        with zipfile.ZipFile(archive) as z, z.open(spec["member"]) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target)
        if file_sha256(partial) != spec["member_sha256"]:
            raise ValueError("Extracted GloVe checksum mismatch")
        partial.replace(member)
    if file_sha256(member) != spec["member_sha256"]:
        raise ValueError("GloVe file checksum mismatch")
    write_json(args.output/"glove_manifest.json", spec)
    print(f"Verified {member}")


if __name__ == "__main__":
    main()
