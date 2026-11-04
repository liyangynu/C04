"""Build a compact GloVe matrix aligned to a training-only vocabulary."""
import argparse
from pathlib import Path

import numpy as np

from .common import file_sha256, read_jsonl, stable_seed, write_json
from .text import build_vocab, text_stats, tokenize


def load_glove(path, vocab, dimension, seed):
    if vocab.get("<pad>") != 0 or vocab.get("<unk>") != 1:
        raise ValueError("Vocabulary must reserve PAD=0 and UNK=1")
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Vocabulary IDs must be contiguous")
    rng = np.random.default_rng(stable_seed(seed, "embedding-initialization"))
    matrix = rng.normal(0, 0.05, (len(vocab), dimension)).astype(np.float32)
    matrix[0] = 0
    found = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            token, separator, vector = line.rstrip().partition(" ")
            if not separator or token not in vocab or token in {"<pad>", "<unk>"}:
                continue
            values = np.fromstring(vector, sep=" ", dtype=np.float32)
            if values.shape != (dimension,) or not np.isfinite(values).all():
                raise ValueError(f"Malformed GloVe vector: {token}")
            if token in found:
                raise ValueError(f"Duplicate GloVe token: {token}")
            matrix[vocab[token]] = values
            found.add(token)
    return matrix, found


def build_assets(train, glove, output, seed, dimension=100, max_length=40):
    rows = read_jsonl(train)
    vocab = build_vocab(rows)
    matrix, found = load_glove(glove, vocab, dimension, seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"vocab.json", vocab)
    np.save(output/"embeddings.npy", matrix, allow_pickle=False)
    tokens = [token for row in rows for token in tokenize(row["text"])]
    report = {"seed": seed, "dimension": dimension, "vocab_size": len(vocab),
              "train_sha256": file_sha256(train), "glove_sha256": file_sha256(glove),
              "vocab_sha256": file_sha256(output/"vocab.json"),
              "matrix_sha256": file_sha256(output/"embeddings.npy"),
              "covered_types": len(found), "total_types": len(vocab)-2,
              "type_coverage": len(found)/(len(vocab)-2) if len(vocab)>2 else 0.0,
              "token_coverage": sum(t in found for t in tokens)/len(tokens) if tokens else 0.0,
              "text_stats": text_stats(rows, vocab, max_length)}
    write_json(output/"manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report = build_assets(args.train, args.glove, args.output, args.seed)
    print(f"vocab={report['vocab_size']}; GloVe types={report['type_coverage']:.2%}; tokens={report['token_coverage']:.2%}")


if __name__ == "__main__":
    main()
