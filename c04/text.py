"""Training-only vocabulary and deterministic short-text tokenization."""
from collections import Counter
import re

PAD = "<pad>"
UNK = "<unk>"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]", re.ASCII)


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(rows, min_frequency=1):
    if min_frequency < 1 or not rows:
        raise ValueError("A nonempty training split and positive frequency are required")
    if any(not r["id"].startswith("train:") for r in rows):
        raise ValueError("Vocabulary accepts training rows only")
    counts = Counter(token for row in rows for token in tokenize(row["text"]))
    tokens = sorted(t for t, n in counts.items() if n >= min_frequency and t not in {PAD, UNK})
    return {token: i for i, token in enumerate([PAD, UNK, *tokens])}


def encode(text, vocab, max_length):
    if max_length < 1:
        raise ValueError("max_length must be positive")
    return [vocab.get(token, 1) for token in tokenize(text)[:max_length]] or [1]


def text_stats(rows, vocab, max_length):
    tokens = [tokenize(r["text"]) for r in rows]
    total = sum(map(len, tokens))
    unknown = sum(token not in vocab for row in tokens for token in row)
    return {"rows": len(rows), "tokens": total, "unknown_tokens": unknown,
            "unknown_token_rate": unknown / total if total else 0.0,
            "truncated_rows": sum(len(row) > max_length for row in tokens),
            "max_observed_length": max(map(len, tokens), default=0)}
