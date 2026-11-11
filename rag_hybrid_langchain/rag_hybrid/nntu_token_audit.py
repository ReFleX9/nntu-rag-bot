"""Count the tokens in the exact text sent to the embedding model."""

import argparse
from collections import Counter
from itertools import batched
import json
from pathlib import Path

from corpus import index_text

MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--tokenizer-path", type=Path, help="Локальный снимок указанной ревизии")
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path) if args.tokenizer_path else MODEL,
        revision=None if args.tokenizer_path else REVISION, cache_dir=args.cache_dir,
        local_files_only=bool(args.tokenizer_path), trust_remote_code=False,
    )
    lengths = Counter()
    over_limit = 0
    examples = []
    with args.chunks.open(encoding="utf-8-sig") as stream:
        rows = (json.loads(line) for line in stream if line.strip())
        for batch in batched((row for row in rows if row["indexable"]), 128):
            inputs = tokenizer(
                [index_text(row) for row in batch], add_special_tokens=True,
                padding=False, truncation=False, return_attention_mask=False,
            )["input_ids"]
            for row, tokens in zip(batch, inputs, strict=True):
                length = len(tokens)
                lengths[length] += 1
                if length > args.max_length:
                    over_limit += 1
                    if len(examples) < 20:
                        examples.append({"chunk_id": row["chunk_id"], "tokens": length})
    report = {
        "model": MODEL, "revision": REVISION, "chunks": sum(lengths.values()),
        "max_tokens": max(lengths, default=0), "limit": args.max_length,
        "over_limit": over_limit, "examples": examples, "histogram": dict(sorted(lengths.items())),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print({key: value for key, value in report.items() if key != "histogram"})


if __name__ == "__main__":
    main()
