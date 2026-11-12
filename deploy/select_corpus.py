"""Prepare the first NNTU index or a random benchmark sample of it."""

import argparse
import json
import random
from pathlib import Path

CORE = {
    "01_поступающим",
    "02_студентам",
    "03_образовательные_программы",
    "04_кампусная_жизнь",
    "07_об_университете",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--include-documents", type=Path,
                        help="UTF-8 file with one additional document_id per line")
    parser.add_argument("--sample", type=int)
    args = parser.parse_args()
    if args.sample is not None and args.sample < 1:
        parser.error("--sample must be positive")
    extra = set()
    if args.include_documents:
        extra = {line.strip() for line in
                 args.include_documents.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip()}

    def eligible():
        with args.source.open(encoding="utf-8-sig") as source:
            for line in source:
                row = json.loads(line)
                if not row["indexable"] or row.get("indexing_exclusions"):
                    continue
                if row.get("category") in CORE or row["document_id"] in extra:
                    yield line

    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    sample_indices = None
    if args.sample is not None:
        total = sum(1 for _ in eligible())
        sample_indices = set(random.Random(42).sample(range(total), min(total, args.sample)))
    count = 0
    temp = args.output.with_suffix(args.output.suffix + ".partial")
    with temp.open("w", encoding="utf-8", newline="\n") as output:
        try:
            for i, line in enumerate(eligible()):
                if sample_indices is None or i in sample_indices:
                    output.write(line.rstrip("\n") + "\n")
                    count += 1
        except BaseException:
            output.close()
            temp.unlink()
            raise
    if not count:
        temp.unlink()
        parser.error("No eligible chunks")
    temp.rename(args.output)
    print(f"{count:,} chunks -> {args.output}")


if __name__ == "__main__":
    main()
