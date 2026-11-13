#!/usr/bin/env python3
"""Find PDF pages whose nonempty native text should be checked with OCR."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
LATIN = re.compile(r"[A-Za-z]")
UPPER_LATIN = re.compile(r"[A-Z]")


def signals(text: str) -> dict:
    cyr = len(CYRILLIC.findall(text))
    latin = len(LATIN.findall(text))
    upper = len(UPPER_LATIN.findall(text))
    letters = cyr + latin
    replacements = text.count("\ufffd")
    return {"cyrillic": cyr, "latin": latin,
            "cyrillic_share": round(cyr / max(1, letters), 4),
            "latin_upper_share": round(upper / max(1, latin), 4),
            "replacement_chars": replacements,
            "text_chars": len(text)}


def suspicious(text: str, info: dict) -> str | None:
    if info["text_chars"] < 120:
        return None
    if info["replacement_chars"] >= max(3, info["text_chars"] // 500):
        return "replacement_characters"
    # Broken embedded fonts often turn Russian text into long pseudo-Latin runs.
    if (info["latin"] >= 150 and info["cyrillic_share"] < .12
            and info["latin_upper_share"] > .25):
        return "pseudo_latin_font"
    if (info["latin"] >= 150 and info["cyrillic_share"] < .08
            and len(re.findall(r"[~<>]{2,}|[A-Za-z]\d[A-Za-z]", text)) >= 8):
        return "garbled_glyphs"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=Path, default=ROOT / "data/nntu_site/extracted/blocks.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "work/v2/ocr_candidates.jsonl")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".partial")
    counts = Counter()
    seen = set()
    with args.blocks.open(encoding="utf-8") as source, temp.open("w", encoding="utf-8", newline="\n") as target:
        for line in source:
            row = json.loads(line)
            path = row.get("source_path") or ""
            if not path.lower().endswith(".pdf") or row.get("ocr"):
                continue
            info = signals(row.get("text") or "")
            reason = suspicious(row.get("text") or "", info)
            if not reason:
                continue
            key = (row.get("doc_id"), row.get("page"))
            if key in seen:
                continue
            seen.add(key)
            target.write(json.dumps({"doc_id": key[0], "page": key[1],
                                     "source_path": path, "title": row.get("title"),
                                     "category": row.get("category"), "reason": reason,
                                     **info}, ensure_ascii=False) + "\n")
            counts[reason] += 1
    temp.replace(args.output)
    print(json.dumps({"candidate_pages": sum(counts.values()), "by_reason": counts},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
