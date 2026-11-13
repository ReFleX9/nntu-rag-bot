#!/usr/bin/env python3
"""Summarize the selected corpus and verify essential NNTU sources are present."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ESSENTIAL = {
    "student_platforms": "elektronnaya-obrazovatelnaya-sreda",
    "transfer_expulsion_policy": "polog-porjdok-perevod-otchisl-vosstanovl.pdf",
    "admissions": "/abit/",
    "schedule": "raspisanie",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    categories, sources, documents = Counter(), Counter(), Counter()
    examples = {}
    essential = {name: None for name in ESSENTIAL}
    low_confidence = 0
    short = 0
    hashes = set()
    duplicates = 0
    with args.source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            categories[row.get("category") or "unknown"] += 1
            sources[row.get("source_type") or "unknown"] += 1
            doc_id = row["document_id"]
            documents[doc_id] += 1
            url = (row.get("source_urls") or [""])[0]
            examples.setdefault(doc_id, {"title": row.get("title"), "url": url})
            for name, part in ESSENTIAL.items():
                if essential[name] is None and part in url.lower():
                    essential[name] = {"title": row.get("title"), "url": url}
            low_confidence += bool(row.get("is_ocr") and row.get("ocr_confidence") is not None
                                   and row["ocr_confidence"] < 60)
            short += len(row.get("text") or "") < 80
            digest = row.get("content_sha256")
            duplicates += digest in hashes
            hashes.add(digest)
    report = {"chunks": sum(categories.values()), "documents": len(documents),
              "by_category": categories, "by_source_type": sources,
              "low_confidence_ocr_chunks": low_confidence, "short_chunks": short,
              "exact_text_duplicates": duplicates, "essential_sources": essential,
              "largest_documents": [{"chunks": n, **examples[doc_id]}
                                    for doc_id, n in documents.most_common(20)]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
