#!/usr/bin/env python3
"""Assemble a new extracted corpus from old files, repaired PDF pages and clean HTML."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_atomic(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", encoding="utf-8", newline="\n") as out:
        for row in items:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/nntu_site/extracted")
    parser.add_argument("--work", type=Path, default=ROOT / "work/v2")
    args = parser.parse_args()
    html = {x["doc_id"]: x for x in rows(args.work / "html_blocks.jsonl")}
    repairs = {(x["doc_id"], x["page"]): x for x in rows(args.work / "ocr_repairs.jsonl") if x["accepted"]}
    counts = Counter()
    emitted_html = set()

    def blocks():
        for original in rows(args.source / "blocks.jsonl"):
            doc_id = original.get("doc_id")
            if original.get("source_type") == "html":
                if doc_id in html and doc_id not in emitted_html:
                    emitted_html.add(doc_id)
                    counts["clean_html"] += 1
                    yield html[doc_id]
                else:
                    counts["old_html_removed"] += 1
                continue
            repair = repairs.get((doc_id, original.get("page")))
            if repair:
                original.update(text=repair["text"], ocr=True,
                                ocr_confidence=repair["ocr_confidence"], method="tesseract_v2")
                counts["pdf_pages_repaired"] += 1
            yield original
            counts["original_file_blocks"] += 1
        for doc_id, block in html.items():
            if doc_id not in emitted_html:
                counts["new_html"] += 1
                yield block

    destination = args.work / "corpus/extracted"
    write_atomic(destination / "blocks.jsonl", blocks())

    def documents():
        for doc in rows(args.source / "documents.jsonl"):
            block = html.get(doc.get("doc_id"))
            if block:
                doc["category"] = block["category"]
                doc["text_chars"] = len(block["text"])
            yield doc

    write_atomic(destination / "documents.jsonl", documents())
    (args.work / "assembly_report.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
