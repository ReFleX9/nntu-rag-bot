"""Audit table documents across v2 and produce an original-file catalogue."""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

from review_corpus_v3 import records

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_hybrid_langchain/rag_hybrid"))
from file_lookup import document_type, groups, official_file, years


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    work = args.work
    source = work / "v3_candidate_r4"
    core = list(records(source / "core.jsonl"))
    extra = list(records(source / "supplemental.jsonl"))
    core_ids = {row["chunk_id"] for row in core}
    extra_ids = {row["chunk_id"] for row in extra}
    core_docs = {row["document_id"] for row in core}
    extra_docs = {row["document_id"] for row in extra}
    docs = {}
    for row in records(work / "v2/chunks.jsonl"):
        url = next((url for url in row["source_urls"] if official_file(url)), None)
        if not url:
            continue
        key = row["document_id"]
        heading = row["text"].splitlines()[0][:200] if row["text"].splitlines() else ""
        name = unquote(row["title"])
        if re.match(r"radio\+|политехник\b", name, re.I):
            continue
        kind = document_type(name)
        if kind is None and row["chunk_index"] == 0:
            kind = document_type(heading)
        info = docs.setdefault(key, {"document_id": key, "title": name, "url": url,
                                     "type": kind, "groups": set(), "years": years(name),
                                     "counts": Counter()})
        chunk_id = row["chunk_id"]
        bucket = "core" if chunk_id in core_ids else "supplemental" if chunk_id in extra_ids else "outside"
        info["counts"][bucket] += 1
        info["counts"]["tables"] += row["kind"] == "table"
        info["counts"]["ocr"] += bool(row["is_ocr"])
        if info["type"] in {"calendar", "schedule", "exam", "study_plan"}:
            info["groups"].update(groups(row["text"]))
        if row["chunk_index"] == 0:
            info["years"].update(years(heading))
    output = args.output or work / "v3_candidate_r5"
    output.mkdir()
    catalog = []
    direct = set()
    for key, info in docs.items():
        c = info["counts"]
        if c["tables"] == 0 and not info["type"]:
            continue
        info["scope"] = "core" if key in core_docs else "supplemental" if key in extra_docs else "outside"
        info["groups"] = sorted(info["groups"])
        info["years"] = sorted(info["years"])
        # Preserve prose in mixed documents. Complete schedules and calendars are file lookups.
        info["file_only"] = info["type"] in {"schedule", "exam", "calendar"}
        if info["file_only"]:
            direct.add(key)
        catalog.append(info)

    def write(name, rows):
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    suppressed = [row for row in core + extra if row["document_id"] in direct
                  and row.get("editorial_scope") != "navigation_only"]
    removed = {row["chunk_id"] for row in suppressed}
    write("core.jsonl", (row for row in core if row["chunk_id"] not in removed))
    write("supplemental.jsonl", (row for row in extra if row["chunk_id"] not in removed))
    write("raw_table_review.jsonl", suppressed)
    write("file_catalog.jsonl", catalog)
    print("catalog", len(catalog), Counter((row["scope"], row["type"] or "table_other") for row in catalog))
    print("removed from text search", len(removed), "core", len(removed & core_ids), "supplemental", len(removed & extra_ids))


if __name__ == "__main__":
    main()
