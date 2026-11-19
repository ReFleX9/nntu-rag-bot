"""Audit category 03 and select traceable bibliography excerpts and original files."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

HEADING = re.compile(
    r"(?<!\w)(?:основная|дополнительная|учебно[- ]методическая)\s+литература\b", re.I
)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
AUTHOR = re.compile(r"[А-ЯЁ][а-яё]{2,},?\s+[А-ЯЁ]\.\s*(?:[А-ЯЁ]\.)?")
ISBN = re.compile(r"\bISBN\s*[:\-]?\s*[\dXхХ-]{8,}", re.I)
LIBRARY = re.compile(r"https?://[^\s<>]*(?:lanbook|urait|znanium|biblioclub|ibooks)[^\s<>]*", re.I)
TABLE = re.compile(r"автор\(?ы?\)?.{0,100}(?:заглавие|название|издательств)|"
                   r"кол-?во.{0,30}(?:экз|библи)|столбец\s+\d|\|", re.I | re.S)
TOC = re.compile(r"\.{4,}|(?:основная|дополнительная)\s+литература\s+\d{1,3}\s*(?:\n|$)", re.I)


def write(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def decide(row):
    text = row["text"]
    match = HEADING.search(text)
    if not match:
        return "no_literature_heading", None
    tail = text[match.end():]
    evidence = {"years": sorted(set(YEAR.findall(tail))),
                "authors": AUTHOR.findall(tail), "isbn": ISBN.findall(tail),
                "library_urls": LIBRARY.findall(tail)}
    if len(tail.strip()) < 100 or not (
        evidence["years"] and (evidence["authors"] or evidence["isbn"] or evidence["library_urls"])
    ):
        return "heading_without_bibliographic_evidence", evidence
    if TOC.search(text):
        return "possible_contents_page", evidence
    if not row.get("indexable") or row.get("indexing_exclusions"):
        return "source_not_indexable", evidence
    if row.get("is_ocr") or row.get("needs_review") or "\ufffd" in text:
        return "quality_review", evidence
    if row["kind"] == "table" or TABLE.search(tail):
        return "original_file", evidence
    return "text_excerpt", evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    counts, kinds, families = Counter(), Counter(), Counter()
    docs, file_candidates = defaultdict(list), {}
    routes = defaultdict(set)
    examples = defaultdict(list)
    with args.source.open("rb") as source:
        source_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    with (
        args.source.open(encoding="utf-8") as source,
        (args.output / "category03_raw.jsonl").open("w", encoding="utf-8") as raw,
        (args.output / "literature_text.jsonl").open("w", encoding="utf-8") as selected,
        (args.output / "decisions.jsonl").open("w", encoding="utf-8") as decisions,
    ):
        for line in source:
            row = json.loads(line)
            if not (row.get("category") or "").startswith("03_"):
                continue
            write(raw, row)
            kinds[row["kind"]] += 1
            families[re.split("[_-]", row["title"])[0].casefold()] += 1
            docs[row["document_id"]].append(row)
            route, evidence = decide(row)
            counts[route] += 1
            routes[route].add(row["document_id"])
            write(decisions, {"chunk_id": row["chunk_id"], "document_id": row["document_id"],
                              "title": row["title"], "page": row.get("page"),
                              "route": route, "evidence": evidence})
            if route == "text_excerpt":
                start = HEADING.search(row["text"]).start()
                excerpt = row["text"][start:]
                digest = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
                derived_id = hashlib.sha256((row["chunk_id"] + ":literature:" + digest).encode()).hexdigest()[:32]
                copy = dict(row, text=excerpt, chunk_id=derived_id,
                            source_chunk_id=row["chunk_id"], source_text_offset=start,
                            content_sha256=digest, length_chars=len(excerpt), token_count=None,
                            prev_chunk_id=None, next_chunk_id=None,
                            section_title="Литература по дисциплине",
                            section_path=["Литература по дисциплине"],
                            corpus_inclusion_reason="bibliography_heading_year_author_or_identifier")
                write(selected, copy)
            if route in {"text_excerpt", "original_file", "quality_review"}:
                for url in row.get("source_urls", []):
                    host = urlsplit(url).hostname or ""
                    if host == "nntu.ru" or host.endswith(".nntu.ru"):
                        key = (row["document_id"], url)
                        item = file_candidates.setdefault(key, {
                            "document_id": row["document_id"], "title": row["title"],
                            "url": url, "purpose": "literature_list_in_source_document",
                            "pages": set(), "chunk_ids": [], "routes": set()})
                        if row.get("page") is not None:
                            item["pages"].add(row["page"])
                        item["chunk_ids"].append(row["chunk_id"])
                        item["routes"].add(route)
            if evidence and len(examples[route]) < 6:
                examples[route].append({"title": row["title"], "page": row.get("page"),
                                        "chunk_id": row["chunk_id"], "text": row["text"]})
    with (args.output / "literature_originals.jsonl").open("w", encoding="utf-8") as stream:
        for item in file_candidates.values():
            write(stream, {**item, "pages": sorted(item["pages"]), "routes": sorted(item["routes"])})
    report = {"source": str(args.source), "source_sha256": source_sha256,
              "category_chunks": sum(kinds.values()), "category_documents": len(docs),
              "by_kind": kinds, "top_filename_families": families.most_common(20),
              "decisions": counts, "documents_by_decision": {k: len(v) for k, v in routes.items()},
              "original_file_entries": len(file_candidates), "examples": examples}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
