"""Prepare two search corpora and preserve the raw OCR of the reviewed schedule."""
import argparse
import hashlib
import json
from pathlib import Path

from review_corpus_v3 import records

SCHEDULE_ID = "f3e27a7558459634403242f4f3ee6fde0c18a91c032469c7bf141ac72e058ffd"
TITLE = "Расписание ИЯЭиТФ — 1 курс, очная форма, первый семестр 2026/2027"
SUMMARY = (TITLE + ". Откройте PDF по ссылке источника и найдите столбец своей группы. "
           "Время, аудиторию и номер пары сверяйте по таблице оригинала.")


def write_rows(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work"))
    args = parser.parse_args()
    core = list(records(args.work / "v3_candidate_r3/core.jsonl"))
    supplemental = list(records(args.work / "v3_candidate_r2/supplemental.jsonl"))
    raw = [row for row in core if row["document_id"] == SCHEDULE_ID]
    first = next((row for row in raw if row["chunk_index"] == 0), None)
    if first is None or not all(row["is_ocr"] for row in raw):
        parser.error("Ожидалась OCR-версия расписания из кандидата r3")
    digest = hashlib.sha256(SUMMARY.encode("utf-8")).hexdigest()
    card = dict(first, text=SUMMARY, title=TITLE, kind="prose", page=None,
                citation_label=TITLE, content_sha256=digest, length_chars=len(SUMMARY),
                document_length_chars=len(SUMMARY), chunk_id=digest[:32],
                prev_chunk_id=None, next_chunk_id=None, is_ocr=False, ocr_confidence=None,
                quality_flags=[], sources=[], source_pages=[], editorial_scope="navigation_only")
    main_rows = [row for row in core if row["document_id"] != SCHEDULE_ID] + [card]
    output = args.work / "v3_candidate_r4"
    output.mkdir()
    write_rows(output / "core.jsonl", main_rows)
    write_rows(output / "supplemental.jsonl", supplemental)
    write_rows(output / "schedule_ocr_review.jsonl", raw)
    report = {"core_chunks": len(main_rows), "supplemental_chunks": len(supplemental),
              "schedule_ocr_chunks": len(raw), "schedule_ocr_pages": sorted({r["page"] for r in raw}),
              "schedule_link": card["source_urls"][0], "navigation_chunk_id": card["chunk_id"]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
