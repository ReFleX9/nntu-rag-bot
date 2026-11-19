"""Restore selected source-text chunks from categories 01–03 into r7."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit
from urllib.parse import unquote

from review_corpus_v3 import records
from expand_student_corpus import write

ADMISSION_PAGES = {
    "/content/abiturientam/napravleniya",
    "/content/abiturientam/vstupitelnye-ispytaniya",
    "/content/abiturientam/pravila-priema",
    "/content/abiturientam/informaciya-dlya-inostrancev",
    "/content/abiturientam/celevoi-priem",
    "/content/abiturientam/priem-v-magistraturu",
    "/content/abiturientam/podacha-dokumentov",
    "/content/abiturientam/podgotovitelnye-kursy",
}


def reason(row):
    category = row["category"][:2]
    title = row["title"]
    text = row["text"]
    idx = row["chunk_index"]
    urls = row["source_urls"]
    if (not row["indexable"] or row.get("indexing_exclusions") or row.get("needs_review")
            or len(text) < 100 or row["kind"] not in {"html", "prose", "ocr_prose", "list"}):
        return None
    if row["is_ocr"] and (row.get("ocr_confidence") or 0) < 90:
        return None
    if category == "01" and any(urlsplit(url).path in ADMISSION_PAGES for url in urls):
        if row["source_type"] == "html" and not re.search(r"архив|(?:202[0-5])\s*г", text, re.I):
            return "admission_navigation"
    if category == "02" and title == "instr-pb-obchezh-studgorodok-pril11.pdf":
        if idx <= 2:
            return "dormitory_fire_rules"
    if category == "03" and title == "Praktiki_NGTU.pdf" and 5 <= idx <= 11:
        if not re.search(r"\bтаблица\s+\d|срок действия|лист регистрации", text, re.I):
            return "practice_procedure"
    if (category == "03" and re.match(r"^OPVO_", title, re.I)
            and any(urlsplit(url).path.startswith("/sveden/files/education/") for url in urls)
            and not row["is_ocr"] and idx <= 12
            and not re.search(r"оглавление|сертификат ЭП|столбец|таблица\s+\d", text, re.I)):
        if re.search(r"квалификаци[яию]|форм[аы]\s+обучения|срок\s+освоения", text, re.I):
            return "program_profile"
    return None


PROGRAM_PATH = re.compile(r"^/sveden/files/education/\d{2}\.\d{2}\.\d{2}/", re.I)
OVERVIEW = re.compile(
    r"цели?\s+(?:освоения|дисциплины)|место дисциплины|"
    r"краткое содержание дисциплины|аннотация рабочей программы|"
    r"характеристика (?:профессиональной деятельности|образовательной программы)|"
    r"общая характеристика основной профессиональной образовательной программы",
    re.I,
)


def expanded_reason(row):
    text = row["text"]
    category = row["category"][:2]
    title = row["title"].casefold()
    idx = row["chunk_index"]
    paths = [unquote(urlsplit(url).path).casefold() for url in row["source_urls"]]
    if (not row["indexable"] or row.get("indexing_exclusions") or row.get("needs_review")
            or row["kind"] not in {"prose", "html", "list"} or row["is_ocr"]
            or len(text) < 100):
        return None
    if category == "01" and title == "pamyatka-abit-celevoe-ob.pdf" and idx == 0:
        if any("/abiturientam/2025/" in path for path in paths):
            return "targeted_admission_guide_2025"
    if category == "02" and title == "prikaz_n35_30.01.2026.pdf" and idx == 0:
        if any("/studgorodok/" in path for path in paths):
            return "dormitory_improved_room_order_2026"
    if (category == "03" and re.match(r"^(?:rpp|rpd|rp|dis|ann|opvo|opop)[_-]", title)
            and any(PROGRAM_PATH.match(path) for path in paths)
            and idx <= 10 and OVERVIEW.search(text)
            and not re.search(r"оглавление|сертификат ЭП|столбец|таблица\s+\d|"
                              r"сведения о сертификате|лист регистрации изменений|\.{5,}", text, re.I)):
        return "program_or_course_overview"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expand", action="store_true")
    args = parser.parse_args()
    work = args.work
    base = work / ("v3_candidate_r9" if args.expand else "v3_candidate_r7")
    core = list(records(base / "core.jsonl"))
    present = {row["chunk_id"] for row in core}
    seen = {hashlib.sha256(row["text"].encode()).hexdigest() for row in core}
    restored = []
    reasons = {}
    by_document = defaultdict(list)
    raw_counts = Counter()
    raw_docs = defaultdict(set)
    catalog_docs = {row["document_id"] for row in records(base / "file_catalog.jsonl")}
    raw_in_catalog = Counter()
    raw_in_text = Counter()
    for row in records(work / "v2/chunks.jsonl"):
        category = row["category"][:2]
        if category not in {"01", "02", "03"}:
            continue
        raw_counts[category] += 1
        raw_docs[category].add(row["document_id"])
        raw_in_catalog[category] += row["document_id"] in catalog_docs
        if row["chunk_id"] in present:
            raw_in_text[category] += 1
            continue
        why = expanded_reason(row) if args.expand else reason(row)
        digest = hashlib.sha256(row["text"].encode()).hexdigest()
        if not why or digest in seen:
            continue
        if why == "program_or_course_overview" and len(by_document[row["document_id"]]) >= 3:
            continue
        row["editorial_scope"] = "source_excerpt"
        row["corpus_inclusion_reason"] = why
        if why == "practice_procedure" and not row.get("section_title"):
            row["section_title"] = "Практическая подготовка обучающихся"
            row["section_path"] = [row["section_title"]]
        if why == "program_profile" and not row.get("section_title"):
            row["section_title"] = "Характеристика образовательной программы"
            row["section_path"] = [row["section_title"]]
        labels = {
            "targeted_admission_guide_2025": "Целевое обучение — памятка приёма 2025 года",
            "dormitory_improved_room_order_2026": "Проживание в помещениях улучшенного типа — приказ от 30.01.2026",
            "program_or_course_overview": "Описание конкретной образовательной программы или дисциплины",
        }
        if why in labels and not row.get("section_title"):
            row["section_title"] = labels[why]
            row["section_path"] = [labels[why]]
        restored.append(row)
        reasons[row["chunk_id"]] = why
        by_document[row["document_id"]].append(row)
        seen.add(digest)
    # Connect only consecutive restored excerpts. Holes remain explicit.
    for rows in by_document.values():
        indices = {row["chunk_index"]: row["chunk_id"] for row in rows}
        for row in rows:
            row["prev_chunk_id"] = indices.get(row["chunk_index"] - 1)
            row["next_chunk_id"] = indices.get(row["chunk_index"] + 1)

    out = args.output or work / "v3_candidate_r8"
    out.mkdir()
    write(out / "core.jsonl", [*core, *restored])
    write(out / "added_text.jsonl", restored)
    for name in ("file_catalog.jsonl", "supplemental.jsonl", "raw_table_review.jsonl"):
        rows = records(base / name)
        if name == "supplemental.jsonl":
            # Preserve only links whose target stays in this file.
            kept = [row for row in rows if row["chunk_id"] not in reasons]
            kept_ids = {row["chunk_id"] for row in kept}
            for row in kept:
                for key in ("prev_chunk_id", "next_chunk_id"):
                    if row.get(key) not in kept_ids:
                        row[key] = None
            rows = kept
        write(out / name, rows)
    report = {"source_version": "r9" if args.expand else "r7",
              "core_before": len(core), "core_after": len(core) + len(restored),
              "category_coverage": {
                  category: {
                      "raw_chunks": raw_counts[category],
                      "raw_documents": len(raw_docs[category]),
                      "raw_chunks_in_text_before": raw_in_text[category],
                      "raw_chunks_in_text_after": raw_in_text[category] + sum(
                          row["category"][:2] == category for row in restored),
                      "raw_chunks_in_catalog_documents": raw_in_catalog[category],
                      "raw_documents_in_catalog": len(raw_docs[category] & catalog_docs),
                  } for category in ("01", "02", "03")
              },
              "added_chunks_by_category": Counter(row["category"][:2] for row in restored),
              "added_chunks_by_reason": Counter(reasons.values()),
              "source_documents_by_category": {
                  category: len({row["document_id"] for row in restored if row["category"][:2] == category})
                  for category in ("01", "02", "03")
              },
              "additions": [{"chunk_id": row["chunk_id"], "title": row["title"], "page": row["page"],
                             "url": row["source_urls"][0], "reason": reasons[row["chunk_id"]]}
                            for row in restored]}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print({key: value for key, value in report.items() if key != "additions"})


if __name__ == "__main__":
    main()
