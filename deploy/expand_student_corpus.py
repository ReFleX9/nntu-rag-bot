"""Add reviewed navigation cards and educational originals to the r6 corpus."""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

from review_corpus_v3 import records

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_hybrid_langchain/rag_hybrid"))
from file_lookup import PROGRAM, document_type, groups, official_file, years

FAMILIES = {
    "syllabus": r"^(?:rpp|rpd|rp|dis|ann)[_-]",
    "study_plan": r"^(?:ucheb|up)[_-]",
    "program_description": r"^(?:opop|opvo|progr)[_-]",
    "assessment_material": r"^(?:os|fos|kkh)[_-]",
    "educational_material": r"^(?:posob|metod)[_-]",
}
SPECIAL = {
    "3-5_kurs_vesna_2026.xlsx": ("schedule", "2025/2026"),
    "1-2_kurs_vesna_2026.xlsx": ("schedule", "2025/2026"),
    "mag_vesna_2026.xlsx": ("schedule", "2025/2026"),
    "och_ITS_1_2_3_4_5k_ekzam_leto_2026.xlsx": ("exam", "2025/2026"),
    "baza-praktik-26.xlsx": ("practice_catalog", None),
    "bazy_praktik.xlsx": ("practice_catalog", None),
    "dogovor_praktiki.docx": ("practice_form", None),
    "Praktiki_NGTU.pdf": ("practice_policy", None),
    "pamyatka-abit-celevoe-ob.pdf": ("admission_guide", None),
}
CARDS = {
    "https://nntu.ru/sveden/files/education/Praktiki_NGTU.pdf":
        ("Положение о практической подготовке НГТУ",
         "В оригинале положение о практической подготовке обучающихся НГТУ, код ПВД-11.3/80-20. "
         "Для условий и порядка практики откройте документ и уточните применимую редакцию в университете."),
    "https://nntu.ru/content/abiturientam/obshhezhitiya":
        ("Общежития для поступающих",
         "Страница НГТУ для иногородних поступающих с информацией об общежитиях. "
         "Сведения о количестве мест на странице относятся к набору 2026 года."),
}


def write(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work"))
    args = parser.parse_args()
    work = args.work
    previous = work / "v3_candidate_r6"
    catalog = {row["document_id"]: row for row in records(previous / "file_catalog.jsonl")}
    candidates = {}
    cards = {}
    for row in records(work / "v2/chunks.jsonl"):
        category = row["category"][:2]
        title = row["title"]
        urls = row["source_urls"]
        for url in urls:
            if url in CARDS and row["chunk_index"] == 0:
                cards.setdefault(url, row)
        if category not in {"01", "02", "03"} and title not in SPECIAL:
            continue
        url = next((value for value in urls if official_file(value)), None)
        if url is None:
            continue
        path = unquote(urlsplit(url).path)
        if re.search(r"/(?:nomera|politehnik|radio\+|rezultaty)/", path, re.I):
            continue
        name = unquote(title)
        kind, period = SPECIAL.get(title, (None, None))
        if not kind and category == "03" and path.startswith("/sveden/files/education/"):
            kind = next((family for family, pattern in FAMILIES.items() if re.match(pattern, name, re.I)),
                        document_type(name))
        if not kind and category in {"01", "02"} and any(
            key in path.lower() for key in ("/studentam/", "/abiturientam/")
        ):
            kind = document_type(name)
        if not kind:
            continue
        key = row["document_id"]
        entry = candidates.setdefault(key, dict(document_id=key, title=name, url=url, type=kind,
                                               category=category, scope="core" if category == "03" else "outside",
                                               groups=set(), years=set(), program_codes=set(),
                                               source_chunks=0))
        entry["source_chunks"] += 1
        entry["years"].update(years(name))
        entry["program_codes"].update(PROGRAM.findall(path))
        if period:
            entry["years"].add(period)
        if row["chunk_index"] == 0:
            entry["years"].update(years(row["text"].split("\n", 1)[0]))
        if kind in {"schedule", "exam", "calendar"}:
            entry["groups"].update(groups(row["text"]))

    for key, row in candidates.items():
        for field in ("groups", "years", "program_codes"):
            row[field] = sorted(row[field])
        if key in catalog:
            previous_entry = catalog[key]
            row["scope"] = previous_entry["scope"] if row["category"] != "03" else "core"
            for field in ("groups", "years"):
                row[field] = sorted(set(row[field]) | set(previous_entry[field]))
        if row["type"] in {"practice_form", "practice_policy"} and row["title"] in SPECIAL:
            row["scope"] = "core"
        catalog[key] = {**catalog.get(key, {}), **row, "file_only": True}

    out = work / "v3_candidate_r7"
    out.mkdir()
    text_rows = list(records(previous / "core.jsonl"))
    extra = list(records(previous / "supplemental.jsonl"))
    suppressed = [r for r in text_rows + extra if r["title"] in SPECIAL
                  and SPECIAL[r["title"]][0] in {"schedule", "exam", "practice_catalog", "practice_form"}]
    removed = {r["chunk_id"] for r in suppressed}
    new_core = [r for r in text_rows if r["chunk_id"] not in removed]
    for url, source in cards.items():
        if any(url in row["source_urls"] for row in new_core):
            continue
        title, summary = CARDS[url]
        digest = hashlib.sha256((url + "\n" + summary).encode("utf-8")).hexdigest()
        new_core.append(dict(source, title=title, text=summary, section_title=None, section_path=[],
                             page=None, citation_label=title, kind="prose", token_count=None,
                             length_chars=len(summary), document_length_chars=len(summary),
                             chunk_id=digest[:32], content_sha256=hashlib.sha256(summary.encode()).hexdigest(),
                             source_urls=[url], sources=[], source_pages=[], prev_chunk_id=None,
                             next_chunk_id=None, is_ocr=False, ocr_confidence=None, needs_review=False,
                             indexable=True, indexing_exclusions=[], quality_flags=[],
                             editorial_scope="navigation_only"))
    new_extra = [r for r in extra if r["chunk_id"] not in removed]
    write(out / "core.jsonl", new_core)
    write(out / "supplemental.jsonl", new_extra)
    write(out / "file_catalog.jsonl", catalog.values())
    write(out / "raw_table_review.jsonl", [*records(previous / "raw_table_review.jsonl"), *suppressed])
    report = dict(core=len(new_core), supplemental=len(new_extra), catalog=len(catalog),
                  reviewed_catalog_documents=Counter(row["category"] for row in candidates.values()),
                  reviewed_catalog_types=Counter(row["type"] for row in candidates.values()),
                  new_navigation_cards=len(new_core) - len(text_rows) + sum(
                      row["chunk_id"] in removed for row in text_rows),
                  newly_removed_table_chunks=len(suppressed))
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
