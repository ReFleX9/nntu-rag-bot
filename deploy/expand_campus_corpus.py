"""Compare campus-life selection profiles and build r11 from r10."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

from review_corpus_v3 import records


def classify(rows):
    row = rows[0]
    path = unquote(urlsplit(row["source_urls"][0]).path).casefold()
    name = unquote(row["title"]).casefold()
    if re.search(r"/(?:politehnik|radio\+|nomera)/|/nauka/|/resurvsy/conf/|/biblconf/", path):
        return "publication_archive"
    if re.search(r"/rezultaty/|/nok_obr/|/soc_issled/|/monitoring", path):
        return "survey_report"
    if re.search(r"rasp|kurs_vesna|plan_vosp|calendarn_plan|/plan_year/|plan_kultmass|zhurnal_kuratora", name + " " + path):
        return "plan_or_schedule_original"
    if re.search(r"/dostupnost/|/studgorodok/|/kluby/|/stud_sovet/", path):
        return "student_service"
    if "putevoditel_pervokursnika" in name:
        return "student_service"
    if "mr_profilaktika_v_narkologii" in name:
        return "medical_reference_original"
    if name in {"posob10.doc", "posob12.doc", "posob16.doc",
                "luchshie_prektiki_meropr_antinarc_napr_v_stud_srede.pdf"}:
        return "specialized_manual_original"
    if re.search(r"polozhenie_volonters", name):
        return "student_service"
    if name == "molodezhnaya_politika.pdf":
        return "plan_or_schedule_original"
    if re.search(r"/security/(?:gochs/(?:posobiya|attendance)|antiterror/pamyatki|antinark)/", path):
        return "safety_reference"
    if re.search(r"/library/(?:news|meropr)/|/resurvsy/", path):
        return "library_original"
    if re.search(r"/fgos|/profstandarty/|/passport_spec/|/education/", path):
        return "education_other_category"
    return "manual_review"


def text_decision(row):
    text = row["text"]
    if not row.get("indexable") or row.get("indexing_exclusions"):
        return "not_indexable"
    if row.get("needs_review") or "\ufffd" in text:
        return "damaged_text"
    if row["is_ocr"] and (row.get("ocr_confidence") or 0) < 85:
        return "ocr_review"
    if row["kind"] == "table" or re.search(r"столбец\s+\d|\bтаблица\s+\d|\|", text, re.I):
        return "table_original"
    if len(text) < 160:
        return "short_fragment"
    if re.search(r"лист регистрации изменений|лист ознакомления|\.{5,}|проект вносит|согласовано:", text, re.I):
        return "administrative_or_contents"
    return "usable_text"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("work/v2/chunks.jsonl"))
    parser.add_argument("--base", type=Path, default=Path("work/v3_candidate_r10"))
    parser.add_argument("--output", type=Path, default=Path("work/v3_candidate_r11"))
    args = parser.parse_args()
    core = list(records(args.base / "core.jsonl"))
    present = {r["chunk_id"] for r in core}
    hashes = {hashlib.sha256(r["text"].encode()).hexdigest() for r in core}
    docs = defaultdict(list)
    for row in records(args.source):
        if row["category"].startswith("04_"):
            docs[row["document_id"]].append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    def write(name, rows):
        with (args.output / name).open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    added, decisions, inventory = [], [], []
    doc_counts, chunk_counts, rejects = Counter(), Counter(), Counter()
    catalog = {r["document_id"]: r for r in records(args.base / "file_catalog.jsonl")}
    for doc_id, rows in docs.items():
        kind = classify(rows)
        first = rows[0]
        doc_counts[kind] += 1
        chunk_counts[kind] += len(rows)
        new_count = 0
        for row in rows:
            why = text_decision(row)
            eligible = kind in {"student_service", "safety_reference"} and why == "usable_text"
            digest = hashlib.sha256(row["text"].encode()).hexdigest()
            decision = "outside_profile"
            if row["chunk_id"] in present:
                decision = "already_in_core"
            elif eligible and digest in hashes:
                decision = "duplicate_text"
            elif eligible:
                copy = dict(row, corpus_inclusion_reason="campus_" + kind,
                            editorial_scope="source_excerpt")
                added.append(copy)
                hashes.add(digest)
                new_count += 1
                decision = "added"
            elif kind in {"student_service", "safety_reference"}:
                decision = why
            rejects[decision] += 1
            decisions.append(dict(chunk_id=row["chunk_id"], document_id=doc_id,
                                  title=row["title"], page=row["page"], profile=kind, decision=decision))
        if kind in {"student_service", "safety_reference", "library_original", "plan_or_schedule_original", "medical_reference_original", "specialized_manual_original"}:
            url = next((u for u in first["source_urls"] if urlsplit(u).scheme == "https"
                        and (urlsplit(u).hostname == "nntu.ru" or (urlsplit(u).hostname or "").endswith(".nntu.ru"))
                        and re.search(r"\.(?:pdf|docx?|xlsx?|rtf)$",urlsplit(u).path,re.I)), None)
            if url and doc_id not in catalog:
                catalog[doc_id] = dict(document_id=doc_id, title=unquote(first["title"]), url=url,
                                      type=kind, scope="core", groups=[], years=[], program_codes=[],
                                      file_only=kind in {"library_original","plan_or_schedule_original"},
                                      period_review_required=True)
        inventory.append(dict(document_id=doc_id,title=unquote(first["title"]),
                              urls=first["source_urls"],profile=kind,raw_chunks=len(rows),
                              added_chunks=new_count,example=first["text"][:500]))
    # Keep source gaps explicit and preserve chains already present in r10.
    new_ids = {r["chunk_id"]:r for r in added}
    for row in added:
        for key,reverse in (("prev_chunk_id","next_chunk_id"),("next_chunk_id","prev_chunk_id")):
            target = new_ids.get(row.get(key))
            if target is None or target["document_id"] != row["document_id"] or target.get(reverse) != row["chunk_id"]:
                row[key] = None
    # Source-side links were updated above. Reconcile endpoints once more.
    for row in added:
        for key,reverse in (("prev_chunk_id","next_chunk_id"),("next_chunk_id","prev_chunk_id")):
            target = new_ids.get(row.get(key))
            if target is not None and target.get(reverse) != row["chunk_id"]:
                row[key] = None
    write("core.jsonl", core + added)
    write("added_category04.jsonl", added)
    write("category04_raw.jsonl", (r for rows in docs.values() for r in rows))
    write("category04_decisions.jsonl", decisions)
    write("category04_inventory.jsonl", inventory)
    write("file_catalog.jsonl", catalog.values())
    added_ids = set(new_ids)
    extra = [r for r in records(args.base/"supplemental.jsonl") if r["chunk_id"] not in added_ids]
    extra_ids = {r["chunk_id"] for r in extra}
    for row in extra:
        for key in ("prev_chunk_id","next_chunk_id"):
            if row.get(key) not in extra_ids:
                row[key] = None
    write("supplemental.jsonl",extra)
    write("raw_table_review.jsonl",records(args.base/"raw_table_review.jsonl"))
    report = dict(base=str(args.base), core_before=len(core), core_after=len(core)+len(added),
                  category_raw_chunks=sum(chunk_counts.values()), category_raw_documents=len(docs),
                  document_profiles=doc_counts, chunk_profiles=chunk_counts,
                  decisions=rejects,added_chunks=len(added),
                  added_documents=len({r["document_id"] for r in added}),
                  added_by_profile=Counter(r["corpus_inclusion_reason"] for r in added),
                  catalog_entries=len(catalog))
    report["profile_comparison"] = {
        "student_services": sum(r["corpus_inclusion_reason"] == "campus_student_service" for r in added),
        "services_and_references": len(added),
        "catalog_additions": len(catalog) - sum(1 for _ in records(args.base / "file_catalog.jsonl")),
    }
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False))


if __name__ == "__main__":
    main()
