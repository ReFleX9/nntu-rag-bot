"""Resolve category 07 leftovers and build a traceable category 08 candidate."""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

from review_corpus_v3 import records
from expand_university_corpus import quality
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_hybrid_langchain/rag_hybrid"))
from file_lookup import official_file


def classify07(row, all_rows):
    path = unquote(urlsplit(row["source_urls"][0]).path).lower()
    title = unquote(row["title"]).lower()
    if re.search(r"/covid/|/abiturientam/20(?:1\d|2[0-5])/|/profsouz_sotrudniki/|/pps_izbranie/", path):
        return "dated_or_employee_material"
    if re.search(r"/nok_obr/|bezopasnost_truda_rez|rez_anket", path + title):
        return "quality_monitoring"
    if re.search(r"rasp|расписани|stoimost|стоимость|ball_\d|календар|план|договор|согласие|лиценз", title):
        return "original_form_or_table"
    if re.search(r"/stazhirovki/|/roty/", path):
        return "career_original_dated"
    if re.search(r"/abiturientam/2026/|/abiturientu/2026/|/wp-content/uploads/2026/", path):
        return "admission_or_institute"
    if re.search(r"mr_akad_otpusk|udal-registr|правила", title):
        return "student_instruction"
    if re.search(r"/library/|/history/|galeraya_pamyati",path):
        return "library_history_original"
    if re.search(r"/security/|/norm_docs/",path):
        return "specialized_policy_original"
    if re.search(r"^\d\.\d\.\d|автор|avtor|реквизит|egrul",title):
        return "specialized_original"
    return "unresolved_original"


def classify08(row, all_rows):
    path = unquote(urlsplit(row["source_urls"][0]).path).lower()
    title = unquote(row["title"]).lower()
    opening = " ".join(r["text"] for r in all_rows[:2]).lower()
    if re.search(r"/sotrudnikam/|/zakupki/|/upr_person/|/kadr-rezerv/|/deloproizvodstvo/|/profsouz_sotrudniki/",path) or re.search(r"trud_rasporyadka|трудового распорядка",title):
        return "procurement_or_employee"
    if "skzi" in title or "скзи" in title:
        return "specialized_security_original"
    if re.search(r"/rezultaty/|/nok_obr/|/samoobsledov/|otchet|отчет", path+" "+title):
        return "monitoring_report"
    if re.search(r"/covid/|/abiturientam/20(?:1\d|2[0-5])/|v_2023",path):
        return "archived_period"
    if re.search(r"/nomera/|/politehnik/|/radio\+/",path):
        return "publication"
    if re.search(r"согласие|soglasi|blank|бланк|dogovor|договор|макет|maket|shablon",title):
        return "form_original"
    if re.search(r"fgos|фгос|^\d{2}\.\d{2}\.\d{2}",title):
        return "educational_standard"
    if (re.search(r"^polo[zg]|^полож|^pravila|^правила|^politika",title)
            and "нгту" in opening
            and re.search(r"обучающ|студент|аспирант|практик|аттестаци|библиотек|инвалид|общежит|персональных данных",opening)):
        return "student_policy"
    if re.search(r"федеральный закон|федерального закона|постановление правительства российской федерации|"
                 r"приказ (?:министерства|минобрнауки|рособрнадзора)|государственный стандарт|национальный стандарт|"
                 r"(?:национальн|государственн).{0,35}стандарт|гост\s+р\s+",opening[:1000]):
        return "general_regulation"
    if re.search(r"^fz|[-_]fz|фз|^gost|гост|^fkz|^\d+\.(?:doc|rtf)$",title):
        return "general_regulation"
    if re.search(r"/security/|/eksportnyi_kontrol/",path):
        if re.search(r"инструкц|instr|пропуск|propusk",title):
            return "local_instruction"
        return "specialized_security_original"
    if re.search(r"/norm_docs_ngtu/|/norm_docs/|/norm_doc_i_loc_akty/|/studgorodok/|/profkom_stud/|/studentam/|/dlya_inostrannykh/",path):
        return "student_policy"
    if re.search(r"/metod_docs/|/gos-attestac/|/trebovaniya_zashita/|/docs/",path) and re.search(r"полож|polo[zg]|por[jy]|поряд|prav|прав|instr|инстр",title):
        return "educational_procedure"
    if re.search(r"лиценз|licen|аккред|akkred|устав|ustav",title):
        return "university_status_original"
    return "manual_review_original"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--output", type=Path, default=Path("work/v3_candidate_r16"))
    args = parser.parse_args()
    base_dir = args.work / "v3_candidate_r15"
    core = list(records(base_dir / "core.jsonl"))
    core_ids = {r["chunk_id"] for r in core}
    hashes = {hashlib.sha256(r["text"].encode()).hexdigest() for r in core}
    leftovers = {r["document_id"] for r in records(base_dir / "university_inventory.jsonl")
                 if r["profile"] == "manual_review" and r["category"] == "07"}
    supplemental = list(records(base_dir / "supplemental.jsonl"))
    supplemental_docs = {r["document_id"] for r in supplemental}
    catalog = {r["document_id"]: r for r in records(base_dir / "file_catalog.jsonl")}
    docs = defaultdict(list)
    for r in records(args.work / "v2/chunks.jsonl"):
        if r["category"].startswith("08_") or r["document_id"] in leftovers and r["category"].startswith("07_"):
            docs[r["document_id"]].append(r)
    added, inventory, decisions = [], [], []
    profiles = defaultdict(Counter)
    outcome = defaultdict(Counter)
    examples = defaultdict(list)
    args.output.mkdir(parents=True, exist_ok=False)
    for key, rows in docs.items():
        first = rows[0]
        category = first["category"][:2]
        kind = classify07(first,rows) if category == "07" else classify08(first,rows)
        if key in supplemental_docs:
            kind = "already_in_supplemental"
        profiles[category][kind] += len(rows)
        count = 0
        for row in rows:
            why = quality(row)
            eligible = kind in {"admission_or_institute", "student_instruction",
                                "student_policy", "local_instruction", "educational_procedure"}
            digest = hashlib.sha256(row["text"].encode()).hexdigest()
            if row["chunk_id"] in core_ids:
                why = "already_in_core"
            elif not eligible:
                why = "original_or_deferred"
            elif why == "usable" and digest in hashes:
                why = "duplicate_text"
            elif why == "usable":
                why = "added"
                copy = dict(row, corpus_inclusion_reason=category+"_"+kind, editorial_scope="source_excerpt")
                added.append(copy)
                hashes.add(digest)
                count += 1
                if len(examples[category+"_"+kind]) < 3:
                    examples[category+"_"+kind].append(dict(title=row["title"],page=row["page"],text=row["text"]))
            outcome[category][why] += 1
            decisions.append(dict(document_id=key,chunk_id=row["chunk_id"],category=category,
                                  title=row["title"],profile=kind,decision=why))
        url = next((u for u in first["source_urls"] if official_file(u)),None)
        if url and key not in catalog and kind not in {
            "procurement_or_employee","dated_or_employee_material","quality_monitoring",
            "monitoring_report","publication","already_in_supplemental"
        }:
            catalog[key] = dict(document_id=key,title=unquote(first["title"]),url=url,
                                type=kind,scope="core" if kind in {
                                    "student_policy","local_instruction","educational_procedure",
                                    "admission_or_institute","student_instruction","form_original"
                                } else "supplemental",groups=[],years=[],file_only=True,
                                edition_review_required=True)
        inventory.append(dict(document_id=key,title=unquote(first["title"]),category=category,
                              urls=first["source_urls"],profile=kind,raw_chunks=len(rows),
                              added_chunks=count,excerpt=first["text"][:600]))
    new = {r["chunk_id"]:r for r in added}
    for row in added:
        for field,reverse in (("prev_chunk_id","next_chunk_id"),("next_chunk_id","prev_chunk_id")):
            target = new.get(row.get(field))
            if target is None or target["document_id"] != row["document_id"] or target.get(reverse) != row["chunk_id"]:
                row[field] = None
    def write(name,rows):
        with (args.output/name).open("w",encoding="utf8",newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    write("core.jsonl",core+added)
    write("added_policy.jsonl",added)
    write("category08_raw.jsonl",(r for group in docs.values() for r in group if r["category"].startswith("08_")))
    write("inventory.jsonl",inventory)
    write("decisions.jsonl",decisions)
    write("file_catalog.jsonl",catalog.values())
    write("supplemental.jsonl",supplemental)
    write("raw_table_review.jsonl",records(base_dir/"raw_table_review.jsonl"))
    report = dict(core_before=len(core),core_after=len(core)+len(added),
                  added_by_category=Counter(r["category"][:2] for r in added),
                  documents_by_category=Counter(row["category"] for row in inventory),
                  profiles_chunks=profiles,decisions=outcome,catalog_entries=len(catalog),examples=examples)
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:v for k,v in report.items() if k!="examples"},ensure_ascii=False))


if __name__ == "__main__":
    main()
