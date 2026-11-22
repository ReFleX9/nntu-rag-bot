"""Review unclassified category 99 and residual media category 00."""
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
from expand_research_corpus import abstract_candidate, DOI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_hybrid_langchain/rag_hybrid"))
from file_lookup import official_file


def classify(rows):
    first = rows[0]
    if first["category"].startswith("00_"):
        return "media_original"
    title = unquote(first["title"]).lower()
    path = unquote(urlsplit(first["source_urls"][0]).path).lower()
    opening = " ".join(row["text"] for row in rows[:2]).lower()
    if re.search(r"/sotrudnikam/|/zakupki/|/deloproizvodstvo/|/upr_person/|/profsouz_sotrudniki/", path):
        return "employee_or_procurement"
    if re.search(r"/samoobsledov/|otchet|отчет|/rezultaty/", path+" "+title):
        return "report"
    if re.search(r"/nomera/|/nauka/izdaniya/|/nauka/konf/", path) or (
        DOI.search(opening) and re.search(r"ключевые слова|abstract|аннотаци", opening)
    ):
        return "scientific_publication"
    if re.search(r"/library/|каталог|^\w\.doc$",path+" "+title):
        return "library_original"
    if re.search(r"/books/|пособи|posob|uchebn",path+" "+title):
        return "educational_original"
    if re.search(r"/dissertacii/|otzyv|отзыв|recenz",path+" "+title):
        return "dissertation_original"
    if re.search(r"квитанц|согласие|soglasi|blank|бланк|договор|dogovor",title):
        return "form_original"
    if re.search(r"/stazhirovki/|/vakansii/|/roty/",path):
        return "career_original_dated"
    if re.search(r"прейскурант|тариф|расписан|rasp|цены|путевок",opening[:500]):
        return "table_original_dated"
    if re.search(r"федеральный закон|российская федерация.{0,40}закон|^fz|фз|^\d+-fz",opening[:300]+" "+title):
        return "general_regulation_original"
    if re.search(r"памятка|инструкция|положение|порядок",opening[:1000]) and (
        re.search(r"студент|обучающ|абитуриент|поступающ|общежит|аспирант",opening)
        and re.search(r"/studentam/|/studgorodok/|/norm_docs_ngtu/|/dostupnost/",path)
    ):
        return "student_instruction"
    return "unresolved_original"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--output", type=Path, default=Path("work/v3_candidate_r23"))
    args = parser.parse_args()
    base_dir = args.work / "v3_candidate_r22"
    core = list(records(base_dir / "core.jsonl"))
    ids = {row["chunk_id"] for row in core}
    hashes = {hashlib.sha256(row["text"].encode()).hexdigest() for row in core}
    catalog = {row["document_id"]:row for row in records(base_dir / "file_catalog.jsonl")}
    docs = defaultdict(list)
    for row in records(args.work / "v2/chunks.jsonl"):
        if row["category"][:2] in {"00","99"}:
            docs[row["document_id"]].append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    added, inventory, decisions = [], [], []
    profiles, doc_profiles, counts = Counter(), Counter(), Counter()
    for key, rows in docs.items():
        first = rows[0]
        profile = classify(rows)
        profiles[profile] += len(rows)
        doc_profiles[profile] += 1
        selected = 0
        for row in rows:
            text, reason = None, "original_or_deferred"
            if row["chunk_id"] in ids:
                reason = "already_in_core"
            elif profile == "scientific_publication":
                text, reason = abstract_candidate(row)
            elif profile == "student_instruction":
                reason = quality(row)
                if reason == "usable":
                    text = row["text"]
            if text is not None:
                digest = hashlib.sha256(text.encode()).hexdigest()
                if digest in hashes:
                    reason = "duplicate_text"
                else:
                    chunk_id = row["chunk_id"] if text == row["text"] else hashlib.sha256(
                        (row["chunk_id"]+":reclassified:"+digest).encode()).hexdigest()[:32]
                    new = dict(row,text=text,chunk_id=chunk_id,content_sha256=digest,
                               length_chars=len(text),token_count=None,source_chunk_id=row["chunk_id"],
                               prev_chunk_id=None,next_chunk_id=None,
                               corpus_inclusion_reason="category99_"+profile,
                               editorial_scope="research_abstract" if profile=="scientific_publication" else "source_excerpt")
                    added.append(new)
                    hashes.add(digest)
                    selected += 1
                    reason = "added"
            counts[reason] += 1
            decisions.append(dict(chunk_id=row["chunk_id"],document_id=key,
                                  category=row["category"],profile=profile,decision=reason))
        url = next((url for url in first["source_urls"] if official_file(url)),None)
        if url and key not in catalog and profile not in {
            "employee_or_procurement","report","unresolved_original","media_original"
        }:
            catalog[key] = dict(document_id=key,title=unquote(first["title"]),url=url,
                                type="research_publication" if profile=="scientific_publication" else profile,
                                scope="core" if profile in {"scientific_publication","student_instruction","educational_original"} else "supplemental",
                                groups=[],years=[],program_codes=[],file_only=True,
                                edition_review_required=True)
        inventory.append(dict(document_id=key,category=first["category"],title=unquote(first["title"]),
                              urls=first["source_urls"],raw_chunks=len(rows),profile=profile,
                              added_chunks=selected,excerpt=first["text"][:500]))
    def write(name, rows):
        with (args.output/name).open("w",encoding="utf8",newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    write("core.jsonl",core+added)
    write("added_unclassified.jsonl",added)
    write("category99_raw.jsonl",(r for a in docs.values() for r in a if r["category"].startswith("99_")))
    write("category00_raw.jsonl",(r for a in docs.values() for r in a if r["category"].startswith("00_")))
    write("inventory.jsonl",inventory)
    write("decisions.jsonl",decisions)
    write("file_catalog.jsonl",catalog.values())
    for name in ("supplemental.jsonl","raw_table_review.jsonl"):
        write(name,records(base_dir/name))
    report = dict(core_before=len(core),core_after=len(core)+len(added),
                  source_chunks=sum(profiles.values()),source_documents=len(docs),
                  profiles_chunks=profiles,profiles_documents=doc_profiles,decisions=counts,
                  added_chunks=len(added),added_documents=len({r["document_id"] for r in added}),
                  catalog_entries=len(catalog))
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False))


if __name__ == "__main__":
    main()
