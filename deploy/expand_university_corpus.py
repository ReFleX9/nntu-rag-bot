"""Audit university category 07 and revisit selected campus documents."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

from review_corpus_v3 import records


def profile(row):
    path = unquote(urlsplit(row["source_urls"][0]).path).lower()
    title = unquote(row["title"]).lower()
    if path in {"", "/"}:
        return "home_navigation"
    if title == "osobennosy_priema_v_porydke_perevoda_v_ngtu_v_2023.pdf":
        return "archived_transfer_2023"
    if re.search(r"^\d+fz-|^ustav_11\.03\.16|^kriterii_aior",title):
        return "general_law_or_accreditation_original"
    if re.search(r"интеллектуальная электротехника|smart electrical engineering",title):
        return "publication"
    if re.search(r"/(?:politehnik|radio\+|nomera)/|/nauka/izdaniya/|/library/news/|/library/resurvsy/conf/", path):
        return "publication"
    if re.search(r"/dissertacii/|/dissert_sovety/|obyavleniya-o-zashhitah", path):
        return "dissertation"
    if re.search(r"/(?:rezultaty|soc_issled)/|rez_anket|otchet|отчет",path+" "+title):
        return "report"
    if re.search(r"/vacans|/vakans|/trudoustroistvo/vakansii/",path):
        return "vacancy_dated"
    if path in {"/sveden/employees", "/sveden/objects"}:
        return "directory_original"
    if path == "/sveden/education":
        return "education_links"
    if re.search(r"/nauka/|nauchno-tekhnicheskie-dostizheniya|/prioritet2030/",path):
        return "research_strategy_original"
    if re.search(r"/zhizn/|/uchen_sovet/resheniya/|/zgt/|eksportnyi_kontrol|/deloproizvodstvo/",path):
        return "dated_news_or_internal"
    if row["source_type"] == "html":
        if re.search(r"\b\d{2}\.\d{2}\.20\d{2}\s+\d{2}:\d{2}\b",row["text"][:200]):
            return "dated_news_or_internal"
        return "institutional_page"
    if re.search(r"/studgorodok/|/profkom_stud/|/kluby/|/dostupnost/|/norm_docs_ngtu/|/priem/|/licen_akkred/",path):
        return "student_document"
    if re.search(r"инструкц|положени|polo[zg]|ustav|устав",title):
        return "policy_candidate"
    return "manual_review"


def quality(row):
    text=row["text"]
    if not row.get("indexable") or row.get("indexing_exclusions"):
        return "not_indexable"
    if len(text)<100:
        return "short"
    if row.get("needs_review") or "\ufffd" in text:
        return "damaged"
    if row.get("is_ocr") and (row.get("ocr_confidence") or 0)<85:
        return "ocr_review"
    if row["kind"]=="table" or re.search(r"столбец\s+\d|\|",text,re.I):
        return "table_original"
    if re.search(r"лист регистрации изменений|лист ознакомления|\.{5,}|сведения о сертификате",text,re.I):
        return "administrative"
    return "usable"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("work/v2/chunks.jsonl"))
    parser.add_argument("--base",type=Path,default=Path("work/v3_candidate_r13"))
    parser.add_argument("--output",type=Path,default=Path("work/v3_candidate_r14"))
    args=parser.parse_args()
    base=list(records(args.base/"core.jsonl"))
    present={r["chunk_id"] for r in base}
    hashes={hashlib.sha256(r["text"].encode()).hexdigest() for r in base}
    docs=defaultdict(list)
    for row in records(args.source):
        if row["category"][:2] in {"04","07"}:
            docs[row["document_id"]].append(row)
    args.output.mkdir(parents=True,exist_ok=False)
    added=[]; decisions=[];inventory=[]; raw=[]
    profiles=Counter(); profiles_docs=Counter(); counts=Counter()
    catalog={r["document_id"]:r for r in records(args.base/"file_catalog.jsonl")}
    supplemental_docs={r["document_id"] for r in records(args.base/"supplemental.jsonl")}
    for key,rows in docs.items():
        first=rows[0];cat=first["category"][:2]
        if cat=="07":
            kind=profile(first); raw.extend(rows)
            if key in supplemental_docs:
                kind="already_in_supplemental"
            profiles[kind]+=len(rows);profiles_docs[kind]+=1
        else:
            name=unquote(first["title"]).lower()
            if not re.search(r"pologenie_studklub|polozhenie_patriot_club|polozhenie_volonters|положение.*(?:конкурс|кулибин|авиамодел)",name):
                continue
            if "донской" in first["text"].lower():
                continue
            kind="campus_revisit"
        n=0
        for row in rows:
            decision=quality(row)
            eligible=kind in {"institutional_page","education_links","student_document","policy_candidate","campus_revisit"}
            digest=hashlib.sha256(row["text"].encode()).hexdigest()
            if row["chunk_id"] in present:
                decision="already_in_core"
            elif not eligible:
                decision="outside_text_profile"
            elif decision=="usable" and digest in hashes:
                decision="duplicate_text"
            elif decision=="usable":
                row=dict(row,corpus_inclusion_reason="university_"+kind,
                         editorial_scope="source_excerpt")
                added.append(row);hashes.add(digest);n+=1;decision="added"
            counts[decision]+=1
            decisions.append(dict(chunk_id=row["chunk_id"],document_id=key,category=cat,
                                  title=unquote(row["title"]),page=row["page"],profile=kind,decision=decision))
        inventory.append(dict(document_id=key,category=cat,title=unquote(first["title"]),url=first["source_urls"][0],
                              raw_chunks=len(rows),added_chunks=n,profile=kind,
                              excerpt=first["text"][:600]))
        if kind in {"directory_original","student_document","policy_candidate","research_strategy_original","general_law_or_accreditation_original"}:
            for url in first["source_urls"]:
                host=urlsplit(url).hostname or ""
                if urlsplit(url).scheme!="https" or not (host=="nntu.ru" or host.endswith(".nntu.ru")):
                    continue
                if urlsplit(url).path in {"/sveden/employees","/sveden/objects"}:
                    # A searchable navigation card points to the official directory.
                    summary=first["title"]+". Официальный раздел НГТУ. Сведения о конкретной записи сверяйте по таблице на странице источника."
                    sha=hashlib.sha256(summary.encode()).hexdigest()
                    card=dict(first,text=summary,chunk_id=hashlib.sha256((url+summary).encode()).hexdigest()[:32],
                              content_sha256=sha,length_chars=len(summary),token_count=None,
                              kind="html",is_ocr=False,ocr_confidence=None,needs_review=False,
                              prev_chunk_id=None,next_chunk_id=None,page=None,
                              sources=[],source_pages=[],editorial_scope="navigation_only")
                    if sha not in hashes:
                        added.append(card);hashes.add(sha)
                elif re.search(r"\.(?:pdf|docx?|xlsx?|rtf)$",urlsplit(url).path,re.I):
                    catalog.setdefault(key,dict(document_id=key,title=unquote(first["title"]),url=url,type=kind,
                                                scope="core",groups=[],years=[],file_only=True))
    # Source gaps terminate the new excerpt chains.
    new={r["chunk_id"]:r for r in added}
    for r in added:
        for field,reverse in (("prev_chunk_id","next_chunk_id"),("next_chunk_id","prev_chunk_id")):
            target=new.get(r.get(field))
            if target is None or target["document_id"]!=r["document_id"] or target.get(reverse)!=r["chunk_id"]:
                r[field]=None
    def write(name,rows):
        with (args.output/name).open("w",encoding="utf8",newline="\n") as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    write("core.jsonl",base+added);write("added_university.jsonl",added)
    write("category07_raw.jsonl",raw);write("university_decisions.jsonl",decisions)
    write("university_inventory.jsonl",inventory);write("file_catalog.jsonl",catalog.values())
    write("supplemental.jsonl",records(args.base/"supplemental.jsonl"))
    write("raw_table_review.jsonl",records(args.base/"raw_table_review.jsonl"))
    report=dict(core_before=len(base),core_after=len(base)+len(added),profiles_chunks=profiles,
                profiles_documents=profiles_docs,decisions=counts,
                added_by_category=Counter(r["category"][:2] for r in added),catalog_entries=len(catalog))
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":
    main()
