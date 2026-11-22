"""Review category 08 leftovers and select research abstracts from category 06."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

from review_corpus_v3 import records
from expand_university_corpus import quality

ABSTRACT = re.compile(r"(?i)(?<!\w)(?:аннотаци[яи]|abstract|резюме)\s*[.:]?")
KEYWORDS = re.compile(r"(?i)(?:ключевые\s+слова|key\s*words)\s*:")
DOI = re.compile(r"(?i)\b10\.\d{4,9}/[-._;()/:a-z0-9]+")
STUDENT_TITLE = re.compile(
    r"metod_rekom_oform_prakt_rabot|kontrol_dvizhenie_otnosheniya_dpo|"
    r"polozh_stip_prez_rf_aspiranty|polog_strukt_i_poryad_rpd|"
    r"polozhenie_ob-ekspert-eko-ngtu|ob_organizacii|правила пользования", re.I
)


def scientific_profile(row):
    path = unquote(urlsplit(row["source_urls"][0]).path).lower()
    name = unquote(row["title"]).lower()
    if re.search(r"/dissertacii/|otzyv|отзыв|recenz|реценз", path+" "+name):
        return "dissertation_review"
    if re.search(r"sbornik|сборник|/konf/|/nomera/", path+" "+name):
        return "collection_or_journal"
    if re.search(r"perechen_recenz|перечень|spisok", name):
        return "directory_original"
    if re.search(r"/kandidat_ekzamen/|/passport_spec/|/programmamy/",path):
        return "postgraduate_curriculum"
    return "research_document"


def abstract_candidate(row):
    text = row["text"]
    if quality(row) != "usable" or row.get("is_ocr"):
        return None, "quality_or_layout"
    marker = KEYWORDS.search(text)
    heading = ABSTRACT.search(text)
    if not marker:
        return None, "no_keywords_marker"
    # Keywords alone also occur in references and author instructions.
    if not (heading or re.search(r"(?i)\b(?:doi|удк|udc)\b", text[:700])):
        return None, "no_abstract_identity"
    start = heading.start() if heading else 0
    if heading is None and re.search(r"\s//\s|список\s+литературы|\breferences\b", text[:marker.start()], re.I):
        return None, "mixed_with_references"
    # Include the keyword line and its hyphenated continuation, then stop.
    end = text.find("\n", marker.end())
    if end < 0:
        end = len(text)
    elif not text[marker.end():end].strip():
        following = text.find("\n", end + 1)
        end = following if following >= 0 else len(text)
    for _ in range(2):
        if end >= len(text) or not text[:end].rstrip().endswith("-"):
            break
        following = text.find("\n", end + 1)
        end = following if following >= 0 else len(text)
    excerpt = text[start:end].strip()
    if excerpt.endswith("-"):
        return None, "unfinished_keyword_block"
    if len(excerpt) < 180 or marker.start() - start < 100:
        return None, "too_little_abstract_text"
    if re.search(r"стать[яи].{0,40}(?:должна содержать|оформля|объем)|оформлени[ея] стат",excerpt,re.I):
        return None, "author_guidelines"
    return excerpt, "research_abstract"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--output", type=Path, default=Path("work/v3_candidate_r20"))
    args = parser.parse_args()
    base_dir = args.work/"v3_candidate_r19"
    core = list(records(base_dir/"core.jsonl"))
    ids = {r["chunk_id"] for r in core}
    hashes = {hashlib.sha256(r["text"].encode()).hexdigest() for r in core}
    catalog = {r["document_id"]:r for r in records(base_dir/"file_catalog.jsonl")}
    docs = defaultdict(list)
    for row in records(args.work/"v2/chunks.jsonl"):
        if row["category"][:2] in {"06","08"}:
            docs[row["document_id"]].append(row)
    args.output.mkdir(parents=True,exist_ok=False)
    added, decisions, inventory, raw06 = [], [], [], []
    counts = defaultdict(Counter)
    profiles = Counter()
    for key, rows in docs.items():
        first = rows[0]
        category = first["category"][:2]
        profile = scientific_profile(first) if category=="06" else "policy_revisit"
        if category=="06":
            profiles[profile] += len(rows)
            raw06.extend(rows)
        added_count = 0
        for row in rows:
            reason = "not_in_revisit_scope"
            excerpt = None
            if row["chunk_id"] in ids:
                reason = "already_in_core"
            elif category=="08":
                if STUDENT_TITLE.search(unquote(row["title"])) and quality(row)=="usable":
                    excerpt = row["text"]
                    reason = "educational_instruction"
            elif profile not in {"dissertation_review","directory_original","postgraduate_curriculum"}:
                excerpt, reason = abstract_candidate(row)
            if excerpt is not None:
                digest = hashlib.sha256(excerpt.encode()).hexdigest()
                if digest in hashes:
                    reason = "duplicate_text"
                else:
                    new_id = row["chunk_id"] if excerpt==row["text"] else hashlib.sha256(
                        (row["chunk_id"]+":abstract:"+digest).encode()).hexdigest()[:32]
                    copy = dict(row,text=excerpt,chunk_id=new_id,content_sha256=digest,
                                length_chars=len(excerpt),token_count=None,
                                prev_chunk_id=None,next_chunk_id=None,
                                corpus_inclusion_reason=reason)
                    if category=="06":
                        copy["source_chunk_id"]=row["chunk_id"]
                        copy["section_title"]="Аннотация и ключевые слова научной публикации"
                        copy["section_path"]=[copy["section_title"]]
                        copy["editorial_scope"]="research_abstract"
                        copy["doi_candidates"]=DOI.findall(row["text"])
                    added.append(copy)
                    hashes.add(digest)
                    reason="added_"+reason
                    added_count+=1
            counts[category][reason]+=1
            decisions.append(dict(chunk_id=row["chunk_id"],document_id=key,category=category,
                                  title=row["title"],page=row["page"],profile=profile,decision=reason))
        if category=="06":
            url=next((u for u in first["source_urls"]
                      if urlsplit(u).scheme=="https"
                      and ((urlsplit(u).hostname or "")=="nntu.ru" or (urlsplit(u).hostname or "").endswith(".nntu.ru"))
                      and re.search(r"\.(?:pdf|docx?|rtf)$",urlsplit(u).path,re.I)),None)
            if url and profile!="dissertation_review":
                catalog.setdefault(key,dict(document_id=key,title=unquote(first["title"]),url=url,
                                            type="research_publication",scope="core",groups=[],years=[],
                                            program_codes=[],file_only=True))
        inventory.append(dict(document_id=key,category=category,title=unquote(first["title"]),
                              urls=first["source_urls"],profile=profile,raw_chunks=len(rows),
                              added_chunks=added_count,excerpt=first["text"][:400]))
    def write(name,rows):
        with (args.output/name).open("w",encoding="utf8",newline="\n") as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    write("core.jsonl",core+added)
    write("added_research_policy.jsonl",added)
    write("category06_raw.jsonl",raw06)
    write("decisions.jsonl",decisions)
    write("inventory.jsonl",inventory)
    write("file_catalog.jsonl",catalog.values())
    write("supplemental.jsonl",records(base_dir/"supplemental.jsonl"))
    write("raw_table_review.jsonl",records(base_dir/"raw_table_review.jsonl"))
    report = dict(core_before=len(core),core_after=len(core)+len(added),
                  added_by_category=Counter(r["category"][:2] for r in added),
                  source_documents_by_category=Counter(r["category"] for r in inventory),
                  research_profiles=profiles,decisions=counts,catalog_entries=len(catalog))
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":
    main()
