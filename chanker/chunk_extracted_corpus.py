#!/usr/bin/env python3
"""Chunk extracted NNTU JSONL into source-linked RAG records."""
from __future__ import annotations
import argparse, hashlib, json, re, statistics, sys, unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"vendor/python"))
sys.path.insert(0,str(ROOT/"chanker"))
from text_cleaner import clean_text
SCHEMA="nntu-rag-chunk-v1"
LINK=re.compile(r"^\[Ссылка:\s*(.*?)\s+—\s+(.*?)\s*\]$")
SENTENCE=re.compile(r"(?<=[.!?…])\s+(?=[А-ЯЁA-Z0-9«])")

def rows(path):
    with path.open(encoding="utf-8-sig",errors="replace") as f:
        for n,line in enumerate(f,1):
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError as e: raise ValueError(f"{path}:{n}: {e}") from e

def clean(s):
    return clean_text(unicodedata.normalize("NFC",s),kind="unknown").text.replace("\u00ad","")

def budget(doc,b):
    c=str(doc.get("category") or b.get("category") or "")
    if b.get("source_type")=="html": return 1500
    if c.startswith(("01_","02_","08_")): return 1750
    if c.startswith("03_"): return 2100
    if c.startswith(("06_","07_")): return 2300
    if b.get("element") in {"sheet","table"}: return 1800
    return 2000

def split_long(text,limit):
    if len(text)<=limit:return [text]
    units=[x.strip() for x in SENTENCE.split(text) if x.strip()]
    if len(units)<2: units=[text]
    out=[]; cur=""
    for unit in units:
        while len(unit)>limit:
            cut=unit.rfind(" ",0,limit+1)
            if cut<limit//2:cut=limit
            out.append(unit[:cut].strip());unit=unit[cut:].lstrip()
        if cur and len(cur)+len(unit)+1>limit:out.append(cur);cur=unit
        else:cur=(cur+"\n"+unit).strip()
    if cur:out.append(cur)
    return out

def units_for(b):
    text=clean(b.get("text",""))
    if not text.strip():return []
    if b.get("source_type")=="html":
        result=[]; section=None; seen_links=set()
        for line in (x.strip() for x in text.splitlines()):
            if not line:continue
            if any(x in line for x in ("cookies,","использовании cookies","Вы можете отказаться от использования cookies")):continue
            m=LINK.match(line)
            if m:
                label,href=m.groups()
                if href.strip().lower().startswith("javascript:"):continue
                if href.strip() in {"", "#"}:continue
                href=urljoin("https://www.nntu.ru/",href.strip())
                key=(label.strip().casefold(),href.strip().casefold())
                if key in seen_links:continue
                seen_links.add(key)
                result.append((f"Ресурс: {label.strip()}\nСсылка: {href.strip()}","link",section));continue
            if line.startswith("https://") or line.startswith("http://"):continue
            if line.lower() in {"согласен","цвет сайта:","размер шрифта:","изображения:"}:continue
            heading=len(line)<=110 and (line.isupper() or line.endswith(":"))
            if heading:section=line.rstrip(":")
            result.append((line,"heading" if heading else "text",section))
        return result
    if b.get("element") in {"sheet","table"} and " | " in text:
        lines=[line.strip() for line in text.splitlines() if line.strip()]
        rows=[line.split(" | ") for line in lines]
        result=[]
        for row_no,cells in enumerate(rows,1):
            values=[(i,c.strip()) for i,c in enumerate(cells) if re.search(r"\w",c,re.UNICODE)]
            if not values:continue
            if not re.search(r"\w"," ".join(v for _,v in values),re.UNICODE):continue
            if len(values)==1 and row_no==1:
                result.append((values[0][1],"heading",None));continue
            labeled=[f"столбец {i+1}: {value}" for i,value in values]
            result.append((f"Строка {row_no}: "+" | ".join(labeled),"table_row",None))
        return result
    result=[]
    for para in re.split(r"\n\s*\n+",text):
        ls=[x.strip() for x in para.splitlines() if x.strip()]
        if not ls:continue
        if len(ls)>1 and all(" | " in x for x in ls):
            result.extend((x,"table_row",None) for x in ls if re.search(r"\w",x,re.UNICODE))
        elif len(ls)>1 and all(re.match(r"^(?:[-*•‣]|\d+[.)])\s",x) for x in ls):
            result.extend((x,"list_item",None) for x in ls)
        else:result.append((" ".join(ls),"ocr" if b.get("ocr") else "text",None))
    return result

def chunk_doc(doc_id,blocks,doc):
    grouped=[]; cur=[]; nchar=0; boundary=None; section=None
    document_length_chars=sum(len(b.get("text") or "") for b in blocks)
    def flush():
        nonlocal cur,nchar
        if cur:grouped.append(cur);cur=[];nchar=0
    for b in blocks:
        lim=budget(doc,b); page=b.get("page") or 1
        key=(b.get("source_member"),page,b.get("sheet"),b.get("slide"))
        for text,kind,heading in units_for(b):
            if kind=="heading":section=heading
            for part in split_long(text,lim):
                u={"text":part,"kind":kind,"block":b,"page":page,"key":key,"section":heading or section}
                if cur and (key!=boundary or nchar+len(part)+1>lim):flush()
                cur.append(u);nchar+=len(part)+1;boundary=key
    flush()
    out=[]; total=max([int(b.get("page") or 1) for b in blocks] or [1])
    title=doc.get("filename") or doc.get("title") or doc_id
    for i,group in enumerate(grouped):
        text="\n".join(u["text"] for u in group).strip()
        pages=sorted({int(u["page"]) for u in group})
        refs=[]
        for u in group:
            b=u["block"];r={"source_path":b.get("source_path"),"page":u["page"]}
            if b.get("source_member"):r["source_member"]=b["source_member"]
            if r not in refs:refs.append(r)
        conf=[u["block"].get("ocr_confidence") for u in group if isinstance(u["block"].get("ocr_confidence"),(int,float))]
        isocr=any(u["block"].get("ocr") for u in group); confidence=round(statistics.mean(conf),1) if conf else None
        flags=[]
        if confidence is not None and confidence<60:flags.append("low_ocr_confidence")
        if "\ufffd" in text:flags.append("replacement_character")
        kinds={u["kind"] for u in group}
        kind="link" if kinds=={"link"} else "html" if any(u["block"].get("source_type")=="html" for u in group) else "table" if "table_row" in kinds else "list" if "list_item" in kinds else "ocr_prose" if isocr else "prose"
        section=next((u["section"] for u in group if u.get("section")),None)
        cid=hashlib.sha256(json.dumps([SCHEMA,doc_id,i,refs,text],ensure_ascii=False).encode()).hexdigest()[:32]
        b=group[0]["block"]
        out.append({"schema_version":SCHEMA,"chunk_id":cid,"document_id":doc_id,"title":title,
          "page":pages[0],"page_count":total,"section_title":section,"section_path":[section] if section else [],
          "kind":kind,"text":text,"sources":refs,"token_count":None,"length_chars":len(text),
          "is_ocr":bool(isocr),"ocr_confidence":confidence,"quality_flags":flags,
          "document_status":doc.get("status","ok"),"indexable":True,"indexing_exclusions":[],
          "document_length_chars":document_length_chars,
          "needs_review":bool(flags),"chunk_index":i,"prev_chunk_id":None,"next_chunk_id":None,
          "citation_label":f"{title}, стр. {pages[0]}" if doc.get("extension")==".pdf" else title,
          "content_sha256":hashlib.sha256(text.encode()).hexdigest(),"category":doc.get("category") or b.get("category"),
          "extension":doc.get("extension"),"source_type":b.get("source_type"),
          "source_urls":doc.get("source_urls",b.get("source_urls",[])),
          "source_pages":doc.get("source_pages",b.get("source_pages",[])),
          **({"source_member":b["source_member"]} if b.get("source_member") else {})})
    for i,row in enumerate(out):
        row["prev_chunk_id"]=out[i-1]["chunk_id"] if i else None
        row["next_chunk_id"]=out[i+1]["chunk_id"] if i+1<len(out) else None
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("corpus",nargs="?",default="data/nntu_site");ap.add_argument("--output");ap.add_argument("--report")
    a=ap.parse_args();root=Path(a.corpus).resolve();ed=root/"extracted"
    dest=Path(a.output).resolve() if a.output else ed/"chunks.jsonl";reportpath=Path(a.report).resolve() if a.report else ed/"chunking_report.json"
    docs={r.get("doc_id"):r for r in rows(ed/"documents.jsonl")}
    classification=root/"classification.jsonl"
    cats=({r.get("stored_path"):r.get("category") for r in rows(classification)}
          if classification.is_file() else {})
    for d in docs.values():
        if d.get("source_path") in cats:d["category"]=cats[d["source_path"]]
    counts=Counter();lens=[];bycat=Counter();bykind=Counter();ocr=Counter();chunk_docs=set()
    dest.parent.mkdir(parents=True,exist_ok=True)
    temporary=dest.with_suffix(dest.suffix+".tmp")
    with temporary.open("w",encoding="utf-8",newline="\n") as f:
        active=None;group=[]
        def emit(docid,bs):
            if not docid:return
            for r in chunk_doc(docid,bs,docs.get(docid,{})):
                f.write(json.dumps(r,ensure_ascii=False)+"\n");counts["chunks"]+=1;lens.append(r["length_chars"])
                chunk_docs.add(docid)
                bycat[r.get("category") or "unknown"]+=1;bykind[r["kind"]]+=1
                if r["is_ocr"]:
                    ocr["chunks"]+=1
                    if r["ocr_confidence"] is not None and r["ocr_confidence"]<60:ocr["low_confidence_chunks"]+=1
                if r["needs_review"]:counts["needs_review"]+=1
        for b in rows(ed/"blocks.jsonl"):
            docid=b.get("doc_id")
            if active is not None and docid!=active:emit(active,group);group=[]
            active=docid;group.append(b);counts["input_blocks"]+=1
        emit(active,group)
    temporary.replace(dest)
    lens.sort()
    report={"schema_version":SCHEMA,"input_blocks":counts["input_blocks"],"documents_with_chunks":len(chunk_docs),
      "chunks":counts["chunks"],"needs_review":counts["needs_review"],"ocr":dict(ocr),"by_category":dict(bycat),
      "by_kind":dict(bykind),"length_chars":{"min":min(lens) if lens else 0,"median":lens[len(lens)//2] if lens else 0,
      "p95":lens[int((len(lens)-1)*.95)] if lens else 0,"max":max(lens) if lens else 0},
      "token_count":"null: measured later with the selected embedding tokenizer",
      "spelling_policy":"No guessed OCR spelling changes; conservative Unicode cleanup only; low-confidence OCR flagged.",
      "policy":{"HTML":"1500 chars; link cards atomic; remove cookie notice and javascript:void(0) controls",
      "admissions_students_administration":"1750 chars","educational_programs":"2100 chars",
      "science_university":"2300 chars","other":"2000 chars","overlap":"0; neighbor IDs preserve context",
      "tables":"keep extracted rows intact and group to the selected character budget"}}
    reportpath.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
