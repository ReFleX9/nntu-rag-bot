#!/usr/bin/env python3
"""Remove repeated extraction rows produced by duplicate URLs/resume retries."""
from __future__ import annotations
import argparse, json, os, tempfile
from collections import defaultdict
from pathlib import Path

def read(path):
    with path.open(encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError: continue

def write_atomic(path, rows):
    temp = path.with_name(path.name + ".dedupe.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("corpus",nargs="?",default="data/nntu_site")
    args=ap.parse_args(); out=Path(args.corpus).resolve()/"extracted"
    if out.name != "extracted" or out.parent != Path(args.corpus).resolve(): raise RuntimeError("Unexpected extraction directory")
    root=out.parent
    source_rows=defaultdict(list)
    for row in read(root/"files.jsonl"):
        source_rows[row.get("stored_path")].append(row)
    categories={x.get("stored_path"):x for x in read(root/"classification.jsonl")}
    counts={}
    for name in ("documents.jsonl","errors.jsonl"):
        path=out/name; by_id={}
        for row in read(path):
            source=source_rows.get(row.get("source_path"),[])
            if source:
                first=source[0]
                row.setdefault("sha256",first.get("sha256")); row.setdefault("size_bytes",first.get("size_bytes"))
                row.setdefault("source_urls",sorted({x.get("url") for x in source if x.get("url")}))
                row.setdefault("source_pages",sorted({x.get("source_page") for x in source if x.get("source_page")}))
                row.setdefault("category",categories.get(row.get("source_path"),{}).get("category"))
                row.setdefault("extension",Path(first.get("stored_path","")).suffix.casefold())
            by_id[row.get("doc_id") or row.get("source_path")]=row
        rows=list(by_id.values()); counts[name]={"kept":len(rows)}; write_atomic(path,rows)
    # Error entries written before a later successful retry are stale. Keep an
    # error only when the latest document record for that ID is still failed.
    document_rows = {r.get("doc_id"): r for r in read(out/"documents.jsonl")}
    document_status = {k: r.get("status") for k, r in document_rows.items()}
    error_path = out/"errors.jsonl"
    error_rows = {r.get("doc_id"): r for r in read(error_path)
                  if document_status.get(r.get("doc_id")) not in {"ok", "empty"}}
    stale_errors = counts["errors.jsonl"]["kept"] - len(error_rows)
    missing_errors = sum(1 for doc_id, doc in document_rows.items()
                         if doc.get("status") == "error" and doc_id not in error_rows)
    for doc_id, doc in document_rows.items():
        if doc.get("status") == "error" and doc_id not in error_rows:
            error_rows[doc_id] = {k: doc.get(k) for k in
                                  ("doc_id", "source_path", "filename", "error", "sha256",
                                   "size_bytes", "source_urls", "source_pages", "category", "extension")}
    errors = list(error_rows.values())
    write_atomic(error_path, errors)
    counts["errors.jsonl"] = {"kept":len(errors), "stale_retries_removed":stale_errors,
                              "missing_errors_recovered_from_documents":missing_errors}
    path=out/"blocks.jsonl"; temp=path.with_name(path.name+".dedupe.tmp"); seen=set(); kept=0; removed=0
    with temp.open("w",encoding="utf-8",newline="\n") as f:
        for row in read(path):
            # Documents unpacked from one ZIP share the archive's doc_id. Keep
            # their member path in the identity so page 1 of two PDFs (or
            # paragraph 1 of two DOCX files) does not collapse into one row.
            identity=[row.get("doc_id"),row.get("source_member"),row.get("page"),row.get("element"),row.get("ordinal"),row.get("table"),row.get("sheet"),row.get("slide")]
            key=json.dumps(identity,ensure_ascii=False,separators=(",",":"))
            if key in seen: removed+=1; continue
            seen.add(key); kept+=1; f.write(json.dumps(row,ensure_ascii=False)+"\n")
    os.replace(temp,path); counts["blocks.jsonl"]={"kept":kept,"removed":removed}
    print(json.dumps(counts,ensure_ascii=False,indent=2))
    return 0
if __name__=="__main__": raise SystemExit(main())
