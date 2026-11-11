#!/usr/bin/env python3
"""Summarize extraction coverage, OCR quality, and failures."""
from __future__ import annotations
import argparse, json, statistics
from collections import Counter, defaultdict
from pathlib import Path

def rows(path):
    if not path.exists(): return
    with path.open(encoding="utf-8-sig",errors="replace") as f:
        for line in f:
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError: continue

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("corpus",nargs="?",default="data/nntu_site"); args=ap.parse_args()
    root=Path(args.corpus).resolve(); out=root/"extracted"
    docs={}
    for d in rows(out/"documents.jsonl"): docs[d.get("doc_id") or d.get("source_path")]=d
    status=Counter(d.get("status","unknown") for d in docs.values())
    extensions=Counter(d.get("extension") or Path(d.get("filename","")).suffix.casefold() or "(без расширения)" for d in docs.values())
    categories=Counter(d.get("category") or "99_неопределено_на_проверку" for d in docs.values())
    errors=Counter()
    error_rows=[]
    for e in rows(out/"errors.jsonl"):
        key=(e.get("extension") or Path(e.get("filename","")).suffix.casefold() or "unknown",e.get("error","unknown").split(":",1)[0])
        errors[key]+=1; error_rows.append(e)
    conf=[]; ocr_blocks=0; text_blocks=0; links=0; page_blocks=0; block_count=0
    for b in rows(out/"blocks.jsonl"):
        block_count+=1; text=b.get("text",""); text_blocks+=bool(text)
        if b.get("page"): page_blocks+=1
        if b.get("ocr"):
            ocr_blocks+=1
            if isinstance(b.get("ocr_confidence"),(int,float)): conf.append(float(b["ocr_confidence"]))
        if "[Ссылка:" in text: links+=1
    report={"unique_documents":len(docs),"statuses":dict(status),"extensions":dict(extensions),"categories":dict(categories),
            "blocks":block_count,"nonempty_text_blocks":text_blocks,"pdf_page_blocks":page_blocks,
            "ocr_blocks":ocr_blocks,"ocr_confidence_mean":round(statistics.mean(conf),1) if conf else None,
            "ocr_confidence_below_60":sum(x<60 for x in conf),"html_blocks_with_links":links,
            "errors":{f"{ext}: {typ}":n for (ext,typ),n in errors.items()},
            "ocr_pages":sum(int(d.get("ocr_pages",0)) for d in docs.values()),
            "text_chars":sum(int(d.get("text_chars",0)) for d in docs.values())}
    (out/"quality_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    md=["# Отчёт по извлечению корпуса НГТУ","",f"Уникальных файлов: {len(docs):,}; текстовых блоков: {block_count:,}; символов: {report['text_chars']:,}.",
        f"Статусы: успешно — {status['ok']:,}; пустой текст — {status['empty']:,}; ошибки — {status['error']:,}.",
        f"Страниц с OCR: {report['ocr_pages']:,}; OCR-блоков с оценкой уверенности: {len(conf):,}; средняя уверенность: {report['ocr_confidence_mean']}%; ниже 60%: {report['ocr_confidence_below_60']:,}.",
        f"HTML-блоков со ссылками: {links:,}.","","## По форматам","","| Расширение | Файлов |","|---|---:|"]
    for ext,n in sorted(extensions.items(),key=lambda x:-x[1]): md.append(f"| `{ext}` | {n:,} |")
    md += ["","## По категориям","","| Категория | Файлов |","|---|---:|"]
    for cat,n in sorted(categories.items(),key=lambda x:-x[1]): md.append(f"| `{cat}` | {n:,} |")
    md += ["","## Ошибки","","| Тип | Файлов |","|---|---:|"]
    for key,n in sorted(errors.items(),key=lambda x:-x[1]): md.append(f"| {key[0]} — {key[1]} | {n:,} |")
    (out/"quality_report.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
