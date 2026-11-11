#!/usr/bin/env python3
"""Extract downloaded NNTU files and crawled HTML into page/block JSONL.

PDF OCR is selective: native text is extracted first and only pages with very
little text (or scanned-image coverage) are rendered and passed to Tesseract.
Original downloads are never modified. Use --resume to continue an interrupted
run, and --limit for a pilot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import subprocess
import tempfile
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "python"))
SUPPORTED_EXT = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".xlsm", ".ppt", ".pptx",
                 ".rtf", ".txt", ".csv", ".tsv", ".xml", ".html", ".htm", ".json",
                 ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".zip"}
OCR_MAX_PIXELS = 18_000_000


def effective_extension(path: Path, row: dict) -> str:
    ext = path.suffix.casefold()
    if ext in SUPPORTED_EXT: return ext
    mime = str(row.get("content_type") or "").split(";",1)[0].strip().casefold()
    mapping = {"application/pdf":".pdf", "application/msword":".doc",
               "application/vnd.openxmlformats-officedocument.wordprocessingml.document":".docx",
               "application/vnd.ms-excel":".xls", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":".xlsx",
               "application/vnd.ms-powerpoint":".ppt", "application/vnd.openxmlformats-officedocument.presentationml.presentation":".pptx",
               "application/rtf":".rtf", "text/rtf":".rtf", "text/html":".html", "application/xhtml+xml":".html",
               "text/xml":".xml", "application/xml":".xml", "text/csv":".csv", "text/plain":".txt",
               "image/jpeg":".jpg", "image/png":".png", "image/tiff":".tif", "image/webp":".webp"}
    if mime in mapping: return mapping[mime]
    try:
        with path.open("rb") as stream: head = stream.read(512)
        if head.startswith(b"%PDF-"): return ".pdf"
        if head.startswith(b"{\\rtf"): return ".rtf"
        if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
            return ".doc" if "word" in mime else ".xls" if "excel" in mime else ".ppt"
        if head.startswith(b"\x89PNG\r\n\x1a\n"): return ".png"
        if head.startswith(b"\xff\xd8\xff"): return ".jpg"
        if head.lstrip().startswith((b"<?xml", b"<urlset", b"<sitemapindex")): return ".xml"
        if head.lstrip().lower().startswith((b"<!doctype html", b"<html")): return ".html"
        if head.startswith(b"PK\x03\x04"):
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
            if "word/document.xml" in names: return ".docx"
            if "xl/workbook.xml" in names: return ".xlsx"
            if "ppt/presentation.xml" in names: return ".pptx"
            return ".zip"
    except Exception:
        pass
    return ext


def jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def clean(s: Any) -> str:
    s = str(s or "").replace("\x00", " ").replace("\r", "\n")
    return re.sub(r"[ \t\f\v]+", " ", re.sub(r"\n{3,}", "\n\n", s)).strip()


def init_ocr():
    import pytesseract
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    tess = ROOT / "vendor" / "tesseract" / "tesseract.exe"
    data = ROOT / "vendor" / "tesseract" / "tessdata"
    if tess.exists():
        pytesseract.pytesseract.tesseract_cmd = str(tess)
        os.environ["TESSDATA_PREFIX"] = str(data)
    return pytesseract


def ocr_png(payload: bytes, pytesseract):
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(payload))
    data = pytesseract.image_to_data(img, lang="rus+eng", config="--psm 3",
                                     output_type=pytesseract.Output.DICT)
    words, confidences = [], []
    for word, score in zip(data.get("text", []), data.get("conf", [])):
        word = clean(word)
        try: score = float(score)
        except (ValueError, TypeError): score = -1
        if word:
            words.append(word)
            if score >= 0: confidences.append(score)
    return clean(" ".join(words)), round(sum(confidences) / len(confidences), 1) if confidences else None


def extract_pdf(path: Path, doc_id: str, meta: dict, pytesseract, dpi: int, workers: int):
    import pymupdf
    blocks, ocr_pages, pages = [], 0, 0
    with pymupdf.open(path) as pdf:
        pages = len(pdf)
        page_data, ocr_results, batch = [], {}, []
        pool = ThreadPoolExecutor(max_workers=workers) if pytesseract else None
        def run_batch(items):
            if not items or pool is None: return
            print(f"    OCR {path.name}: pages " + ",".join(str(x[0]) for x in items), flush=True)
            futures = [pool.submit(ocr_png, item[2], pytesseract) for item in items]
            for item, future in zip(items, futures):
                try: ocr_results[item[0]] = (*future.result(), item[3])
                except Exception as e:
                    meta.setdefault("warnings", []).append(f"page {item[0]} OCR: {type(e).__name__}: {e}")
                    ocr_results[item[0]] = (None, None, item[3])
        for i, page in enumerate(pdf):
            native = clean(page.get_text("text", sort=True))
            image_area = 0.0
            try:
                rect = page.rect
                total_area = max(1.0, rect.width * rect.height)
                image_area = min(1.0, sum(r.width * r.height for im in page.get_images(full=True)
                                          for r in page.get_image_rects(im[0])) / total_area)
            except Exception:
                pass
            # Low text alone also describes blank covers and title pages. OCR
            # only when sparse native text coincides with a substantial page image.
            candidate = (len(native) < 80 and image_area > 0.08) or (len(native) < 250 and image_area > 0.35)
            page_data.append({"number": i+1, "native": native})
            if candidate and pytesseract:
                try:
                    scale = dpi / 72
                    nominal_pixels = page.rect.width * scale * page.rect.height * scale
                    effective_dpi = dpi
                    if nominal_pixels > OCR_MAX_PIXELS:
                        scale *= (OCR_MAX_PIXELS / nominal_pixels) ** 0.5
                        effective_dpi = max(30, round(scale * 72))
                        meta.setdefault("warnings", []).append(
                            f"page {i+1}: OCR resolution reduced from {dpi} to {effective_dpi} dpi to cap image at {OCR_MAX_PIXELS:,} pixels")
                    png = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")
                    batch.append((i+1, native, png, effective_dpi))
                    if len(batch) >= workers:
                        run_batch(batch)
                        batch = []
                except Exception as e:
                    meta.setdefault("warnings", []).append(f"page {i+1} render: {type(e).__name__}: {e}")
        run_batch(batch)
        if pool: pool.shutdown(wait=True)
        for p in page_data:
            native = p["native"]
            text, method, conf = native, "pymupdf", None
            if p["number"] in ocr_results:
                ocr, conf, effective_dpi = ocr_results[p["number"]]
                if ocr and (len(ocr) > len(native) * 1.15 or not native):
                    text, method = ocr, "tesseract"
                    ocr_pages += 1
                elif ocr and len(ocr) > 100:
                    text, method = native + "\n\n[OCR supplement]\n" + ocr, "pymupdf+tesseract"
                    ocr_pages += 1
                elif not ocr:
                    method = "pymupdf;ocr_empty"
            if text:
                blocks.append(block(doc_id, meta, text, page=p["number"], method=method,
                                    ocr=method != "pymupdf", ocr_confidence=conf,
                                    ocr_dpi=effective_dpi if p["number"] in ocr_results else None))
    return blocks, pages, ocr_pages


def block(doc_id, meta, text, **extra):
    return {"doc_id": doc_id, "text": text, "source_type": meta.get("source_type"),
            "source_path": meta.get("source_path"), "source_urls": meta.get("source_urls", []),
            "source_pages": meta.get("source_pages", []), "source_member": meta.get("source_member"),
            "title": meta.get("title"),
            "category": meta.get("category", "99_неопределено_на_проверку"), **extra}


def extract_office(path: Path, doc_id: str, meta: dict, pytesseract=None, dpi: int = 150, workers: int = 6):
    ext = path.suffix.casefold()
    out = []
    if ext == ".docx":
        from docx import Document
        d = Document(path)
        for n, p in enumerate(d.paragraphs):
            t = clean(p.text)
            if t: out.append(block(doc_id, meta, t, element="paragraph", ordinal=n, style=p.style.name if p.style else None))
        for ti, table in enumerate(d.tables):
            rows = [" | ".join(clean(c.text) for c in row.cells) for row in table.rows]
            t = clean("\n".join(rows))
            if t: out.append(block(doc_id, meta, t, element="table", table=ti+1))
    elif ext in {".xlsx", ".xlsm"}:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for ws in wb.worksheets:
            lines = []
            for row in ws.iter_rows(values_only=True):
                vals = [clean(v) for v in row]
                if any(vals): lines.append(" | ".join(vals))
            if lines: out.append(block(doc_id, meta, "\n".join(lines), element="sheet", sheet=ws.title))
        wb.close()
    elif ext == ".xls":
        import xlrd
        wb = xlrd.open_workbook(path, on_demand=True)
        for ws in wb.sheets():
            lines = [" | ".join(clean(ws.cell_value(r,c)) for c in range(ws.ncols))
                     for r in range(ws.nrows)]
            text = clean("\n".join(lines))
            if text: out.append(block(doc_id, meta, text, element="sheet", sheet=ws.name))
        wb.release_resources()
    elif ext == ".pptx":
        from pptx import Presentation
        prs = Presentation(path)
        for n, slide in enumerate(prs.slides, 1):
            texts = [clean(sh.text) for sh in slide.shapes if getattr(sh, "has_text_frame", False) and clean(sh.text)]
            for sh in slide.shapes:
                if getattr(sh, "has_table", False):
                    texts += [" | ".join(clean(c.text) for c in row.cells) for row in sh.table.rows]
            if texts: out.append(block(doc_id, meta, "\n".join(texts), element="slide", slide=n))
    elif ext == ".rtf":
        from striprtf.striprtf import rtf_to_text
        text = clean(rtf_to_text(path.read_text(encoding="utf-8", errors="replace")))
        if text: out.append(block(doc_id, meta, text, element="document"))
    elif ext in {".txt", ".csv", ".tsv", ".xml", ".html", ".htm", ".json"}:
        text = clean(path.read_text(encoding="utf-8-sig", errors="replace"))
        if ext in {".html", ".htm"}:
            from bs4 import BeautifulSoup
            text = clean(BeautifulSoup(text, "html.parser").get_text("\n", strip=True))
        if text: out.append(block(doc_id, meta, text, element="document"))
    elif ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}:
        # Images are a small minority of the crawl; OCR only document-like
        # images and skip tiny logos/icons to avoid wasting time.
        from PIL import Image
        with Image.open(path) as image:
            if image.width >= 500 and image.height >= 300 and pytesseract:
                image = image.convert("RGB")
                pixels = image.width * image.height
                effective_dpi = 240
                if pixels > OCR_MAX_PIXELS:
                    scale = (OCR_MAX_PIXELS / pixels) ** 0.5
                    image = image.resize((round(image.width*scale), round(image.height*scale)), Image.Resampling.LANCZOS)
                    effective_dpi = max(30, round(240*scale))
                data = pytesseract.image_to_data(image, lang="rus+eng", config="--psm 3",
                                                 output_type=pytesseract.Output.DICT)
                words = [clean(w) for w in data.get("text", []) if clean(w)]
                text = clean(" ".join(words))
                if len(text) >= 35:
                    out.append(block(doc_id, meta, text, element="image_ocr", method="tesseract", ocr_dpi=effective_dpi,
                                     ocr=True, image_size=[image.width,image.height]))
    elif ext in {".doc", ".ppt"}:
        # Antiword handles legacy Word binary documents; legacy PowerPoint is
        # rendered by the installed Microsoft PowerPoint through a PS helper.
        if ext == ".doc":
            antiword = Path(__file__).resolve().parents[1] / "vendor" / "antiword" / "bin" / "antiword.exe"
            if antiword.exists():
                env = os.environ.copy()
                env["HOME"] = str(antiword.parent.parent)
                result = subprocess.run([str(antiword), "-m", "UTF-8.txt", str(path)],
                                        capture_output=True, text=True, encoding="utf-8", errors="replace",
                                        timeout=120, env=env, cwd=str(antiword.parent))
                text = clean(result.stdout)
                if result.returncode == 0 and len(text) >= 20:
                    out.append(block(doc_id, meta, text, element="document", method="antiword"))
                    return out
                raise RuntimeError(f"Antiword failed (exit={result.returncode}): {clean(result.stderr)[:300] or 'no text extracted'}")
        if ext == ".ppt":
            helper = Path(__file__).with_name("convert_ppt_to_pdf.ps1")
            powershell = os.environ.get("POWERSHELL_EXE", "powershell.exe")
            with tempfile.TemporaryDirectory(prefix="nntu_ppt_") as temp:
                converted = Path(temp) / "presentation.pdf"
                result = subprocess.run(
                    [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                     "-File", str(helper), "-InputPath", str(path.resolve()), "-OutputPath", str(converted)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
                if result.returncode != 0 or not converted.exists():
                    detail = clean(result.stderr or result.stdout)[:500]
                    raise RuntimeError(f"PowerPoint PDF conversion failed (exit={result.returncode}): {detail or 'no details'}")
                pdf_blocks, pdf_pages, pdf_ocr = extract_pdf(converted, doc_id, meta, pytesseract, dpi, workers)
                meta["legacy_ppt_pages"] = pdf_pages
                meta["legacy_ppt_ocr_pages"] = pdf_ocr
                meta.setdefault("warnings", []).append("Legacy PPT converted to PDF with installed Microsoft PowerPoint")
                return pdf_blocks
        raise RuntimeError("legacy Office binary format: not converted (DOC needs Antiword; PPT needs Microsoft PowerPoint)")
    elif ext == ".zip":
        supported = SUPPORTED_EXT - {".zip", ".ppt"}
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            total_size = sum(m.file_size for m in members)
            if len(members) > 500 or total_size > 1_000_000_000:
                raise RuntimeError(f"ZIP safety limit exceeded: {len(members)} files, {total_size} uncompressed bytes")
            with tempfile.TemporaryDirectory(prefix="nntu_zip_") as temp:
                temp_root = Path(temp)
                for index, member in enumerate(members, 1):
                    member_ext = Path(member.filename).suffix.casefold()
                    if member_ext not in supported:
                        meta.setdefault("warnings", []).append(f"ZIP member skipped (unsupported): {member.filename}")
                        continue
                    if member.file_size > 200_000_000 or (member.compress_size and member.file_size / member.compress_size > 1000):
                        meta.setdefault("warnings", []).append(f"ZIP member skipped (size/ratio limit): {member.filename}")
                        continue
                    member_path = temp_root / f"member_{index:04d}{member_ext}"
                    member_meta = dict(meta)
                    member_meta["title"] = f"{meta.get('title') or path.name} :: {member.filename}"
                    member_meta["source_member"] = member.filename
                    try:
                        with archive.open(member) as source, member_path.open("wb") as target:
                            while chunk := source.read(1024 * 1024):
                                target.write(chunk)
                        if member_ext == ".pdf":
                            member_blocks, member_pages, member_ocr = extract_pdf(
                                member_path, doc_id, member_meta, pytesseract, dpi, workers)
                            meta["archive_pages"] = meta.get("archive_pages", 0) + member_pages
                            meta["archive_ocr_pages"] = meta.get("archive_ocr_pages", 0) + member_ocr
                        else:
                            member_blocks = extract_office(member_path, doc_id, member_meta,
                                                           pytesseract, dpi, workers)
                        out.extend(member_blocks)
                    except Exception as error:
                        meta.setdefault("warnings", []).append(
                            f"ZIP member failed ({member.filename}): {type(error).__name__}: {error}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", nargs="?", default="data/nntu_site")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--retry-errors", action="store_true", help="повторно обработать записи со статусом error при --resume")
    ap.add_argument("--reprocess-empty", action="store_true", help="повторно разобрать пустые записи, например после добавления парсера формата")
    ap.add_argument("--only-weird-ext", action="store_true", help="отобрать файлы с расширением, которое не распознано как поддерживаемый формат")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", help="substring match on filename or URL")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--ocr-workers", type=int, default=6, help="число параллельных процессов OCR (по умолчанию 6)")
    ap.add_argument("--no-ocr", action="store_true")
    args = ap.parse_args()
    root = Path(args.corpus).resolve()
    outdir = root / "extracted"
    outdir.mkdir(parents=True, exist_ok=True)
    files = list(jsonl(root / "files.jsonl"))
    classes = {r.get("stored_path"): r for r in jsonl(root / "classification.jsonl")}
    by_sha = defaultdict(list)
    for row in files:
        if row.get("stored_path"): by_sha[row.get("sha256") or row["stored_path"]].append(row)
    docs = []
    for sha, rows in by_sha.items():
        row = rows[0]
        source = root / row["stored_path"]
        docs.append((sha, rows, source))
    priority = {"01_поступающим":0, "02_студентам":1, "03_образовательные_программы":2,
                "04_кампусная_жизнь":3, "07_об_университете":4,
                "08_нормативные_и_административные":5, "05_преподавателям_и_сотрудникам":6,
                "06_наука_и_публикации":7, "99_неопределено_на_проверку":8,
                "00_медиа_не_для_текста":9}
    def priority_key(item):
        _, rows, path = item
        category = classes.get(rows[0].get("stored_path"), {}).get("category", "99_неопределено_на_проверку")
        return (priority.get(category, 8), int(rows[0].get("size_bytes") or 0), path.name.casefold())
    docs.sort(key=priority_key)
    if args.only:
        needle = args.only.casefold()
        docs = [x for x in docs if needle in str(x[1][0].get("filename", "")).casefold() or needle in str(x[1][0].get("stored_path", "")).casefold() or needle in str(x[1][0].get("url", "")).casefold()]
    if args.only_weird_ext:
        docs = [x for x in docs if Path(x[2]).suffix.casefold() not in SUPPORTED_EXT]
    if args.limit: docs = docs[:args.limit]
    page_rows = list(jsonl(root / "pages.jsonl"))
    if args.only: page_rows = [r for r in page_rows if args.only.casefold() in str(r.get("url", "")).casefold()]
    done = set()
    if args.resume:
        skip_statuses = {"ok"} | ({"empty"} if not args.reprocess_empty else set()) | ({"error"} if not args.retry_errors else set())
        done = {r.get("doc_id") for r in jsonl(outdir / "documents.jsonl") if r.get("status") in skip_statuses}
    pytes = None if args.no_ocr else init_ocr()
    mode = "a" if args.resume else "w"
    stats, t0 = Counter(), time.time()
    with (outdir / "blocks.jsonl").open(mode, encoding="utf-8") as bf, \
         (outdir / "documents.jsonl").open(mode, encoding="utf-8") as df, \
         (outdir / "errors.jsonl").open(mode, encoding="utf-8") as ef:
        for idx, (sha, rows, path) in enumerate(docs, 1):
            doc_id = sha or hashlib.sha256(str(path).encode()).hexdigest()
            if doc_id in done: stats["skipped_resume"] += 1; continue
            cls = classes.get(rows[0].get("stored_path"), {})
            meta = {"source_type":"file", "source_path":rows[0].get("stored_path"),
                    "source_urls":sorted({r.get("url") for r in rows if r.get("url")}),
                    "source_pages":sorted({r.get("source_page") for r in rows if r.get("source_page")}),
                    "title":rows[0].get("filename"), "category":cls.get("category", "99_неопределено_на_проверку")}
            try:
                if not path.exists(): raise FileNotFoundError(path)
                ext = effective_extension(path, rows[0])
                if idx == 1 or idx % 10 == 0:
                    print(f"[{idx}/{len(docs)}] {path.name} ({ext})", flush=True)
                if ext == ".pdf":
                    blocks, pages, ocrn = extract_pdf(path, doc_id, meta, pytes, args.dpi, max(1,args.ocr_workers))
                else:
                    blocks = extract_office(path, doc_id, meta, pytes, args.dpi, max(1,args.ocr_workers))
                    pages = meta.get("archive_pages", meta.get("legacy_ppt_pages", 0))
                    ocrn = meta.get("archive_ocr_pages", meta.get("legacy_ppt_ocr_pages", 0))
                for b in blocks: bf.write(json.dumps(b, ensure_ascii=False) + "\n")
                status = "ok" if blocks else "empty"
                rowout = {"doc_id":doc_id,"status":status,"filename":rows[0].get("filename"),
                          "source_path":rows[0].get("stored_path"),"sha256":sha,"size_bytes":rows[0].get("size_bytes"),
                          "extension":ext,"source_extension":path.suffix.casefold(),"category":meta["category"],"source_urls":meta["source_urls"],
                          "source_pages":meta["source_pages"],"blocks":len(blocks),"pages":pages,"ocr_pages":ocrn,
                          "text_chars":sum(len(b["text"]) for b in blocks),"warnings":meta.get("warnings",[])}
                df.write(json.dumps(rowout, ensure_ascii=False)+"\n")
                stats[status]+=1; stats["ocr_pages"]+=ocrn; stats["blocks"]+=len(blocks)
            except Exception as e:
                err={"doc_id":doc_id,"source_path":rows[0].get("stored_path"),"filename":rows[0].get("filename"),
                     "error":f"{type(e).__name__}: {e}"}
                ef.write(json.dumps(err, ensure_ascii=False)+"\n")
                df.write(json.dumps({"doc_id":doc_id,"status":"error",**err},ensure_ascii=False)+"\n")
                stats["errors"]+=1
            if idx % 100 == 0 or idx == len(docs):
                print(f"files {idx}/{len(docs)}; ok={stats['ok']} empty={stats['empty']} errors={stats['errors']} OCR-pages={stats['ocr_pages']} elapsed={int(time.time()-t0)}s", flush=True)
        for n, row in enumerate(page_rows, 1):
            url = row.get("url", "")
            doc_id = "html:" + hashlib.sha256(url.encode()).hexdigest()
            if doc_id in done: continue
            stored = row.get("stored_html")
            html_path = root / stored if stored else None
            title = row.get("title") or url
            text = clean(row.get("text"))
            if html_path and html_path.exists():
                try:
                    from bs4 import BeautifulSoup
                    html = BeautifulSoup(html_path.read_text(encoding="utf-8",errors="replace"),"html.parser")
                    for tag in html(["script","style","noscript"]): tag.decompose()
                    # Keep link labels and their destinations so the assistant can direct users to resources.
                    for a in html.find_all("a", href=True):
                        label=clean(a.get_text(" ",strip=True)); href=a["href"].strip()
                        if label and href: a.append(f" [Ссылка: {label} — {href}]")
                    text=clean(html.get_text("\n",strip=True)) or text
                except Exception: pass
            topics = set(row.get("topics") or [])
            topic_category = {"admission":"01_поступающим", "study":"02_студентам",
                              "campus_life":"04_кампусная_жизнь", "university":"07_об_университете",
                              "science":"06_наука_и_публикации", "staff":"05_преподавателям_и_сотрудникам"}
            category = next((topic_category[t] for t in ("admission","study","campus_life","staff","science","university") if t in topics),
                            "99_неопределено_на_проверку")
            meta={"source_type":"html","source_path":stored,"source_urls":[url],"source_pages":[row.get("source_page")] if row.get("source_page") else [],
                  "title":title,"category":category}
            if text:
                bf.write(json.dumps(block(doc_id,meta,text,element="webpage"),ensure_ascii=False)+"\n")
                df.write(json.dumps({"doc_id":doc_id,"status":"ok","filename":title,"source_path":stored,
                                     "extension":".html","category":meta["category"],"source_urls":[url],"source_pages":meta["source_pages"],
                                     "blocks":1,"pages":1,"ocr_pages":0,"text_chars":len(text)},ensure_ascii=False)+"\n")
                stats["html_pages"]+=1; stats["blocks"]+=1
    latest_docs = {}
    for row in jsonl(outdir / "documents.jsonl"):
        latest_docs[row.get("doc_id", row.get("source_path"))] = row
    all_docs = list(latest_docs.values())
    global_stats = Counter(r.get("status", "unknown") for r in all_docs)
    by_ext = Counter(r.get("extension", "") for r in all_docs)
    by_category = Counter(r.get("category", "99_неопределено_на_проверку") for r in all_docs)
    versions = {"python":sys.version.split()[0]}
    try:
        import pymupdf
        versions["pymupdf"] = pymupdf.VersionBind
    except Exception: pass
    try:
        if pytes: versions["tesseract"] = str(pytes.get_tesseract_version())
    except Exception: pass
    report={"created_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"unique_files_selected":len(docs),"html_selected":len(page_rows),
            "stats_this_run":dict(stats),"stats_cumulative":dict(global_stats),
            "ocr_pages_total":sum(int(r.get("ocr_pages",0)) for r in all_docs),
            "blocks_total":sum(int(r.get("blocks",0)) for r in all_docs),
            "text_chars_total":sum(int(r.get("text_chars",0)) for r in all_docs),
            "by_extension":dict(by_ext),"by_category":dict(by_category),
            "elapsed_seconds_this_run":round(time.time()-t0,1),"ocr_dpi":args.dpi,"ocr_workers":max(1,args.ocr_workers),
            "ocr_enabled":not args.no_ocr,"versions":versions,
            "policy":"native PDF text first; OCR only if <80 chars with >8% embedded image area, or <250 chars with >35% image area"}
    (outdir/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
