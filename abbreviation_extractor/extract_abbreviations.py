"""Find abbreviation candidates in PDFs with local NER. Python 3.10+.

The output contains model predictions, NOT verified abbreviation expansions.
Offsets refer to the extracted page text saved in progress.sqlite3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path
from corpus_export import export_results

DEFAULTS = {
    "gliner": ("urchade/gliner_multi-v2.1", "443d26d654e0324125a96bebd8e796c14ff2efe6"),
    "specialist": ("kaejo98/acronym-definition-detection", "24af86d7b264660f34806c045340c56c6f7a3e6a"),
}
# This exact checkpoint exports LABEL_0..4 instead of its documented BIO tags.
# Mapping follows the author's model-card tag list; checked on explicit SC/PCA/NLP
# and Russian mixed-text examples. Never apply it to an unrelated checkpoint.
SPECIALIST_TAGS = {0: "O", 1: "B-AC", 2: "I-AC", 3: "B-LF", 4: "I-LF"}
LOG = logging.getLogger("abbreviations")
# Inference is local; the first run only needs model/config/tokenizer downloads.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"


def canonical(text):
    """A grouping key; preserve original spelling separately. No transliteration."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def parse_pages(spec):
    if not spec:
        return None
    pages = set()
    for part in spec.split(","):
        pieces = part.strip().split("-")
        if len(pieces) not in (1, 2):
            raise ValueError("Invalid page range")
        first, last = int(pieces[0]), int(pieces[-1])
        if first < 1 or last < first:
            raise ValueError("Pages must be positive, increasing physical page numbers")
        pages.update(range(first, last + 1))
    return pages


def windows(text, tokenizer, budget=320, overlap=64):
    """Overlapping subtoken windows; char offsets remain relative to original text.

    Cut at whitespace when possible, so a short form is not cut in half. A
    pathological very long unbroken token may still need splitting.
    """
    if not 0 < overlap < budget:
        raise ValueError("Require 0 < overlap < max_tokens")
    offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True, verbose=False)["offset_mapping"]
    offsets = [(a, b) for a, b in offsets if b > a]
    start = 0
    while start < len(offsets):
        stop = min(start + budget, len(offsets))
        a, b = offsets[start][0], offsets[stop - 1][1]
        if start and a > 0 and not text[a - 1].isspace():
            # Previous overlapping window covers this partial first word.
            while a < b and not text[a].isspace():
                a += 1
        if stop < len(offsets) and b < len(text) and not text[b].isspace():
            cut = b
            while cut > a and not text[cut - 1].isspace():
                cut -= 1
            if cut > a:
                b = cut
        if b > a and text[a:b].strip():
            yield a, text[a:b]
        if stop == len(offsets):
            break
        start = stop - overlap


def deduplicate(entities):
    """Merge window duplicates; retain a flat, non-overlapping set of mentions."""
    best = {}
    for e in entities:
        key = (e["start"], e["end"])
        if key not in best or e["score"] > best[key]["score"]:
            best[key] = e
    picked = []
    for e in sorted(best.values(), key=lambda x: (-x["score"], -(x["end"] - x["start"]))):
        if not any(e["start"] < p["end"] and p["start"] < e["end"] for p in picked):
            picked.append(e)
    return sorted(picked, key=lambda x: x["start"])


def join_subword_entities(entities, text):
    """Repair repeated B-AC on contiguous subpieces in the specialist checkpoint.

    Do NOT join across whitespace: 'SQL XML' represents separate mentions.
    """
    result = []
    for e in sorted(entities, key=lambda x: (x["start"], x["end"])):
        e = dict(e)
        if result and e["start"] <= result[-1]["end"] and e["label"] == result[-1]["label"]:
            last = result[-1]
            last["end"] = max(last["end"], e["end"])
            last["score"] = min(last["score"], e["score"])
        else:
            result.append(e)
    return result


def clean_abbreviation_span(text, start, end):
    """Conservative shape/boundary filter for technical initialisms.

    Reject partial words (GU inside GUC) and ordinary Titlecase names. This filter
    can miss lowercase abbreviations; --keep-unfiltered disables it for auditing.
    """
    while start < end and text[start] in " \t\n-–—.,:;()[]{}":
        start += 1
    while end > start and text[end - 1] in " \t\n-–—,:;()[]{}":
        end -= 1
    term = text[start:end]
    # Detach a Russian suffix in an expression such as SQL-команд.
    match = re.fullmatch(r"([A-ZА-ЯЁ][A-ZА-ЯЁ0-9]{1,})-[а-яё]+", term)
    if match:
        end = start + len(match.group(1))
        term = text[start:end]
    if not term or (start and (text[start - 1].isalnum() or text[start - 1] == "_")):
        return None
    if end < len(text) and (text[end].isalnum() or text[end] == "_"):
        return None
    letters = [c for c in term if c.isalpha()]
    dotted = bool(re.fullmatch(r"(?:[A-Za-zА-Яа-яЁё]{1,3}\.){2,}", term))
    if len(letters) < 2 or len(term) > 40 or (sum(c.isupper() for c in letters) < 2 and not dotted):
        return None
    return start, end


class Extractor:
    def __init__(self, args):
        import torch
        self.torch = torch
        torch.set_num_threads(args.cpu_threads)
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Install a compatible PyTorch build or use --device cpu.")
        self.args = args
        self.batch_size = args.batch_size
        kwargs = {"revision": args.revision} if args.revision else {}
        LOG.info("Loading %s (%s) on %s", args.model, args.revision or "main", self.device)
        if args.backend == "gliner":
            from gliner import GLiNER
            self.model = GLiNER.from_pretrained(args.model, **kwargs).to(self.device).eval()
            self.tokenizer = self.model.data_processor.transformer_tokenizer
        else:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, **kwargs)
            self.model = AutoModelForTokenClassification.from_pretrained(args.model, **kwargs).eval()
            if args.model == DEFAULTS["specialist"][0] and args.revision == DEFAULTS["specialist"][1]:
                if set(self.model.config.id2label.values()) == {f"LABEL_{i}" for i in range(5)}:
                    self.model.config.id2label = dict(SPECIALIST_TAGS)
                    self.model.config.label2id = {v: k for k, v in SPECIALIST_TAGS.items()}
            self.pipe = pipeline("token-classification", model=self.model, tokenizer=self.tokenizer,
                                 aggregation_strategy="simple", device=self.device)
            groups = {str(label).removeprefix("B-").removeprefix("I-")
                      for label in self.model.config.id2label.values()}
            if not groups.intersection({"AC", "ABBR", "ABBREVIATION", "short"}):
                raise ValueError(f"Model has no supported abbreviation tag: {groups}")
        if not self.tokenizer.is_fast:
            raise ValueError("A fast tokenizer with character offsets is required")
        # Explicitly enforce a conservative input budget. GLiNER prepends labels.
        if args.max_tokens > 320:
            raise ValueError("This configuration supports --max-tokens <= 320")

    def predict(self, texts):
        try:
            with self.torch.inference_mode():
                if self.args.backend == "gliner":
                    return self.model.inference(
                        texts, self.args.labels, threshold=self.args.threshold,
                        flat_ner=True, batch_size=self.batch_size)
                results = self.pipe(texts, batch_size=self.batch_size,
                                    ignore_labels=["O", "B-O", "I-O"])
                pages = []
                for text, entities in zip(texts, results):
                    abbreviations = [{"start": int(e["start"]), "end": int(e["end"]),
                                      "label": e["entity_group"], "score": float(e["score"])}
                                     for e in entities if e["entity_group"] in {"AC", "ABBR", "ABBREVIATION", "short"}]
                    joined = join_subword_entities(abbreviations, text)
                    pages.append([e for e in joined if e["score"] >= self.args.threshold])
                return pages
        except self.torch.cuda.OutOfMemoryError:
            if self.batch_size <= 1:
                raise
            self.batch_size = max(1, self.batch_size // 2)
            self.torch.cuda.empty_cache()
            LOG.warning("GPU memory exhausted; retrying with batch size %s", self.batch_size)
            return self.predict(texts)

    def extract(self, text):
        chunks = list(windows(text, self.tokenizer, self.args.max_tokens, self.args.overlap))
        if not chunks:
            return []
        predictions = self.predict([chunk for _, chunk in chunks])
        if len(predictions) != len(chunks):
            raise RuntimeError("Model returned the wrong number of predictions")
        mentions = []
        for (base, chunk), entities in zip(chunks, predictions):
            for e in entities:
                a, b = int(e["start"]), int(e["end"])
                if not 0 <= a < b <= len(chunk):
                    raise RuntimeError("Invalid model character offsets")
                a, b = base + a, base + b
                if not self.args.keep_unfiltered:
                    cleaned = clean_abbreviation_span(text, a, b)
                    if cleaned is None:
                        continue
                    a, b = cleaned
                term = text[a:b]
                if not any(c.isalpha() for c in term):
                    continue
                mentions.append({"start": a, "end": b, "abbreviation": term,
                                 "canonical_key": canonical(term), "label": e["label"],
                                 "score": float(e["score"])})
        return deduplicate(mentions)


def open_db(path, config):
    db = sqlite3.connect(path)
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, config TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY, path TEXT UNIQUE, sha256 TEXT, title TEXT, family TEXT, page_count INTEGER);
    CREATE TABLE IF NOT EXISTS pages (
        doc_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
        page INTEGER, printed_page TEXT, status TEXT, text TEXT, error TEXT, section TEXT DEFAULT '',
        PRIMARY KEY(doc_id,page));
    CREATE TABLE IF NOT EXISTS mentions (
        doc_id TEXT, page INTEGER, start INTEGER, end INTEGER, data TEXT,
        PRIMARY KEY(doc_id,page,start,end),
        FOREIGN KEY(doc_id,page) REFERENCES pages(doc_id,page) ON DELETE CASCADE);
    """)
    serialized = json.dumps(config, sort_keys=True, ensure_ascii=False)
    old = db.execute("SELECT config FROM settings WHERE id=1").fetchone()
    if old and old[0] != serialized:
        db.close()
        raise ValueError("Output folder belongs to different input/model/settings. Use another --output folder.")
    db.execute("INSERT OR IGNORE INTO settings VALUES(1,?)", (serialized,))
    db.commit()
    return db


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def process_pdf(path, root, db, extractor, selected):
    import pymupdf
    rel = path.relative_to(root).as_posix()
    sha = digest_file(path)
    doc_id = hashlib.sha256((rel + "\0" + sha).encode()).hexdigest()
    # Invalidate an old version even when the replacement PDF is damaged.
    with db:
        db.execute("DELETE FROM documents WHERE path=? AND id<>?", (rel, doc_id))
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError("Password-protected PDF")
        if selected and max(selected) > len(pdf):
            raise ValueError(f"Requested page exceeds PDF length ({len(pdf)})")
        with db:
            db.execute("INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?,?)",
                       (doc_id, rel, sha, pdf.metadata.get("title", "") or "",
                        Path(rel).parts[0] if len(Path(rel).parts) > 1 else "", len(pdf)))
        # Read the whole document before NER so context can cross page boundaries.
        # Each page is committed separately and reused on a resumed run.
        headings = {}
        stack = []
        for level, title, page_number in pdf.get_toc():
            if page_number < 1:
                continue
            stack = [(lev, name) for lev, name in stack if lev < level]
            stack.append((level, title))
            headings[page_number] = " > ".join(name for _, name in stack)
        section = ""
        for number in range(1, len(pdf) + 1):
            section = headings.get(number, section)
            row = db.execute("SELECT status FROM pages WHERE doc_id=? AND page=?", (doc_id, number)).fetchone()
            if row and row[0] != "text_error":
                continue
            try:
                page = pdf[number - 1]
                raw = page.get_text("text", sort=True)
                status = "needs_ocr" if sum(c.isalnum() for c in raw) < 20 else "pending"
                values = (doc_id, number, page.get_label(), status, raw, None, section)
            except Exception as error:
                LOG.exception("Cannot read %s page %s", rel, number)
                values = (doc_id, number, "", "text_error", "", str(error), section)
            with db:
                db.execute("DELETE FROM pages WHERE doc_id=? AND page=?", (doc_id, number))
                db.execute("INSERT INTO pages VALUES(?,?,?,?,?,?,?)", values)
            if number % 200 == 0 or number == len(pdf):
                LOG.info("Чтение текста %s: %s/%s страниц", rel, number, len(pdf))
        page_numbers = sorted(selected) if selected else range(1, len(pdf) + 1)
        last_page = max(page_numbers, default=0)
        errors, cached, processed = 0, 0, 0
        for number in page_numbers:
            status, raw = db.execute("SELECT status,text FROM pages WHERE doc_id=? AND page=?", (doc_id, number)).fetchone()
            if status in {"ok", "needs_ocr"}:
                cached += 1
                continue
            if status == "text_error":
                errors += 1
                continue
            try:
                entities = extractor.extract(raw)
                with db:
                    db.execute("DELETE FROM mentions WHERE doc_id=? AND page=?", (doc_id, number))
                    db.executemany("INSERT INTO mentions VALUES(?,?,?,?,?)",
                                   [(doc_id, number, e["start"], e["end"], json.dumps(e, ensure_ascii=False)) for e in entities])
                    db.execute("UPDATE pages SET status='ok',error=NULL WHERE doc_id=? AND page=?", (doc_id, number))
                processed += 1
                if processed == 1 or processed % 20 == 0 or number == last_page:
                    LOG.info("NER %s: страница %s/%s, найдено на странице: %s", rel, number, len(pdf), len(entities))
            except Exception as error:
                LOG.exception("Failed: %s page %s", rel, number)
                with db:
                    db.execute("DELETE FROM mentions WHERE doc_id=? AND page=?", (doc_id, number))
                    db.execute("UPDATE pages SET status='error',error=? WHERE doc_id=? AND page=?", (str(error), doc_id, number))
                errors += 1
        LOG.info("%s: новых страниц %s, сохранённых %s, ошибок %s", rel, processed, cached, errors)
        return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=Path("pdfs"), help="One PDF or a root folder (recursively)")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--backend", choices=DEFAULTS, default="specialist")
    parser.add_argument("--model", help="Optional compatible model or local model directory")
    parser.add_argument("--revision", help="Model commit; defaults are pinned")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--labels", nargs="+", default=["abbreviation"], help="GLiNER entity labels; English labels are intentional")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--keep-unfiltered", action="store_true", help="Keep raw NER spans, including partial words and Titlecase names")
    parser.add_argument("--export-only", action="store_true", help="Rebuild tables from the saved database without loading NER")
    parser.add_argument("--pages", help="Physical pages, e.g. 428,713,2730-2734; omitted = all pages")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    if not 0 < args.threshold < 1 or args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("Invalid threshold, batch size or thread count")
    if not 0 < args.overlap < args.max_tokens <= 320:
        parser.error("Require 0 < overlap < max-tokens <= 320")
    try:
        selected = parse_pages(args.pages)
    except ValueError as error:
        parser.error(str(error))
    source = args.input.expanduser().resolve()
    root = source if source.is_dir() else source.parent
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf") if source.is_dir() else [source]
    if not files or any(not p.is_file() or p.suffix.lower() != ".pdf" for p in files):
        parser.error("No valid input PDF files found")
    default_model, default_revision = DEFAULTS[args.backend]
    if not args.model:
        args.model = default_model
        args.revision = args.revision or default_revision
    config = {"code_version": 4, "input": str(source), "backend": args.backend, "model": args.model,
              "revision": args.revision, "threshold": args.threshold, "labels": args.labels,
              "max_tokens": args.max_tokens, "overlap": args.overlap, "keep_unfiltered": args.keep_unfiltered,
              "specialist_tag_map": SPECIALIST_TAGS if args.backend == "specialist" else None}
    args.output.mkdir(parents=True, exist_ok=True)
    db = open_db(args.output / "progress.sqlite3", config)
    (args.output / "RUN_STATUS.txt").write_text("running\nОбработка идёт. CSV будут обновлены при завершении.\n", encoding="utf-8-sig")
    LOG.info("Найдено PDF: %s. Результаты: %s", len(files), args.output.resolve())
    errors, exit_code = [], 0
    try:
        if not args.export_only:
            import razdel  # Check the export dependency before a long run.
            extractor = Extractor(args)
        present_paths = {path.relative_to(root).as_posix() for path in files}
        with db:
            for (old_path,) in db.execute("SELECT path FROM documents").fetchall():
                if old_path not in present_paths:
                    db.execute("DELETE FROM documents WHERE path=?", (old_path,))
        for i, path in enumerate([] if args.export_only else files, 1):
            LOG.info("PDF %s/%s: %s", i, len(files), path.name)
            try:
                if process_pdf(path, root, db, extractor, selected):
                    exit_code = 1
            except Exception as error:
                LOG.exception("Cannot process %s", path)
                errors.append({"source_file": path.relative_to(root).as_posix(), "status": "file_error", "error": str(error)})
                exit_code = 1
    except KeyboardInterrupt:
        LOG.warning("Interrupted; completed pages saved. Run the same command to continue.")
        exit_code = 130
    except Exception as error:
        LOG.exception("Extraction stopped")
        errors.append({"status": "run_error", "error": str(error)})
        exit_code = 1
    finally:
        try:
            report = export_results(db, args.output, config, errors, expected_files=[p.relative_to(root).as_posix() for p in files],
                                    interrupted=(exit_code == 130))
            print(f"\nРезультаты: {args.output.resolve()}")
            print(f"Упоминаний: {report['mentions']}; сокращений: {report['unique_surface_forms']}; статус: {report['status']}")
        finally:
            db.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
