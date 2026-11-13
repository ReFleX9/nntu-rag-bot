#!/usr/bin/env python3
"""OCR only audited PDF pages with broken text layers; resume from checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/python"))
from audit_pdf_text_v2 import signals


def recover(path: Path):
    done = set()
    if not path.exists():
        return done
    with path.open("rb+") as stream:
        end = 0
        while line := stream.readline():
            if not line.endswith(b"\n"):
                break
            try:
                row = json.loads(line)
                done.add((row["doc_id"], row["page"]))
                end = stream.tell()
            except (ValueError, KeyError):
                break
        stream.truncate(end)
    return done


def recognize(png: bytes):
    import io
    import pytesseract
    from PIL import Image
    image = Image.open(io.BytesIO(png))
    data = pytesseract.image_to_data(image, lang="rus+eng", config="--psm 3",
                                     output_type=pytesseract.Output.DICT)
    lines = defaultdict(list)
    confidence = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        lines[(data["block_num"][i], data["par_num"][i], data["line_num"][i])].append(word)
        try:
            score = float(data["conf"][i])
            if score >= 0:
                confidence.append(score)
        except (ValueError, TypeError):
            pass
    text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
    return text, round(sum(confidence) / len(confidence), 1) if confidence else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/nntu_site")
    parser.add_argument("--candidates", type=Path, default=ROOT / "work/v2/ocr_candidates.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "work/v2/ocr_repairs.jsonl")
    parser.add_argument("--backup", type=Path, help="Checkpoint copy, refreshed every 50 pages and at completion")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", help="Substring of source PDF filename for a pilot")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 100 <= args.dpi <= 300 or not 1 <= args.workers <= 8:
        parser.error("dpi must be 100..300 and workers 1..8")
    import pymupdf
    import pytesseract
    from extract_corpus import init_ocr
    init_ocr()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = recover(args.output)
    by_file = defaultdict(list)
    with args.candidates.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if args.only and args.only.lower() not in row["source_path"].lower():
                continue
            if (row["doc_id"], row["page"]) not in done:
                by_file[row["source_path"]].append(row)
    counts = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
         args.output.open("a", encoding="utf-8", newline="\n") as target:
        for source_path, candidates in by_file.items():
            path = (args.root / source_path.replace("\\", "/")).resolve()
            if not path.is_relative_to(args.root.resolve()) or not path.is_file():
                counts["missing_or_unsafe_path"] += len(candidates)
                continue
            try:
                with pymupdf.open(path) as pdf:
                    jobs = []
                    for row in candidates:
                        page = pdf[row["page"] - 1]
                        scale = args.dpi / 72
                        if page.rect.width * page.rect.height * scale * scale > 18_000_000:
                            scale *= (18_000_000 / (page.rect.width * page.rect.height * scale * scale)) ** .5
                        png = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")
                        jobs.append((row, pool.submit(recognize, png)))
                    for row, future in jobs:
                        text, confidence = future.result()
                        quality = signals(text)
                        accepted = (quality["cyrillic"] >= 80 and quality["cyrillic_share"] >= .35
                                    and confidence is not None and confidence >= 45)
                        result = {"doc_id": row["doc_id"], "page": row["page"],
                                  "source_path": source_path, "accepted": accepted,
                                  "ocr_confidence": confidence, "ocr_quality": quality,
                                  "text": text if accepted else ""}
                        target.write(json.dumps(result, ensure_ascii=False) + "\n")
                        counts["accepted" if accepted else "rejected"] += 1
                        if sum(counts.values()) % 10 == 0:
                            target.flush()
                            os.fsync(target.fileno())
                            print("checkpoint", sum(counts.values()), dict(counts), flush=True)
                            if args.backup and sum(counts.values()) % 50 == 0:
                                args.backup.parent.mkdir(parents=True, exist_ok=True)
                                temporary = args.backup.with_suffix(args.backup.suffix + ".partial")
                                shutil.copyfile(args.output, temporary)
                                os.replace(temporary, args.backup)
                        if args.limit and sum(counts.values()) >= args.limit:
                            break
            except Exception as error:
                print(f"ERROR {source_path}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                counts["error_docs"] += 1
            if args.limit and sum(counts.values()) >= args.limit:
                break
        target.flush()
        os.fsync(target.fileno())
    if args.backup:
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.backup.with_suffix(args.backup.suffix + ".partial")
        shutil.copyfile(args.output, temporary)
        os.replace(temporary, args.backup)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
