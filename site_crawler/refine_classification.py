#!/usr/bin/env python3
"""Refine the crawl's provisional file categories using extracted text."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from categorize_corpus import classify


def read_jsonl(path):
    if not path.exists(): return
    with path.open(encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError: continue


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", nargs="?", default="data/nntu_site")
    ap.add_argument("--rebuild-view", action="store_true", help="пересоздать hard-link organized_view по обновлённым категориям")
    args = ap.parse_args()
    root = Path(args.corpus).resolve()
    output = root / "extracted" / "blocks.jsonl"
    snippets = defaultdict(str)
    # Take text from the earliest blocks, with a per-document cap to keep RAM bounded.
    for b in read_jsonl(output):
        doc_id = b.get("doc_id")
        text = b.get("text", "")
        if doc_id and len(snippets[doc_id]) < 4000 and text:
            snippets[doc_id] += "\n" + text[:max(0, 4000-len(snippets[doc_id]))]

    manifest = defaultdict(list)
    for row in read_jsonl(root / "files.jsonl"):
        if row.get("stored_path"):
            manifest[row["stored_path"]].append(row)

    records = list(read_jsonl(root / "classification.jsonl"))
    changed = Counter()
    for item in records:
        current = item.get("category", "99_неопределено_на_проверку")
        if current != "99_неопределено_на_проверку" and item.get("confidence") != "low":
            continue
        sample = snippets.get(item.get("sha256", ""), "")
        if not sample.strip():
            continue
        rows = manifest.get(item.get("stored_path"), [])
        augmented = []
        for row in rows:
            r = dict(row)
            r["link_context"] = (r.get("link_context", "") + "\n" + sample[:4000])
            augmented.append(r)
        if not augmented:
            augmented = [{"filename":item.get("filename", ""), "link_context":sample[:4000]}]
        result = classify(augmented)
        if result["category"] != "99_неопределено_на_проверку" and result["confidence"] in {"medium", "high"}:
            old = current
            item["category"] = result["category"]
            item["confidence"] = result["confidence"]
            item["signals"] = sorted(set(item.get("signals", [])) | set(result["signals"]))
            item["secondary_categories"] = result["secondary"]
            item["classification_method"] = "crawl metadata + extracted text"
            changed[f"{old} -> {item['category']}"] += 1

    (root / "classification.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in records), encoding="utf-8")
    with (root / "classification.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["category","confidence","filename","size_bytes","sha256","stored_path","signals","secondary_categories","source_urls"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for x in records:
            row = {field:x.get(field, "") for field in fields}
            row["signals"] = "; ".join(x.get("signals", []))
            row["secondary_categories"] = "; ".join(x.get("secondary_categories", []))
            row["source_urls"] = "; ".join(x.get("source_urls", []))
            w.writerow(row)
    categories = Counter(x.get("category", "unknown") for x in records)
    already_refined = sum(x.get("classification_method") == "crawl metadata + extracted text" for x in records)
    reclassified_total = already_refined + sum(changed.values())
    report = {"file_records":len(records), "text_snippets":len(snippets),
              "reclassified":reclassified_total, "reclassified_this_run":sum(changed.values()),
              "changes_this_run":dict(changed), "categories":dict(categories),
              "unknown_remaining":categories["99_неопределено_на_проверку"]}
    (root / "extracted" / "classification_refinement.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    confidence_counts = defaultdict(Counter)
    for x in records: confidence_counts[x.get("category", "unknown")][x.get("confidence", "low")] += 1
    md = ["# Тематическая классификация корпуса НГТУ", "",
          "Категории первоначально определены по адресу и подписи ссылки. Низкоуверенные и неопределённые файлы уточнены по первым 4 000 символам извлечённого текста.",
          f"", f"Всего файлов: {len(records):,}; переклассифицировано по тексту: {reclassified_total:,}; осталось неопределёнными: {categories['99_неопределено_на_проверку']:,}.",
          "", "| Категория | Файлов | Низкая уверенность | Средняя | Высокая |", "|---|---:|---:|---:|---:|"]
    for cat, n in sorted(categories.items()):
        cc=confidence_counts[cat]
        md.append(f"| `{cat}` | {n:,} | {cc['low']:,} | {cc['medium']:,} | {cc['high']:,} |")
    (root / "classification_report.md").write_text("\n".join(md)+"\n", encoding="utf-8")

    if args.rebuild_view:
        view = (root / "organized_view").resolve()
        if view != root / "organized_view" or view.parent != root:
            raise RuntimeError("Refusing to rebuild view outside corpus root")
        if view.exists():
            for p in sorted(view.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if p.is_file() or p.is_symlink(): p.unlink()
                elif p.is_dir(): p.rmdir()
            view.rmdir()
        for x in records:
            src = (root / x["stored_path"]).resolve()
            if src.is_file():
                stable = (x.get("sha256") or hashlib.sha256(x["stored_path"].encode()).hexdigest())[:24]
                dest = view / x.get("category", "99_неопределено_на_проверку") / (stable + src.suffix)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists(): os.link(src, dest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
