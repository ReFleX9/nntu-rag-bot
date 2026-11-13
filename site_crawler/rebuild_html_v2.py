#!/usr/bin/env python3
"""Re-extract saved HTML without the repeated site chrome; resume per page."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "python"))
from bs4 import BeautifulSoup


def body_node(soup: BeautifulSoup):
    # Templates observed on nntu.ru, its /sveden pages and WordPress sub-sites.
    selectors = (".case.article", ".case.custom", "#container_content",
                 "#content", ".entry-content", ".page-body .col-lg-9 .case",
                 ".page-body .col-lg-9", ".page-body", "main", "article")
    for selector in selectors:
        node = soup.select_one(selector)
        if node and len(node.get_text(" ", strip=True)) >= 80:
            return node, selector
    return None, None


def extract_html(path: Path, url: str):
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    node, selector = body_node(soup)
    if node is None:
        return "", None, 0
    # Remove page controls and containers that recur independently of article text.
    for tag in list(node.find_all(["script", "style", "noscript", "svg", "form", "nav", "footer", "header"])):
        tag.decompose()
    for tag in list(node.find_all(class_=lambda x: x and any(
            marker in str(x).lower() for marker in (
                "breadcrumb", "sidebar", "pagination", "cookie", "social", "share", "mobile-nav", "navbar", "calendar")))):
        tag.decompose()
    for link in node.find_all("a", href=True):
        label = link.get_text(" ", strip=True)
        href = link["href"].strip()
        if not label or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        resolved = urljoin(url, href)
        if urlsplit(resolved).scheme not in {"http", "https"}:
            continue
        link.append(f" [Ссылка: {label} — {resolved}]")
    lines = [line.strip() for line in node.get_text("\n", strip=True).splitlines()]
    text = "\n".join(line for line in lines if line)
    return text, selector, len(node.find_all("a", href=True))


def recover_jsonl(path: Path):
    done = set()
    if not path.exists():
        return done
    with path.open("rb+") as stream:
        valid_end = 0
        while True:
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                break
            try:
                row = json.loads(line)
                done.add(row["doc_id"])
                valid_end = stream.tell()
            except (ValueError, KeyError):
                break
        stream.truncate(valid_end)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/nntu_site")
    parser.add_argument("--output", type=Path, default=ROOT / "work/v2/html_blocks.jsonl")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = recover_jsonl(args.output)
    counts = Counter()
    with (args.root / "pages.jsonl").open(encoding="utf-8") as source, \
         args.output.open("a", encoding="utf-8", newline="\n") as target:
        for line in source:
            row = json.loads(line)
            url = row.get("url", "")
            doc_id = "html:" + hashlib.sha256(url.encode()).hexdigest()
            if doc_id in done:
                counts["resume_skipped"] += 1
                continue
            path = args.root / (row.get("stored_html") or "")
            if not path.is_file():
                counts["missing_file"] += 1
                continue
            text, selector, links = extract_html(path, url)
            if len(text) < 80:
                counts["no_body"] += 1
                continue
            old_length = len(row.get("text") or "")
            block = {"doc_id": doc_id, "text": text, "source_type": "html",
                     "source_path": row.get("stored_html"), "source_urls": [url],
                     "source_pages": [row["source_page"]] if row.get("source_page") else [],
                     "title": row.get("title") or url,
                     "category": "02_студентам" if "/studentam/" in url else
                                 "01_поступающим" if any(x in url for x in ("/abit/", "/abiturientam/")) else
                                 "08_нормативные_и_административные" if "/sveden/document" in url else
                                 "07_об_университете",
                     "element": "webpage", "html_selector": selector,
                     "old_text_chars": old_length, "links_kept": links}
            target.write(json.dumps(block, ensure_ascii=False) + "\n")
            done.add(doc_id)
            counts["pages"] += 1
            counts["old_chars"] += old_length
            counts["new_chars"] += len(text)
            counts[selector] += 1
            if counts["pages"] % 25 == 0:
                target.flush()
                os.fsync(target.fileno())
                print("checkpoint", counts["pages"], flush=True)
            if args.limit and counts["pages"] >= args.limit:
                break
        target.flush()
        os.fsync(target.fileno())
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
