#!/usr/bin/env python3
"""Сводный аудит JSONL-корпуса краулера НГТУ."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from crawl_nntu import norm_url


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                yield {"_parse_error": f"line {line_no}: {exc}"}


def gib(n: int | float) -> str:
    return f"{n / (1024 ** 3):.2f} GiB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="?", default="data/nntu_site")
    args = parser.parse_args()
    root = Path(args.corpus).resolve()
    pages = list(read_jsonl(root / "pages.jsonl"))
    links = list(read_jsonl(root / "links.jsonl"))
    files = list(read_jsonl(root / "files.jsonl"))
    errors = list(read_jsonl(root / "errors.jsonl"))

    successful_urls = {norm_url(row.get("url", "")) for row in files if row.get("url")}
    # Preserve the append-only error log, but do not count old transient failures
    # as outstanding when a later retry downloaded the same canonical URL.
    unresolved_errors = [row for row in errors
                         if row.get("url") and norm_url(row.get("url", "")) not in successful_urls]
    error_urls = {norm_url(row.get("url", "")) for row in unresolved_errors if row.get("url")}
    download_links = {norm_url(row.get("url", "")) for row in links
                      if row.get("is_download") and row.get("url")}
    missing = sorted(download_links - successful_urls - error_urls)

    type_rows = {}
    for row in files:
        ctype = row.get("content_type") or "unknown"
        stat = type_rows.setdefault(ctype, {"count": 0, "bytes_recorded": 0})
        stat["count"] += 1
        stat["bytes_recorded"] += int(row.get("size_bytes") or 0)

    page_topics, file_topics = Counter(), Counter()
    page_hosts, file_hosts = Counter(), Counter()
    for row in pages:
        page_hosts[urlparse(row.get("url", "")).hostname or "unknown"] += 1
        page_topics.update(row.get("topics") or [])
    for row in files:
        file_hosts[urlparse(row.get("url", "")).hostname or "unknown"] += 1
        file_topics.update(row.get("topics") or [])

    categories = Counter()
    error_unique: dict[str, set[str]] = {}
    for row in unresolved_errors:
        message = row.get("error", "")
        if message.startswith("404"):
            category = "404 not found"
        elif "timed out" in message.casefold() or "timeout" in message.casefold():
            category = "timeout"
        elif "больше лимита" in message.casefold() or "превысил лимит" in message.casefold():
            category = "file size limit"
        elif row.get("_parse_error"):
            category = "JSON parse error"
        else:
            category = "other request error"
        categories[category] += 1
        error_unique.setdefault(category, set()).add(norm_url(row.get("url", "")))

    pdf_404 = sorted({row.get("url", "") for row in unresolved_errors
                      if row.get("error", "").startswith("404")
                      and Path(urlparse(row.get("url", "")).path).suffix.casefold() == ".pdf"})
    oversized = {}
    for row in unresolved_errors:
        if "больше лимита" in row.get("error", "").casefold():
            url = row.get("url", "")
            oversized[url] = max(oversized.get(url, 0), int(row.get("content_length") or 0))

    files_dir = root / "files"
    physical_paths = [p for p in files_dir.rglob("*") if p.is_file()]
    physical_bytes = sum(p.stat().st_size for p in physical_paths)
    recorded_paths = {row.get("stored_path") for row in files if row.get("stored_path")}
    existing_recorded = {str((root / rel).resolve()) for rel in recorded_paths
                         if (root / rel).is_file()}
    physical_set = {str(p.resolve()) for p in physical_paths}
    missing_paths = sorted(rel for rel in recorded_paths if not (root / rel).is_file())
    untracked_count = len(physical_set - existing_recorded)

    lines = [
        "# Аудит выгрузки сайта НГТУ",
        "",
        f"Сформировано: {datetime.now(timezone.utc).isoformat()}",
        f"Каталог: `{root}`",
        "",
        "## Объём и покрытие",
        "",
        f"- Строк страниц: {len(pages):,}; уникальных URL: {len({norm_url(x.get('url','')) for x in pages if x.get('url')}):,}.",
        f"- Строк ссылок: {len(links):,}; уникальных URL всех ссылок: {len({norm_url(x.get('url','')) for x in links if x.get('url')}):,}.",
        f"- Уникальных ссылок на файлы: {len(download_links):,}; успешно загружены: {len(download_links & successful_urls):,}; записаны как ошибка: {len(download_links & error_urls):,}; пока без результата: {len(missing):,}.",
        f"- Строк манифеста файлов: {len(files):,}; уникальных URL: {len(successful_urls):,}; SHA-256: {len({x.get('sha256') for x in files if x.get('sha256')}):,}.",
        f"- Физических файлов: {len(physical_paths):,}; размер каталога: {gib(physical_bytes)}.",
        f"- Путь из манифеста отсутствует на диске: {len(missing_paths):,}; файлов без записи манифеста: {untracked_count:,}.",
        "",
        "## Форматы по строкам манифеста",
        "",
        "| MIME | Строк | Учтённый объём (с повторными ссылками) |",
        "|---|---:|---:|",
    ]
    for mime, stats in sorted(type_rows.items(), key=lambda item: -item[1]["count"]):
        lines.append(f"| `{mime}` | {stats['count']:,} | {gib(stats['bytes_recorded'])} |")
    lines += ["", "## Ошибки", "", "| Категория | Строк | Уникальных URL |", "|---|---:|---:|"]
    for category, count in categories.most_common():
        lines.append(f"| {category} | {count:,} | {len(error_unique[category]):,} |")
    lines += ["", "### PDF с 404", ""]
    lines += [f"- `{url}`" for url in pdf_404] or ["Нет."]
    lines += ["", "### Пропущенные из-за ограничения размера", ""]
    lines += [f"- `{url}` — {gib(size)}" for url, size in sorted(oversized.items())] or ["Нет."]
    lines += ["", "## Метки тем (эвристика по заголовкам и тексту)", "",
              "Страницы: " + (", ".join(f"`{key}` {value:,}" for key, value in page_topics.most_common()) or "нет"),
              "", "Файлы: " + (", ".join(f"`{key}` {value:,}" for key, value in file_topics.most_common()) or "нет"),
              "", "## Домены", "", "Страницы: " + ", ".join(f"`{k}` {v:,}" for k, v in page_hosts.most_common(15)),
              "", "Файлы: " + ", ".join(f"`{k}` {v:,}" for k, v in file_hosts.most_common(15)), ""]

    report = root / "audit_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "pages_rows": len(pages), "pages_unique_urls": len({norm_url(x.get("url", "")) for x in pages if x.get("url")}),
        "links_rows": len(links), "download_links_unique": len(download_links),
        "downloaded_urls": len(successful_urls), "download_links_with_errors": len(download_links & error_urls),
        "download_links_without_result": len(missing), "file_manifest_rows": len(files),
        "file_urls_unique": len(successful_urls), "hashes_unique": len({x.get("sha256") for x in files if x.get("sha256")}),
        "physical_files": len(physical_paths), "physical_bytes": physical_bytes,
        "missing_local_paths": missing_paths, "untracked_physical_files": untracked_count,
        "errors": {k: {"rows": v, "unique_urls": len(error_unique[k])} for k, v in categories.items()},
        "404_pdf_urls": pdf_404, "oversized_urls": oversized, "undownloaded_link_urls": missing,
        "content_types": type_rows, "page_topics": dict(page_topics), "file_topics": dict(file_topics),
        "page_hosts": dict(page_hosts), "file_hosts": dict(file_hosts),
    }
    (root / "audit_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report)
    print("\n".join(lines[:24]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
