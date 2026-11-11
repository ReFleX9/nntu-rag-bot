#!/usr/bin/env python3
"""Сбор публичных страниц и файлов сайта НГТУ в JSONL + локальное хранилище.

Скрипт не входит в аккаунты и не обходит внешние сайты. Все найденные ссылки
сохраняются с подписью и контекстом (в том числе текстом строки HTML-таблицы).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import mimetypes
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urldefrag, urlencode, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

ROOT = "https://www.nntu.ru/"
DEFAULT_SEEDS = [
    "https://www.nntu.ru/",
    "https://www.nntu.ru/content/abit/common",
    "https://www.nntu.ru/content/studentam/raspisanie_zanyatiy",
    "https://www.nntu.ru/content/studentam/elektronnaya-obrazovatelnaya-sreda",
]
USER_AGENT = "NNTU-RAG-ResearchBot/0.1 (public pages; contact: project owner)"
TEXT_TYPES = {"text/html", "application/xhtml+xml"}
SKIP_SCHEMES = {"mailto", "tel", "javascript", "data", "sms"}
TRACKING = {
    "utm_source", "utm_medium", "utm_campaign", "yclid", "_ga", "f_acc", "?f_acc",
    "_wpnonce", "attachment_id", "dbutton", "pbutton", "obutton", "sbutton",
    "editbuttons", "v",
}
PRIVATE_HOSTS = {"lks.nntu.ru", "my.nntu.ru"}
TOPICS = {
    "admission": ("абитуриент", "поступлен", "приемная комиссия", "вступительн", "правила приема", "поступающ"),
    "study": ("студент", "расписание", "учебн", "образовательн", "сессия", "экзамен", "пересдач"),
    "campus_life": ("общежит", "стипенд", "профком", "студенческ", "питани", "спорт", "социальн"),
    "university": ("университет", "нгту", "институт", "направлени", "специальност", "кафедр", "контакт"),
}
PAGE_EXTENSIONS = {".html", ".htm", ".php", ".aspx", ".asp", ".jsp"}
FILE_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods",
    ".rtf", ".txt", ".csv", ".zip", ".rar", ".7z", ".xml", ".json", ".ics",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm_url(url: str) -> str:
    url = (url or "").strip()
    url, _ = urldefrag(url)
    p = urlparse(url)
    if p.path.casefold().endswith("/viewer.php"):
        embedded = dict(parse_qsl(p.query)).get("file")
        if embedded:
            return norm_url(embedded)
    if p.scheme not in {"http", "https"} or not p.hostname:
        return ""
    host = p.hostname.casefold()
    if host.startswith("www."):
        host = host[4:]
    scheme = "https" if host == "nntu.ru" or host.endswith(".nntu.ru") else p.scheme
    netloc = host + (f":{p.port}" if p.port and p.port not in (80, 443) else "")
    path = re.sub(r"/{2,}", "/", p.path or "/")
    # Percent-escape hex digits are case-insensitive (e.g. %d0 == %D0).
    path = re.sub(r"%[0-9a-fA-F]{2}", lambda match: match.group(0).upper(), path)
    if path != "/":
        path = path.rstrip("/")
    query_pairs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
                   if k.casefold() not in TRACKING
                   and not (p.path.startswith("/news/") and k.casefold() == "tag")]
    query = urlencode(sorted(query_pairs))
    return urlunparse((scheme, netloc, path, "", query, ""))


def domain_allowed(url: str, allowed: set[str]) -> bool:
    host = (urlparse(url).hostname or "").casefold().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in allowed)


def public_host(url: str) -> bool:
    host = (urlparse(url).hostname or "").casefold().rstrip(".")
    return host not in PRIVATE_HOSTS


def text_of(node) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def page_text(soup: BeautifulSoup) -> str:
    soup = copy.deepcopy(soup)
    for node in soup.select("script, style, noscript, svg, template, nav, footer, header"):
        node.decompose()
    main = soup.select_one("main, article, [role=main], .content, .page-content") or soup.body or soup
    return text_of(main)


def topical_labels(text: str) -> list[str]:
    folded = text.casefold().replace("ё", "е")
    return [topic for topic, words in TOPICS.items() if any(w.replace("ё", "е") in folded for w in words)]


def write_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_seen(path: Path) -> set[str]:
    seen = set()
    if not path.exists():
        return seen
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                value = json.loads(line)["url"]
                seen.add(norm_url(value) or value)
            except (ValueError, KeyError):
                continue
    return seen


def clean_filename(value: str) -> str:
    value = re.sub(r"[^\w.()-]+", "_", value, flags=re.UNICODE).strip("._")
    return value[:140] or "download"


class Crawler:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.output).resolve()
        self.pages_dir = self.out / "pages"
        self.files_dir = self.out / "files"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.allowed = {d.casefold().strip(".") for d in args.domains}
        self.robots: dict[str, RobotFileParser] = {}
        self.visited = load_seen(self.out / "pages.jsonl") if args.resume else set()
        self.file_seen = load_seen(self.out / "files.jsonl") if args.resume else set()
        self.page_count = 0
        self.file_count = 0
        self.max_total_bytes = int(args.max_total_gb * 1024 ** 3)
        self.stored_bytes = sum(p.stat().st_size for p in self.files_dir.rglob("*") if p.is_file())

    def write_summary(self, interrupted: bool = False) -> None:
        totals = {}
        for name in ("pages.jsonl", "links.jsonl", "files.jsonl", "errors.jsonl"):
            path = self.out / name
            count = 0
            if path.exists():
                with path.open(encoding="utf-8") as stream:
                    count = sum(1 for line in stream if line.strip())
            totals[name.removesuffix(".jsonl")] = count
        size = sum(p.stat().st_size for p in self.files_dir.glob("*") if p.is_file())
        summary = {
            "root": ROOT, "domains": sorted(self.allowed),
            "pages_this_run": self.page_count, "files_this_run": self.file_count,
            "records": totals, "stored_files_bytes": size,
            "visited_this_run": len(self.visited), "finished_at": now(),
            "interrupted": interrupted, "max_pages": self.args.max_pages,
            "max_depth": self.args.max_depth,
        }
        (self.out / "crawl_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def allowed_by_robots(self, url: str) -> bool:
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        if origin not in self.robots:
            rp = RobotFileParser()
            rp.set_url(origin + "/robots.txt")
            try:
                response = self.session.get(rp.url, timeout=self.args.timeout)
                if response.status_code == 200:
                    rp.parse(response.text.splitlines())
                else:
                    rp.parse([])
            except requests.RequestException as exc:
                logging.warning("robots.txt не прочитан (%s): %s", origin, exc)
                # При проблеме с robots.txt не начинаем массовый обход вслепую.
                rp.parse(["User-agent: *", "Disallow: /"])
            self.robots[origin] = rp
        return self.robots[origin].can_fetch(USER_AGENT, url)

    def sitemap_urls(self) -> list[str]:
        """Прочитать Sitemap-директивы robots.txt, если сайт их публикует."""
        found = set()
        sitemap_timeout = self.args.sitemap_timeout
        for domain in self.allowed:
            for scheme in ("https", "http"):
                robots_url = f"{scheme}://{domain}/robots.txt"
                try:
                    response = self.session.get(robots_url, timeout=sitemap_timeout)
                    if response.status_code != 200:
                        continue
                    for line in response.text.splitlines():
                        if line.strip().casefold().startswith("sitemap:"):
                            found.add(line.split(":", 1)[1].strip())
                except requests.RequestException:
                    continue
        urls = []
        for sitemap in found:
            if not domain_allowed(sitemap, self.allowed):
                continue
            try:
                response = self.session.get(sitemap, timeout=sitemap_timeout)
                response.raise_for_status()
                root = ET.fromstring(response.content)
                for node in root.iter():
                    if node.tag.rsplit("}", 1)[-1] == "loc" and node.text:
                        url = norm_url(node.text.strip())
                        if url and domain_allowed(url, self.allowed):
                            urls.append(url)
            except (requests.RequestException, ET.ParseError) as exc:
                logging.warning("Не удалось прочитать sitemap %s: %s", sitemap, exc)
        return urls

    def save_file(self, url: str, source_url: str, label: str, context: str, response) -> None:
        max_bytes = int(self.args.max_file_mb * 1024 * 1024)
        try:
            declared_size = int(response.headers.get("Content-Length", "0"))
        except ValueError:
            declared_size = 0
        if declared_size > max_bytes:
            response.close()
            write_jsonl(self.out / "errors.jsonl", {"url": url, "source_page": source_url,
                        "error": f"Пропущен файл больше лимита {self.args.max_file_mb} MiB",
                        "content_length": declared_size, "at": now()})
            return
        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                response.close()
                write_jsonl(self.out / "errors.jsonl", {"url": url, "source_page": source_url,
                            "error": f"Потоковый размер превысил лимит {self.args.max_file_mb} MiB", "at": now()})
                return
            chunks.append(chunk)
        content = b"".join(chunks)
        digest = hashlib.sha256(content).hexdigest()
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
        name = Path(urlparse(url).path).name or "download"
        suffix = Path(name).suffix.lower()
        if not suffix:
            suffix = mimetypes.guess_extension(content_type) or ".bin"
        stored_name = f"{digest[:16]}_{clean_filename(name)}"
        if not Path(stored_name).suffix:
            stored_name += suffix
        target = self.files_dir / stored_name
        if not target.exists():
            if self.stored_bytes + len(content) > self.max_total_bytes:
                write_jsonl(self.out / "errors.jsonl", {
                    "url": url, "source_page": source_url,
                    "error": f"Общий лимит корпуса {self.args.max_total_gb} GiB превышен",
                    "size_bytes": len(content), "stored_bytes": self.stored_bytes, "at": now(),
                })
                return
            target.write_bytes(content)
            self.stored_bytes += len(content)
        row = {
            "url": url, "source_page": source_url, "link_text": label,
            "link_context": context, "filename": name, "stored_path": str(target.relative_to(self.out)),
            "content_type": content_type, "size_bytes": len(content), "sha256": digest,
            "downloaded_at": now(), "http_status": response.status_code,
            "topics": topical_labels(" ".join((label, context, name))),
        }
        write_jsonl(self.out / "files.jsonl", row)
        self.file_seen.add(url)
        self.file_count += 1

    def visit(self, url: str, source_url: str = "", depth: int = 0) -> list[str]:
        url = norm_url(url)
        if not url or not domain_allowed(url, self.allowed) or not public_host(url) or url in self.visited:
            return []
        # Sitemap endpoints are parsed by sitemap_urls(); do not re-request them
        # as ordinary pages, since some dynamic maps respond very slowly.
        if urlparse(url).path.rstrip("/").casefold().endswith("/sitemap/get"):
            self.visited.add(url)
            return []
        if not self.allowed_by_robots(url):
            logging.info("robots disallow: %s", url)
            self.visited.add(url)
            return []
        try:
            response = self.session.get(url, timeout=self.args.timeout, stream=True, allow_redirects=True)
            response.raise_for_status()
        except requests.RequestException as exc:
            write_jsonl(self.out / "errors.jsonl", {"url": url, "source_page": source_url, "error": str(exc), "at": now()})
            self.visited.add(url)
            logging.warning("Не удалось загрузить %s: %s", url, exc)
            return []
        final_url = norm_url(response.url)
        if not domain_allowed(final_url, self.allowed):
            logging.info("Внешний редирект пропущен: %s", response.url)
            self.visited.add(url)
            return []
        self.visited.add(url)
        ctype = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
        if ctype not in TEXT_TYPES and not ("<html" in response.text[:500].lower() if ctype.startswith("text/") else False):
            self.save_file(final_url, source_url, "", "", response)
            time.sleep(self.args.delay)
            return []

        soup = BeautifulSoup(response.content, "html.parser")
        title = text_of(soup.title)
        body = page_text(soup)
        page_id = hashlib.sha256(final_url.encode()).hexdigest()[:20]
        html_path = self.pages_dir / f"{page_id}.html"
        html_path.write_bytes(response.content)
        row = {
            "url": final_url, "source_page": source_url, "depth": depth,
            "title": title, "text": body, "stored_html": str(html_path.relative_to(self.out)),
            "content_type": ctype, "http_status": response.status_code,
            "last_modified": response.headers.get("Last-Modified"), "etag": response.headers.get("ETag"),
            "crawled_at": now(), "topics": topical_labels(" ".join((title, body[:12000], final_url))),
        }
        write_jsonl(self.out / "pages.jsonl", row)
        self.page_count += 1
        discovered = []
        for anchor in soup.select("a[href]"):
            target = norm_url(urljoin(final_url, anchor.get("href", "")))
            if not target:
                continue
            label = text_of(anchor) or anchor.get("aria-label", "") or anchor.get("title", "")
            row_node = anchor.find_parent("tr")
            context_node = row_node or anchor.parent
            context = text_of(context_node)
            extension = Path(urlparse(target).path).suffix.casefold()
            file_link = extension in FILE_EXTENSIONS
            record = {"url": target, "source_page": final_url, "link_text": label,
                      "link_title": anchor.get("title", ""), "link_context": context,
                      "is_download": file_link, "found_at": now()}
            write_jsonl(self.out / "links.jsonl", record)
            if domain_allowed(target, self.allowed) and public_host(target) and file_link:
                if target not in self.file_seen and self.allowed_by_robots(target):
                    try:
                        file_response = self.session.get(target, timeout=self.args.timeout, stream=True, allow_redirects=True)
                        file_response.raise_for_status()
                        self.save_file(norm_url(file_response.url), final_url, label, context, file_response)
                    except requests.RequestException as exc:
                        write_jsonl(self.out / "errors.jsonl", {"url": target, "source_page": final_url, "error": str(exc), "at": now()})
            elif domain_allowed(target, self.allowed) and public_host(target) and (not extension or extension in PAGE_EXTENSIONS):
                discovered.append(target)
        time.sleep(self.args.delay)
        logging.info("Страниц: %d; файлов: %d; %s", self.page_count, self.file_count, final_url)
        return discovered

    def crawl(self):
        seeds = list(self.args.seeds)
        if self.args.url_list:
            with Path(self.args.url_list).open(encoding="utf-8-sig") as stream:
                seeds.extend(line.strip() for line in stream if line.strip() and not line.lstrip().startswith("#"))
        # The in-memory BFS queue disappears when a run is interrupted. Rebuild
        # its frontier from recorded links so --resume continues beyond the seeds.
        if self.args.resume and (self.out / "links.jsonl").exists():
            with (self.out / "links.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        link = json.loads(line)
                        target = norm_url(link.get("url", ""))
                        extension = Path(urlparse(target).path).suffix.casefold()
                        if target and (not extension or extension in PAGE_EXTENSIONS):
                            seeds.append(target)
                    except (ValueError, TypeError):
                        continue
        queue = deque((norm_url(s), 0, "") for s in seeds)
        queued = {u for u, _, _ in queue}
        for sitemap_url in self.sitemap_urls():
            if sitemap_url not in queued and sitemap_url not in self.visited:
                queue.append((sitemap_url, 0, "robots.txt sitemap"))
                queued.add(sitemap_url)
        while queue and self.page_count < self.args.max_pages:
            url, depth, source = queue.popleft()
            if depth > self.args.max_depth or url in self.visited:
                continue
            for child in self.visit(url, source, depth):
                if child not in queued and child not in self.visited and depth < self.args.max_depth:
                    queue.append((child, depth + 1, url))
                    queued.add(child)
        self.write_summary()
        print((self.out / "crawl_summary.json").read_text(encoding="utf-8"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="nntu_site_corpus", help="Каталог результатов")
    p.add_argument("--seed", dest="seeds", action="append", help="Стартовый URL; можно указать несколько")
    p.add_argument("--url-list", help="Файл: по одному публичному URL сайта в строке (например, из поисковой выдачи)")
    p.add_argument("--domain", dest="domains", action="append", help="Разрешённый домен; можно указать несколько")
    p.add_argument("--max-pages", type=int, default=5000)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--delay", type=float, default=0.8, help="Пауза между запросами в секундах")
    p.add_argument("--timeout", type=float, default=25)
    p.add_argument("--sitemap-timeout", type=float, default=8, help="Короткий тайм-аут для медленных sitemap")
    p.add_argument("--max-file-mb", type=float, default=1024, help="Пропустить файл, если он больше лимита")
    p.add_argument("--max-total-gb", type=float, default=50, help="Не превышать общий объём загруженных файлов")
    p.add_argument("--resume", action="store_true", help="Продолжить по уже сохранённым URL")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    # On resume, use the saved link frontier below instead of re-fetching the
    # original seed pages. Explicit --seed values still add new starting points.
    args.seeds = args.seeds if args.seeds is not None else ([] if args.resume or args.url_list else DEFAULT_SEEDS)
    args.domains = args.domains or ["nntu.ru"]
    if args.max_pages < 1 or args.max_depth < 0 or args.delay < 0 or args.max_file_mb <= 0 or args.sitemap_timeout <= 0 or args.max_total_gb <= 0:
        p.error("max-pages должен быть >0; max-depth/delay — неотрицательными; лимиты размера и времени — >0")
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    crawler = None
    try:
        crawler = Crawler(args)
        crawler.crawl()
        return 0
    except KeyboardInterrupt:
        if crawler is not None:
            crawler.write_summary(interrupted=True)
        print("Остановлено. Уже записанные JSONL-файлы сохранены; повторите с --resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
