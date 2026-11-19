"""Small authenticated web UI for the already built NNTU RAG index."""
from __future__ import annotations

import argparse
import json
import logging
import re
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from kb import answer, load_kb
from file_lookup import find_file

LOG = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 8192
LANGUAGES = {"auto", "ru", "en", "tk", "uz"}
DEPTHS = {
    "normal": dict(top_k=6, candidate_k=35, neighbors=0, max_context_chars=7000,
                   num_ctx=8192, num_predict=350),
    "detailed": dict(top_k=10, candidate_k=60, neighbors=1, max_context_chars=14000,
                     num_ctx=16384, num_predict=700),
}

STATIC_DIR = Path(__file__).resolve().parent / "static"
PAGE = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
MARK = (STATIC_DIR / "nntu-mark.png").read_bytes()


def question_args(base: SimpleNamespace, language: str, depth: str) -> SimpleNamespace:
    return SimpleNamespace(**{**vars(base), **DEPTHS[depth], "language": language})


def public_sources(sources: list[dict], answer_text: str) -> list[dict]:
    cited = {int(number) for number in re.findall(r"\[(\d+)\]", answer_text)}
    result, seen = [], set()
    for item in sources:
        if item["number"] not in cited:
            continue
        url = next((url for url in item.get("source_urls") or []
                    if isinstance(url, str) and url.startswith(("https://", "http://"))), None)
        page = item.get("page")
        if url and url.lower().split("?", 1)[0].endswith(".pdf") and isinstance(page, int) and page > 0:
            url = url.split("#", 1)[0] + f"#page={page}"
        key = (url, page)
        if key in seen:
            continue
        seen.add(key)
        result.append({"number": item["number"],
                       "label": item.get("citation_label") or item.get("title") or item.get("document_id"),
                       "url": url})
    return result


def make_handler(kb, base: SimpleNamespace, token: str, supplemental_kb=None, file_catalog=None):
    lock = threading.Lock()  # One GPU inference at a time on this L4.
    page = PAGE
    if supplemental_kb is not None:
        page = PAGE.replace(
            '<!-- SCOPE_SELECTOR -->',
            '<div><label for="scope">Где искать</label><select id="scope">'
            '<option value="core">Основная база</option>'
            '<option value="supplemental">Архив и специальные документы</option>'
            '</select></div>',
        )

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status: HTTPStatus, content: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'")
            self.end_headers()
            self.wfile.write(content)

        def send_json(self, status: HTTPStatus, value: dict):
            self.send_bytes(status, json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            if self.path == "/":
                self.send_bytes(HTTPStatus.OK, page.encode(), "text/html; charset=utf-8")
            elif self.path == "/nntu-mark.png":
                self.send_bytes(HTTPStatus.OK, MARK, "image/png")
            elif self.path == "/health":
                self.send_json(HTTPStatus.OK, {"ready": True, "chunks": len(kb.documents)})
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Не найдено"})

        def do_POST(self):
            if self.path != "/api/ask":
                return self.send_json(HTTPStatus.NOT_FOUND, {"error": "Не найдено"})
            supplied = self.headers.get("X-Chat-Token", "")
            if not secrets.compare_digest(supplied, token):
                return self.send_json(HTTPStatus.UNAUTHORIZED, {"error": "Неверный код доступа"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = 0
            if not 0 < size <= MAX_REQUEST_BYTES:
                return self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Слишком большой запрос"})
            try:
                data = json.loads(self.rfile.read(size))
                question = data.get("question", "")
                language = data.get("language", "auto")
                depth = data.get("depth", "normal")
                scope = data.get("scope", "core")
                if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1500:
                    raise ValueError("Вопрос должен содержать от 1 до 1500 символов")
                question = question.strip()
                if language not in LANGUAGES or depth not in DEPTHS:
                    raise ValueError("Неизвестный язык или глубина поиска")
                if scope not in ("core", "supplemental") or (
                    scope == "supplemental" and supplemental_kb is None
                ):
                    raise ValueError("Область поиска недоступна")
            except (ValueError, AttributeError, TypeError):
                return self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Некорректный запрос"})
            if not lock.acquire(blocking=False):
                return self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Бот отвечает на другой вопрос. Повторите через минуту."})
            try:
                file_result = find_file(question, file_catalog, scope) if file_catalog is not None else None
                if file_result is not None:
                    message, files = file_result
                    return self.send_json(HTTPStatus.OK, {
                        "answer": message, "sources": [],
                        "files": [{"title": file["title"], "url": file["url"]} for file in files],
                    })
                settings = question_args(base, language, depth)
                selected_kb = kb
                if scope == "supplemental":
                    settings.mode = "bm25"
                    selected_kb = supplemental_kb
                result = answer(selected_kb, question, settings)
                self.send_json(HTTPStatus.OK, {
                    "answer": result["answer"], "sources": public_sources(result["sources"], result["answer"]),
                    "retrieval_seconds": result["retrieval_seconds"],
                    "answer_language": result["answer_language"],
                })
            except Exception:
                LOG.exception("Не удалось ответить на вопрос")
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Не удалось сформировать ответ. Проверьте журнал службы."})
            finally:
                lock.release()

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Локальный веб-интерфейс к индексу НГТУ")
    parser.add_argument("--index", required=True)
    parser.add_argument("--supplemental-chunks", help="JSONL для адресного поиска по архивным материалам")
    parser.add_argument("--file-catalog", type=Path, help="JSONL каталога оригиналов")
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--llm-model", default="qwen3:4b-instruct")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = Path(args.token_file).read_text(encoding="utf-8").strip()
    if len(token) < 24:
        parser.error("Код доступа должен содержать не менее 24 символов")
    base = SimpleNamespace(index=args.index, chunks=None, mode="hybrid", device=args.device,
                           local_files_only=True, bm25_weight=.5, faiss_weight=.5, rrf_k=60,
                           source_folder=None, document_id=None, kind=None,
                           llm_model=args.llm_model, ollama_url=args.ollama_url)
    kb = load_kb(base)
    supplemental_kb = None
    if args.supplemental_chunks:
        supplemental_kb = load_kb(SimpleNamespace(chunks=args.supplemental_chunks, mode="bm25"))
    file_catalog = None
    if args.file_catalog:
        with args.file_catalog.open(encoding="utf-8") as stream:
            file_catalog = [json.loads(line) for line in stream]
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(kb, base, token, supplemental_kb, file_catalog))
    LOG.info("NNTU chat ready: %s chunks, %s:%s", len(kb.documents), args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
