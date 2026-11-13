"""Small authenticated web UI for the already built NNTU RAG index."""
from __future__ import annotations

import argparse
import json
import logging
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from kb import answer, load_kb

LOG = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 8192
LANGUAGES = {"auto", "ru", "en", "tk", "uz"}
DEPTHS = {
    "normal": dict(top_k=6, candidate_k=35, neighbors=0, max_context_chars=7000,
                   num_ctx=8192, num_predict=350),
    "detailed": dict(top_k=10, candidate_k=60, neighbors=1, max_context_chars=14000,
                     num_ctx=16384, num_predict=700),
}

PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>НГТУ · Поиск по базе знаний</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;background:#f3f6fb;color:#17243a;margin:0}
main{max-width:850px;margin:2rem auto;padding:0 1rem}h1{font-size:1.65rem;margin-bottom:.25rem}
.sub{color:#53647c;margin-top:0}.card{background:#fff;border-radius:14px;padding:1.2rem;margin:1rem 0;box-shadow:0 2px 14px #1e3a5f12}
label{display:block;font-weight:600;margin:.7rem 0 .3rem}textarea,input,select,button{box-sizing:border-box;font:inherit}
textarea,input,select{width:100%;padding:.7rem;border:1px solid #b8c7d9;border-radius:8px;background:#fff}
textarea{min-height:100px;resize:vertical}.row{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
button{background:#0759a7;color:white;border:0;border-radius:8px;padding:.7rem 1.1rem;margin-top:1rem;cursor:pointer}
button:disabled{opacity:.6;cursor:wait}.answer{white-space:pre-wrap}.sources{padding-left:1.4rem}
.sources li{margin:.6rem 0;overflow-wrap:anywhere}.error{color:#a5222e}.muted{color:#617089;font-size:.9rem}
@media(max-width:600px){.row{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Вопросы об НГТУ</h1><p class="sub">Поиск по открытым страницам и документам университета. Проверяйте даты и первоисточники.</p>
<form id="form" class="card">
<label for="token">Код доступа</label><input id="token" type="password" autocomplete="off" required>
<label for="question">Ваш вопрос</label><textarea id="question" maxlength="1500" placeholder="Например: где найти расписание занятий?" required></textarea>
<div class="row"><div><label for="language">Язык ответа</label><select id="language"><option value="auto">Определить автоматически</option><option value="ru">Русский</option><option value="en">English</option><option value="tk">Türkmen</option><option value="uz">O‘zbek</option></select></div>
<div><label for="depth">Глубина поиска</label><select id="depth"><option value="normal">Обычная</option><option value="detailed">Подробная</option></select></div></div>
<button id="submit" type="submit">Получить ответ</button><span id="status" class="muted" role="status"></span>
</form><section id="result" class="card" hidden><h2>Ответ</h2><div id="answer" class="answer"></div><h3>Источники</h3><ol id="sources" class="sources"></ol><p id="timing" class="muted"></p></section>
<p class="muted">Ответ создаётся моделью на основе найденных фрагментов и может содержать ошибки. Источники открываются по ссылкам ниже.</p>
</main><script>
const $=id=>document.getElementById(id); $('token').value=sessionStorage.getItem('nntu-chat-token')||'';
$('form').addEventListener('submit',async e=>{e.preventDefault();const button=$('submit');button.disabled=true;$('status').textContent='Ищу сведения и готовлю ответ…';$('result').hidden=true;
try{const token=$('token').value.trim();sessionStorage.setItem('nntu-chat-token',token);
const response=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json','X-Chat-Token':token},body:JSON.stringify({question:$('question').value,language:$('language').value,depth:$('depth').value})});
const data=await response.json();if(!response.ok)throw Error(data.error||'Ошибка сервера');
$('answer').textContent=data.answer||'';const list=$('sources');list.replaceChildren();
for(const source of data.sources||[]){const li=document.createElement('li');const label='['+source.number+'] '+(source.label||'Источник');
if(source.url&&/^https?:\/\//i.test(source.url)){const a=document.createElement('a');a.href=source.url;a.target='_blank';a.rel='noopener noreferrer';a.textContent=label;li.append(a)}else li.textContent=label;list.append(li)}
$('timing').textContent='Поиск: '+data.retrieval_seconds+' с';$('result').hidden=false;$('status').textContent='Готово';
}catch(error){$('status').textContent='';$('answer').textContent=error.message;$('answer').className='answer error';$('sources').replaceChildren();$('timing').textContent='';$('result').hidden=false}
finally{button.disabled=false;setTimeout(()=>{$('answer').className='answer'},100)}});
</script></body></html>"""


def question_args(base: SimpleNamespace, language: str, depth: str) -> SimpleNamespace:
    return SimpleNamespace(**{**vars(base), **DEPTHS[depth], "language": language})


def public_sources(sources: list[dict]) -> list[dict]:
    return [{"number": item["number"],
             "label": item.get("citation_label") or item.get("title") or item.get("document_id"),
             "url": next((url for url in item.get("source_urls") or []
                          if isinstance(url, str) and url.startswith(("https://", "http://"))), None)}
            for item in sources]


def make_handler(kb, base: SimpleNamespace, token: str):
    lock = threading.Lock()  # One GPU inference at a time on this L4.

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status: HTTPStatus, content: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'")
            self.end_headers()
            self.wfile.write(content)

        def send_json(self, status: HTTPStatus, value: dict):
            self.send_bytes(status, json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            if self.path == "/":
                self.send_bytes(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
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
                if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1500:
                    raise ValueError("Вопрос должен содержать от 1 до 1500 символов")
                question = question.strip()
                if language not in LANGUAGES or depth not in DEPTHS:
                    raise ValueError("Неизвестный язык или глубина поиска")
            except (ValueError, AttributeError, TypeError):
                return self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Некорректный запрос"})
            if not lock.acquire(blocking=False):
                return self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Бот отвечает на другой вопрос. Повторите через минуту."})
            try:
                result = answer(kb, question, question_args(base, language, depth))
                self.send_json(HTTPStatus.OK, {
                    "answer": result["answer"], "sources": public_sources(result["sources"]),
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
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--llm-model", default="qwen3:4b")
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
    server = ThreadingHTTPServer((args.host, args.port), make_handler(kb, base, token))
    LOG.info("NNTU chat ready: %s chunks, %s:%s", len(kb.documents), args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
