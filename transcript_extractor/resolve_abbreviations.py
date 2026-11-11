#!/usr/bin/env python3
"""Дополнение словаря после NER/Schwartz–Hearst. Python 3.10+, только stdlib.

Не угадывает расшифровки: ищет их в исходных контекстах. См. README.md.

python .\resolve_abbreviations.py `
>>   --dictionary .\glasary\dictionary_schwartz_hearst.csv `
>>   --abbreviations .\glasary\abbreviations.csv `
>>   --chunks .\glasary\chunks.csv `
>>   --documents .\glasary\documents.csv `
>>   --output-dir .\glasary\resolved

"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

VERSION = "1.0.0"
METHOD = "explicit_definition_resolver_v1"
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
WORD = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9.+/_-]{1,31}")
STOP = set("a an the of for and to in on at by with or as de и в во на по для с со из от к о об при или над под".split())
CODE = set("NULL SELECT FROM WHERE INSERT UPDATE DELETE DROP ALTER CREATE TABLE VALUES VALUE SET WITH AS IF ELSE THEN WHEN CASE END BEGIN RETURN RETURNS OUT IN INTO ON OFF JOIN LEFT RIGHT INNER OUTER GROUP ORDER BY LIMIT HAVING DISTINCT TRUE FALSE CHECK DEFAULT INT INTEGER SMALLINT BIGINT VARCHAR CHAR TEXT DATE TIME TIMESTAMP FLOAT DOUBLE NUMERIC DECIMAL BOOLEAN BOOL INDEX VIEW FUNCTION PROCEDURE TRIGGER GRANT REVOKE VACUUM ANALYZE EXPLAIN SHOW USING UNION ALL ANY SOME NOT AND OR IS LIKE ILIKE BETWEEN EXISTS DO LOOP WHILE FOR TO OF ROW ROWS FETCH DECLARE COMMIT ROLLBACK SAVEPOINT CASCADE PRIMARY FOREIGN KEY UNIQUE REFERENCES CONSTRAINT EXECUTE CALL EXCEPTION NEW OLD ASC DESC NO YES NONE".casefold().split())
BAD_LONG = re.compile(r"https?://|www\.|[{}\[\]<>_=\\/@]|\b(?:SELECT|WHERE|INSERT|VARCHAR|RETURNS|sudo|return|const)\b", re.I)
BAD_WORDS = set("например пример примере рисунок таблица таблице см далее подробнее нажмите выберите установите запустите позволяет используется используется означает представляет содержит должен должна будет можно необходимо example see click set this these используется".split())
LANGUAGE = re.compile(r"^\s*(?:(?:от\s+)?(?:англ(?:ийск(?:ого|ий))?\.?|лат\.?|рус\.?|eng(?:lish)?\.?)(?=$|[\s,;:–—-])\s*[,;:–—-]?\s*|(?:сокр\.?\s+от|сокращение\s+от)\s+)", re.I)
DEFINE = re.compile(r"^\s*(?:расшифровывается\s+как|является\s+сокращением\s+от|сокращение\s+от|stands\s+for|is\s+short\s+for|означает)\s+", re.I)
SF_MARKER = re.compile(r"^\s*(?:далее(?:\s+по\s+тексту|\s+в\s+тексте)?|сокращ[её]нно|сокр\.?|hereinafter(?:\s+referred\s+to\s+as)?)\s*[-—–:,]?\s*", re.I)
GLOSSARY = re.compile(r"сокращени[яйе]|аббревиатур|расшифров|термин[ыов].{0,30}определени|abbreviations|acronyms|glossary", re.I)
ADDED_COLUMNS = ["status", "expansion_count", "expansions_json", "evidence_count", "evidence_refs_json", "confirmation_method"]
PAIR_COLUMNS = ["abbreviation", "canonical_key", "expansion", "status", "evidence_count", "patterns_json", "evidence_json"]
REVIEW_COLUMNS = ["abbreviation", "canonical_key", "proposed_expansion", "reason", "pattern", "reference", "quote", "evidence_json"]


def norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value or "").casefold().split())


def expansion_key(value: str) -> str:
    return " ".join(WORD.findall(norm(value)))


def row_key(row: dict) -> str:
    return norm(row.get("canonical_key") or row.get("abbreviation") or "")


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def json_list(value: str, label: str) -> list:
    if not value:
        return []
    try:
        result = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Некорректный JSON в {label}: {exc}") from exc
    if not isinstance(result, list):
        raise ValueError(f"В {label} ожидался JSON-массив")
    return result


def csv_reader(path: Path):
    stream = path.open(encoding="utf-8-sig", newline="")
    sample = stream.read(8192)
    stream.seek(0)
    # У многострочных контекстов Sniffer может ошибаться; заголовок однострочный.
    header = sample.splitlines()[0] if sample else ""
    delimiter = max(";\t,", key=header.count)
    return stream, csv.DictReader(stream, delimiter=delimiter)


def read_csv(path: Path, required: tuple[str, ...] = ()) -> tuple[list[str], list[dict]]:
    stream, reader = csv_reader(path)
    with stream:
        fields = reader.fieldnames or []
        missing = set(required) - set(fields)
        if missing:
            raise ValueError(f"{path}: отсутствуют столбцы {sorted(missing)}")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"{path}: число полей не совпадает с заголовком")
    return fields, rows


def short_reason(short: str) -> str:
    """Фильтр только для нового этапа; уже confirmed никогда не меняются."""
    if norm(short) in CODE:
        return "sql_or_code_keyword"
    if not re.fullmatch(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9.+/-]{1,23}", short):
        return "unsupported_short_form"
    letters = [c for c in short if c.isalpha()]
    if len(letters) < 2 or sum(c.isupper() for c in letters) < 2:
        return "not_acronym_shape"
    if re.search(r"[А-ЯЁ]{2}[а-яё]+$", short):
        return "inflected_short_form"
    return ""


def long_ok(short: str, phrase: str) -> bool:
    tokens = WORD.findall(phrase)
    if not 2 <= len(tokens) <= 14 or len(phrase) > 220 or BAD_LONG.search(phrase):
        return False
    if any(len(w) > 36 or w.isdigit() for w in tokens):
        return False
    if any(norm(w) in BAD_WORDS or norm(w) == norm(short) for w in tokens):
        return False
    if sum(w.isupper() and len(w) > 1 for w in tokens) == len(tokens):
        return False
    if any(len(w) == 1 and norm(w) not in STOP for w in tokens):
        return False
    # Не превращаем законченные предложения и отрицания в именные термины.
    if re.search(r"\b(?:не|нет|not|without|котор\w*|это|такой|такая|такие|этот|эта)\b", phrase, re.I):
        return False
    return True


def alignment(short: str, phrase: str, allow_internal: bool = False) -> str | None:
    """Все значимые слова обязательны. Пропускаются только служебные слова.

    В строгом режиме используются начала слов и CamelCase. В сильной
    конструкции дополнительно допускаются до двух внутрисловных букв
    (либо пропуск начальной e/i в латинском слове):
    Secure SHell, Binary Large OBject, GeneralIzed Search Tree.
    Это ограниченное выравнивание, а не поиск произвольной подпоследовательности.
    """
    if not long_ok(short, phrase):
        return None
    chars = "".join(c for c in norm(short) if c.isalnum())
    raw_words = WORD.findall(phrase)
    # CamelCase — структурная граница, а не свободное совпадение внутри слова.
    words = []
    for word in raw_words:
        words.extend(re.sub(r"(?<=[a-zа-яё])(?=[A-ZА-ЯЁ])", " ", word).split())
    surface_words = tuple(words)
    words = tuple(norm(w) for w in words)

    @lru_cache(None)
    def solve(i: int, j: int, budget: int) -> bool:
        if i == len(words):
            return j == len(chars)
        word = words[i]
        if word in STOP and solve(i + 1, j, budget):
            return True
        if j >= len(chars):
            return False
        # XML: eXtensible, GIN: iNdex. Не дальше второй буквы слова,
        # только в сильной конструкции, с расходом общего бюджета.
        if word[0] != chars[j]:
            if (allow_internal and budget > 0 and len(word) > 1
                    and word[0] in "ei" and word[1] == chars[j]
                    and word not in STOP and not surface_words[i].isupper()):
                return solve(i + 1, j + 1, budget - 1)
            return False
        if solve(i + 1, j + 1, budget):
            return True
        if not allow_internal or budget <= 0:
            return False
        # Каждое слово потребляет не более трёх букв SF, первая — начальная.
        pos = 0
        for length in range(2, min(3, len(chars) - j, budget + 1) + 1):
            pos = word.find(chars[j + length - 1], pos + 1)
            if pos < 0:
                break
            # Последнее слово нельзя дописывать за автора: Client Access (CAS)
            # не содержит слова Server; Oldest active (OAT) — Transaction.
            # SHell/IDentifier/OBject разрешены как буквальный префикс.
            if i == len(words) - 1 and not word.startswith(chars[j:j+length]):
                continue
            if solve(i + 1, j + length, budget - length + 1):
                return True
        return False

    if solve(0, 0, 0):
        return "word_initials"
    if allow_internal and len(chars) >= 3 and solve(0, 0, min(2, len(chars) // 2)):
        return "bounded_internal_letters"
    return None


@dataclass(frozen=True)
class Proposal:
    key: str
    short: str
    expansion: str
    pattern: str
    start: int
    end: int
    accepted: bool
    reason: str


def strip_language(text: str, offset: int) -> tuple[str, int]:
    original = text
    text = text.lstrip(" \t\r\n\"'«")
    offset += len(original) - len(text)
    m = LANGUAGE.match(text)
    if m:
        offset += m.end()
        text = text[m.end():]
        original = text
        text = text.lstrip(" \t\r\n\"'«")
        offset += len(original) - len(text)
    return text, offset


def head_phrase(text: str, offset: int) -> tuple[str, int, int]:
    """Часть до перевода/комментария, без нормализации исходных координат."""
    text, offset = strip_language(text, offset)
    boundary = re.search(r"[,;:!?()\[\]]|[.]\s|[.]$|\s+[—–-]\s+|\n\s*\n|\s+[«\"]", text)
    end = boundary.start() if boundary else len(text)
    raw = text[:end].rstrip(" \t\r\n.\"'»")
    return raw, offset, offset + len(raw)


def make_proposal(key: str, short: str, phrase: str, pattern: str,
                  start: int, end: int, *, internal: bool, review: bool = True) -> Proposal | None:
    if not long_ok(short, phrase):
        return None
    matched = alignment(short, phrase, internal)
    if matched:
        return Proposal(key, short, " ".join(phrase.split()), pattern, start, end, True, matched)
    if review:
        return Proposal(key, short, " ".join(phrase.split()), pattern, start, end, False, "no_reliable_letter_alignment")
    return None


def extract(text: str, eligible: dict[str, str]) -> list[Proposal]:
    result: list[Proposal] = []
    table_spans: list[tuple[int, int]] = []

    # Сериализованные PDF-таблицы: именованные столбцы или «Столбец N».
    for line in re.finditer(r"[^\n]+", text):
        cells = list(re.finditer(r"(?:^|;)\s*([^:;]{1,90}):\s*([^;]*)", line.group()))
        nonempty = [c for c in cells if c.group(2).strip()]
        if len(nonempty) < 2:
            continue
        first = nonempty[0]
        short = first.group(2).strip()
        key = norm(short)
        if key not in eligible or short_reason(short):
            continue
        value = nonempty[1]
        header = norm(first.group(1) + " " + value.group(1))
        if not re.search(r"столбец|сокращ|аббрев|термин|обознач|расшиф|определ|описан", header):
            continue
        a = line.start() + first.start(2)
        value_start = line.start() + value.start(2)
        phrase, _, b = head_phrase(value.group(2), value_start)
        table_is_glossary = bool(GLOSSARY.search(header) or GLOSSARY.search(text[max(0,line.start()-700):line.start()]))
        p = make_proposal(key, short, phrase, "table_row", a, b, internal=True, review=table_is_glossary)
        if p:
            result.append(p)
        table_spans.append((line.start(), line.end()))

    for token in TOKEN.finditer(text):
        short = token.group().rstrip(".")
        key = norm(short)
        if key not in eligible or short_reason(short):
            continue
        start, end = token.start(), token.start() + len(short)
        # Никаких частичных совпадений внутри пути, кода, C_API или API-key.
        if (start and (text[start-1].isalnum() or text[start-1] in "_/\\.")) or (end < len(text) and (text[end].isalnum() or text[end] in "_/\\")):
            continue
        if any(a <= start < b for a, b in table_spans):
            continue
        right = text[end:end + 450]

        # SF (LF[, перевод]) и SF (от англ. LF).
        paren = re.match(r"\s*\(\s*([^()]{1,280})\)", right)
        if paren:
            content = paren.group(1)
            phrase, _, b = head_phrase(content, end + paren.start(1))
            explicit_hint = bool(LANGUAGE.match(content) or GLOSSARY.search(text[max(0,start-500):start]))
            p = make_proposal(key, short, phrase, "SF(parenthesized_LF)", start, end + paren.end(), internal=True, review=explicit_hint)
            if p:
                result.append(p)

        # LF (SF), LF (далее — SF). Слева не перескакиваем через предложение.
        left = text[max(0, start-350):start]
        open_paren = left.rfind("(")
        if open_paren >= 0 and re.match(r"\s*\)", right):
            middle = left[open_paren+1:]
            if not middle.strip() or SF_MARKER.fullmatch(middle):
                prefix = left[:open_paren].rstrip()
                cut = list(re.finditer(r"[.!?;:]\s+|\n\s*\n|[()]", prefix))
                if cut:
                    prefix = prefix[cut[-1].end():]
                words = list(WORD.finditer(prefix))
                for w in reversed(words[-14:]):
                    phrase = prefix[w.start():].strip(" \t\r\n«»\"'—–-")
                    match = alignment(short, phrase, True)
                    if match:
                        # Точное начало находим по подстроке; не по обрезанным пробелам.
                        a = text.rfind(phrase, max(0, start-350), start)
                        result.append(Proposal(key, short, " ".join(phrase.split()), "LF(parenthesized_SF)", a, end + re.match(r"\s*\)", right).end(), True, match))
                        break

        # SF — LF / SF: LF / SF stands for LF.
        connector = re.match(r"\s*(?:[—–:]\s*|-\s+|=\s*)", right)
        defining = DEFINE.match(right)
        if connector or defining:
            delimiter = defining or connector
            raw_start = end + delimiter.end()
            phrase, _, b = head_phrase(text[raw_start:raw_start+260], raw_start)
            # Тире в обычном предложении слабее скобок/таблицы.
            p = make_proposal(key, short, phrase, "definition_marker" if defining else "SF_separator_LF", start, b, internal=bool(defining), review=bool(defining))
            if p:
                result.append(p)

        # Словарь без скобок: «DDL Data Definition Language, Язык ...».
        # Запрещён произвольный отбор нескольких удобных слов из предложения.
        if re.match(r"\s+[A-Za-zА-Яа-яЁё]", right) and not defining:
            raw_start = end + len(right) - len(right.lstrip())
            raw = text[raw_start:raw_start+260]
            phrase, a, b = head_phrase(raw, raw_start)
            words = WORD.findall(phrase)
            glossary_near = bool(GLOSSARY.search(text[max(0, start-800):min(len(text), end+300)]))
            title_case = len(words) >= 2 and all(w[0].isupper() or norm(w) in STOP for w in words)
            # В обычном тексте обязательно видимое завершение LF и Title Case.
            terminated = b < len(text) and text[b:b+1] in ",;\n(—–."
            if glossary_near or (title_case and terminated):
                p = make_proposal(key, short, phrase, "glossary_plain", start, b, internal=glossary_near, review=False)
                if p:
                    result.append(p)

    # Одно и то же место не получает несколько голосов за разные шаблоны.
    seen = set()
    unique = []
    for p in result:
        k = p.key, expansion_key(p.expansion), p.start, p.accepted
        if k not in seen and p.start >= 0 and p.end > p.start:
            seen.add(k)
            unique.append(p)
    return unique


@dataclass
class Context:
    text: str
    chunk_id: str
    document_id: str
    title: str
    page: str
    kind: str
    chunk_ids: list[str]
    source_type: str
    source_row: int
    exact_chunk: bool = False
    is_ocr: bool = False


def contexts_from_ner(path: Path, chunks: dict, documents: dict, stats: Counter) -> Iterator[Context]:
    stream, reader = csv_reader(path)
    required = {"chunk_id", "context", "document_id"}
    if not required.issubset(reader.fieldnames or []):
        stream.close()
        raise ValueError(f"{path}: нужны столбцы {sorted(required)}")
    seen = set()
    with stream:
        for number, row in enumerate(reader, 1):
            stats["ner_rows"] += 1
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"{path}, запись {number}: неверное число полей")
            chunk_id = row["chunk_id"]
            text = row["context"]
            if not chunk_id or not text.strip():
                stats["empty_contexts"] += 1
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            identity = row["document_id"], chunk_id, digest
            if identity in seen:
                continue
            seen.add(identity)
            meta = chunks.get(chunk_id, {})
            doc_id = row["document_id"] or meta.get("document_id", "")
            doc = documents.get(doc_id, {})
            if documents and not doc:
                stats["unknown_documents"] += 1
                continue
            if chunks and not meta:
                stats["unknown_anchor_chunks"] += 1
                continue
            if meta.get("document_id") and doc_id != meta["document_id"]:
                stats["mismatched_document_ids"] += 1
                continue
            if meta.get("status") not in (None, "", "ok") or doc.get("status") not in (None, "", "ok", "complete"):
                stats["non_ok_sources"] += 1
                continue
            ids = json_list(row.get("context_chunk_ids", ""), f"context_chunk_ids, запись {number}")
            if not ids:
                ids = [x for x in (row.get("prev_chunk_id"), chunk_id, row.get("next_chunk_id")) if x]
            ids = list(dict.fromkeys(str(x) for x in ids))
            if chunk_id not in ids:
                ids.append(chunk_id)
            if any(chunks.get(x, {}).get("document_id", doc_id) != doc_id for x in ids):
                stats["cross_document_contexts"] += 1
                continue
            if chunks and any(x not in chunks or chunks[x].get("status") not in (None, "", "ok") for x in ids):
                stats["unknown_or_failed_context_neighbors"] += 1
                continue
            stats["ner_contexts_scanned"] += 1
            yield Context(text, chunk_id, doc_id, row.get("title") or meta.get("title") or doc.get("title") or doc_id,
                          row.get("page") or meta.get("page", ""), row.get("kind") or meta.get("kind", ""), ids,
                          "ner_context", number)


def contexts_from_jsonl(path: Path, stats: Counter) -> Iterator[Context]:
    """Не склеивает разные чанки: каждый получает собственное точное доказательство."""
    seen = set()
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("metadata") or {}
            row = {**meta, **row}
            chunk_id, text = str(row.get("chunk_id") or ""), row.get("text") or row.get("page_content") or ""
            if not chunk_id or not isinstance(text, str):
                raise ValueError(f"{path}:{number}: нужны chunk_id и строковый text/page_content")
            if chunk_id in seen:
                raise ValueError(f"{path}:{number}: повторный chunk_id {chunk_id}")
            seen.add(chunk_id)
            if not text.strip() or row.get("indexable") is False or row.get("document_status") not in (None, "ok", "complete"):
                stats["jsonl_skipped"] += 1
                continue
            stats["jsonl_contexts_scanned"] += 1
            yield Context(text, chunk_id, str(row.get("document_id") or ""), str(row.get("source_filename") or row.get("title") or row.get("document_id") or ""),
                          str(row.get("page") or ""), str(row.get("kind") or ""), [chunk_id], "chunks_jsonl", number, True, bool(row.get("is_ocr")))


def evidence(p: Proposal, ctx: Context, chunks: dict) -> dict:
    ids = ctx.chunk_ids
    refs = []
    for cid in ids:
        m = chunks.get(cid, {})
        refs.append({"chunk_id": cid, "document_id": m.get("document_id") or ctx.document_id,
                     "title": m.get("title") or ctx.title,
                     "page": m.get("page") or (ctx.page if cid == ctx.chunk_id else "")})
    if ctx.exact_chunk:
        reference = f"{ctx.title} | PDF page {ctx.page} | chunk {ctx.chunk_id}"
    else:
        # Контекст NER содержит соседей. Страница якоря НЕ является доказанной
        # страницей определения; список всех чанков позволяет открыть источник.
        parts = [f"{r['chunk_id']} (PDF page {r['page'] or '?'})" for r in refs]
        reference = f"{ctx.title} | NER context: " + ", ".join(parts)
    return {"document_id": ctx.document_id, "title": ctx.title,
            "page": ctx.page if ctx.exact_chunk else "", "chunk_id": ctx.chunk_id,
            "anchor_page": ctx.page, "citation_scope": "exact_chunk" if ctx.exact_chunk else "context_group",
            "context_chunk_ids": ids, "source_chunks": refs, "pattern": p.pattern,
            "alignment": p.reason, "matched_expansion": p.expansion, "reference": reference,
            "quote": ctx.text[p.start:p.end], "quote_start": p.start, "quote_end": p.end,
            "offset_scope": "context" if not ctx.exact_chunk else "text",
            "context": ctx.text[max(0, p.start-90):min(len(ctx.text), p.end+90)],
            "source_type": ctx.source_type, "source_record": ctx.source_row,
            "context_sha256": hashlib.sha256(ctx.text.encode()).hexdigest()}


class Collector:
    def __init__(self, max_evidence: int):
        self.max_evidence = max_evidence
        self.found: dict[str, dict[str, dict]] = defaultdict(dict)
        self.reviews: dict[tuple, dict] = {}

    def add(self, p: Proposal, ctx: Context, chunks: dict):
        ev = evidence(p, ctx, chunks)
        source_kinds = [ctx.kind] + [chunks.get(cid, {}).get("kind", "") for cid in ctx.chunk_ids]
        ocr = ctx.is_ocr or any("ocr" in k.lower() or k.lower() == "image_text" for k in source_kinds)
        if not p.accepted or ocr:
            reason = "ocr_requires_review" if p.accepted else p.reason
            key = p.key, expansion_key(p.expansion), reason
            self.reviews.setdefault(key, {"abbreviation": p.short, "canonical_key": p.key, "proposed_expansion": p.expansion,
                                         "reason": reason, "pattern": p.pattern, "reference": ev["reference"], "quote": ev["quote"], "evidence_json": dumps(ev)})
            return
        k = expansion_key(p.expansion)
        rec = self.found[p.key].setdefault(k, {"variants": Counter(), "patterns": Counter(), "evidence": [], "seen": set()})
        # Overlap чанков и повторные NER-упоминания не наращивают число доказательств.
        # Одинаковая цитата в одном документе считается одним доказательством.
        signature = ctx.document_id, norm(ev["quote"])
        if signature in rec["seen"]:
            return
        rec["seen"].add(signature)
        rec["variants"][p.expansion] += 1
        rec["patterns"][p.pattern] += 1
        if len(rec["evidence"]) < self.max_evidence:
            rec["evidence"].append(ev)

    def expansions(self, key: str) -> list[dict]:
        records = []
        for rec in self.found.get(key, {}).values():
            variants = sorted(rec["variants"], key=lambda x: (-rec["variants"][x], len(x), x))
            records.append({"expansion": variants[0],
                            "variants": [{"text": x, "hits": rec["variants"][x]} for x in variants],
                            "evidence_count": len(rec["seen"]), "patterns": dict(rec["patterns"]),
                            "evidence": rec["evidence"], "confirmation_method": METHOD})
        return sorted(records, key=lambda x: (-x["evidence_count"], x["expansion"].casefold()))


def atomic_write(path: Path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig" if path.suffix.lower() == ".csv" else "utf-8", newline="") as stream:
            writer(stream)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_csv(path: Path, fields: list[str], rows):
    def write(stream):
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    atomic_write(path, write)


def run(args) -> dict:
    headers, rows = read_csv(args.dictionary, ("abbreviation",))
    if not rows:
        raise ValueError("Словарь пуст")
    keys = [row_key(r) for r in rows]
    if not all(keys) or len(keys) != len(set(keys)):
        raise ValueError("Пустые или повторные canonical_key: сначала исправьте входной словарь")
    stats = Counter()
    before = sum(norm(r.get("status", "")) == "confirmed" for r in rows)
    candidates = {row_key(r): r for r in rows if norm(r.get("status", "candidate")) in {"", "candidate"}}
    eligible = {}
    skipped = {}
    for key, row in candidates.items():
        reason = short_reason(row["abbreviation"])
        if reason:
            skipped[key] = reason
        else:
            eligible[key] = row["abbreviation"]
    chunks = {}
    if args.chunks:
        _, chunk_rows = read_csv(args.chunks, ("chunk_id", "document_id"))
        chunks = {r["chunk_id"]: r for r in chunk_rows}
        if len(chunks) != len(chunk_rows):
            raise ValueError("Повторные chunk_id в chunks.csv")
    documents = {}
    if args.documents:
        _, doc_rows = read_csv(args.documents, ("document_id",))
        documents = {r["document_id"]: r for r in doc_rows}
    collector = Collector(args.max_evidence)
    sources = []
    if args.abbreviations:
        sources.append(contexts_from_ner(args.abbreviations, chunks, documents, stats))
    if args.chunks_jsonl:
        sources.append(contexts_from_jsonl(args.chunks_jsonl, stats))
    scanned = 0
    for source in sources:
        for ctx in source:
            proposals = extract(ctx.text, eligible)
            stats["proposal_hits"] += len(proposals)
            for p in proposals:
                collector.add(p, ctx, chunks)
            scanned += 1
            if args.progress_every and scanned % args.progress_every == 0:
                print(f"Проверено контекстов: {scanned}; новых сокращений: {len(collector.found)}", file=sys.stderr)
    if not scanned:
        raise ValueError("Нет пригодных контекстов: проверьте входные файлы и соответствие chunk_id")

    output_rows, pairs, unresolved = [], [], []
    added = 0
    for base in rows:
        row = dict(base)
        key = row_key(row)
        expansions = collector.expansions(key)
        if key in candidates and expansions:
            # Существующие неподтвержденные гипотезы не получают confirmed вместе
            # с новой парой. Они остаются в исходном файле/отчёте проверки.
            old = json_list(row.get("expansions_json", ""), f"expansions_json {key}")
            if old:
                collector.reviews[(key, "old_hypotheses", "not_revalidated")] = {
                    "abbreviation": row["abbreviation"], "canonical_key": key,
                    "proposed_expansion": dumps(old), "reason": "previous_candidate_hypotheses_not_revalidated",
                    "pattern": "input_dictionary", "reference": "", "quote": "", "evidence_json": "[]"}
            row.update(status="confirmed", expansion_count=str(len(expansions)), expansions_json=dumps(expansions),
                       evidence_count=str(sum(x["evidence_count"] for x in expansions)),
                       evidence_refs_json=dumps(list(dict.fromkeys(ev["reference"] for x in expansions for ev in x["evidence"]))),
                       confirmation_method=METHOD)
            added += 1
            for x in expansions:
                pairs.append({"abbreviation": row["abbreviation"], "canonical_key": key,
                              "expansion": x["expansion"], "status": "confirmed", "evidence_count": x["evidence_count"],
                              "patterns_json": dumps(x["patterns"]), "evidence_json": dumps(x["evidence"])})
        elif key in candidates:
            if not row.get("status"):
                row["status"] = "candidate"
            has_review = any(k[0] == key for k in collector.reviews)
            reason = skipped.get(key) or ("only_unverified_definitions" if has_review else "no_explicit_definition_found")
            unresolved.append({"abbreviation": row["abbreviation"], "canonical_key": key,
                               "occurrences": row.get("occurrences", ""), "reason": reason})
        output_rows.append(row)

    fields = headers + [x for x in ADDED_COLUMNS if x not in headers]
    out = args.output_dir
    destinations = {"dictionary": args.output_dictionary or out / "dictionary_resolved.csv",
                    "pairs": out / "new_pairs.csv", "review": out / "review_pairs.csv",
                    "unresolved": out / "unresolved.csv", "summary": out / "resolver_summary.json"}
    input_paths = {x.resolve() for x in [args.dictionary, args.abbreviations, args.chunks, args.documents, args.chunks_jsonl] if x}
    paths = [p.resolve() for p in destinations.values()]
    if len(paths) != len(set(paths)) or set(paths) & input_paths:
        raise ValueError("Выходные файлы должны отличаться друг от друга и от входных. Используйте отдельный output-dir.")
    write_csv(destinations["dictionary"], fields, output_rows)
    write_csv(destinations["pairs"], PAIR_COLUMNS, pairs)
    reviews = sorted(collector.reviews.values(), key=lambda r: (r["canonical_key"], r["reason"], r["proposed_expansion"]))
    write_csv(destinations["review"], REVIEW_COLUMNS, reviews)
    write_csv(destinations["unresolved"], ["abbreviation", "canonical_key", "occurrences", "reason"], unresolved)
    summary = {"version": VERSION, "method": METHOD, "dictionary_rows": len(rows), "confirmed_before": before,
               "candidate_rows_scanned": len(candidates), "eligible_candidates": len(eligible),
               "skipped_by_shape_or_keyword": len(skipped), "newly_confirmed": added,
               "new_expansion_pairs": len(pairs), "confirmed_after": before+added,
               "unresolved_candidates": len(unresolved), "review_pairs": len(reviews),
               "unresolved_reasons": dict(Counter(x["reason"] for x in unresolved)), **dict(stats),
               "max_evidence_per_expansion": args.max_evidence,
               "outputs": {k: str(v) for k, v in destinations.items()},
               "notes": ["confirmed means an explicit textual pair passed deterministic rules; it is not a calibrated probability.",
                         "Existing confirmed rows are preserved verbatim by CSV field value; they are not revalidated.",
                         "NER context citations cover the full context_chunk_ids group, not just the anchor page.",
                         "OCR-only evidence and non-aligned definitions require review and never auto-confirm.",
                         "Different meanings are retained separately; no stemming or guessed translations are used.",
                         "Same quote in one document is deduplicated even across overlapping contexts."]}
    atomic_write(destinations["summary"], lambda stream: json.dump(summary, stream, ensure_ascii=False, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dictionary", required=True, type=Path, help="dictionary_schwartz_hearst.csv или исходный dictionary.csv")
    parser.add_argument("--abbreviations", type=Path, help="NER abbreviations.csv с колонкой context")
    parser.add_argument("--chunks", type=Path, help="Метаданные chunks.csv из той же NER-выгрузки")
    parser.add_argument("--documents", type=Path, help="Метаданные documents.csv")
    parser.add_argument("--chunks-jsonl", type=Path, help="Дополнительный полный корпус JSONL: text, chunk_id, document_id, page")
    parser.add_argument("--output-dir", type=Path, default=Path("resolved"))
    parser.add_argument("--output-dictionary", type=Path, help="Отдельный путь к обновленному словарю (по умолчанию output-dir/dictionary_resolved.csv)")
    parser.add_argument("--max-evidence", type=int, default=3, help="Максимум ссылок на одну новую расшифровку (по умолчанию 3)")
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args(argv)
    if not args.abbreviations and not args.chunks_jsonl:
        parser.error("Укажите --abbreviations и/или --chunks-jsonl. chunks.csv не содержит текстов.")
    if args.max_evidence < 1 or args.progress_every < 0:
        parser.error("max-evidence >= 1; progress-every >= 0")
    try:
        summary = run(args)
    except (ValueError, OSError, csv.Error) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")
    print(dumps({k: summary[k] for k in ["dictionary_rows", "confirmed_before", "newly_confirmed", "new_expansion_pairs", "confirmed_after", "unresolved_candidates"]}))


if __name__ == "__main__":
    main()
