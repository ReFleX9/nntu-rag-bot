"""PDF -> текст/таблицы/OCR -> очищенные чанки для RAG.

Версия 3 делает две вещи, которых не хватало предыдущему пайплайну:
1) автоматически исключает повторяющиеся колонтитулы/логотипы и служебные
   подписи вида «Рисунок 12»;
2) объединяет мелкие соседние текстовые блоки одной страницы и одного раздела,
   не требуя почти совпадающих координат колонок.

Embeddings и векторная БД остаются следующим этапом.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import re
import sys
import unicodedata
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import pymupdf
import pdf_media as _pdf_media

_extract = getattr(_pdf_media, "extract_pdf_media", None)
_parameters = inspect.signature(_extract).parameters if callable(_extract) else {}
_image_fields = getattr(getattr(_pdf_media, "ImageData", None), "__dataclass_fields__", {})
_missing_media = [name for name, present in (
    ("create_ocr_engine", callable(getattr(_pdf_media, "create_ocr_engine", None))),
    ("extract_pdf_media(..., on_page=...)", "on_page" in _parameters),
    ("extract_pdf_media(..., ocr_engine=...)", "ocr_engine" in _parameters),
    ("ImageData.rag_word_indices", "rag_word_indices" in _image_fields),
) if not present]
if _missing_media:
    raise ImportError(
        f"Несовместимый pdf_media.py: {getattr(_pdf_media, '__file__', '<unknown>')}. "
        f"Отсутствует: {', '.join(_missing_media)}. Замените pdf_media.py целиком."
    )

from pdf_media import MediaConfig, PageMedia, TableData, create_ocr_engine, extract_pdf_media
from text_cleaner import clean_text
from pipeline_metadata import (
    assign_sections,
    enrich_chunk,
    finalize_chunks,
    make_sections,
    resolve_document_metadata,
    scalar_metadata,
    stable_hash,
)

PIPELINE_VERSION = "3.2.0"


@dataclass
class PipelineConfig:
    media: MediaConfig = field(default_factory=MediaConfig)
    chunk_size_chars: int = 1800
    chunk_overlap_chars: int = 200
    min_chunk_chars: int = 350
    text_kind: str = "prose"
    known_words: tuple[str, ...] = ()
    join_wrapped_lines: bool = True
    include_partial: bool = False
    corpus_id: str = "default"
    tokenizer_name: str | None = None
    max_chunk_tokens: int | None = None
    remove_boilerplate: bool = True
    boilerplate_min_pages: int = 3
    boilerplate_min_fraction: float = 0.18
    filter_image_text: bool = True

    def validate(self) -> None:
        self.media.validate()
        if self.chunk_size_chars < 256:
            raise ValueError("chunk_size_chars должен быть не меньше 256")
        if not 0 <= self.chunk_overlap_chars < self.chunk_size_chars // 2:
            raise ValueError("chunk_overlap_chars должен быть меньше половины chunk_size_chars")
        if not 0 <= self.min_chunk_chars <= self.chunk_size_chars:
            raise ValueError("min_chunk_chars должен быть в диапазоне 0..chunk_size_chars")
        if self.text_kind not in {"unknown", "prose"}:
            raise ValueError("text_kind: unknown или prose")
        if self.join_wrapped_lines and self.text_kind != "prose":
            raise ValueError("join_wrapped_lines разрешён только для text_kind='prose'")
        if not isinstance(self.corpus_id, str) or not self.corpus_id.strip():
            raise ValueError("corpus_id должен быть непустой строкой")
        if self.max_chunk_tokens is not None and (
            type(self.max_chunk_tokens) is not int or self.max_chunk_tokens < 1
        ):
            raise ValueError("max_chunk_tokens должен быть положительным целым числом")
        if self.boilerplate_min_pages < 2:
            raise ValueError("boilerplate_min_pages должен быть >= 2")
        if not 0 < self.boilerplate_min_fraction <= 1:
            raise ValueError("boilerplate_min_fraction должен быть в диапазоне (0, 1]")


@dataclass
class PreparedDocument:
    document: dict[str, Any]
    chunks: list[dict[str, Any]]
    directory: Path
    document_path: Path
    chunks_path: Path


_PAGE_NUMBER_ONLY = re.compile(
    r"^\s*(?:(?:стр\.?|страница|page)\s*)?\d+(?:\s*(?:из|of|/)\s*\d+)?\.?\s*$",
    re.IGNORECASE,
)
_LABEL_ONLY = re.compile(
    r"^\s*(?:рисунок|figure|таблица|table)\s+\d+(?:[.\-]\d+)*(?:[а-яa-z])?\.?\s*$",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def _key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold().strip()


def _bbox(boxes: Iterable) -> list[float]:
    boxes = list(boxes)
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        round(min(b[0] for b in boxes), 3),
        round(min(b[1] for b in boxes), 3),
        round(max(b[2] for b in boxes), 3),
        round(max(b[3] for b in boxes), 3),
    ]


def _inside(box, region) -> bool:
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= x <= region[2] and region[1] <= y <= region[3]


def _table_regions(tables: list[TableData]):
    regions = []
    for table in tables:
        if table.duplicate_of:
            continue
        for cell in table.cells:
            if cell["bbox"] and cell["raw_value"]:
                regions.append((cell["bbox"], {_key(w) for w in cell["raw_value"].split()}, table.table_id))
        if table.header_external and table.header_bbox:
            words = {_key(w) for name in table.header_names if name for w in name.split()}
            regions.append((table.header_bbox, words, table.table_id))
    return regions


def _kept_intervals(mask: list[bool]) -> Iterator[tuple[int, int]]:
    start = None
    for index, keep in enumerate([*mask, False]):
        if keep and start is None:
            start = index
        elif not keep and start is not None:
            yield start, index
            start = None


def _text_block(*, block_id: str, kind: str, bbox, source_text: str,
                fragments: list[dict], config: PipelineConfig, **extra) -> dict:
    raw_lines = [source_text[f["start"]:f["end"]] for f in fragments]
    raw = "\n".join(raw_lines)
    cleaning_kind = config.text_kind if kind == "text" else "unknown"
    cleaned = clean_text(
        raw,
        kind=cleaning_kind,
        known_words=config.known_words,
        join_wrapped_lines=config.join_wrapped_lines if kind == "text" else False,
    )
    return {
        "block_id": block_id,
        "kind": kind,
        "bbox": list(bbox),
        "source_text": source_text,
        "raw_text": raw,
        "text": cleaned.text,
        "source_fragments": fragments,
        "cleaning": {
            "kind": cleaning_kind,
            "line_sources": cleaned.line_sources,
            "changes": cleaned.changes,
            "warnings": cleaned.warnings,
        },
        **extra,
    }


def _native_blocks(page: pymupdf.Page, media: PageMedia, config: PipelineConfig) -> list[dict]:
    flags = pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    raw = page.get_text("rawdict", flags=flags, sort=False)
    regions = _table_regions(media.tables)
    blocks = []
    for number, block in enumerate(raw["blocks"]):
        if block["type"] != 0:
            continue
        source_lines, fragments, removed = [], [], []
        source_offset = 0
        for line_number, line in enumerate(block["lines"], 1):
            chars = [char for span in line["spans"] for char in span["chars"]]
            text = "".join(char["c"] for char in chars)
            boxes = [char["bbox"] for char in chars for _ in char["c"]]
            source_lines.append(text)
            keep = [True] * len(text)
            for match in re.finditer(r"\S+", text):
                word_box = _bbox(boxes[match.start():match.end()])
                represented_by = next(
                    (table_id for area, words, table_id in regions
                     if _inside(word_box, area) and _key(match[0]) in words),
                    None,
                )
                if represented_by:
                    keep[match.start():match.end()] = [False] * len(match[0])
                    removed.append({
                        "start": source_offset + match.start(),
                        "end": source_offset + match.end(),
                        "table_id": represented_by,
                        "bbox": word_box,
                    })
            for start, end in _kept_intervals(keep):
                if text[start:end].strip():
                    fragments.append({
                        "start": source_offset + start,
                        "end": source_offset + end,
                        "line": line_number,
                        "bbox": _bbox(boxes[start:end]),
                    })
            source_offset += len(text) + 1
        source_text = "\n".join(source_lines)
        blocks.append(_text_block(
            block_id=f"p{media.page:04d}_text_{number:04d}",
            kind="text",
            bbox=block["bbox"],
            source_text=source_text,
            fragments=fragments,
            config=config,
            extraction_method="pdf_text",
            represented_in_tables=removed,
        ))
    return blocks


def _image_blocks(media: PageMedia, config: PipelineConfig) -> list[dict]:
    blocks = []
    for image in media.images:
        lines: dict[tuple, list[tuple[int, dict]]] = {}
        for index, word in enumerate(image.words):
            lines.setdefault(tuple(word["line"]), []).append((index, word))
        selected = set(image.rag_word_indices)
        source_lines, fragments = [], []
        offset = 0
        for line_number, words in enumerate(lines.values(), 1):
            line = " ".join(word["text"] for _, word in words)
            source_lines.append(line)
            spans, cursor = [], 0
            for index, word in words:
                spans.append((cursor, cursor + len(word["text"]), index, word["bbox"]))
                cursor += len(word["text"]) + 1
            for first, last in _kept_intervals([index in selected for _, _, index, _ in spans]):
                fragment = spans[first:last]
                fragments.append({
                    "start": offset + fragment[0][0],
                    "end": offset + fragment[-1][1],
                    "line": line_number,
                    "bbox": _bbox(item[3] for item in fragment),
                    "word_indices": [item[2] for item in fragment],
                })
            offset += len(line) + 1
        source_text = "\n".join(source_lines)
        blocks.append(_text_block(
            block_id=image.image_id,
            kind="image_text",
            bbox=image.bbox,
            source_text=source_text,
            fragments=fragments,
            config=config,
            extraction_method="image_ocr",
            image_path=Path(image.image_path).name,
            status=image.status,
            mean_confidence=image.mean_confidence,
        ))
    return blocks


def _table_block(table: TableData) -> dict:
    return {
        "block_id": table.table_id,
        "kind": "table",
        "bbox": list(table.bbox),
        "text": table.markdown,
        "table": asdict(table),
        "extraction_method": table.method,
        "duplicate_of": table.duplicate_of,
    }


def _is_edge_block(block: dict, page_height: float) -> bool:
    if page_height <= 0:
        return False
    y0, y1 = block["bbox"][1], block["bbox"][3]
    return y0 <= page_height * 0.12 or y1 >= page_height * 0.90


def _boilerplate_signature(text: str, *, ocr: bool) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    tokens = _WORD_RE.findall(normalized)
    if ocr:
        # OCR часто добавляет 1-2 мусорных символа к логотипу. Берём только
        # первые содержательные токены: «ОТКРЫТАЯ МОБИЛЬНАЯ ПЛАТФОРМА x»
        # стабильно превращается в одну сигнатуру.
        tokens = [token for token in tokens if len(token) >= 3 and not token.isdigit()]
        return " ".join(tokens[:3])
    compact = " ".join(tokens)
    compact = re.sub(r"\b\d+\b", "#", compact)
    return compact[:160]


def _image_text_noise_reason(block: dict) -> str | None:
    """Консервативно отфильтровать только явно бесполезный OCR-фрагмент.

    Мы не пытаемся исправлять OCR и не выкидываем длинный текст только из-за
    низкой confidence. Фильтр нужен для обрывков вроде ``Я``, ``DN``, ``[М``
    и коротких наборов символов, которые иначе становятся отдельными чанками.
    """
    text = block.get("text", "").strip()
    if not text:
        return None

    tokens = _WORD_RE.findall(text)
    if not tokens:
        return "image_text_symbol_only"

    alnum_chars = sum(len(token) for token in tokens)
    if alnum_chars < 8:
        return "image_text_too_short"

    # Одиночный короткий OCR-токен почти не даёт поискового контекста.
    # Слова вроде Bluetooth/КИБЕРПРОТЕКТ при этом сохраняются.
    if len(tokens) == 1 and len(tokens[0]) < 8:
        return "image_text_too_short"

    # Для коротких фрагментов дополнительно отсеиваем сильно зашумлённый OCR.
    # Длинный текст не удаляем по этим эвристикам: там даже частично ошибочный
    # OCR может содержать полезные подписи интерфейса или схемы.
    if len(text) <= 80:
        visible_chars = sum(not char.isspace() for char in text)
        alnum_ratio = alnum_chars / max(1, visible_chars)
        if alnum_ratio < 0.55:
            return "image_text_symbol_heavy"

        short_ratio = sum(len(token) <= 2 for token in tokens) / len(tokens)
        long_tokens = sum(len(token) >= 3 for token in tokens)
        if short_ratio >= 0.60 and long_tokens < 2:
            return "image_text_fragmented"

        confidence = block.get("mean_confidence")
        if confidence is not None and confidence < 50:
            return "image_text_low_confidence_fragment"

    return None


def _mark_document_noise(captured: dict[int, dict], page_heights: dict[int, float],
                         config: PipelineConfig) -> dict[str, int]:
    """Пометить блоки, которые не должны попадать в retrieval corpus."""
    stats: defaultdict[str, int] = defaultdict(int)
    if not captured:
        return dict(stats)

    occurrences: dict[tuple[str, str], set[int]] = defaultdict(set)
    for page_number, capture in captured.items():
        height = page_heights.get(page_number, 0)
        for block in capture["blocks"]:
            if block.get("duplicate_of") or not block.get("text", "").strip():
                continue
            if block["kind"] not in {"text", "image_text"} or not _is_edge_block(block, height):
                continue
            text = block["text"].strip()
            if len(text) > 180:
                continue
            signature = _boilerplate_signature(text, ocr=block["kind"] == "image_text")
            if signature:
                occurrences[(block["kind"], signature)].add(page_number)

    threshold = max(
        config.boilerplate_min_pages,
        math.ceil(len(captured) * config.boilerplate_min_fraction),
    )
    repeated = {
        key for key, pages in occurrences.items()
        if config.remove_boilerplate and len(pages) >= threshold
    }

    for page_number, capture in captured.items():
        height = page_heights.get(page_number, 0)
        for block in capture["blocks"]:
            text = block.get("text", "").strip()
            if not text or block.get("duplicate_of"):
                continue
            reason = None
            if config.filter_image_text and block["kind"] == "image_text":
                reason = _image_text_noise_reason(block)
            if reason is None and _LABEL_ONLY.fullmatch(text):
                reason = "standalone_figure_or_table_label"
            elif reason is None and _PAGE_NUMBER_ONLY.fullmatch(text) and _is_edge_block(block, height):
                reason = "standalone_page_number"
            elif reason is None and not _WORD_RE.search(text):
                reason = "symbol_only"
            elif reason is None and block["kind"] in {"text", "image_text"} and _is_edge_block(block, height):
                signature = _boilerplate_signature(text, ocr=block["kind"] == "image_text")
                if (block["kind"], signature) in repeated:
                    reason = "repeated_boilerplate"
            if reason:
                block["excluded_from_chunks"] = reason
                stats[reason] += 1
    return dict(stats)


def _text_windows(text: str, size: int, overlap: int, min_size: int = 0) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for separator in ("\n\n", "\n", ". ", "; ", " "):
                split = text.rfind(separator, start + size // 2, end)
                if split >= 0:
                    end = split + len(separator)
                    break
        windows.append((start, end))
        if end == len(text):
            break
        start = max(start + 1, end - overlap)

    # Не оставляем микрочанк в хвосте. Увеличиваем только перекрытие последнего
    # окна, не меняя предыдущий чанк и не превышая size.
    if len(windows) >= 2 and min_size > 0:
        last_start, last_end = windows[-1]
        if last_end - last_start < min_size:
            desired = max(0, last_end - min_size)
            search_from = max(0, desired - 80)
            split = text.find(" ", search_from, desired + 1)
            if split >= 0:
                desired = split + 1
            windows[-1] = (desired, last_end)
    return windows


def _text_reference(block: dict, start: int, end: int) -> dict:
    first_line = block["text"].count("\n", 0, start)
    last_line = block["text"].count("\n", 0, max(start, end - 1))
    raw_lines = sorted({
        number
        for group in block["cleaning"]["line_sources"][first_line:last_line + 1]
        for number in group
    })
    fragments = [
        block["source_fragments"][number - 1]
        for number in raw_lines
        if 0 < number <= len(block["source_fragments"])
    ]
    return {
        "block_id": block["block_id"],
        "clean_start": start,
        "clean_end": end,
        # Подробные source_spans нужны в document.json для аудита, но не будут
        # копироваться в итоговый chunks.jsonl (см. enrich_chunk).
        "source_spans": fragments,
    }


def _combined_text_chunks(blocks: list[dict], config: PipelineConfig) -> list[dict]:
    """Склеить текстовые блоки страницы/раздела и разрезать уже общий текст."""
    if not blocks:
        return []

    combined_parts: list[str] = []
    placements: list[dict] = []
    cursor = 0
    for block in blocks:
        original = block["text"]
        left = len(original) - len(original.lstrip())
        right = len(original.rstrip())
        if right <= left:
            continue
        piece = original[left:right]
        if combined_parts:
            combined_parts.append("\n\n")
            cursor += 2
        start = cursor
        combined_parts.append(piece)
        cursor += len(piece)
        placements.append({
            "start": start,
            "end": cursor,
            "block": block,
            "local_start": left,
        })

    text = "".join(combined_parts)
    if not text.strip():
        return []

    chunks = []
    for start, end in _text_windows(
        text,
        config.chunk_size_chars,
        config.chunk_overlap_chars,
        config.min_chunk_chars,
    ):
        refs, boxes = [], []
        for item in placements:
            inter_start = max(start, item["start"])
            inter_end = min(end, item["end"])
            if inter_start >= inter_end:
                continue
            local_start = item["local_start"] + (inter_start - item["start"])
            local_end = item["local_start"] + (inter_end - item["start"])
            refs.append(_text_reference(item["block"], local_start, local_end))
            boxes.append(item["block"]["bbox"])
        if refs and text[start:end].strip():
            chunks.append({
                "text": text[start:end].strip(),
                "kind": "text",
                "bbox": _bbox(boxes),
                "oversized": False,
                "sources": refs,
            })
    return chunks


def _markdown_table(header: list, rows: list[list]) -> str:
    width = max([len(header)] + [len(row) for row in rows])

    def row_text(row):
        values = [
            (str(v) if v is not None else "")
            .replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\n", "<br>")
            for v in row
        ]
        return "| " + " | ".join(values + [""] * (width - len(values))) + " |"

    return "\n".join([row_text(header), row_text(["---"] * width)] + [row_text(row) for row in rows])


def _table_chunks(block: dict, size: int) -> Iterator[dict]:
    table = block["table"]
    rows = table["rows"]
    if not rows or not any(value for row in rows for value in row):
        return
    width = max(len(row) for row in rows)
    header = table["header_names"]
    header_rows = []
    if header and not table["header_external"] and header == rows[0]:
        header_rows = [0]
        first_data = 1
    else:
        first_data = 0
    if not header:
        header = [f"Столбец {i + 1}" for i in range(width)]
    repeat_first = not table["header_names"]
    cursor = first_data
    while cursor < len(rows) or (cursor == first_data == len(rows)):
        end = cursor
        context_rows = [0] if repeat_first and cursor > 0 else []
        selected = [rows[i] for i in context_rows]
        while end < len(rows):
            candidate = _markdown_table(header, selected + rows[cursor:end + 1])
            if end > cursor and len(candidate) > size:
                break
            end += 1
            if len(candidate) > size:
                break
        text = _markdown_table(header, selected + rows[cursor:end])
        row_indices = sorted(set(header_rows + context_rows + list(range(cursor, end))))
        boxes = [
            cell["bbox"] for cell in table["cells"]
            if cell["row"] in row_indices and cell["bbox"]
        ]
        if table["header_external"] and table["header_bbox"]:
            boxes.append(table["header_bbox"])
        yield {
            "text": text,
            "kind": "table",
            "bbox": _bbox(boxes) if boxes else block["bbox"],
            "oversized": len(text) > size,
            "sources": [{
                "block_id": block["block_id"],
                "row_start": cursor,
                "row_end": end,
            }],
        }
        if end == len(rows):
            break
        cursor = end


def _page_chunks(blocks: list[dict], config: PipelineConfig) -> list[dict]:
    """Чанкинг страницы: текст объединяем, таблицы/OCR оставляем отдельными."""
    chunks: list[dict] = []
    text_buffer: list[dict] = []
    buffer_section = None

    def flush_text() -> None:
        nonlocal buffer_section
        if text_buffer:
            chunks.extend(_combined_text_chunks(text_buffer, config))
            text_buffer.clear()
        buffer_section = None

    for block in blocks:
        if (
            not block.get("text", "").strip()
            or block.get("duplicate_of")
            or block.get("excluded_from_chunks")
        ):
            continue

        if block["kind"] == "text":
            section_key = (block.get("section_id"), block.get("section_status"))
            if text_buffer and section_key != buffer_section:
                flush_text()
            if not text_buffer:
                buffer_section = section_key
            text_buffer.append(block)
            continue

        flush_text()
        if block["kind"] == "table":
            chunks.extend(_table_chunks(block, config.chunk_size_chars))
        else:
            # OCR одного изображения не смешиваем с обычным текстом: так можно
            # отдельно фильтровать качество и при необходимости показать asset.
            for start, end in _text_windows(
                block["text"],
                config.chunk_size_chars,
                config.chunk_overlap_chars,
                config.min_chunk_chars,
            ):
                if block["text"][start:end].strip():
                    chunks.append({
                        "text": block["text"][start:end].strip(),
                        "kind": block["kind"],
                        "bbox": block["bbox"],
                        "oversized": False,
                        "sources": [_text_reference(block, start, end)],
                    })
    flush_text()
    return chunks


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _validate_token_counter(config: PipelineConfig, counter) -> None:
    if counter is not None and (not callable(counter) or not config.tokenizer_name):
        raise ValueError("Для token_counter задайте tokenizer_name")
    if counter is None and (config.tokenizer_name or config.max_chunk_tokens is not None):
        raise ValueError("Для проверки токенов передайте token_counter или --tokenizer-path")


REVIEW_FIELDS = (
    "document_id", "page", "kind", "chunk_id", "title",
    "section_title", "indexable", "quality_flags",
    "indexing_exclusions", "ocr_confidence", "token_count", "text_preview",
)


def _review_row(chunk: dict) -> dict:
    row = {}
    for key in REVIEW_FIELDS:
        value = chunk["text"][:240].replace("\n", " ") if key == "text_preview" else chunk.get(key)
        if isinstance(value, list):
            value = " | ".join(str(item) for item in value)
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
            value = "'" + value
        row[key] = value
    return row


def _public_chunk(chunk: dict) -> dict:
    """Явный whitelist: случайное внутреннее поле не утечёт в chunks.jsonl."""
    keys = (
        "schema_version", "chunk_id", "document_id", "title",
        "page", "page_label", "page_count",
        "section_title", "section_path", "kind", "text", "bbox", "sources",
        "token_count", "is_ocr", "ocr_confidence", "quality_flags",
        "document_status", "indexable", "indexing_exclusions", "needs_review",
        "chunk_index", "prev_chunk_id", "next_chunk_id", "citation_label",
        "content_sha256",
    )
    return {key: chunk.get(key) for key in keys}


def prepare_document(
    pdf_path: str | Path,
    *,
    corpus_root: str | Path,
    output_dir: str | Path,
    config: PipelineConfig | None = None,
    ocr_engine=None,
    token_counter: Callable[[str], int] | None = None,
    run_id: str | None = None,
) -> PreparedDocument:
    config = config or PipelineConfig()
    config.validate()
    _validate_token_counter(config, token_counter)
    output_dir = Path(output_dir).resolve()
    captured: dict[int, dict] = {}
    pdf_info: dict[str, Any] = {}
    processed_at = datetime.now(timezone.utc).isoformat()

    def on_page(page: pymupdf.Page, media_page: PageMedia):
        if not pdf_info:
            pdf_info.update(metadata=page.parent.metadata or {}, toc=page.parent.get_toc())
        native = _native_blocks(page, media_page, config)
        blocks = native + [_table_block(t) for t in media_page.tables] + _image_blocks(media_page, config)
        blocks.sort(key=lambda b: (round(b["bbox"][1] / 4), b["bbox"][0], b["block_id"]))
        captured[media_page.page] = {
            "raw_native_text": "\n".join(b["source_text"] for b in native),
            "blocks": blocks,
            "page_label": page.get_label() or None,
            "page_label_source": "pdf_page_labels" if page.get_label() else "unknown",
        }

    media = extract_pdf_media(
        pdf_path,
        corpus_root=corpus_root,
        output_dir=output_dir / "documents",
        config=config.media,
        on_page=on_page,
        ocr_engine=ocr_engine,
    )
    directory = Path(media.asset_directory)
    page_heights = {page.page: page.height for page in media.pages}
    noise_stats = _mark_document_noise(captured, page_heights, config)

    metadata = resolve_document_metadata(
        media.document_id,
        pdf_info.get("metadata", {}),
        config.corpus_id,
    )
    sections, ambiguous_pages, warnings = make_sections(
        metadata["document_key"],
        pdf_info.get("toc", []),
        media.page_count,
    )

    pages, chunks = [], []
    block_index = 0
    for media_page in media.pages:
        capture = captured.get(media_page.page, {
            "raw_native_text": "", "blocks": [], "page_label": None, "page_label_source": "unknown"
        })
        errors = media_page.errors + [error for image in media_page.images for error in image.errors]
        if media_page.page not in captured and not errors:
            errors.append("Текстовый обработчик страницы не завершился")

        assign_sections(capture["blocks"], media_page.page, sections, ambiguous_pages)
        for index_on_page, block in enumerate(capture["blocks"]):
            block.update(page=media_page.page, block_index=block_index, block_index_on_page=index_on_page)
            block_index += 1

        page_record = {
            "page": media_page.page,
            "width": media_page.width,
            "height": media_page.height,
            "original_rotation": media_page.original_rotation,
            "errors": errors,
            "native_text_chars": len(capture["raw_native_text"]),
            "image_count": len(media_page.images),
            "table_count": sum(not table.duplicate_of for table in media_page.tables),
            "page_label": capture["page_label"],
            "page_label_source": capture["page_label_source"],
            "raw_native_text": capture["raw_native_text"],
            "blocks": capture["blocks"],
        }

        page_chunks = _page_chunks(capture["blocks"], config)
        page_record["chunk_count"] = len(page_chunks)
        pages.append(page_record)

        for chunk in page_chunks:
            chunk.update({
                "schema_version": PIPELINE_VERSION,
                "document_id": media.document_id,
                "page": media_page.page,
            })
            enrich_chunk(
                chunk,
                blocks=capture["blocks"],
                page_record=page_record,
                media_page=media_page,
                document_metadata=metadata,
                page_count=media.page_count,
                token_counter=token_counter,
            )

            # ID не зависит от служебных данных запуска и времени.
            identity = json.dumps({
                "version": PIPELINE_VERSION,
                "document_id": media.document_id,
                "sha256": media.sha256,
                "page": media_page.page,
                "sources": chunk["sources"],
                "text": chunk["text"],
            }, ensure_ascii=False, sort_keys=True)
            chunk["chunk_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
            chunk["content_sha256"] = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
            chunks.append(chunk)

    has_errors = any(page["errors"] for page in pages)
    status = "partial" if has_errors else ("ok" if chunks else "empty")
    finalize_chunks(
        chunks,
        status=status,
        include_partial=config.include_partial,
        max_chunk_tokens=config.max_chunk_tokens,
    )

    if not config.media.ocr_enabled:
        warnings.append("OCR отключён: текст изображений и сканов не распознавался")
    if any(chunk.get("oversized") for chunk in chunks):
        warnings.append("Есть строки таблиц длиннее chunk_size_chars; они сохранены целиком")
    if token_counter is None:
        warnings.append("Лимит embedding-модели не проверен: токенизатор не задан")
    if any("token_limit_exceeded" in chunk["indexing_exclusions"] for chunk in chunks):
        warnings.append("Чанки сверх лимита токенов сохранены локально, но исключены из общего индекса")

    pages_without_chunks = [page["page"] for page in pages if not page["chunk_count"]]
    # Страница только с декоративным/повторяющимся содержимым теперь допустима.
    meaningful_empty = [
        page for page in pages_without_chunks
        if any(
            block.get("text", "").strip() and not block.get("excluded_from_chunks")
            for block in pages[page - 1]["blocks"]
        )
    ]
    if meaningful_empty:
        warnings.append(f"Страницы с содержимым, но без чанков: {meaningful_empty}")

    media_payload = asdict(media)
    media_payload["asset_directory"] = directory.relative_to(output_dir).as_posix()
    for page in media_payload["pages"]:
        for image in page["images"]:
            image["image_path"] = Path(image["image_path"]).name

    document = {
        "schema_version": PIPELINE_VERSION,
        "document_id": media.document_id,
        "sha256": media.sha256,
        "document_key": metadata["document_key"],
        "processed_at_utc": processed_at,
        "run_id": run_id,
        "title": metadata["title"],
        "sections": sections,
        "page_count": media.page_count,
        "status": status,
        "has_errors": has_errors,
        "config": {
            "chunk_size_chars": config.chunk_size_chars,
            "chunk_overlap_chars": config.chunk_overlap_chars,
            "min_chunk_chars": config.min_chunk_chars,
            "remove_boilerplate": config.remove_boilerplate,
            "filter_image_text": config.filter_image_text,
            "native_table_strategy": config.media.native_table_strategy,
            "max_chunk_tokens": config.max_chunk_tokens,
            "tokenizer_name": config.tokenizer_name,
            "ocr_enabled": config.media.ocr_enabled,
            "ocr_languages": config.media.languages,
        },
        "versions": media.versions,
        "warnings": warnings,
        "noise_removed": noise_stats,
        "pages": pages,
        "media": media_payload,
        "chunk_count": len(chunks),
        "indexed_chunk_count": sum(chunk["indexable"] for chunk in chunks),
        "needs_review_count": sum(chunk["needs_review"] for chunk in chunks),
        "pages_without_chunks": pages_without_chunks,
    }

    document_path, chunks_path = directory / "document.json", directory / "chunks.jsonl"
    _write_json(document_path, document)
    _write_jsonl(chunks_path, (_public_chunk(chunk) for chunk in chunks))
    return PreparedDocument(document, chunks, directory, document_path, chunks_path)


def prepare_corpus(
    input_path: str | Path,
    *,
    output_dir: str | Path = "prepared",
    corpus_root: str | Path | None = None,
    config: PipelineConfig | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> dict:
    config = config or PipelineConfig()
    config.validate()
    _validate_token_counter(config, token_counter)

    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    root = Path(corpus_root).expanduser().resolve() if corpus_root else (
        source if source.is_dir() else source.parent
    )
    files = sorted(
        (p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.as_posix(),
    ) if source.is_dir() else [source]
    if not files:
        raise ValueError("PDF-файлы не найдены")
    for file in files:
        file.resolve().relative_to(root)
        if file.suffix.lower() != ".pdf":
            raise ValueError(f"Нужен PDF: {file}")

    engine = create_ocr_engine(config.media)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = Path(output_dir).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True)

    manifest = {
        "schema_version": PIPELINE_VERSION,
        "run_id": run_id,
        "status": "running",
        "corpus_id": config.corpus_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_root": str(root),
        "file_count": len(files),
        "chunk_count": 0,
        "excluded_chunk_count": 0,
        "needs_review_count": 0,
        "has_errors": False,
        "chunks_file": "chunks.jsonl",
        "review_file": "review.csv",
        "warnings": [],
        "documents": [],
        "config": {
            "chunk_size_chars": config.chunk_size_chars,
            "chunk_overlap_chars": config.chunk_overlap_chars,
            "min_chunk_chars": config.min_chunk_chars,
            "remove_boilerplate": config.remove_boilerplate,
            "filter_image_text": config.filter_image_text,
            "native_table_strategy": config.media.native_table_strategy,
            "max_chunk_tokens": config.max_chunk_tokens,
            "tokenizer_name": config.tokenizer_name,
            "ocr_enabled": config.media.ocr_enabled,
            "ocr_languages": config.media.languages,
        },
    }
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    temporary_chunks = run_dir / "chunks.jsonl.tmp"
    review_path = run_dir / "review.csv"
    temporary_review = run_dir / "review.csv.tmp"

    with temporary_chunks.open("w", encoding="utf-8", newline="\n") as stream, \
            temporary_review.open("w", encoding="utf-8-sig", newline="") as review:
        review_writer = csv.DictWriter(review, fieldnames=REVIEW_FIELDS, delimiter=";")
        review_writer.writeheader()

        for index, file in enumerate(files, 1):
            document_id = file.resolve().relative_to(root).as_posix()
            print(f"[{index}/{len(files)}] {document_id}", file=sys.stderr, flush=True)
            try:
                prepared = prepare_document(
                    file,
                    corpus_root=root,
                    output_dir=run_dir,
                    config=config,
                    ocr_engine=engine,
                    token_counter=token_counter,
                    run_id=run_id,
                )
            except Exception as error:
                entry = {
                    "document_id": document_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "chunk_count": 0,
                    "indexed_chunk_count": 0,
                }
                manifest["has_errors"] = True
            else:
                accepted = [chunk for chunk in prepared.chunks if chunk["indexable"]]
                for chunk in accepted:
                    stream.write(json.dumps(_public_chunk(chunk), ensure_ascii=False) + "\n")
                # review.csv теперь содержит только действительно проблемные чанки.
                for chunk in prepared.chunks:
                    if chunk["needs_review"]:
                        review_writer.writerow(_review_row(chunk))
                review.flush()
                stream.flush()

                doc = prepared.document
                entry = {
                    "document_id": document_id,
                    "status": doc["status"],
                    "sha256": doc["sha256"],
                    "page_count": doc["page_count"],
                    "chunk_count": len(prepared.chunks),
                    "indexed_chunk_count": len(accepted),
                    "title": doc["title"],
                    "noise_removed": doc["noise_removed"],
                    "needs_review_count": doc["needs_review_count"],
                    "pages_without_chunks": doc["pages_without_chunks"],
                    "warnings": doc["warnings"],
                    "document_file": prepared.document_path.relative_to(run_dir).as_posix(),
                    "chunks_file": prepared.chunks_path.relative_to(run_dir).as_posix(),
                }
                manifest["has_errors"] |= doc["has_errors"]
                manifest["chunk_count"] += len(accepted)
                manifest["excluded_chunk_count"] += len(prepared.chunks) - len(accepted)
                manifest["needs_review_count"] += doc["needs_review_count"]

            manifest["documents"].append(entry)
            _write_json(manifest_path, manifest)

    temporary_chunks.replace(run_dir / "chunks.jsonl")
    temporary_review.replace(review_path)
    manifest["status"] = "completed_with_errors" if manifest["has_errors"] else "completed"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    return {
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "chunks": str(run_dir / "chunks.jsonl"),
        "review": str(review_path),
        "file_count": len(files),
        "chunk_count": manifest["chunk_count"],
        "excluded_chunk_count": manifest["excluded_chunk_count"],
        "needs_review_count": manifest["needs_review_count"],
        "has_errors": manifest["has_errors"],
    }


def iter_langchain_documents(chunks_path: str | Path, *, include_partial: bool = False):
    from langchain_core.documents import Document

    with Path(chunks_path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk["document_status"] == "partial" and not include_partial:
                continue
            exclusions = set(chunk.get("indexing_exclusions", []))
            if include_partial:
                exclusions.discard("partial_document")
            if exclusions or not chunk.get("indexable", True):
                continue
            yield Document(
                id=chunk["chunk_id"],
                page_content=chunk["text"],
                metadata=scalar_metadata(chunk),
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("prepared"))
    parser.add_argument("--corpus-id", default="default")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--max-chunk-tokens", type=int)
    parser.add_argument("--languages", default="rus+eng")
    parser.add_argument("--tesseract-cmd")
    parser.add_argument("--tessdata-dir")
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-raster-tables", action="store_true")
    parser.add_argument("--full-page-ocr", action="store_true")
    parser.add_argument("--borderless-tables", action="store_true")
    parser.add_argument("--native-table-strategy", choices=["lines", "lines_strict", "text"], default="lines_strict")
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--min-chunk-size", type=int, default=350)
    parser.add_argument("--text-kind", choices=["unknown", "prose"], default="prose")
    parser.add_argument("--no-join-wrapped-lines", action="store_true")
    parser.add_argument("--known-words", type=Path)
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--keep-boilerplate", action="store_true")
    parser.add_argument("--keep-noisy-image-text", action="store_true")
    args = parser.parse_args()

    try:
        token_counter = None
        if args.tokenizer_path:
            if not args.tokenizer_path.is_dir():
                raise ValueError("--tokenizer-path должен указывать на локальную папку")
            try:
                from transformers import AutoTokenizer
            except ImportError as error:
                raise ImportError("Для --tokenizer-path установите transformers") from error
            tokenizer = AutoTokenizer.from_pretrained(
                str(args.tokenizer_path), local_files_only=True, trust_remote_code=False
            )
            token_counter = lambda text: len(tokenizer.encode(text, add_special_tokens=True))

        words = tuple(
            word.strip()
            for word in args.known_words.read_text(encoding="utf-8-sig").splitlines()
            if word.strip()
        ) if args.known_words else ()

        config = PipelineConfig(
            media=MediaConfig(
                languages=args.languages,
                dpi=args.dpi,
                tesseract_cmd=args.tesseract_cmd,
                tessdata_dir=args.tessdata_dir,
                ocr_enabled=not args.no_ocr,
                raster_tables=not args.no_raster_tables,
                full_page_ocr=args.full_page_ocr,
                borderless_tables=args.borderless_tables,
                native_table_strategy=args.native_table_strategy,
            ),
            chunk_size_chars=args.chunk_size,
            chunk_overlap_chars=args.chunk_overlap,
            min_chunk_chars=args.min_chunk_size,
            text_kind=args.text_kind,
            known_words=words,
            join_wrapped_lines=not args.no_join_wrapped_lines,
            include_partial=args.include_partial,
            corpus_id=args.corpus_id,
            tokenizer_name=str(args.tokenizer_path.resolve()) if args.tokenizer_path else None,
            max_chunk_tokens=args.max_chunk_tokens,
            remove_boilerplate=not args.keep_boilerplate,
            filter_image_text=not args.keep_noisy_image_text,
        )
        result = prepare_corpus(
            args.input,
            corpus_root=args.corpus_root,
            output_dir=args.output,
            config=config,
            token_counter=token_counter,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result["has_errors"] else 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
