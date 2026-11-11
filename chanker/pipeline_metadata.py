"""Минимальные автоматически извлекаемые метаданные для RAG.

Внешний metadata.json не используется. Название документа берётся из PDF
metadata, а при его отсутствии — из имени файла. Разделы берутся только из
встроенного PDF outline/закладок.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import PurePosixPath
from typing import Any, Callable

# Единственное описательное поле документа, которое имеет смысл дублировать в
# поисковых чанках. Остальное уже задаётся document_id/page/section/kind.
DOCUMENT_FIELDS = ("title",)

# Эти поля нужны внутри пайплайна для корректного чанкинга по разделам.
SECTION_FIELDS = (
    "section_id", "section_title", "section_path", "section_level",
    "section_source", "section_status", "section_start_page", "section_start_y",
)


def stable_hash(value: Any, length: int = 32) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:length]


def resolve_document_metadata(document_id: str, pdf_metadata: dict, corpus_id: str) -> dict:
    """Получить только автоматически доступные базовые сведения о документе."""
    title = (pdf_metadata.get("title") or "").strip() or PurePosixPath(document_id).stem
    return {
        "title": title,
        "document_key": stable_hash([corpus_id, document_id]),
    }


def make_sections(document_key: str, toc: list, page_count: int) -> tuple[list[dict], set[int], list[str]]:
    """Построить разделы только из PDF outline/закладок.

    PDF outline часто не содержит вертикальную координату. Если несколько
    закладок указывают на одну физическую страницу, точный раздел не угадываем:
    блоки этой страницы получают section_status='ambiguous'.
    """
    entries = [
        {"level": level, "title": title, "page": page, "source": "pdf_outline"}
        for level, title, page in toc
        if 1 <= page <= page_count and str(title).strip()
    ]
    entries.sort(key=lambda item: item["page"])

    page_counts = Counter(entry["page"] for entry in entries)
    ambiguous = {page for page, count in page_counts.items() if count > 1}
    warnings: list[str] = []
    if ambiguous:
        warnings.append(
            "На некоторых страницах несколько PDF-закладок; точный раздел для них не определяется"
        )

    result: list[dict] = []
    stack: list[dict] = []
    for index, entry in enumerate(entries):
        level = entry.get("level", 1)
        while stack and stack[-1]["level"] >= level:
            stack.pop()
        stack.append(entry)
        path = [item["title"].strip() for item in stack]
        result.append({
            "section_id": stable_hash([document_key, index, entry]),
            "section_title": entry["title"].strip(),
            "section_path": path,
            "section_level": level,
            "section_source": "pdf_outline",
            "section_status": "page_approximation",
            "section_start_page": entry["page"],
            "section_start_y": 0,
        })
    return result, ambiguous, warnings


def assign_sections(blocks: list[dict], page: int, sections: list[dict], ambiguous: set[int]) -> None:
    for block in blocks:
        empty = {key: None for key in SECTION_FIELDS}
        empty.update(section_path=[], section_status="unknown", section_source="unknown")
        if page in ambiguous:
            empty.update(section_status="ambiguous", section_source="pdf_outline")
            block.update(empty)
            continue

        position = (page, block["bbox"][1])
        selected = next(
            (section for section in reversed(sections)
             if (section["section_start_page"], section["section_start_y"]) <= position),
            None,
        )
        block.update(selected or empty)


def enrich_chunk(
    chunk: dict,
    *,
    blocks: list[dict],
    page_record: dict,
    media_page: Any,
    document_metadata: dict,
    page_count: int,
    token_counter: Callable[[str], int] | None,
) -> None:
    """Добавить только поля, необходимые поиску, цитированию и QA."""
    by_id = {block["block_id"]: block for block in blocks}
    selected = [by_id[ref["block_id"]] for ref in chunk["sources"]]
    image_map = {image.image_id: image for image in media_page.images}

    compact_sources: list[dict] = []
    confidences: list[float] = []
    cleaning_warnings: list[str] = []
    is_ocr = False

    for reference, block in zip(chunk["sources"], selected):
        compact = {"block_id": reference["block_id"]}
        for key in ("clean_start", "clean_end", "row_start", "row_end"):
            if key in reference:
                compact[key] = reference[key]

        if block["kind"] == "image_text":
            is_ocr = True
            compact["image_path"] = block.get("image_path")
            image = image_map.get(block["block_id"])
            if image is not None and image.mean_confidence is not None:
                confidences.append(float(image.mean_confidence))
        elif block["kind"] == "table" and block.get("table", {}).get("image_id"):
            is_ocr = True

        cleaning_warnings.extend(block.get("cleaning", {}).get("warnings", []))
        compact_sources.append(compact)

    confidence = round(sum(confidences) / len(confidences), 2) if confidences else None
    quality_flags: list[str] = []
    if confidence is not None and confidence < 60:
        quality_flags.append("low_ocr_confidence")
    if chunk.get("oversized"):
        quality_flags.append("oversized_chars")
    if page_record["errors"]:
        quality_flags.append("page_errors")
    if cleaning_warnings:
        quality_flags.append("cleaning_warnings")

    first = selected[0] if selected else {}
    token_count = token_counter(chunk["text"]) if token_counter else None
    if token_count is not None and (type(token_count) is not int or token_count < 0):
        raise ValueError("token_counter должен возвращать неотрицательное целое число")

    chunk.update({
        "title": document_metadata["title"],
        "page_count": page_count,
        "page_label": page_record["page_label"],
        "section_title": first.get("section_title"),
        "section_path": first.get("section_path", []),
        "sources": compact_sources,
        "is_ocr": is_ocr,
        "ocr_confidence": confidence,
        "token_count": token_count,
        "quality_flags": sorted(set(quality_flags)),
        "citation_label": f"{document_metadata['title']}, PDF стр. {chunk['page']}",
    })


def finalize_chunks(
    chunks: list[dict],
    *,
    status: str,
    include_partial: bool,
    max_chunk_tokens: int | None,
) -> None:
    """Определить индексируемость и связать соседние индексируемые чанки."""
    for index, chunk in enumerate(chunks):
        exclusions: list[str] = []
        if status == "partial" and not include_partial:
            exclusions.append("partial_document")
        if (
            max_chunk_tokens is not None
            and chunk.get("token_count") is not None
            and chunk["token_count"] > max_chunk_tokens
        ):
            exclusions.append("token_limit_exceeded")
        chunk.update({
            "document_status": status,
            "indexable": not exclusions,
            "indexing_exclusions": exclusions,
            "needs_review": bool(chunk.get("quality_flags") or exclusions),
            "chunk_index": index,
            "prev_chunk_id": None,
            "next_chunk_id": None,
        })

    accepted = [chunk for chunk in chunks if chunk["indexable"]]
    for index, chunk in enumerate(accepted):
        chunk["prev_chunk_id"] = accepted[index - 1]["chunk_id"] if index else None
        chunk["next_chunk_id"] = accepted[index + 1]["chunk_id"] if index + 1 < len(accepted) else None


def scalar_metadata(chunk: dict) -> dict:
    """Компактные scalar metadata для LangChain/vector store."""
    keep = (
        "document_id", "title", "page", "page_label", "page_count",
        "section_title", "kind", "is_ocr", "ocr_confidence", "token_count",
        "citation_label", "chunk_index", "prev_chunk_id", "next_chunk_id",
        "document_status",
    )
    result: dict[str, Any] = {}
    null_fields: list[str] = []
    for key in keep:
        value = chunk.get(key)
        if value is None:
            null_fields.append(key)
        elif isinstance(value, (str, bool, int, float)):
            result[key] = value
        else:
            raise TypeError(f"Неподдерживаемое scalar metadata: {key}={type(value)}")

    result["section_path_json"] = json.dumps(chunk.get("section_path", []), ensure_ascii=False)
    result["sources_json"] = json.dumps(chunk.get("sources", []), ensure_ascii=False)
    result["quality_flags_json"] = json.dumps(chunk.get("quality_flags", []), ensure_ascii=False)
    result["null_fields_json"] = json.dumps(null_fields, ensure_ascii=False)
    return result
