"""Чтение и аудит готовых чанков. Исходный JSONL не изменяется."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_chunks(path: str | Path) -> list[dict[str, Any]]:
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("ожидается JSON-объект")
                for field in ("chunk_id", "document_id", "text"):
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        raise ValueError(f"пустое или некорректное поле {field}")
                if row["chunk_id"] in seen:
                    raise ValueError(f"повторный chunk_id: {row['chunk_id']}")
                if type(row.get("indexable")) is not bool:
                    raise ValueError("indexable должен быть bool")
                if type(row.get("page")) is not int or row["page"] < 1:
                    raise ValueError("page должен быть положительным целым числом")
                if row.get("page_count") is not None and row["page"] > row["page_count"]:
                    raise ValueError("page превышает page_count")
                seen.add(row["chunk_id"])
                rows.append(row)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}, строка {line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"Нет чанков в {path}")
    return rows


def audit_chunks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {row["chunk_id"]: row for row in rows}
    issues = defaultdict(list)
    for row in rows:
        actual_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        if row.get("content_sha256") and row["content_sha256"] != actual_hash:
            issues["content_hash_mismatch"].append(row["chunk_id"])
        for direction, opposite in (("prev_chunk_id", "next_chunk_id"),
                                    ("next_chunk_id", "prev_chunk_id")):
            target_id = row.get(direction)
            if not target_id:
                continue
            target = ids.get(target_id)
            pair = [row["chunk_id"], target_id]
            if target is None:
                issues["missing_neighbor"].append(pair)
                continue
            if target["document_id"] != row["document_id"]:
                issues["cross_document_neighbor"].append(pair)
            if target.get(opposite) != row["chunk_id"]:
                issues["nonreciprocal_neighbor"].append(pair)
            if direction == "next_chunk_id" and isinstance(row.get("chunk_index"), int):
                if target.get("chunk_index") != row["chunk_index"] + 1:
                    issues["chunk_index_gap"].append(pair)
    by_document = defaultdict(list)
    for row in rows:
        by_document[row["document_id"]].append(row)
    for doc_id, group in by_document.items():
        indexes = [row.get("chunk_index") for row in group]
        if len(set(indexes)) != len(indexes):
            issues["duplicate_index_in_document"].append(doc_id)
        # Validate both endpoint count and reachability: a detached cycle has no endpoint.
        starts = [r for r in group if not r.get("prev_chunk_id")]
        reached = set()
        current = starts[0] if len(starts) == 1 else None
        while current and current["chunk_id"] not in reached:
            if current["document_id"] != doc_id:
                break
            reached.add(current["chunk_id"])
            current = ids.get(current.get("next_chunk_id"))
        if len(starts) != 1 or len(reached) != len(group) or current is not None:
            issues["incomplete_neighbor_chain"].append(doc_id)
    lengths = sorted(r["token_count"] for r in rows if isinstance(r.get("token_count"), int))
    hashes = [hashlib.sha256(r["text"].encode("utf-8")).hexdigest() for r in rows]
    return {
        "chunks": len(rows), "documents": len(by_document),
        "schema_versions": dict(Counter(r.get("schema_version") for r in rows)),
        "kinds": dict(Counter(r.get("kind") for r in rows)),
        "indexable": sum(r["indexable"] for r in rows),
        "needs_review": sum(bool(r.get("needs_review")) for r in rows),
        "quality_flags": dict(Counter(f for r in rows for f in r.get("quality_flags", []))),
        "exact_text_duplicates_beyond_first": len(rows) - len(set(hashes)),
        "without_section": sum(not r.get("section_title") for r in rows),
        "stated_tokens": {"min": lengths[0], "median": lengths[len(lengths)//2],
                          "p95": lengths[int((len(lengths)-1)*.95)], "max": lengths[-1],
                          "sum": sum(lengths)} if lengths else None,
        "missing_token_counts": len(rows) - len(lengths),
        "issues": {key: {"count": len(value), "examples": value[:8]}
                   for key, value in issues.items()},
    }


def select_chunks(rows: list[dict[str, Any]], *, min_ocr_confidence: float | None = None,
                  exclude_needs_review: bool = False) -> tuple[list[dict], dict]:
    if min_ocr_confidence is not None and not 0 <= min_ocr_confidence <= 100:
        raise ValueError("min_ocr_confidence должен находиться в диапазоне 0..100")
    accepted, skipped = [], Counter()
    for row in rows:
        if not row["indexable"] or row.get("indexing_exclusions"):
            skipped["not_indexable"] += 1
        elif exclude_needs_review and row.get("needs_review"):
            skipped["needs_review"] += 1
        elif row.get("is_ocr") and min_ocr_confidence is not None and (
            row.get("ocr_confidence") is None or row["ocr_confidence"] < min_ocr_confidence
        ):
            skipped["ocr_confidence"] += 1
        else:
            accepted.append(row)
    if not accepted:
        raise ValueError("После фильтрации не осталось чанков")
    return accepted, dict(skipped)


def index_text(row: dict[str, Any]) -> str:
    """Метаданные для поиска; исходный text и citation_label остаются прежними."""
    if row.get("schema_version") == "nntu-rag-chunk-v1":
        labels = [f"Документ: {row['title']}"] if row.get("title") else []
        if row.get("section_title"):
            labels.append(f"Раздел: {row['section_title']}")
        return "\n\n".join(["\n".join(labels), row["text"]]) if labels else row["text"]
    source = row["document_id"].replace("\\", "/")
    filename = source.rsplit("/", 1)[-1]
    # Strip only the pipeline's 12-character SHA filename prefix.
    import re
    filename = re.sub(r"^[0-9a-f]{12}_", "", filename, flags=re.I)
    title = row.get("title") or ""
    section = " > ".join(str(s) for s in row.get("section_path", []) if s)
    section = section or row.get("section_title") or ""
    labels = [f"Источник: {source.split('/')[0]}; {filename}"]
    if title and title not in filename and title != filename.rsplit(".", 1)[0]:
        labels.append(f"Документ: {title}")
    if section:
        labels.append(f"Раздел: {section}")
    return "\n".join(labels) + "\n\n" + row["text"]
