#!/usr/bin/env python3
"""Stream-audit a chunk JSONL without loading its text corpus into memory."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", nargs="?", default="data/nntu_site/extracted/chunks.jsonl")
    parser.add_argument("--output", default="data/nntu_site/extracted/chunking_audit.json")
    args = parser.parse_args()
    path = Path(args.chunks)

    counts = Counter()
    categories, kinds, extensions, sources = Counter(), Counter(), Counter(), Counter()
    flags, ocr_bands = Counter(), Counter()
    char_lengths, token_missing = [], 0
    ids: dict[str, tuple[str, int, str | None, str | None]] = {}
    text_hashes: set[str] = set()
    duplicate_text = 0
    errors: list[dict] = []

    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected JSON object")
                for field in ("chunk_id", "document_id", "text"):
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        raise ValueError(f"invalid {field}")
                if type(row.get("indexable")) is not bool:
                    raise ValueError("indexable is not boolean")
                if type(row.get("page")) is not int or row["page"] < 1:
                    raise ValueError("page is not a positive integer")
                cid = row["chunk_id"]
                if cid in ids:
                    raise ValueError(f"duplicate chunk_id {cid}")
                text = row["text"]
                actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if row.get("content_sha256") and row["content_sha256"] != actual_hash:
                    errors.append({"line": line_number, "issue": "content_hash_mismatch", "id": cid})
                if actual_hash in text_hashes:
                    duplicate_text += 1
                else:
                    text_hashes.add(actual_hash)
                idx = row.get("chunk_index")
                if type(idx) is not int:
                    errors.append({"line": line_number, "issue": "invalid_chunk_index", "id": cid})
                    idx = -1
                ids[cid] = (row["document_id"], idx, row.get("prev_chunk_id"), row.get("next_chunk_id"))
                counts["chunks"] += 1
                counts["documents"] = 0  # replaced below with a set cardinality
                char_lengths.append(len(text))
                categories[row.get("category") or "unknown"] += 1
                kinds[row.get("kind") or "unknown"] += 1
                extensions[row.get("extension") or "unknown"] += 1
                sources[row.get("source_type") or "unknown"] += 1
                if row.get("is_ocr"):
                    conf = row.get("ocr_confidence")
                    band = "unscored" if conf is None else "low_<60" if conf < 60 else "60_to_85" if conf < 85 else "85_plus"
                    ocr_bands[band] += 1
                if row.get("token_count") is None:
                    token_missing += 1
                for flag in row.get("quality_flags", []):
                    flags[flag] += 1
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                errors.append({"line": line_number, "issue": str(exc)})

    docs = {item[0] for item in ids.values()}
    chain_issues = Counter()
    for cid, (doc_id, idx, prev_id, next_id) in ids.items():
        for target_id, opposite, direction in ((prev_id, "next", "prev"), (next_id, "prev", "next")):
            if not target_id:
                continue
            target = ids.get(target_id)
            if target is None:
                chain_issues["missing_neighbor"] += 1
            elif target[0] != doc_id:
                chain_issues["cross_document_neighbor"] += 1
            elif target[2 if opposite == "prev" else 3] != cid:
                chain_issues["nonreciprocal_neighbor"] += 1
            elif direction == "next" and target[1] != idx + 1:
                chain_issues["chunk_index_gap"] += 1

    char_lengths.sort()
    n = len(char_lengths)
    result = {
        "input": str(path), "chunks": n, "documents": len(docs),
        "unique_chunk_ids": len(ids), "duplicate_text_chunks_beyond_first": duplicate_text,
        "length_chars": {"min": min(char_lengths, default=0), "median": char_lengths[n // 2] if n else 0,
                          "p95": char_lengths[int((n - 1) * .95)] if n else 0, "max": max(char_lengths, default=0)},
        "by_category": dict(categories), "by_kind": dict(kinds), "by_extension": dict(extensions),
        "by_source_type": dict(sources), "ocr_confidence_bands": dict(ocr_bands),
        "quality_flags": dict(flags), "token_count_missing": token_missing,
        "neighbor_chain_issues": dict(chain_issues), "errors": errors[:100], "error_count": len(errors),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if errors or chain_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
