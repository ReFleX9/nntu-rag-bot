#!/usr/bin/env python3
"""Run repeatable source-recall checks against an NNTU FAISS/BM25 index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "rag_hybrid_langchain" / "rag_hybrid"))
from retrieval import KnowledgeBase  # noqa: E402

CASES = [
    ("Как отчислиться из НГТУ?", "polog-porjdok-perevod-otchisl-vosstanovl.pdf"),
    ("Как перевестись на другое направление?", "polog-porjdok-perevod-otchisl-vosstanovl.pdf"),
    ("Какие документы нужны для поступления?", "/abit/"),
    ("Где узнать расписание занятий?", "raspisanie"),
    ("Где находится электронная образовательная среда?", "elektronnaya-obrazovatelnaya-sreda"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", choices=("hybrid", "bm25", "faiss"), default="hybrid")
    args = parser.parse_args()
    kb = KnowledgeBase.load(args.index, device="cuda", local_files_only=True)
    results = []
    for question, expected in CASES:
        hits = kb.search(question, top_k=args.top_k, candidate_k=60,
                         bm25_weight=0 if args.mode == "faiss" else 1,
                         faiss_weight=0 if args.mode == "bm25" else 1, rrf_k=60)
        sources = []
        rank = None
        for i, hit in enumerate(hits, 1):
            url = (hit.metadata.get("source_urls") or [""])[0]
            title = hit.metadata.get("title") or ""
            sources.append({"rank": i, "title": title, "url": url,
                            "page": hit.metadata.get("page"),
                            "excerpt": hit.page_content[:180]})
            if rank is None and expected in url.lower():
                rank = i
        results.append({"question": question, "expected_url_fragment": expected,
                        "expected_rank": rank, "top_sources": sources})
        print(f"{question} -> expected source rank: {rank}", flush=True)
    report = {"index": str(args.index), "mode": args.mode, "cases": results,
              "recall_at_10": sum(x["expected_rank"] is not None for x in results) / len(results)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"recall_at_10": report["recall_at_10"]}, indent=2))


if __name__ == "__main__":
    main()
