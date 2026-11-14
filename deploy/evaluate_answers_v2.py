#!/usr/bin/env python3
"""Check generated answers and citations on the candidate index before rollout."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "rag_hybrid_langchain" / "rag_hybrid"))
from kb import answer, load_kb  # noqa: E402

QUESTIONS = (
    "Как отчислиться из НГТУ?",
    "Какие документы нужны для поступления?",
    "Где находится электронная образовательная среда?",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = SimpleNamespace(index=args.index, chunks=None, mode="hybrid", device="cuda",
                               local_files_only=True, top_k=6, candidate_k=35,
                               bm25_weight=.5, faiss_weight=.5, rrf_k=60,
                               neighbors=0, max_context_chars=7000,
                               source_folder=None, document_id=None, kind=None,
                               language="ru", llm_model="qwen3:4b-instruct",
                               ollama_url="http://127.0.0.1:11434",
                               num_ctx=8192, num_predict=350)
    kb = load_kb(settings)
    results = []
    for question in QUESTIONS:
        result = answer(kb, question, settings)
        results.append({"question": question, "answer": result["answer"],
                        "sources": result["sources"],
                        "retrieval_seconds": result["retrieval_seconds"]})
        print(json.dumps({"question": question, "answer": result["answer"][:700],
                          "sources": [(s.get("citation_label"),
                                       (s.get("source_urls") or [""])[0])
                                      for s in result["sources"][:6]]},
                         ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
