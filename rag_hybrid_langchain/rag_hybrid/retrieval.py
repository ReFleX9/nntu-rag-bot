"""FAISS + BM25 + reciprocal rank fusion; JSON persistence and cited context."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
import tempfile
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from nltk.stem.snowball import RussianStemmer
from pydantic import Field

from corpus import audit_chunks, index_text, read_chunks, select_chunks, sha256_file
from embeddings import EmbeddingConfig, QwenEmbeddings

LOG = logging.getLogger(__name__)
FORMAT_VERSION = 1
TOKENIZER_VERSION = "ru_snowball_technical_v1"
TOKEN_RE = re.compile(r"--?[a-zа-я0-9][a-zа-я0-9_./:-]*|[a-zа-я0-9]+(?:[_./:-][a-zа-я0-9]+)*(?:\+\+|#)?")
RU = RussianStemmer()
# Keep negations and domain abbreviations. No downloaded NLTK resources are needed.
STOP = set("и в во на с со к ко у по из от до для о об а или как что это этой этот того ли же the a an of to in and is are how what".split())


@lru_cache(maxsize=100000)
def stem_word(word: str) -> str:
    return RU.stem(word) if len(word) > 3 and re.fullmatch("[а-я]+", word) else word


def tokenize(text: str) -> list[str]:
    result = []
    text = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    for token in TOKEN_RE.findall(text):
        if token in STOP:
            continue
        result.append(stem_word(token))
        # Preserve pg_hba.conf/--foo-bar and additionally match their components.
        parts = re.findall(r"[a-zа-я0-9]+", token)
        if len(parts) > 1 or token.startswith("-"):
            result.extend(stem_word(p) for p in parts if p not in STOP)
    return result


def normalize_vectors(vectors) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32, order="C")
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("Некорректная матрица эмбеддингов")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("Модель вернула нулевой эмбеддинг")
    return matrix / norms[:, None]


def to_document(row: dict) -> Document:
    metadata = {k: v for k, v in row.items() if k != "text"}
    metadata["source_folder"] = row["document_id"].replace("\\", "/").split("/")[0]
    return Document(id=row["chunk_id"], page_content=row["text"], metadata=metadata)


def make_store(docs: list[Document], index, embeddings: Embeddings) -> FAISS:
    return FAISS(
        embedding_function=embeddings, index=index,
        docstore=InMemoryDocstore({d.metadata["chunk_id"]: d for d in docs}),
        index_to_docstore_id={i: d.metadata["chunk_id"] for i, d in enumerate(docs)},
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def build_index(chunks_path: str | Path, output: str | Path, embeddings: Embeddings,
                embedding_spec: dict, *, min_ocr_confidence: float | None = None,
                exclude_needs_review: bool = False, block_size: int = 128) -> dict:
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(f"Каталог уже существует: {target}. Укажите новый --out.")
    if block_size < 1:
        raise ValueError("block_size должен быть положительным")
    rows = read_chunks(chunks_path)
    report = audit_chunks(rows)
    for fatal in ("content_hash_mismatch", "cross_document_neighbor", "duplicate_index_in_document"):
        if fatal in report["issues"]:
            raise ValueError(f"Ошибка корпуса: {fatal}; сначала выполните audit.")
    accepted, skipped = select_chunks(rows, min_ocr_confidence=min_ocr_confidence,
                                      exclude_needs_review=exclude_needs_review)
    if report["issues"]:
        LOG.warning("Особенности связей чанков: %s", report["issues"])
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        index = None
        with (staging / "documents.jsonl").open("w", encoding="utf-8") as stream:
            for start in range(0, len(accepted), block_size):
                block = accepted[start:start+block_size]
                vectors = normalize_vectors(embeddings.embed_documents([index_text(r) for r in block]))
                if len(vectors) != len(block):
                    raise ValueError("Число эмбеддингов не совпадает с числом чанков")
                if index is None:
                    index = faiss.IndexFlatIP(vectors.shape[1])
                if index.d != vectors.shape[1]:
                    raise ValueError("Размерность эмбеддингов изменилась в процессе индексации")
                index.add(vectors)
                for row in block:
                    doc = to_document(row)
                    stream.write(json.dumps({"page_content": doc.page_content, "metadata": doc.metadata},
                                            ensure_ascii=False) + "\n")
                LOG.info("Индексировано %s / %s", min(start+len(block), len(accepted)), len(accepted))
        # Python writes bytes, so Cyrillic Windows paths do not go through FAISS fopen.
        (staging / "index.faiss").write_bytes(faiss.serialize_index(index).tobytes())
        manifest = {
            "format_version": FORMAT_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_filename": Path(chunks_path).name, "source_sha256": sha256_file(chunks_path),
            "embedding": embedding_spec, "dimension": index.d, "count": index.ntotal,
            "metric": "cosine_via_normalized_inner_product", "tokenizer_version": TOKENIZER_VERSION,
            "bm25": {"k1": 1.5, "b": .75}, "audit": report,
            "filter": {"min_ocr_confidence": min_ocr_confidence,
                       "exclude_needs_review": exclude_needs_review, "skipped": skipped},
            "packages": {name: version(name) for name in
                         ("langchain-core", "langchain-community", "rank-bm25", "faiss-cpu", "nltk")},
            "files": {name: sha256_file(staging / name) for name in ("documents.jsonl", "index.faiss")},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.rename(target)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


class KnowledgeBase:
    def __init__(self, documents: list[Document], store: FAISS | None = None, manifest: dict | None = None):
        if not documents:
            raise ValueError("Пустая база знаний")
        self.documents, self.store, self.manifest = documents, store, manifest or {}
        self.by_id = {d.metadata["chunk_id"]: d for d in documents}
        if len(self.by_id) != len(documents):
            raise ValueError("Повторные chunk_id в базе")
        search_docs = [Document(page_content=index_text({"text": d.page_content, **d.metadata}),
                                metadata=d.metadata) for d in documents]
        LOG.info("Создание BM25 для %s чанков", len(documents))
        self.bm25 = BM25Retriever.from_documents(
            search_docs, preprocess_func=tokenize,
            bm25_params=self.manifest.get("bm25", {"k1": 1.5, "b": .75}),
        )

    @classmethod
    def from_chunks(cls, path: str | Path, **selection):
        rows, _ = select_chunks(read_chunks(path), **selection)
        return cls([to_document(r) for r in rows])

    @classmethod
    def load(cls, path: str | Path, *, device: str = "auto", batch_size: int = 8,
             local_files_only: bool = False, dense: bool = True,
             embeddings: Embeddings | None = None):
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["format_version"] != FORMAT_VERSION or manifest["tokenizer_version"] != TOKENIZER_VERSION:
            raise ValueError("Версия индекса или токенизации изменилась. Пересоздайте индекс.")
        for name in ("documents.jsonl", "index.faiss"):
            if sha256_file(path / name) != manifest["files"][name]:
                raise ValueError(f"Повреждён или заменён файл индекса: {name}")
        docs = []
        with (path / "documents.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                docs.append(Document(id=row["metadata"]["chunk_id"], **row))
        if len(docs) != manifest["count"]:
            raise ValueError("Число документов отличается от manifest")
        store = None
        if dense:
            spec = dict(manifest["embedding"])
            backend = spec.pop("backend")
            if embeddings is None:
                if backend != "qwen3_sentence_transformers":
                    raise ValueError("Тестовый индекс: передайте соответствующий embeddings явно")
                embeddings = QwenEmbeddings(EmbeddingConfig(**spec), device, batch_size, local_files_only)
            index = faiss.deserialize_index(np.frombuffer((path / "index.faiss").read_bytes(), dtype=np.uint8).copy())
            if index.ntotal != len(docs) or index.d != manifest["dimension"]:
                raise ValueError("FAISS и documents.jsonl не согласованы")
            if index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise ValueError("Ожидался индекс с метрикой inner product")
            store = make_store(docs, index, embeddings)
        return cls(docs, store, manifest)

    def search(self, query: str, *, top_k: int = 8, candidate_k: int = 40,
               bm25_weight: float = .5, faiss_weight: float = .5, rrf_k: int = 60,
               filters: dict[str, Any] | None = None) -> list[Document]:
        if not query.strip():
            raise ValueError("Пустой запрос")
        if top_k < 1 or candidate_k < top_k or rrf_k < 1:
            raise ValueError("Требуется candidate_k >= top_k >= 1 и rrf_k >= 1")
        if not all(math.isfinite(w) and w >= 0 for w in (bm25_weight, faiss_weight)) or bm25_weight+faiss_weight == 0:
            raise ValueError("Веса должны быть конечными, неотрицательными и не оба нулевыми")
        if faiss_weight and self.store is None:
            raise ValueError("FAISS не загружен. Для BM25 задайте faiss_weight=0.")
        allowed_fields = {"source_folder", "document_id", "kind"}
        filters = filters or {}
        if set(filters) - allowed_fields:
            raise ValueError(f"Поддерживаются фильтры: {sorted(allowed_fields)}")
        matches: Callable[[dict], bool] = lambda metadata: all(metadata.get(k) == v for k, v in filters.items())
        ranking = defaultdict(lambda: {"rrf_score": 0.0})
        total_weight = bm25_weight + faiss_weight
        if bm25_weight:
            terms = tokenize(query)
            if terms:
                scores = self.bm25.vectorizer.get_scores(terms)
                # BM25 otherwise returns arbitrary documents when every score is zero.
                unique_terms = set(terms)
                eligible = [i for i, freq in enumerate(self.bm25.vectorizer.doc_freqs)
                            if matches(self.documents[i].metadata) and any(t in freq for t in unique_terms)]
                ordered = sorted(eligible, key=lambda i: (-float(scores[i]), self.documents[i].metadata["chunk_id"]))
                for rank, i in enumerate(ordered[:candidate_k], 1):
                    item = ranking[self.documents[i].metadata["chunk_id"]]
                    item.update(bm25_rank=rank, bm25_score=float(scores[i]))
                    item["rrf_score"] += bm25_weight / total_weight / (rrf_k + rank)
        if faiss_weight:
            vector = normalize_vectors([self.store.embedding_function.embed_query(query)])[0].tolist()
            if len(vector) != self.store.index.d:
                raise ValueError("Размерность query-эмбеддинга отличается от индекса")
            hits = self.store.similarity_search_with_score_by_vector(
                vector, k=candidate_k, filter=matches if filters else None,
                fetch_k=len(self.documents) if filters else candidate_k,
            )
            for rank, (doc, score) in enumerate(hits, 1):
                item = ranking[doc.metadata["chunk_id"]]
                item.update(faiss_rank=rank, cosine_score=float(score))
                item["rrf_score"] += faiss_weight / total_weight / (rrf_k + rank)
        # Questions asking where to go should prefer a relevant website page
        # over a policy PDF that merely repeats the same terminology.
        if re.search(r"\b(?:где|найти|открыть|перейти|ссылка|адрес|находится|расположен)\b",
                     query.casefold()):
            query_terms = set(tokenize(query))
            for cid, item in ranking.items():
                metadata = self.by_id[cid].metadata
                if metadata.get("source_type") != "html":
                    continue
                title_terms = set(tokenize(str(metadata.get("title") or "")))
                overlap = len(query_terms & title_terms)
                if overlap >= 2:
                    item["navigation_boost"] = .008 * overlap
                    item["rrf_score"] += item["navigation_boost"]
        ranked_ids = sorted(ranking, key=lambda cid: (-ranking[cid]["rrf_score"], cid))
        selected, seen = [], {}
        for cid in ranked_ids:
            doc = self.by_id[cid]
            key = (doc.metadata["document_id"], hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest())
            if key in seen:
                seen[key].metadata["retrieval"]["duplicate_citations"].append({
                    "chunk_id": cid, "page": doc.metadata["page"], "citation_label": doc.metadata.get("citation_label")})
                continue
            if len(selected) >= top_k:
                continue
            hit = Document(id=cid, page_content=doc.page_content, metadata={
                **doc.metadata, "retrieval": {**ranking[cid], "duplicate_citations": []}})
            seen[key] = hit
            selected.append(hit)
        return selected

    def expand_context(self, hits: list[Document], *, neighbors: int = 1,
                       filters: dict[str, Any] | None = None) -> list[Document]:
        if not 0 <= neighbors <= 5:
            raise ValueError("neighbors: 0..5")
        result, seen = [], set()
        filters = filters or {}

        def add(doc: Document, anchor: Document, reason: str):
            cid = doc.metadata["chunk_id"]
            if cid not in seen:
                result.append(Document(id=cid, page_content=doc.page_content, metadata={
                    **doc.metadata, "context_role": reason,
                    "anchor_chunk_id": anchor.metadata["chunk_id"]}))
                seen.add(cid)

        # Reserve positions for all ranked hits first, then add surrounding context.
        for hit in hits:
            add(hit, hit, "hit")
        for hit in hits:
            for direction in ("prev_chunk_id", "next_chunk_id"):
                current = hit
                walked = {hit.metadata["chunk_id"]}
                for _ in range(neighbors):
                    other = self.by_id.get(current.metadata.get(direction))
                    if other is None or other.metadata["chunk_id"] in walked:
                        break
                    walked.add(other.metadata["chunk_id"])
                    if other.metadata["document_id"] != hit.metadata["document_id"]:
                        break
                    if abs(other.metadata["page"] - hit.metadata["page"]) > 1:
                        break
                    section_a = hit.metadata.get("section_path") or hit.metadata.get("section_title")
                    section_b = other.metadata.get("section_path") or other.metadata.get("section_title")
                    if section_a and section_b and section_a != section_b:
                        break
                    if any(other.metadata.get(k) != v for k, v in filters.items()):
                        break
                    add(other, hit, "neighbor")
                    current = other
        return result

    def as_retriever(self, **kwargs) -> "HybridRetriever":
        return HybridRetriever(knowledge_base=self, search_kwargs=kwargs)


class HybridRetriever(BaseRetriever):
    knowledge_base: Any
    search_kwargs: dict = Field(default_factory=dict)

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        return self.knowledge_base.search(query, **self.search_kwargs)


def format_context(docs: list[Document], max_chars: int = 24000) -> tuple[str, list[dict]]:
    if max_chars < 1:
        raise ValueError("max_chars должен быть положительным")
    blocks, sources, used = [], [], 0
    for doc in docs:
        meta = doc.metadata
        number = len(sources) + 1
        header = (f"[{number}] Документ: {meta.get('title') or meta['document_id']}\n"
                  f"Путь: {meta['document_id']}\nPDF страница: {meta['page']}\n"
                  f"Раздел: {meta.get('section_title') or 'не указан'}\n"
                  f"Тип: {meta.get('kind')}; chunk_id: {meta['chunk_id']}")
        source_url = next((url for url in meta.get("source_urls") or []
                           if isinstance(url, str) and url.startswith(("https://", "http://"))), None)
        if source_url:
            header += f"\nАдрес первоисточника: {source_url}"
        if meta.get("is_ocr"):
            header += f"\nOCR confidence: {meta.get('ocr_confidence')}; возможны ошибки распознавания"
        block = header + "\n" + doc.page_content
        cost = len(block) + (2 if blocks else 0)
        if used + cost > max_chars:
            continue  # Keep passages and Markdown table rows whole.
        blocks.append(block)
        used += cost
        sources.append({"number": number, **meta})
    if docs and not blocks:
        raise ValueError("Ни один полный чанк не помещается в бюджет контекста. Увеличьте --max-context-chars.")
    return "\n\n".join(blocks), sources
