"""Build a Qwen/FAISS index in persistent shards and resume after interruption."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "rag_hybrid_langchain" / "rag_hybrid"))
from corpus import sha256_file  # noqa: E402
from embeddings import EmbeddingConfig, QwenEmbeddings  # noqa: E402
from retrieval import build_index  # noqa: E402


def valid_shard(path: Path, expected_sha: str) -> bool:
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        return (manifest["source_sha256"] == expected_sha
                and all(sha256_file(path / name) == manifest["files"][name]
                        for name in ("index.faiss", "documents.jsonl")))
    except (OSError, ValueError, KeyError):
        return False


def merge(shards: list[Path], target: Path, source: Path, spec: dict) -> None:
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".merge-", dir=target.parent) as scratch:
        stage = Path(scratch)
        merged = None
        count = 0
        with (stage / "documents.jsonl").open("wb") as docs:
            for shard in shards:
                item = faiss.deserialize_index(np.frombuffer(
                    (shard / "index.faiss").read_bytes(), dtype=np.uint8).copy())
                if merged is None:
                    merged = faiss.IndexFlatIP(item.d)
                if item.d != merged.d:
                    raise ValueError("Shard dimensions differ")
                merged.add(item.reconstruct_n(0, item.ntotal))
                count += item.ntotal
                with (shard / "documents.jsonl").open("rb") as part:
                    for block in iter(lambda: part.read(4 * 1024 * 1024), b""):
                        docs.write(block)
        if merged is None:
            raise ValueError("No shards")
        (stage / "index.faiss").write_bytes(faiss.serialize_index(merged).tobytes())
        template = json.loads((shards[0] / "manifest.json").read_text(encoding="utf-8"))
        template.update(created_at_utc=datetime.now(timezone.utc).isoformat(),
                        source_filename=source.name, source_sha256=sha256_file(source),
                        embedding=spec, dimension=merged.d, count=count,
                        audit={"shards": len(shards)},
                        files={name: sha256_file(stage / name)
                               for name in ("documents.jsonl", "index.faiss")})
        (stage / "manifest.json").write_text(
            json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(stage, target)
    print(f"Completed index: {count} vectors in {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.shard_size < 1 or args.batch_size < 1:
        parser.error("shard-size and batch-size must be positive")
    source = args.chunks.resolve()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    with source.open(encoding="utf-8-sig") as counter:
        total_rows = sum(1 for _ in counter)
    config = EmbeddingConfig(model="Qwen/Qwen3-Embedding-0.6B",
                             revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
                             max_length=2400,
                             instruction="Retrieve the passage that directly answers the question about Nizhny Novgorod State Technical University.")
    spec = {"backend": "qwen3_sentence_transformers", **vars(config)}
    identity = {"source_sha256": source_hash, "shard_size": args.shard_size,
                "embedding": spec}
    identity_path = root / "job.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Source or model changed. Choose a different output directory.")
    else:
        pending = identity_path.with_suffix(".partial")
        pending.write_text(json.dumps(identity, indent=2), encoding="utf-8")
        os.replace(pending, identity_path)

    embedder = None
    shards = []
    total = 0
    newly_built = 0
    started = time.monotonic()
    with source.open(encoding="utf-8-sig") as stream:
        number = 0
        while True:
            lines = []
            for _ in range(args.shard_size):
                line = stream.readline()
                if not line:
                    break
                lines.append(line)
            if not lines:
                break
            shard = root / f"shard_{number:05d}"
            digest = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
            if shard.exists():
                if not valid_shard(shard, digest):
                    raise ValueError(f"Incomplete or damaged checkpoint: {shard}")
                print(f"Resuming after {shard.name}", flush=True)
            else:
                if embedder is None:
                    embedder = QwenEmbeddings(config, device="cuda",
                                              batch_size=args.batch_size)
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                                 suffix=".jsonl", dir=root,
                                                 delete=False) as temporary:
                    temporary.writelines(lines)
                    part = Path(temporary.name)
                try:
                    build_index(part, shard, embedder, embedder.specification(),
                                block_size=args.batch_size)
                finally:
                    part.unlink(missing_ok=True)
                newly_built += len(lines)
                rate = newly_built / max(time.monotonic() - started, 1)
                remaining_hours = (total_rows - total - len(lines)) / rate / 3600
                print(f"Checkpoint {shard.name}: {total + len(lines)}/{total_rows} vectors; "
                      f"{rate:.1f} vectors/s; ETA {remaining_hours:.1f} h", flush=True)
            shards.append(shard)
            total += len(lines)
            number += 1
    if not (root / "index" ).exists():
        merge(shards, root / "index", source, spec)
    print(f"DONE {total} vectors", flush=True)


if __name__ == "__main__":
    main()
