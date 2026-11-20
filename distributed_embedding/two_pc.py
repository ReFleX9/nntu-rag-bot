"""Prepare, run and merge a two-computer NNTU embedding job."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_plan(folder: Path) -> dict:
    return json.loads((folder / "plan.json").read_text(encoding="utf-8"))


def unpack(folder: Path, part: dict) -> Path:
    archive = folder / part["archive"]
    if digest(archive) != part["archive_sha256"]:
        raise ValueError(f"Archive checksum mismatch: {archive}")
    raw = folder / part["name"]
    if raw.exists() and digest(raw) == part["sha256"]:
        return raw
    if raw.exists():
        raise ValueError(f"Existing input differs from plan: {raw}")
    stage = raw.with_suffix(".partial")
    with gzip.open(archive, "rb") as source, stage.open("wb") as target:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            target.write(block)
    if digest(stage) != part["sha256"]:
        stage.unlink(missing_ok=True)
        raise ValueError(f"Unpacked checksum mismatch: {raw}")
    os.replace(stage, raw)
    return raw


def prepare(args: argparse.Namespace) -> None:
    source, folder = args.chunks.resolve(), args.folder.resolve()
    if folder.exists():
        raise FileExistsError(folder)
    rows = sum(1 for _ in source.open("rb"))
    if rows < 2:
        raise ValueError("Need at least two chunks")
    folder.mkdir(parents=True)
    sizes = (rows // 2, rows - rows // 2)
    parts = []
    full_hash = hashlib.sha256()
    with source.open("rb") as stream:
        for number, count in enumerate(sizes):
            name = f"part-{number:02d}.jsonl"
            archive = name + ".gz"
            part_hash = hashlib.sha256()
            with gzip.GzipFile(filename=str(folder / archive), mode="wb", compresslevel=6,
                               mtime=0) as output:
                for _ in range(count):
                    line = stream.readline()
                    if not line:
                        raise ValueError("Source changed while preparing")
                    json.loads(line)
                    output.write(line)
                    full_hash.update(line)
                    part_hash.update(line)
            parts.append({"name": name, "archive": archive, "rows": count,
                          "sha256": part_hash.hexdigest(),
                          "archive_sha256": digest(folder / archive)})
        if stream.read(1):
            raise ValueError("Source changed while preparing")
    plan = {"format": 1, "source_filename": source.name, "source_rows": rows,
            "source_sha256": full_hash.hexdigest(), "parts": parts}
    (folder / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({"rows": rows, "parts": sizes,
                      "archives_mb": [round((folder / p["archive"]).stat().st_size / 1e6, 1)
                                      for p in parts]}, indent=2))


def worker(args: argparse.Namespace) -> None:
    folder = args.folder.resolve()
    plan = load_plan(folder)
    part = plan["parts"][args.part]
    source = unpack(folder, part)
    model = args.model_dir.resolve()
    if not (model / "model.safetensors").is_file():
        raise FileNotFoundError(model / "model.safetensors")
    output = args.out.resolve()
    command = [sys.executable, "-u", str(ROOT / "deploy/build_resumable.py"),
               "--chunks", str(source), "--out", str(output),
               "--device", args.device, "--batch-size", str(args.batch_size),
               "--model-dir", str(model)]
    subprocess.run(command, check=True)
    result = {"plan_sha256": digest(folder / "plan.json"), "part": args.part,
              "part_sha256": part["sha256"],
              "model_sha256": digest(model / "model.safetensors"),
              "tokenizer_sha256": digest(model / "tokenizer.json")}
    (output / "two_pc_job.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Transfer this complete folder to the merge computer: {output}")


def merge(args: argparse.Namespace) -> None:
    import faiss
    import numpy as np

    folder = args.folder.resolve()
    plan = load_plan(folder)
    jobs = [args.first.resolve(), args.second.resolve()]
    stage_parent = args.out.resolve().parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    matrices, specs, manifests = [], [], []
    combined_hash = hashlib.sha256()
    seen_ids = set()
    with tempfile.TemporaryDirectory(prefix=".two-pc-", dir=stage_parent) as temporary:
        stage = Path(temporary)
        with (stage / "documents.jsonl").open("wb") as output_docs:
            for number, (part, job) in enumerate(zip(plan["parts"], jobs)):
                metadata = json.loads((job / "two_pc_job.json").read_text(encoding="utf-8"))
                if metadata["part"] != number or metadata["plan_sha256"] != digest(folder / "plan.json"):
                    raise ValueError(f"Wrong plan or part order in {job}")
                if metadata["part_sha256"] != part["sha256"]:
                    raise ValueError(f"Wrong source in {job}")
                specs.append((metadata["model_sha256"], metadata["tokenizer_sha256"]))
                manifest = json.loads((job / "index/manifest.json").read_text(encoding="utf-8"))
                manifests.append(manifest)
                if manifest["source_sha256"] != part["sha256"] or manifest["count"] != part["rows"]:
                    raise ValueError(f"Index count or source mismatch in {job}")
                for filename in ("index.faiss", "documents.jsonl"):
                    path = job / "index" / filename
                    if digest(path) != manifest["files"][filename]:
                        raise ValueError(f"Index file checksum mismatch: {path}")
                index = faiss.deserialize_index(np.frombuffer(
                    (job / "index/index.faiss").read_bytes(), dtype=np.uint8).copy())
                if index.ntotal != part["rows"] or index.metric_type != faiss.METRIC_INNER_PRODUCT:
                    raise ValueError(f"Invalid FAISS index in {job}")
                matrices.append(index)
                raw = unpack(folder, part)
                with raw.open("rb") as inputs, (job / "index/documents.jsonl").open("rb") as documents:
                    for source_line in inputs:
                        combined_hash.update(source_line)
                        document_line = documents.readline()
                        if not document_line:
                            raise ValueError(f"Missing document in {job}")
                        source_id = json.loads(source_line)["chunk_id"]
                        result_id = json.loads(document_line)["metadata"]["chunk_id"]
                        if source_id != result_id or source_id in seen_ids:
                            raise ValueError(f"Document order or duplicate ID in {job}")
                        seen_ids.add(source_id)
                        output_docs.write(document_line)
                    if documents.readline():
                        raise ValueError(f"Extra document in {job}")
        if combined_hash.hexdigest() != plan["source_sha256"]:
            raise ValueError("Combined source differs from original")
        if specs[0] != specs[1] or manifests[0]["embedding"] != manifests[1]["embedding"]:
            raise ValueError("Different model or embedding configuration")
        if matrices[0].d != matrices[1].d:
            raise ValueError("Different vector dimensions")
        index = faiss.IndexFlatIP(matrices[0].d)
        for item in matrices:
            index.add(item.reconstruct_n(0, item.ntotal))
        (stage / "index.faiss").write_bytes(faiss.serialize_index(index).tobytes())
        result = dict(manifests[0])
        result.update(source_filename=plan["source_filename"], source_sha256=plan["source_sha256"],
                      count=index.ntotal, audit={"two_pc_parts": 2, "source_rows": plan["source_rows"]},
                      files={name: digest(stage / name) for name in ("documents.jsonl", "index.faiss")})
        (stage / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(stage, args.out.resolve())
    print(f"Merged {index.ntotal} vectors into {args.out.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("prepare")
    split.add_argument("--chunks", type=Path, required=True)
    split.add_argument("--folder", type=Path, required=True)
    split.set_defaults(run=prepare)
    run = commands.add_parser("worker")
    run.add_argument("--folder", type=Path, required=True)
    run.add_argument("--part", type=int, choices=(0, 1), required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    run.add_argument("--batch-size", type=int, default=4)
    run.set_defaults(run=worker)
    join = commands.add_parser("merge")
    join.add_argument("--folder", type=Path, required=True)
    join.add_argument("--first", type=Path, required=True)
    join.add_argument("--second", type=Path, required=True)
    join.add_argument("--out", type=Path, required=True)
    join.set_defaults(run=merge)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
