#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p work
if [[ ! -s work/core.jsonl ]]; then
    .venv/bin/python deploy/select_corpus.py \
        data/nntu_site/extracted/chunks.jsonl work/core.jsonl
fi
exec bash deploy/build_gpu.sh work/core.jsonl work/kb-core
