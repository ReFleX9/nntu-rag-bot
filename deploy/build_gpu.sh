#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $# -ne 2 ]]; then
    echo "Usage: bash deploy/build_gpu.sh CHUNKS_JSONL OUTPUT_DIRECTORY" >&2
    exit 2
fi

if [[ ! -s "$1" ]] || head -n 1 -- "$1" | grep -q '^version https://git-lfs.github.com/spec/v1'; then
    echo "Missing JSONL payload: $1. Run setup_gpu.sh to fetch the corpus." >&2
    exit 2
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec /usr/bin/time -v .venv/bin/python -u deploy/build_resumable.py \
    --chunks "$1" --out "$2" \
    --shard-size "${SHARD_SIZE:-256}" --batch-size "${BATCH_SIZE:-16}"
