#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

nvidia-smi
sudo apt-get update
sudo apt-get install -y python3-venv git-lfs tmux time
git lfs install --local
git lfs pull --include="data/nntu_site/extracted/chunks.jsonl" --exclude=""

python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r rag_hybrid_langchain/rag_hybrid/requirements.txt
.venv/bin/python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable. Check NVIDIA driver and GPU assignment.")
print(torch.__version__, torch.cuda.get_device_name(0))
tensor = torch.ones((16, 16), device="cuda", dtype=torch.float16)
print("CUDA FP16 matmul:", (tensor @ tensor)[0, 0].item())
PY
