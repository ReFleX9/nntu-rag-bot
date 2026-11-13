#!/usr/bin/env bash
set -euo pipefail

project=/root/nntu-rag-bot
corpus="$project/work/v2/core.jsonl"
[[ -s "$corpus" ]] || { echo "Missing selected v2 corpus: $corpus" >&2; exit 1; }

cat >/etc/systemd/system/nntu-index-v2.service <<SERVICE
[Unit]
Description=Resumable Qwen index for NNTU corpus v2
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$project
Environment=TOKENIZERS_PARALLELISM=false
Environment=BATCH_SIZE=16
ExecStart=/bin/bash $project/deploy/build_gpu.sh work/v2/core.jsonl work/kb-v2
Restart=on-failure
RestartSec=30
TimeoutStopSec=90

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable nntu-index-v2
systemctl restart nntu-index-v2
echo "Indexing started; follow with: journalctl -u nntu-index-v2 -f"
