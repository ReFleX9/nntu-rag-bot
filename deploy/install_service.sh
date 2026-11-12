#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
project_dir="$(pwd -P)"
service_file=/etc/systemd/system/nntu-index.service
cat > "$service_file" <<EOF
[Unit]
Description=NNTU resumable Qwen vector index
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(id -un)
WorkingDirectory=$project_dir
Environment=TOKENIZERS_PARALLELISM=false
ExecStart=/bin/bash $project_dir/deploy/run_index.sh
Restart=on-failure
RestartSec=30
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now nntu-index.service
systemctl status nntu-index.service --no-pager
