#!/usr/bin/env bash
set -euo pipefail
project=/root/nntu-rag-bot
work="$project/work/v2"
[[ -f "$work/ocr_candidates_core.jsonl" && -d "$work/raw/files" ]] || {
  echo "Transfer and extract selected PDFs first" >&2
  exit 1
}
cat >/etc/systemd/system/nntu-ocr-v2.service <<SERVICE
[Unit]
Description=Resumable selective OCR for NNTU corpus v2
After=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$project
Environment=OMP_THREAD_LIMIT=1
ExecStart=$project/.venv/bin/python site_crawler/repair_pdf_text_v2.py --root $work/raw --candidates $work/ocr_candidates_core.jsonl --output $work/ocr_repairs.jsonl --backup $work/backups/ocr_repairs.jsonl --workers 6
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
SERVICE
systemctl daemon-reload
systemctl enable nntu-ocr-v2
systemctl restart nntu-ocr-v2
echo "OCR started; follow with: journalctl -u nntu-ocr-v2 -f"
