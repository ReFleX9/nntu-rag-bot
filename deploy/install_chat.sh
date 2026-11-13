#!/usr/bin/env bash
# Run on the Selectel host after the index has been built.
set -euo pipefail

project=/root/nntu-rag-bot
index="$project/work/kb-core/index"
token_file="$project/work/chat.token"
port="${NNTU_CHAT_PORT:-8080}"

[[ -f "$index/index.faiss" && -f "$index/documents.jsonl" && -f "$index/manifest.json" ]] || {
    echo "Completed index not found at $index" >&2
    exit 1
}
[[ -x "$project/.venv/bin/python" ]] || {
    echo "Python venv not found at $project/.venv; run deploy/setup_gpu.sh" >&2
    exit 1
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
systemctl enable --now ollama
if ! ollama list | grep -q '^qwen3:4b-instruct[[:space:]]'; then
    ollama pull qwen3:4b-instruct
fi

mkdir -p "$project/work"
if [[ ! -s "$token_file" ]]; then
    umask 077
    "$project/.venv/bin/python" -c 'import secrets; print(secrets.token_urlsafe(32))' > "$token_file"
fi
chmod 600 "$token_file"

cat >/etc/systemd/system/nntu-chat.service <<SERVICE
[Unit]
Description=NNTU RAG chat with hybrid search and Ollama
Wants=network-online.target
Requires=ollama.service
After=network-online.target ollama.service

[Service]
Type=simple
User=root
WorkingDirectory=$project/rag_hybrid_langchain/rag_hybrid
Environment=HF_HUB_OFFLINE=1
Environment=TRANSFORMERS_OFFLINE=1
Environment=TOKENIZERS_PARALLELISM=false
ExecStart=$project/.venv/bin/python chat_web.py --index $index --token-file $token_file --host 0.0.0.0 --port $port --device cuda --llm-model qwen3:4b-instruct
Restart=on-failure
RestartSec=15
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable nntu-chat
systemctl restart nntu-chat
echo "Chat service installed. Loading the knowledge base can take several minutes."
echo "Check: systemctl status nntu-chat --no-pager"
echo "Logs:  journalctl -u nntu-chat -f"
echo "Access token (keep private): $token_file"
