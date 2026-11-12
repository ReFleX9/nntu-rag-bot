#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y git git-lfs curl
project=/root/nntu-rag-bot
if [[ -d "$project/.git" ]]; then
    git -C "$project" pull --ff-only
else
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/ReFleX9/nntu-rag-bot.git "$project"
fi
cd "$project"
bash deploy/setup_gpu.sh
bash deploy/install_service.sh
echo "NNTU index service started. Follow with: journalctl -u nntu-index -f"
