#!/usr/bin/env bash
# Runs on the GPU box: install deps, download the embedding model (with stall timeouts), start embed_server.
set -uo pipefail
cd /workspace
pip install -q "sentence-transformers>=5" "transformers>=4.51" accelerate fastapi uvicorn 2>&1 | grep -v -i warning || true
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
for try in 1 2 3 4 5 6 7 8; do
  echo "[$(date +%H:%M:%S)] download try $try"
  timeout 900 python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-8B')" && break
done
export HF_HUB_OFFLINE=1
echo "[$(date +%H:%M:%S)] starting embed server"
exec python3 /workspace/embed_server.py --model Qwen/Qwen3-Embedding-8B --port 8100
