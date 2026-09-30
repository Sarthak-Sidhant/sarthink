#!/usr/bin/env bash
# Runs on the rented GPU box: embed chunks with several models, then score them with the needle test.
# Expects /workspace/sarthink with scripts/index/, processed_data/db/, processed_data/eval/ uploaded.
set -euo pipefail
cd /workspace/sarthink
pip install -q "sentence-transformers>=5" "transformers>=4.51" accelerate python-dotenv tiktoken openai 2>&1 | grep -v -i warning || true
cd scripts/index
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "keyword index"; python3 build_fts.py

# Anonymous HF downloads can hang forever; time out stalled requests and retry (partial files resume).
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
for m in Qwen/Qwen3-Embedding-8B Qwen/Qwen3-Embedding-4B Qwen/Qwen3-Embedding-0.6B Qwen/Qwen3-Reranker-4B; do
  for try in 1 2 3 4 5 6 7 8; do
    log "download $m (try $try)"
    timeout 900 python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$m')" && break
  done
done
export HF_HUB_OFFLINE=1

log "embedding 8b ctx";   python3 build_embeddings.py --model Qwen/Qwen3-Embedding-8B   --variant ctx --batch-size 64
log "embedding 8b raw";   python3 build_embeddings.py --model Qwen/Qwen3-Embedding-8B   --variant raw --batch-size 64
log "embedding 4b ctx";   python3 build_embeddings.py --model Qwen/Qwen3-Embedding-4B   --variant ctx --batch-size 64
log "embedding 0.6b ctx"; python3 build_embeddings.py --model Qwen/Qwen3-Embedding-0.6B --variant ctx --batch-size 128

log "needle eval"
python3 needle_eval.py run --configs \
  8b/raw:dense 8b/ctx:dense 4b/ctx:dense 0.6b/ctx:dense \
  8b/ctx:fts 8b/ctx:hybrid 4b/ctx:hybrid 0.6b/ctx:hybrid \
  8b/ctx:hybrid+rerank 0.6b/ctx:hybrid+rerank
log "ALL DONE"
