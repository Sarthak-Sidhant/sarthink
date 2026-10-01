"""Stateless query-embedding service for the GPU box. Holds only the embedding model — no chat data.

POST /embed {"texts": [...], "query": true} -> {"vectors": [[...], ...]}   (L2-normalised)
GET  /health

Run on the box:  python3 embed_server.py --model Qwen/Qwen3-Embedding-8B --port 8100
Reach it from the laptop through an SSH tunnel (never expose the port publicly).
On an Apple-silicon Mac it runs on the GPU (MPS); scripts/api/mac_encoder.sh serves it to the VPS.
"""
import argparse

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
MODEL = None


class EmbedRequest(BaseModel):
    texts: list[str]
    query: bool = True


@app.get("/health")
def health():
    return {"ok": MODEL is not None, "device": str(MODEL.device) if MODEL else None}


@app.post("/embed")
def embed(req: EmbedRequest):
    prompts = getattr(MODEL, "prompts", {}) or {}
    kw = {"prompt_name": "query"} if req.query and "query" in prompts else {}
    with torch.inference_mode():
        v = MODEL.encode(req.texts, normalize_embeddings=True, convert_to_numpy=True, **kw)
    return {"vectors": v.astype("float32").tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    kw = {"model_kwargs": {"torch_dtype": torch.bfloat16}} if dev in ("cuda", "mps") else {}   # 8B: ~16 GB
    MODEL = SentenceTransformer(a.model, device=dev, **kw)
    MODEL.max_seq_length = 1024
    uvicorn.run(app, host="127.0.0.1", port=a.port)
