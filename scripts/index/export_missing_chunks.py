"""Write the chunks that have no vector yet to a gzipped JSONL ({"id","text"}) for remote_embed_batch.py.

After an import, only new chunks need embedding: run this, embed the file on the GPU box, then
merge_embeddings.py <prefix> to fold the new vectors into the existing file. Texts are built exactly like
build_embeddings.py builds them (header, LLM context line for the ctx variant, chunk text).

Usage: python3 scripts/index/export_missing_chunks.py out.jsonl.gz [--model qwen3-embedding-8b --variant ctx]
"""
import argparse
import gzip
import json

from build_embeddings import OUT_DIR, chunk_texts
from common import connect_index


def main(a):
    base = OUT_DIR / f"emb_{a.model}_{a.variant}"
    meta = json.load(open(f"{base}.ids.json"))
    have = set(meta["ids"])
    ids, texts = chunk_texts(connect_index(), a.variant, meta.get("ctx_model", "deepseek-flash"),
                             meta.get("ctx_version", "v3"))
    missing = [(i, t) for i, t in zip(ids, texts) if i not in have]
    no_ctx = sum(1 for _, t in missing if "\nContext: " not in t)
    with gzip.open(a.out, "wt", encoding="utf-8") as f:
        for cid, text in missing:
            f.write(json.dumps({"id": cid, "text": text}, ensure_ascii=False) + "\n")
    print(f"{len(missing)} chunks without a vector -> {a.out} ({no_ctx} have no context line)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", default="qwen3-embedding-8b")
    ap.add_argument("--variant", default="ctx")
    main(ap.parse_args())
