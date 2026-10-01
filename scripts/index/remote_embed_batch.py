"""Runs ON the GPU box: embed documents from a gzipped JSONL ({"id","text"}) through the already-running
embed_server (localhost, document mode) and save float16 vectors + ids. Avoids loading a second model copy
and avoids shipping float lists over the SSH tunnel.

Usage (on the box): python3 remote_embed_batch.py to_embed.jsonl.gz out_prefix [--url http://127.0.0.1:8100]
"""
import argparse
import gzip
import json
import time
import urllib.error
import urllib.request

import numpy as np


def main(a):
    rows = [json.loads(l) for l in gzip.open(a.src, "rt")]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))      # similar lengths batch well
    vecs = None
    t0 = time.time()
    def embed(idx):
        body = json.dumps({"texts": [rows[i]["text"] for i in idx], "query": False}).encode()
        req = urllib.request.Request(f"{a.url}/embed", data=body, headers={"content-type": "application/json"})
        try:
            return np.asarray(json.load(urllib.request.urlopen(req, timeout=600))["vectors"], dtype=np.float32)
        except urllib.error.HTTPError:
            if len(idx) == 1:
                raise
            h = len(idx) // 2                                  # out of memory: split the batch and retry
            return np.concatenate([embed(idx[:h]), embed(idx[h:])])

    done, pos = 0, 0
    while pos < len(order):
        idx, chars = [], 0
        while pos < len(order) and (not idx or chars + len(rows[order[pos]]["text"]) <= a.max_chars):
            chars += len(rows[order[pos]]["text"])
            idx.append(order[pos])
            pos += 1
        v = embed(idx)
        if vecs is None:
            vecs = np.zeros((len(rows), v.shape[1]), dtype=np.float16)
        vecs[idx] = v.astype(np.float16)
        done += len(idx)
        if done // 500 != (done - len(idx)) // 500:
            print(f"{done}/{len(rows)}  {done / (time.time() - t0):.1f}/s", flush=True)
    np.save(f"{a.out}.npy", vecs)
    json.dump([r["id"] for r in rows], open(f"{a.out}.ids.json", "w"))
    print(f"saved {vecs.shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--max-chars", type=int, default=24000, help="text per request; long chunks get small batches")
    main(ap.parse_args())
