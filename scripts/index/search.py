"""Hybrid retrieval over chunks: dense (numpy exact search) + keyword (FTS5), fused with RRF,
then an optional "me" boost. Returns chunk ids with scores; callers expand to sessions/messages.

    from search import Searcher
    s = Searcher(model="Qwen/Qwen3-Embedding-0.6B", variant="ctx")
    s.search("when did I talk to harsh about wayland", k=10)

CLI: python3 scripts/index/search.py "query" [--mode hybrid|dense|fts] [-k 10]
"""
import argparse
import json
import re

import numpy as np

from common import connect_index
from build_embeddings import OUT_DIR, slug

RRF_K = 60
MODELS = {"0.6b": "Qwen/Qwen3-Embedding-0.6B", "4b": "Qwen/Qwen3-Embedding-4B", "8b": "Qwen/Qwen3-Embedding-8B"}
# Hosted query encoder: the same Qwen3 weights on OpenRouter (vectors match the GPU-built ones, cosine ~0.9998).
# Used with encoder_url="openrouter" (EMBED_URL=openrouter); needs OPENROUTER_API_KEY in the environment or .env.
OPENROUTER_MODELS = {"Qwen/Qwen3-Embedding-8B": "qwen/qwen3-embedding-8b", "Qwen/Qwen3-Embedding-4B": "qwen/qwen3-embedding-4b"}
# Nebius measured fastest (median 0.66s vs 1.3s DeepInfra, 2.0s SiliconFlow fp8) and matches the stored vectors
# best (cosine 0.9999); DeepInfra is the fallback if Nebius is down.
OPENROUTER_ROUTE = {"order": ["nebius", "deepinfra"], "allow_fallbacks": True}
QUERY_PROMPT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
RERANK_INSTRUCTION = "Given a question Sarthak asks about his own chat history, judge whether this chat excerpt answers it"


class Reranker:
    """Qwen3-Reranker scoring as in its model card: P("yes") from the final-token logits."""

    def __init__(self, model="Qwen/Qwen3-Reranker-4B", device=None, max_len=1024):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype).to(self.device).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
                       "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
                       "<|im_start|>user\n")
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.max_len = max_len

    def score(self, query, docs, batch_size=32):
        """Truncate only the document so the instruction and the yes/no suffix always survive."""
        head = f"{self.prefix}<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {query}\n<Document>: "
        fixed = len(self.tok(head + self.suffix, add_special_tokens=False)["input_ids"])
        room = max(64, self.max_len - fixed)
        doc_ids = self.tok(docs, add_special_tokens=False)["input_ids"]
        texts = [head + self.tok.decode(ids[:room]) + self.suffix for ids in doc_ids]
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], padding=True, truncation=False, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                # only the final position is scored; computing full-sequence logits over a 151k vocab is huge
                logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :]
            pair = self.torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            out += self.torch.softmax(pair, dim=1)[:, 1].tolist()
        return out


class Searcher:
    """encoder_url: base URL of embed_server.py (query vectors computed on a remote GPU); without it the
    embedding model is loaded locally. Filters (platform, accounts, since, until, me_only) apply to both
    the dense and keyword legs; accounts is a set of account ids (a person is the set of their accounts)."""

    def __init__(self, model="Qwen/Qwen3-Embedding-0.6B", variant="ctx", device=None, me_boost=0.0, reranker=None,
                 encoder_url=None):
        model = MODELS.get(model, model)
        self.reranker = reranker
        self.encoder_url = encoder_url.rstrip("/") if encoder_url else None
        self.model_name = model
        self._qcache = {}
        self.conn = connect_index()
        base = OUT_DIR / f"emb_{slug(model)}_{variant}"
        self.model = None
        current = [r[0] for r in self.conn.execute("SELECT chunk_id FROM Chunks ORDER BY chunk_id")]
        self.has_vec = None
        if (OUT_DIR / f"{base.name}.npy").exists():
            vecs = np.load(f"{base}.npy")
            with open(f"{base}.ids.json") as f:
                vec_ids = json.load(f)["ids"]
            if vec_ids != current:
                # Chunks were rebuilt after an import but not all re-embedded yet: align vectors by chunk id.
                # New chunks are keyword-only until embedded; vectors of chunks that no longer exist are dropped.
                pos = {cid: i for i, cid in enumerate(vec_ids)}
                rows = np.fromiter((pos.get(cid, -1) for cid in current), dtype=np.int64, count=len(current))
                self.has_vec = rows >= 0
                vecs = vecs[np.maximum(rows, 0)]
                vecs[~self.has_vec] = 0
                print(f"[search] {int((~self.has_vec).sum())} of {len(current)} chunks have no vector yet "
                      f"(keyword search only until embedded)")
            self.vecs = vecs.astype(np.float32)
            self.ids = current
            if not self.encoder_url:
                from sentence_transformers import SentenceTransformer
                import torch
                dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
                kw = {"model_kwargs": {"torch_dtype": torch.bfloat16}} if dev == "cuda" else {}
                self.model = SentenceTransformer(model, device=dev, **kw)
        else:  # keyword-only mode until embeddings are built
            self.vecs = None
            self.ids = current
        self.row = {cid: i for i, cid in enumerate(self.ids)}
        self.me_boost = me_boost
        n = len(self.ids)
        self.embeddable = np.ones(n, dtype=bool)
        self.platform = np.empty(n, dtype=object)
        self.start_ts = np.zeros(n, dtype=np.int64)
        self.end_ts = np.zeros(n, dtype=np.int64)
        self.me_arr = np.zeros(n, dtype=bool)
        self.accounts = [()] * n
        self.me = {}
        for r in self.conn.execute(
                "SELECT chunk_id, index_policy, me_involved, platform, start_ts, end_ts, account_ids FROM Chunks"):
            self.me[r["chunk_id"]] = r["me_involved"]
            i = self.row.get(r["chunk_id"])
            if i is None:
                continue
            self.embeddable[i] = r["index_policy"] == "embed"
            self.platform[i] = r["platform"]
            self.start_ts[i], self.end_ts[i] = r["start_ts"], r["end_ts"]
            self.me_arr[i] = bool(r["me_involved"])
            self.accounts[i] = tuple(json.loads(r["account_ids"]))

    def mask(self, platform=None, accounts=None, since=None, until=None, me_only=False):
        """Boolean row mask for the filters, or None when no filter is set."""
        if not any([platform, accounts, since, until, me_only]):
            return None
        m = np.ones(len(self.ids), dtype=bool)
        if platform:
            m &= self.platform == platform
        if since:
            m &= self.end_ts >= since
        if until:
            m &= self.start_ts <= until
        if me_only:
            m &= self.me_arr
        if accounts:
            want = set(accounts)
            m &= np.fromiter((not want.isdisjoint(a) for a in self.accounts), dtype=bool, count=len(self.accounts))
        return m

    def encode_query(self, q):
        if q in self._qcache:                            # the agent often repeats a phrasing within one answer
            return self._qcache[q]
        v = self._encode_query(q)
        if len(self._qcache) > 2000:
            self._qcache.clear()
        self._qcache[q] = v
        return v

    def _openrouter(self, texts):
        if getattr(self, "_http", None) is None:
            import os
            import httpx
            from dotenv import dotenv_values
            from common import REPO_ROOT
            key = os.environ.get("OPENROUTER_API_KEY") or dotenv_values(REPO_ROOT / ".env").get("OPENROUTER_API_KEY")
            self._http = httpx.Client(timeout=httpx.Timeout(20, connect=5), base_url="https://openrouter.ai/api/v1",
                                      headers={"Authorization": f"Bearer {key}"})
        r = self._http.post("/embeddings", json={"model": OPENROUTER_MODELS[self.model_name],
                                                 "input": [QUERY_PROMPT + t for t in texts], "provider": OPENROUTER_ROUTE},
                            timeout=20 if len(texts) == 1 else 90)      # batches (evals) may queue longer
        r.raise_for_status()
        out = []
        for d in sorted(r.json()["data"], key=lambda d: d["index"]):
            v = np.asarray(d["embedding"], dtype=np.float32)
            out.append(v / (np.linalg.norm(v) or 1))
        return out

    def prefetch(self, qs):
        """Encode several queries in one request (e.g. all phrasings of one agent search) so the dense legs that
        follow are cache hits. Never raises: a failure just leaves dense() to fall back to keywords."""
        todo = [q for q in dict.fromkeys(qs) if q not in self._qcache]
        if not todo or self.vecs is None:
            return
        try:
            vs = self._openrouter(todo) if self.encoder_url == "openrouter" else [self._encode_query(q) for q in todo]
            for q, v in zip(todo, vs):
                self._qcache[q] = v
        except Exception as e:
            print(f"[search] query prefetch failed: {type(e).__name__}: {e}")

    def _encode_query(self, q):
        if self.encoder_url == "openrouter":
            return self._openrouter([q])[0]
        if self.encoder_url:
            if getattr(self, "_http", None) is None:   # keep-alive: skip a tunnel round trip per query
                import httpx
                self._http = httpx.Client(timeout=httpx.Timeout(30, connect=3))
            r = self._http.post(f"{self.encoder_url}/embed", json={"texts": [q], "query": True})
            r.raise_for_status()
            return np.asarray(r.json()["vectors"][0], dtype=np.float32)
        prompts = getattr(self.model, "prompts", {}) or {}
        kw = {"prompt_name": "query"} if "query" in prompts else {}
        return self.model.encode([q], normalize_embeddings=True, **kw)[0].astype(np.float32)

    def dense(self, q, k=50, include_short=False, mask=None):
        if self.vecs is None:
            return []
        try:
            qv = self.encode_query(q)
        except Exception as e:   # remote encoder offline (e.g. the Mac is asleep): keyword results only
            print(f"[search] encoder unavailable, keyword-only: {type(e).__name__}: {e}")
            return []
        scores = self.vecs @ qv
        if not include_short:
            scores = np.where(self.embeddable, scores, -1)
        if self.has_vec is not None:
            scores = np.where(self.has_vec, scores, -np.inf)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]

    def fts(self, q, k=50, mask=None):
        terms = [t for t in re.findall(r"\w+", q.lower()) if len(t) > 1]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        limit = k if mask is None else k * 20   # over-fetch, then keep rows passing the filter
        rows = self.conn.execute(
            "SELECT chunk_id, bm25(ChunksFTS, 0, 2.0, 1.0, 1.0, 0.5) AS s FROM ChunksFTS WHERE ChunksFTS MATCH ? "
            "ORDER BY s LIMIT ?", (match, limit)).fetchall()
        out = [(r["chunk_id"], -r["s"]) for r in rows
               if mask is None or (r["chunk_id"] in self.row and mask[self.row[r["chunk_id"]]])]
        return out[:k]

    def doc_text(self, cid):
        r = self.conn.execute("SELECT header, text FROM Chunks WHERE chunk_id=?", (cid,)).fetchone()
        return f"{r['header']}\n{r['text']}"

    def search(self, q, k=10, mode="hybrid", **filters):
        rerank = mode.endswith("+rerank")
        mode = mode.removesuffix("+rerank")
        if rerank:
            pool = self.search(q, k=30, mode=mode, **filters)  # hybrid R@10 ~0.98: top 30 nearly always holds it
            scores = self.reranker.score(q, [self.doc_text(cid) for cid, _ in pool])
            ranked = sorted(zip([cid for cid, _ in pool], scores), key=lambda x: -x[1])
            return ranked[:k]
        m = self.mask(**filters)
        if mode == "dense":
            ranked = self.dense(q, k=max(k, 50), mask=m)
        elif mode == "fts":
            ranked = self.fts(q, k=max(k, 50), mask=m)
        else:
            fused = {}
            for lst in (self.dense(q, mask=m), self.fts(q, mask=m)):
                for rank, (cid, _) in enumerate(lst):
                    fused[cid] = fused.get(cid, 0) + 1 / (RRF_K + rank + 1)
            ranked = sorted(fused.items(), key=lambda x: -x[1])
        if self.me_boost:
            ranked = sorted(((cid, s * (1 + self.me_boost * self.me.get(cid, 0))) for cid, s in ranked),
                            key=lambda x: -x[1])
        return ranked[:k]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--variant", default="ctx")
    a = ap.parse_args()
    s = Searcher(a.model, a.variant)
    for cid, score in s.search(a.query, a.k, a.mode):
        r = s.conn.execute("SELECT header, substr(text,1,160) t FROM Chunks WHERE chunk_id=?", (cid,)).fetchone()
        print(f"{score:.4f}  {r['header']}\n        {r['t']!r}")
