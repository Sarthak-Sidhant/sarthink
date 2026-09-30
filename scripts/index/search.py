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
    def __init__(self, model="Qwen/Qwen3-Embedding-0.6B", variant="ctx", device=None, me_boost=0.0, reranker=None):
        model = MODELS.get(model, model)
        self.reranker = reranker
        self.conn = connect_index()
        base = OUT_DIR / f"emb_{slug(model)}_{variant}"
        if (OUT_DIR / f"{base.name}.npy").exists():
            from sentence_transformers import SentenceTransformer
            self.vecs = np.load(f"{base}.npy").astype(np.float32)
            with open(f"{base}.ids.json") as f:
                self.ids = json.load(f)["ids"]
            import torch
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            kw = {"model_kwargs": {"torch_dtype": torch.bfloat16}} if dev == "cuda" else {}
            self.model = SentenceTransformer(model, device=dev, **kw)
        else:  # keyword-only mode until embeddings are built
            self.vecs, self.model = None, None
            self.ids = [r[0] for r in self.conn.execute("SELECT chunk_id FROM Chunks ORDER BY chunk_id")]
        self.row = {cid: i for i, cid in enumerate(self.ids)}
        self.me_boost = me_boost
        meta = self.conn.execute("SELECT chunk_id, index_policy, me_involved FROM Chunks").fetchall()
        self.embeddable = np.array([True] * len(self.ids))
        self.me = {}
        for r in meta:
            self.me[r["chunk_id"]] = r["me_involved"]
            if r["index_policy"] != "embed" and r["chunk_id"] in self.row:
                self.embeddable[self.row[r["chunk_id"]]] = False

    def encode_query(self, q):
        prompts = getattr(self.model, "prompts", {}) or {}
        kw = {"prompt_name": "query"} if "query" in prompts else {}
        return self.model.encode([q], normalize_embeddings=True, **kw)[0].astype(np.float32)

    def dense(self, q, k=50, include_short=False):
        if self.vecs is None:
            return []
        scores = self.vecs @ self.encode_query(q)
        if not include_short:
            scores = np.where(self.embeddable, scores, -1)
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]

    def fts(self, q, k=50):
        terms = [t for t in re.findall(r"\w+", q.lower()) if len(t) > 1]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        rows = self.conn.execute(
            "SELECT chunk_id, bm25(ChunksFTS, 0, 2.0, 1.0, 1.0, 0.5) AS s FROM ChunksFTS WHERE ChunksFTS MATCH ? "
            "ORDER BY s LIMIT ?", (match, k)).fetchall()
        return [(r["chunk_id"], -r["s"]) for r in rows]

    def doc_text(self, cid):
        r = self.conn.execute("SELECT header, text FROM Chunks WHERE chunk_id=?", (cid,)).fetchone()
        return f"{r['header']}\n{r['text']}"

    def search(self, q, k=10, mode="hybrid"):
        rerank = mode.endswith("+rerank")
        mode = mode.removesuffix("+rerank")
        if rerank:
            pool = self.search(q, k=30, mode=mode)  # hybrid R@10 is ~0.98, so the top 30 nearly always holds the answer
            scores = self.reranker.score(q, [self.doc_text(cid) for cid, _ in pool])
            ranked = sorted(zip([cid for cid, _ in pool], scores), key=lambda x: -x[1])
            return ranked[:k]
        if mode == "dense":
            ranked = self.dense(q, k=max(k, 50))
        elif mode == "fts":
            ranked = self.fts(q, k=max(k, 50))
        else:
            fused = {}
            for lst in (self.dense(q), self.fts(q)):
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
