"""Needle test: can retrieval find a specific chunk from a natural recall question?

gen  – sample chunks (stratified by platform), have an LLM write the question Sarthak would type to
       find that conversation later (paraphrased, no long quotes). Saved to processed_data/eval/.
run  – score retrieval configs: recall@1/5/10 and MRR, overall and per platform. A hit is the target
       chunk or any chunk sharing >=50% of its messages (overlapping windows).

Usage:
  python3 scripts/index/needle_eval.py gen --n 300
  python3 scripts/index/needle_eval.py run --configs raw:dense ctx:dense ctx:fts ctx:hybrid
"""
import argparse
import asyncio
import json
import random
from collections import defaultdict

from dotenv import dotenv_values
from openai import AsyncOpenAI

from common import REPO_ROOT, connect_index

EVAL_DIR = REPO_ROOT / "processed_data" / "eval"
QUERIES = EVAL_DIR / "needle_queries.jsonl"

GEN_PROMPT = """Below is a snippet of Sarthak's own chat/post history. Imagine that months later Sarthak wants to
find this exact conversation again with a search box over his whole archive.
Write the ONE query he would type. Rules:
- Describe it the way memory works: who, what it was about, roughly when/where. Not a transcript.
- Do not copy more than 3 consecutive words from the snippet.
- It must be specific enough that this snippet is the right answer, not any chat with that person.
- Natural phrasing; Hinglish is fine if the chat is Hinglish.
Return JSON: {"query": "..."}"""


async def gen(a):
    env = dotenv_values(REPO_ROOT / ".env")
    client = AsyncOpenAI(api_key=env["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com", timeout=120)
    conn = connect_index()
    rows = conn.execute("""SELECT chunk_id, platform, header, text FROM Chunks
                           WHERE index_policy='embed' AND token_count >= 60""").fetchall()
    by_plat = defaultdict(list)
    for r in rows:
        by_plat[r["platform"]].append(r)
    rng = random.Random(a.seed)
    # proportional sample, but at least 25 per platform so small sources are measured
    picks = []
    for p, lst in by_plat.items():
        n = max(25, round(a.n * len(lst) / len(rows)))
        picks += rng.sample(lst, min(n, len(lst)))
    sem = asyncio.Semaphore(16)

    async def one(r):
        async with sem:
            for attempt in range(3):
                try:
                    resp = await client.chat.completions.create(
                        model=a.model, temperature=0.7, response_format={"type": "json_object"},
                        extra_body={"thinking": {"type": "disabled"}}, max_tokens=300,
                        messages=[{"role": "system", "content": GEN_PROMPT},
                                  {"role": "user", "content": f"{r['header']}\n{r['text']}"}])
                    return {"chunk_id": r["chunk_id"], "platform": r["platform"],
                            "query": json.loads(resp.choices[0].message.content)["query"]}
                except Exception:
                    await asyncio.sleep(2 ** attempt)
            return None

    out = [x for x in await asyncio.gather(*(one(r) for r in picks)) if x]
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(QUERIES, "w") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} queries to {QUERIES}")


def gold_sets(conn, targets):
    """target chunk + chunks sharing >=50% of its messages"""
    gold = {}
    for cid in targets:
        msgs = {r[0] for r in conn.execute("SELECT msg_id FROM ChunkMessages WHERE chunk_id=?", (cid,))}
        overlap = defaultdict(int)
        for m in msgs:
            for (other,) in conn.execute("SELECT chunk_id FROM ChunkMessages WHERE msg_id=?", (m,)):
                overlap[other] += 1
        gold[cid] = {o for o, n in overlap.items() if n >= 0.5 * len(msgs)} | {cid}
    return gold


def run(a):
    from search import Searcher
    qs = [json.loads(l) for l in open(QUERIES)]
    conn = connect_index()
    gold = gold_sets(conn, [q["chunk_id"] for q in qs])
    searchers, results, reranker = {}, {}, None
    if any(c.endswith("+rerank") for c in a.configs):
        from search import Reranker
        reranker = Reranker(a.reranker)
    for cfg in a.configs:
        spec, mode = cfg.split(":")
        model, variant = spec.split("/") if "/" in spec else (a.model, spec)
        key = (model, variant)
        if key not in searchers:
            searchers[key] = Searcher(model, variant, me_boost=a.me_boost, reranker=reranker)
        s = searchers[key]
        stats = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0, "r10": 0, "mrr": 0.0})
        for q in qs:
            ranked = [cid for cid, _ in s.search(q["query"], k=10, mode=mode)]
            rank = next((i for i, cid in enumerate(ranked) if cid in gold[q["chunk_id"]]), None)
            for bucket in ("ALL", q["platform"]):
                st = stats[bucket]
                st["n"] += 1
                if rank is not None:
                    st["r1"] += rank < 1
                    st["r5"] += rank < 5
                    st["r10"] += 1
                    st["mrr"] += 1 / (rank + 1)
        results[cfg] = stats
        st = stats["ALL"]
        print(f"{cfg:<22} n={st['n']}  R@1 {st['r1'] / st['n']:.2f}  R@5 {st['r5'] / st['n']:.2f}  "
              f"R@10 {st['r10'] / st['n']:.2f}  MRR {st['mrr'] / st['n']:.3f}")
    print("\nR@10 by platform:")
    plats = sorted({p for r in results.values() for p in r if p != "ALL"})
    print(f"{'config':<22} " + " ".join(f"{p:>10}" for p in plats))
    for cfg, stats in results.items():
        print(f"{cfg:<22} " + " ".join(f"{stats[p]['r10'] / max(stats[p]['n'], 1):>10.2f}" for p in plats))
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DIR / "needle_results.json", "w") as f:
        json.dump({"configs": a.configs, "me_boost": a.me_boost, "results": results}, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--n", type=int, default=300)
    g.add_argument("--seed", type=int, default=11)
    g.add_argument("--model", default="deepseek-flash")
    r = sub.add_parser("run")
    r.add_argument("--configs", nargs="+", default=["raw:dense", "ctx:dense", "ctx:fts", "ctx:hybrid"])
    r.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    r.add_argument("--me-boost", type=float, default=0.0)
    r.add_argument("--reranker", default="Qwen/Qwen3-Reranker-4B")
    a = ap.parse_args()
    asyncio.run(gen(a)) if a.cmd == "gen" else run(a)
