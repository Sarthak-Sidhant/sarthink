"""LLM enrichment for sessions: a faithful session summary + a short context line per chunk.

Output is enrichment, never ground truth: every row records model, prompt version and time,
and chunk text itself is untouched. Chunk contexts are prepended to chunks at embedding time.

- Only sessions with >= --min-tokens are summarized (tiny sessions rely on their header).
- The tail of the previous session and head of the next session in the same conversation are
  passed as read-only context, so the summary can note continuity without absorbing their facts.
- Resumable: sessions already summarized with the same model + prompt version are skipped.
- Provider-agnostic (any OpenAI-compatible endpoint); defaults to DeepSeek.

Usage:
  python3 scripts/index/summarize_sessions.py --limit 20 --sample            # pilot
  python3 scripts/index/summarize_sessions.py --concurrency 32               # full run
  python3 scripts/index/summarize_sessions.py --model deepseek-flash --tag bakeoff --limit 20 --sample
"""
import argparse
import asyncio
import datetime as dt
import json
import random
import time

from dotenv import dotenv_values
from openai import AsyncOpenAI

from common import REPO_ROOT, connect_index

PROMPT_VERSION = "v3"  # v2: thinking off; v3: context-derived details must be marked
NEIGHBOR_TOKENS = 500  # cap on each neighbor excerpt (~chars/4)

SYSTEM = """You write memory records for a personal archive of one person's chats ("me" = Sarthak Sidhant).
You receive one CURRENT SESSION split into numbered chunks, plus optional PREVIOUS and NEXT excerpts
from the same conversation that are context only.

Rules:
- Summarize ONLY the current session. Use PREVIOUS/NEXT only to understand references or to note
  continuity (e.g. "continues the earlier discussion of X"); never report their facts as this session's.
  If you use them to resolve a reference, mark it, e.g. "the item (a ZimaBoard, per earlier context)".
- Be faithful: state only what the messages say. No guessing of motives, relationships or feelings
  unless explicit. If something is unclear, leave it out.
- Messages mix English, Hindi and Hinglish slang. Write in English, but keep names, nicknames and
  key Hinglish phrases verbatim where they matter.
- Keep concrete details: names, dates, places, amounts, links' domains, decisions, promises, plans.
- Refer to the owner as "Sarthak".

Return JSON only, with exactly these keys:
{
  "summary": "3-6 dense factual sentences about this session",
  "topics": ["short topic", ...],
  "people": [{"name": "as written", "role": "what they did/said in this session"}],
  "facts": ["atomic factual statement", ...],
  "commitments": [{"who": "...", "what": "...", "when": "date/time if stated, else null"}],
  "keywords": ["search terms, include both Hinglish and English forms where useful"],
  "chunk_contexts": {"<chunk number>": "1-2 sentences situating that chunk within the session: who is talking about what"}
}
chunk_contexts must have an entry for every chunk number given. Use empty lists when nothing applies."""


def load_env():
    return dotenv_values(REPO_ROOT / ".env")


def setup(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS SessionSummaries (
        session_id TEXT, model TEXT, prompt_version TEXT, tag TEXT,
        created_at TEXT, summary TEXT, data TEXT,
        input_tokens INTEGER, output_tokens INTEGER,
        PRIMARY KEY (session_id, model, prompt_version)
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_conv ON Sessions(thread_ids, platform, start_ts);
    CREATE TABLE IF NOT EXISTS ChunkContexts (
        chunk_id TEXT, model TEXT, prompt_version TEXT, context TEXT,
        PRIMARY KEY (chunk_id, model, prompt_version)
    );
    """)


def neighbors(conn, sess):
    """Tail of previous / head of next session sharing the same thread set."""
    prev = conn.execute(
        "SELECT text FROM Sessions WHERE thread_ids=? AND platform=? AND start_ts<? ORDER BY start_ts DESC LIMIT 1",
        (sess["thread_ids"], sess["platform"], sess["start_ts"])).fetchone()
    nxt = conn.execute(
        "SELECT text FROM Sessions WHERE thread_ids=? AND platform=? AND start_ts>? ORDER BY start_ts ASC LIMIT 1",
        (sess["thread_ids"], sess["platform"], sess["start_ts"])).fetchone()
    cap = NEIGHBOR_TOKENS * 4
    p = prev["text"][-cap:] if prev else None
    n = nxt["text"][:cap] if nxt else None
    return p, n


def build_prompt(conn, sess):
    chunks = conn.execute(
        "SELECT chunk_id, text FROM Chunks WHERE session_id=? ORDER BY start_ts, chunk_id", (sess["session_id"],)
    ).fetchall()
    p, n = neighbors(conn, sess)
    parts = [f"SESSION HEADER: {sess['header']}"]
    if p:
        parts.append(f"PREVIOUS (context only, ends right before this session):\n…{p}")
    parts.append("CURRENT SESSION:")
    for i, c in enumerate(chunks, 1):
        parts.append(f"<chunk {i}>\n{c['text']}\n</chunk {i}>")
    if n:
        parts.append(f"NEXT (context only, starts right after this session):\n{n}…")
    return "\n\n".join(parts), [c["chunk_id"] for c in chunks]


async def summarize_one(client, model, sess_id, prompt, chunk_ids, sem, thinking, retries=4):
    async with sem:
        for attempt in range(retries):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=16000,
                    # summarization needs no chain of thought; reasoning otherwise eats the output budget
                    extra_body=None if thinking else {"thinking": {"type": "disabled"}},
                )
                if r.choices[0].finish_reason == "length":
                    raise ValueError("output truncated")
                data = json.loads(r.choices[0].message.content)
                if not isinstance(data.get("summary"), str):
                    raise ValueError("missing summary")
                ctx = data.get("chunk_contexts") or {}
                contexts = {cid: ctx.get(str(i)) for i, cid in enumerate(chunk_ids, 1)}
                usage = r.usage
                return sess_id, data, contexts, usage.prompt_tokens, usage.completion_tokens, None
            except Exception as e:  # network, rate limit, bad JSON: back off and retry
                err = e
                await asyncio.sleep(2 ** attempt + random.random())
        return sess_id, None, None, 0, 0, repr(err)


async def run(a):
    env = load_env()
    client = AsyncOpenAI(api_key=env.get(a.api_key_env), base_url=a.base_url, timeout=180)
    conn = connect_index()
    setup(conn)

    done = {r[0] for r in conn.execute(
        "SELECT session_id FROM SessionSummaries WHERE model=? AND prompt_version=?", (a.model, PROMPT_VERSION))}
    sessions = [s for s in conn.execute(
        "SELECT * FROM Sessions WHERE token_count >= ? ORDER BY start_ts", (a.min_tokens,)).fetchall()
        if s["session_id"] not in done]
    if a.sample:
        random.Random(a.seed).shuffle(sessions)
    if a.limit:
        sessions = sessions[:a.limit]
    print(f"{len(sessions)} sessions to summarize with {a.model} ({len(done)} already done)")

    sem = asyncio.Semaphore(a.concurrency)
    jobs = []
    for s in sessions:
        prompt, chunk_ids = build_prompt(conn, s)
        jobs.append(summarize_one(client, a.model, s["session_id"], prompt, chunk_ids, sem, a.thinking))

    t0, ok, failed, tin, tout = time.time(), 0, 0, 0, 0
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    for i, fut in enumerate(asyncio.as_completed(jobs), 1):
        sid, data, contexts, pin, pout, err = await fut
        if err:
            failed += 1
            print(f"  FAILED {sid}: {err[:200]}")
            continue
        ok += 1
        tin += pin
        tout += pout
        conn.execute("INSERT OR REPLACE INTO SessionSummaries VALUES (?,?,?,?,?,?,?,?,?)",
                     (sid, a.model, PROMPT_VERSION, a.tag, now, data["summary"],
                      json.dumps(data, ensure_ascii=False), pin, pout))
        conn.executemany("INSERT OR REPLACE INTO ChunkContexts VALUES (?,?,?,?)",
                         [(cid, a.model, PROMPT_VERSION, c) for cid, c in contexts.items() if c])
        if i % 50 == 0 or i == len(jobs):
            conn.commit()
            rate = i / (time.time() - t0)
            print(f"  {i}/{len(jobs)} done, {failed} failed, {rate:.1f}/s, tokens in {tin:,} out {tout:,}")
    conn.commit()
    print(f"Finished: {ok} ok, {failed} failed in {time.time() - t0:.0f}s. Tokens in {tin:,}, out {tout:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--min-tokens", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", action="store_true", help="random sample instead of chronological order")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--tag", default="")
    ap.add_argument("--thinking", action="store_true", help="let the model reason before answering (slower, costlier)")
    asyncio.run(run(ap.parse_args()))
