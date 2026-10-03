"""Inbox triage: is each current commitment still pending, or did the conversation settle it?

Summaries pull every promise out of every chat, and most are moment-to-moment ("send a pic", "meet at 7pm") that
the same chat settles within minutes. For each current item (due within the last week or later, or undated and
said in the last 3 weeks) that involves Sarthak (his own, or someone else's in a 1:1 chat or naming him), a cheap
model reads the rest of that conversation and the next one, and answers:
  pending   a real commitment, not shown done, still relevant today
  done      the conversation shows it was done or answered
  moot      cancelled, plans changed, or a one-off moment that has passed
  trivial   in-the-moment chat logistics, not worth a reminder
Verdicts are derived (CommitmentTriage in the index DB, with model and date); the inbox hides non-pending ones
under "Auto-cleared" with the reason. The user's own done/dismissed always wins. Items already judged are not
re-asked unless --recheck (pending ones whose chat has moved on can be re-asked with it).

Usage: python3 scripts/index/triage_inbox.py [--recheck] [--limit N]
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json

from dotenv import dotenv_values
from openai import OpenAI

from common import REPO_ROOT, connect_index

MODEL = "deepseek-flash"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SESSION_CHARS, NEXT_CHARS = 9000, 3000

PROMPT = """Today is {today}. Below is a chat of Sarthak's, then what came next in the same chat. Items were
extracted from it as commitments. For EACH item decide, from the chat:
- "done": the chat shows it was carried out, answered or delivered (a question that got its answer is done).
  Agreeing on or scheduling a plan is NOT doing it: a plan agreed for a later date is "pending"
- "moot": cancelled, plans changed, or a one-off moment (a call, a meetup, "tonight") whose time has passed.
  Each item says how long ago it was said and when it is due: trust those. Something due today or later has NOT
  passed; a same-day errand or meetup said days ago without a later date has
- "trivial": in-the-moment chat logistics or banter not worth a reminder (send a pic, reply later, call now)
- "pending": a real commitment, not shown done, that could still matter as of today
Return JSON: {{"items": [{{"i": <index>, "verdict": "pending|done|moot|trivial", "reason": "<= 12 words"}}]}}

ITEMS
{items}

CHAT ({said_on})
{chat}

WHAT CAME NEXT
{after}"""

SCHEMA = """CREATE TABLE IF NOT EXISTS CommitmentTriage (
    cid TEXT PRIMARY KEY, verdict TEXT, reason TEXT, model TEXT, checked_on TEXT)"""


def current(conn, recheck=False):
    """Current, relevant items without a verdict (or every current one with recheck), by session."""
    conn.execute(SCHEMA)
    rows = conn.execute("""
        WITH members AS (SELECT thread_id, COUNT(DISTINCT account_id) n FROM MessageFacts GROUP BY thread_id)
        SELECT c.cid, c.session_id, c.owner, c.who, c.title, c.what, c.due, c.said_on
        FROM Commitments c JOIN Sessions s USING(session_id)
        LEFT JOIN members mb ON mb.thread_id = json_extract(s.thread_ids, '$[0]')
        LEFT JOIN CommitmentTriage tr ON tr.cid = c.cid
        WHERE ((c.due != '' AND c.due >= date('now', '-7 days'))
               OR ((c.due IS NULL OR c.due = '') AND c.said_on >= date('now', '-21 days')))
          AND (c.owner = 'me' OR COALESCE(mb.n, 99) <= 2 OR lower(c.title || ' ' || c.what) LIKE '%sarthak%')
          AND (tr.cid IS NULL OR (? AND tr.verdict = 'pending'))
        ORDER BY c.session_id""", (int(recheck),)).fetchall()
    by_session = {}
    for r in rows:
        by_session.setdefault(r[1], []).append(r)
    return by_session


def session_text(conn, sid):
    """The session's chunks in order, and the start of the next sessions in the same chat."""
    s = conn.execute("SELECT thread_ids, start_ts FROM Sessions WHERE session_id = ?", (sid,)).fetchone()
    body = "\n".join(t for (t,) in conn.execute(
        "SELECT text FROM Chunks WHERE session_id = ? ORDER BY start_ts, chunk_id", (sid,)))
    nxt = conn.execute("""SELECT session_id FROM Sessions WHERE json_extract(thread_ids, '$[0]') = ? AND start_ts > ?
                          ORDER BY start_ts LIMIT 2""", (json.loads(s[0])[0], s[1])).fetchall()
    after = "\n".join(t for (n,) in nxt for (t,) in conn.execute(
        "SELECT text FROM Chunks WHERE session_id = ? ORDER BY start_ts, chunk_id", (n,)))
    if len(body) > SESSION_CHARS:            # keep the start (where plans are made) and the end (where they land)
        body = body[:SESSION_CHARS // 2] + "\n[…]\n" + body[-SESSION_CHARS // 2:]
    return body, after[:NEXT_CHARS] or "(nothing after this in the archive)"


def ago(day):
    """'4 days ago' / 'today' / 'in 7 days' against today (IST): the model is unreliable at date arithmetic."""
    try:
        d = (dt.date.fromisoformat(day) - dt.datetime.now(IST).date()).days
    except ValueError:
        return day
    return f"{day}, " + ("today" if d == 0 else "tomorrow" if d == 1 else "yesterday" if d == -1
                         else f"in {d} days" if d > 0 else f"{-d} days ago")


def main(a):
    client = OpenAI(api_key=dotenv_values(REPO_ROOT / ".env")["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    conn = connect_index()
    todo = current(conn, a.recheck)
    if a.limit:
        todo = dict(list(todo.items())[:a.limit])
    n_items = sum(len(v) for v in todo.values())
    print(f"{n_items} current items in {len(todo)} conversations to triage")
    today = dt.datetime.now(IST).strftime("%Y-%m-%d")
    texts = {sid: session_text(conn, sid) for sid in todo}

    def ask(sid):
        items = todo[sid]
        lines = "\n".join(f"{i}. [{'Sarthak' if it[2] == 'me' else it[3]}] {it[4]} — {it[5]} "
                          f"(said {ago(it[7])}{'; due ' + ago(it[6]) if it[6] else ''})" for i, it in enumerate(items))
        chat, after = texts[sid]
        for _ in range(3):
            try:
                r = client.chat.completions.create(
                    model=MODEL, response_format={"type": "json_object"}, max_tokens=8000,   # thinking on: without it,
                    # it left past one-off calls and answered questions "pending" (10-chat trial)
                    messages=[{"role": "user", "content": PROMPT.format(
                        today=today, items=lines, said_on=items[0][7], chat=chat, after=after)}])
                out = json.loads(r.choices[0].message.content).get("items", [])
                return items, out, r.usage.prompt_tokens, r.usage.completion_tokens
            except Exception as e:     # provider refusal or bad JSON: leave these unjudged
                err = e
        print(f"  skipped {sid}: {str(err)[:120]}")
        return items, [], 0, 0

    rows, tin, tout, counts = [], 0, 0, {}
    with cf.ThreadPoolExecutor(16) as ex:
        for items, out, pin, pout in ex.map(ask, list(todo)):
            tin, tout = tin + pin, tout + pout
            for o in out:
                i, v = o.get("i"), o.get("verdict")
                if isinstance(i, int) and 0 <= i < len(items) and v in ("pending", "done", "moot", "trivial"):
                    rows.append((items[i][0], v, str(o.get("reason") or "")[:160], MODEL, today))
                    counts[v] = counts.get(v, 0) + 1
    conn.executemany("INSERT OR REPLACE INTO CommitmentTriage VALUES (?,?,?,?,?)", rows)
    conn.commit()
    print(f"triaged {len(rows)} of {n_items}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
          + f" | tokens in {tin:,} out {tout:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recheck", action="store_true", help="ask again about items judged pending before")
    ap.add_argument("--limit", type=int, default=0, help="only this many conversations (a test run)")
    main(ap.parse_args())
