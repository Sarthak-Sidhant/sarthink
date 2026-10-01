"""Commitments inbox: promises and plans found in conversations (from the session summaries), cleaned up.

Each summary lists commitments as {who, what, when}. Many are past actions or noise, so a cheap model
classifies them in batches: keep only forward-looking promises/plans, mark the owner (Sarthak vs someone
else), write a short title, and resolve relative dates ("Monday", "in 3 days") against the conversation date.
Output: Commitments table in the index DB (derived, with session provenance). Status (done / dismissed /
added to calendar) lives in the user's notes DB and survives rebuilds.

Usage: python3 scripts/index/build_commitments.py
"""
import concurrent.futures as cf
import datetime as dt
import hashlib
import json

from dotenv import dotenv_values
from openai import OpenAI

from common import REPO_ROOT, connect_index

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
BATCH = 40

PROMPT = """Below are items extracted from Sarthak's chats, each with the date of the conversation. For EACH item
decide whether it is a forward-looking commitment or plan (someone promising, agreeing or planning to do
something later) — not a past action, not a vague wish, not a joke.
For kept items: owner = "me" if Sarthak is the one who will do it, otherwise "them"; title = short imperative
phrase (max 10 words, e.g. "Share Neo4j pipeline findings with Gareth"); due = YYYY-MM-DD if a date or relative
time ("Monday", "in 3 days", "tomorrow") can be resolved against the conversation date, else null.
Return JSON: {{"items": [{{"i": <index>, "keep": true/false, "owner": "me|them", "title": "...", "due": null}}]}}

{items}"""

SCHEMA = """
DROP TABLE IF EXISTS Commitments;
CREATE TABLE Commitments (
    cid TEXT PRIMARY KEY, session_id TEXT, owner TEXT, who TEXT, title TEXT, what TEXT,
    due TEXT, said_on TEXT, platform TEXT, people TEXT, nodes TEXT
);"""


def main():
    client = OpenAI(api_key=dotenv_values(REPO_ROOT / ".env")["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    conn = connect_index()
    names = {r[0]: r[1] for r in conn.execute("SELECT person_id, name FROM Persons")}
    items = []
    for sid, data, ts, plat, pids, tids in conn.execute("""
            SELECT x.session_id, x.data, s.start_ts, s.platform, s.person_ids, s.thread_ids
            FROM SessionSummaries x JOIN Sessions s USING(session_id)
            WHERE x.model = 'deepseek-flash' AND x.prompt_version = 'v3'"""):
        for cm in json.loads(data).get("commitments") or []:
            if isinstance(cm, dict) and cm.get("what"):
                items.append({"sid": sid, "day": dt.datetime.fromtimestamp(ts, IST).strftime("%Y-%m-%d"),
                              "platform": plat, "pids": json.loads(pids), "tids": json.loads(tids), **cm})

    def classify(batch):
        lines = "\n".join(f"{i}. [{it['day']}] {it.get('who')}: {it['what']} (when: {it.get('when')})"
                          for i, it in enumerate(batch))
        r = client.chat.completions.create(
            model="deepseek-flash", response_format={"type": "json_object"}, max_tokens=4000,
            extra_body={"thinking": {"type": "disabled"}},
            messages=[{"role": "user", "content": PROMPT.format(items=lines)}])
        try:
            return batch, json.loads(r.choices[0].message.content).get("items", [])
        except json.JSONDecodeError:
            return batch, []

    batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    rows = []
    with cf.ThreadPoolExecutor(16) as ex:
        for batch, out in ex.map(classify, batches):
            for o in out:
                i = o.get("i")
                if not isinstance(i, int) or not (0 <= i < len(batch)) or not o.get("keep"):
                    continue
                it = batch[i]
                due = o.get("due")
                try:
                    due = dt.date.fromisoformat(due).isoformat() if due else None
                except ValueError:
                    due = None
                cid = hashlib.sha1(f"{it['sid']}|{it['what']}".encode()).hexdigest()[:16]
                rows.append((cid, it["sid"], "me" if o.get("owner") == "me" else "them", it.get("who"),
                             (o.get("title") or it["what"])[:120], it["what"], due, it["day"], it["platform"],
                             json.dumps([names.get(p, "?") for p in it["pids"] if p != 1]),
                             json.dumps([f"P_{p}" for p in it["pids"]] + [f"T_{t}" for t in it["tids"]])))
    conn.executescript(SCHEMA)
    conn.executemany("INSERT OR REPLACE INTO Commitments VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    mine = sum(r[2] == "me" for r in rows)
    print(f"{len(items)} raw items -> {len(rows)} commitments kept ({mine} yours, {len(rows) - mine} by others; "
          f"{sum(1 for r in rows if r[6])} with a due date)")


if __name__ == "__main__":
    main()
