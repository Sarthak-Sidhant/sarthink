"""Audit the memory layer against the raw messages: how many facts are right, and why the wrong ones are wrong.

Samples facts (stratified over people and you), shows a strong model (deepseek-v4-pro, thinking) each fact with the
messages of the conversations it rests on, and records a verdict and the failure kind:
  correct | partly (right core, wrong detail) | wrong | unverifiable (the sources don't say)
  kinds: wrong_person, joke_or_sarcasm, summary_error, bad_merge, outdated, wrong_date, other
Writes processed_data/eval/facts_audit_<tag>.json and prints precision overall and per kind. Use the same seed to
compare a fix against the last audit.

Usage: python3 scripts/index/audit_facts.py [--n 100] [--tag v1] [--seed 7]
"""
import argparse
import concurrent.futures as cf
import json
import random
from pathlib import Path


import llm
from common import DATA_DIR, connect_index

CHECK = """You are auditing a personal memory system. Below is a FACT it holds about {name}, and the actual messages
of the conversations it claims to rest on (oldest first; "Sarthak Sidhant (me)" is the owner).
Judge the fact strictly against the messages (the best-matching excerpts of the archive for this fact):
- "correct": the messages clearly support it as stated
- "partly": the core is right but a detail (date, number, name, place, scope) is wrong or overstated
- "wrong": the messages contradict it, or it is about someone else, or it takes a joke/sarcasm/hypothetical literally
- "unverifiable": these messages don't say it either way
If not correct, give the kind: wrong_person (it's about someone else), joke_or_sarcasm, summary_error (a misreading of
what was said), bad_merge (blends separate things), outdated (was true, isn't now), wrong_date, other.
Then, using the RECENT messages too: is the fact (as a {status} fact) still true today ({today})? "yes", "no" (the
recent messages show it changed or ended; or it's a plan/episode whose time has passed while marked current), or
"unclear". For a fact marked past, "yes" means it is correctly in the past.
Return JSON: {{"verdict": "...", "kind": "...", "why": "<= 25 words", "corrected": "<the fact as the messages
support it, or empty>", "current_today": "yes|no|unclear"}}

FACT ({status}; {since} – {until}; {n} conversations): {fact}

MESSAGES
{messages}

RECENT MESSAGES ABOUT THIS (last 4 months)
{recent}"""


def evidence(api, fact, person_id, account_id, k=5, since=None):
    """The archive excerpts that best match the fact, from that person's chats (yours, for facts about you):
    a fact's stored sources can point at look-alike conversations, so the evidence is searched for."""
    import urllib.parse
    import urllib.request
    q = {"q": fact, "k": k}
    if since:
        q["since"] = since
    if person_id and person_id != 1:
        q["person"] = f"P_{person_id}"
    elif account_id:
        q["person"] = f"A_{account_id}"
    else:
        q["me_only"] = "true"
    try:
        with urllib.request.urlopen(f"{api}/api/search?{urllib.parse.urlencode(q)}", timeout=60) as r:
            hits = json.load(r)["results"]
    except Exception as e:
        return f"(search failed: {e})"
    return "\n\n".join(f"=== {h['header']}\n{h['text']}" for h in hits)


def main(a):
    conn = connect_index()
    cl = llm.client(timeout=600)
    people = conn.execute("""SELECT e.entity_id, e.name, e.person_id, e.account_id FROM Entities e JOIN Facts f USING(entity_id)
                             WHERE e.kind IN ('person', 'self') GROUP BY e.entity_id ORDER BY COUNT(*) DESC LIMIT 12""").fetchall()
    rnd = random.Random(a.seed)
    sample = []
    per = max(1, a.n // len(people))
    for eid, name, pid, aid in people:
        rows = conn.execute("""SELECT fact_id, statement, status, valid_from, valid_to, confidence, sources FROM Facts
                               WHERE entity_id = ? ORDER BY fact_id""", (eid,)).fetchall()
        for r in rnd.sample(rows, min(per, len(rows))):
            sample.append(((name, *r), pid, aid))
    import datetime as dt
    today = dt.date.today()
    recent_since = (today - dt.timedelta(days=120)).isoformat()
    with cf.ThreadPoolExecutor(6) as ex:
        texts = list(ex.map(lambda x: evidence(a.api, x[0][2], x[1], x[2]), sample))
        recents = list(ex.map(lambda x: evidence(a.api, x[0][2], x[1], x[2], k=3, since=recent_since), sample))
    sample = [x[0] for x in sample]

    def check(item):
        (name, fid, st, status, vf, vt, n, srcs), msgs, recent = item
        for _ in range(3):
            try:
                r = cl.chat.completions.create(
                    model=a.model, response_format={"type": "json_object"}, max_tokens=16000,
                    extra_body=None if a.thinking else {"thinking": {"type": "disabled"}},
                    messages=[{"role": "user", "content": CHECK.format(
                        name=name, fact=st, status=status, since=vf or "?", until=vt or "now", n=n,
                        messages=msgs or "(no source messages)", recent=recent or "(none)", today=today)}])
                v = json.loads(r.choices[0].message.content or "")
                return {"name": name, "fact_id": fid, "fact": st, "support": n, **v}
            except Exception:
                continue
        return {"name": name, "fact_id": fid, "fact": st, "support": n, "verdict": "error"}

    with cf.ThreadPoolExecutor(12) as ex:
        out = list(ex.map(check, zip(sample, texts, recents)))
    path = Path(DATA_DIR) / "eval" / f"facts_audit_{a.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    judged = [o for o in out if o.get("verdict") in ("correct", "partly", "wrong", "unverifiable")]
    count = lambda k, v: sum(1 for o in judged if o.get(k) == v)  # noqa: E731
    print(f"{len(judged)} facts judged: correct {count('verdict', 'correct')}, partly {count('verdict', 'partly')}, "
          f"wrong {count('verdict', 'wrong')}, unverifiable {count('verdict', 'unverifiable')}")
    print(f"precision (correct / judged): {count('verdict', 'correct') / max(1, len(judged)):.0%}; "
          f"correct or partly: {(count('verdict', 'correct') + count('verdict', 'partly')) / max(1, len(judged)):.0%}")
    good = sum(1 for o in judged if o["verdict"] == "correct" and o.get("current_today") != "no")
    stale = sum(1 for o in judged if o.get("current_today") == "no")
    print(f"correct AND current: {good / max(1, len(judged)):.0%}; marked current but no longer true: {stale}")
    kinds = {}
    for o in judged:
        if o["verdict"] != "correct":
            kinds[o.get("kind") or "?"] = kinds.get(o.get("kind") or "?", 0) + 1
    print("not correct, by kind:", ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda x: -x[1])))
    by_support = {}
    for o in judged:
        b = "1" if o["support"] <= 1 else "2-4" if o["support"] <= 4 else "5+"
        by_support.setdefault(b, []).append(o["verdict"] == "correct")
    print("precision by support:", ", ".join(f"{b} conv: {sum(v) / len(v):.0%} (n={len(v)})"
                                             for b, v in sorted(by_support.items())))
    print(f"-> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="a running Sarthink API, to search for evidence")
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    main(ap.parse_args())
