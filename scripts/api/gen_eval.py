"""Generate a second batch of eval questions with checkable ground truth -> processed_data/eval/agent/cases_v2.json

  lookups   25 random conversations not used before; an LLM reads the FULL text and writes a question in an
            assigned style + 2-3 facts that the text states
  stats     15 template questions (random people / months / platforms) with answers computed in SQL
  abstain    5 topics verified to have zero matches in the archive

Usage: python3 scripts/api/gen_eval.py [--api http://127.0.0.1:8000] [--seed 7]
"""
import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "index"))
sys.path.insert(0, str(REPO / "scripts" / "api"))
from agent import SqlTool  # noqa: E402
from common import connect_index  # noqa: E402

OUT = REPO / "processed_data" / "eval" / "agent" / "cases_v2.json"
USED = {"s_786adc7bdfdc2bc5", "s_7e3a3e396ff6290b", "s_aff3a96fd2fd7b26", "s_046d740ba3c7f23c", "s_0d4bf54c5faf9090",
        "s_a51bb21264c2f820", "s_40a59f163a7bb87e", "s_99cae3652f6e16ed", "s_1721232264251f24", "s_1aeac3eb7b787374",
        "s_7b3ac2c444e74a2f", "s_145047b6c9888958", "s_d7a599f253330916", "s_15ad66f8537e89c5", "s_5416a255ea936c84",
        "s_44f0677e5f0c712f", "s_0046df61b16e67eb", "s_d6204e7abf52fa0e", "s_5e87add89cf9d1e9", "s_ea866920ad11a796"}
PLACEHOLDER = ("instagram user", "deleted user", "deleted_user", "twitter_user_", "unknown", "meta ai")
STYLES = ["casual", "vague", "hinglish", "specific", "casual"]

GEN = """Below is one conversation from Sarthak's own chat archive (times are IST). Write ONE question Sarthak
might ask his memory assistant months later whose answer is in this conversation, in this style: {style}.
Styles: casual = everyday wording; vague = how a fuzzy memory sounds ("that time when…"); hinglish = Hindi-English
mix as he texts; specific = precise, names the person/topic.
Rules: the question must identify THIS conversation well enough (who / what / roughly when), must not copy more
than 4 consecutive words, and must be answerable from the text alone. Then list 2-3 short KEY FACTS the answer
must contain, each directly stated in the text (no inference). Avoid phone numbers, emails and addresses.
Return JSON: {{"question": "...", "facts": ["...", "..."]}}

{header}
{text}"""


def api(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def lookups(a, rng, client):
    conn = connect_index()
    quota = {"discord": 7, "instagram": 7, "twitter": 5, "reddit": 5, "facebook": 1}
    out, i = [], 0
    for plat, n in quota.items():
        rows = [r[0] for r in conn.execute(
            "SELECT session_id FROM Sessions WHERE platform=? AND me_involved=1 AND token_count BETWEEN 300 AND 2500",
            (plat,)) if r[0] not in USED]
        for sid in rng.sample(rows, min(n, len(rows))):
            s = api(f"{a.api}/api/session/{sid}")
            style = STYLES[i % len(STYLES)]
            i += 1
            r = client.chat.completions.create(
                model="deepseek-v4-pro", response_format={"type": "json_object"}, max_tokens=6000,
                messages=[{"role": "user", "content": GEN.format(style=style, header=s["header"], text=s["text"][:9000])}])
            try:
                g = json.loads(r.choices[0].message.content)
                out.append({"type": f"lookup-{style}", "question": g["question"], "facts": g["facts"][:3],
                            "abstain": False, "session_id": sid, "platform": plat})
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def named(p):
    return not any(x in (p or "").lower() for x in PLACEHOLDER)


def stats(rng):
    t = SqlTool()
    q = lambda sql: t.run(sql)["rows"]  # noqa: E731
    people = [(pid, name) for pid, name, n in q("""
        SELECT o.person_id, o.person, COUNT(*) n FROM messages o
        WHERE o.is_me=0 AND o.thread_id IN (SELECT thread_id FROM messages WHERE is_me=1)
        GROUP BY o.person_id HAVING n >= 300 ORDER BY n DESC LIMIT 60""") if named(name)]
    pick = rng.sample(people, 9)
    out = []
    for pid, name in pick[:3]:
        d = q(f"""SELECT MIN(day) FROM messages WHERE thread_id IN (SELECT thread_id FROM messages WHERE person_id={pid})
                  AND thread_id IN (SELECT thread_id FROM messages WHERE is_me=1)""")[0][0]
        out.append({"type": "sql-first", "question": f"When did I first talk to {name}?", "facts": [d], "abstain": False})
    for pid, name in pick[3:6]:
        m, n = q(f"""SELECT substr(day,1,7) m, COUNT(*) n FROM messages WHERE thread_id IN
                     (SELECT thread_id FROM messages WHERE person_id={pid}) AND thread_id IN
                     (SELECT thread_id FROM messages WHERE is_me=1) GROUP BY m ORDER BY n DESC LIMIT 1""")[0]
        out.append({"type": "sql-busiest", "question": f"Which month was I most active with {name}?",
                    "facts": [f"{m} (about {n} messages in their shared chats)"], "abstain": False})
    for pid, name in pick[6:9]:
        n = q(f"""SELECT COUNT(*) FROM messages WHERE person_id={pid} AND thread_id IN
                  (SELECT thread_id FROM messages WHERE is_me=1)""")[0][0]
        out.append({"type": "sql-count", "question": f"Roughly how many messages has {name} sent me?",
                    "facts": [f"about {n} messages (within ~15%)"], "abstain": False})
    months = [f"{y}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)]
    for month in rng.sample(months, 3):
        rows = [r for r in q(f"""SELECT person, COUNT(*) n FROM messages WHERE is_me=0 AND day LIKE '{month}%'
                    AND thread_id IN (SELECT thread_id FROM messages WHERE is_me=1)
                    GROUP BY person_id ORDER BY n DESC LIMIT 8""") if named(r[0])]
        if rows:
            import calendar
            y, mm = month.split("-")
            label = f"{calendar.month_name[int(mm)]} {y}"
            out.append({"type": "sql-window", "question": f"Who did I talk to the most in {label}?",
                        "facts": [f"{rows[0][0]} (~{rows[0][1]} messages) among named people"], "abstain": False})
    for plat, year in rng.sample([(p, y) for p in ("twitter", "reddit", "discord", "instagram") for y in (2024, 2025)], 3):
        n = q(f"SELECT COUNT(*) FROM messages WHERE is_me=1 AND platform='{plat}' AND day LIKE '{year}%'")[0][0]
        out.append({"type": "sql-platform", "question": f"How many messages did I send on {plat.title()} in {year}?",
                    "facts": [f"about {n} messages (within ~10%)"], "abstain": False})
    return out


def abstains():
    t = SqlTool()
    topics = [("When did I go scuba diving in the Maldives?", "maldives"),
              ("What did I name my pet parrot?", "parrot"),
              ("How was my trip to Paris?", "paris trip"),
              ("What did I say at my PhD thesis defense?", "thesis defense"),
              ("When did I buy my Tesla?", "my tesla"),
              ("What happened at my sister's wedding in Goa?", "wedding in goa"),
              ("Which marathon did I run in 2024?", "ran a marathon")]
    out = []
    for qtext, needle in topics:
        n = t.run(f"SELECT COUNT(*) FROM messages WHERE is_me=1 AND lower(content) LIKE '%{needle}%'")["rows"][0][0]
        if n == 0:
            out.append({"type": "abstain", "question": qtext, "facts": [], "abstain": True})
    return out[:5]


def main(a):
    rng = random.Random(a.seed)
    client = OpenAI(api_key=dotenv_values(REPO / ".env")["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    cases = lookups(a, rng, client) + stats(rng) + abstains()
    for i, c in enumerate(cases):
        c["id"] = 101 + i
    OUT.write_text(json.dumps(cases, indent=1, ensure_ascii=False))
    print(f"wrote {len(cases)} cases to {OUT}")
    for c in cases:
        print(f"#{c['id']} [{c['type']}] {c['question']}\n      -> {c['facts']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--seed", type=int, default=7)
    main(ap.parse_args())
