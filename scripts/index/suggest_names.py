"""Guess real names for the biggest deleted/deactivated-account chats, only from explicit evidence
(self-introduction, being addressed by name, a signature). Stored in PersonNameGuesses, which
build_identity.py applies as "Name? (Deactivated Instagram account …)". Re-run build_identity afterwards.

Usage: python3 scripts/index/suggest_names.py [--min-messages 100]
"""
import argparse
import json
import random
import re

from dotenv import dotenv_values
from openai import OpenAI

from build_identity import placeholder_name
from common import REPO_ROOT, connect_index

PROMPT = """These are messages from one private chat between Sarthak and another person, "THEM", whose account was
later deleted, so their name is lost. Does the chat reveal THEM's real first name (or the name Sarthak calls them)?
Only answer with a name if it is explicit: THEM introduces themselves, Sarthak or THEM states it, or THEM is
clearly addressed by it. Nicknames are fine if clearly used for THEM. If unsure, return null.
Return JSON: {{"name": "Name" or null, "evidence": "short exact quote showing it" or null}}

{lines}"""


def verified(name, msgs, them, me_users):
    """Accept a name only from THEM introducing themselves, or Sarthak greeting/thanking them by it."""
    n = re.escape(name.lower().strip("@"))
    intro = re.compile(rf"\b(i'?m|i am|my name is|this is|call me)\s+{n}\b")
    greet = re.compile(rf"\b(hi|hey|hello|bye|thanks|thank you|thx|ok|okay|good night|gn|gm)\s*,?\s+{n}\b")
    for au, c in msgs:
        c = (c or "").lower()
        if au == them and intro.search(c):
            return True
        if au in me_users and greet.search(c):
            return True
    return False


def main(a):
    client = OpenAI(api_key=dotenv_values(REPO_ROOT / ".env")["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    conn = connect_index()
    rows = conn.execute("""
        SELECT pt.user_id, pt.thread_id, p.message_count FROM PersonThreads pt
        JOIN Persons p ON p.person_id = pt.person_id
        WHERE p.message_count >= ? ORDER BY p.message_count DESC""", (a.min_messages,)).fetchall()
    me_users = {r[0] for r in conn.execute("SELECT user_id FROM PersonAliases WHERE person_id = 1")}
    rng = random.Random(3)
    found = 0
    conn.execute("DELETE FROM PersonNameGuesses")
    for user_id, thread_id, n in rows:
        u = conn.execute("SELECT id, platform, raw_id, display_name FROM src.Users WHERE id = ?", (user_id,)).fetchone()
        t0, t1, title = conn.execute("""SELECT MIN(m.timestamp_utc), MAX(m.timestamp_utc), t.title FROM src.Messages m
            JOIN src.Threads t ON t.id = m.thread_id WHERE m.author_id = ? AND m.thread_id = ?""",
                                     (user_id, thread_id)).fetchone()
        label = placeholder_name(u, t0, t1, title)          # honest base label, independent of earlier guesses
        msgs = conn.execute("SELECT author_id, content FROM src.Messages WHERE thread_id = ? AND content != '' "
                            "ORDER BY timestamp_utc", (thread_id,)).fetchall()
        pick = msgs[:80] + rng.sample(msgs[80:], min(120, max(0, len(msgs) - 80)))
        lines = "\n".join(f"{'Sarthak' if au in me_users else ('THEM' if au == user_id else 'other')}: "
                          f"{(c or '')[:200]}" for au, c in pick)
        r = client.chat.completions.create(
            model="deepseek-flash", response_format={"type": "json_object"}, max_tokens=400,
            extra_body={"thinking": {"type": "disabled"}},
            messages=[{"role": "user", "content": PROMPT.format(lines=lines[:24000])}])
        try:
            g = json.loads(r.choices[0].message.content)
        except json.JSONDecodeError:
            continue
        name, ev = (g.get("name") or "").strip(), (g.get("evidence") or "").strip()
        if not (name and ev and len(name) <= 30 and verified(name, msgs, user_id, me_users)):
            continue
        base, _, span = label.partition(" (")
        new = f"{name.title()}? ({base.lower()}, {span.rstrip(')')})" if span else f"{name.title()}? ({base.lower()})"
        conn.execute("INSERT OR REPLACE INTO PersonNameGuesses VALUES (?,?,?,?)", (user_id, thread_id, new, ev))
        found += 1
        print(f"{n:>6} msgs | {label}\n         -> {new}   evidence: {ev[:90]!r}")
    conn.commit()
    print(f"\nnamed {found} of {len(rows)} placeholder chats; now re-run build_identity.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-messages", type=int, default=100)
    main(ap.parse_args())
