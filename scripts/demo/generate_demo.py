"""Generate the fictional demo archive ("Aarav Mehta") into demo/db/sarthink_memory.db.

Everything here is invented. Conversations are written by an LLM from storyline specs (platform, people,
dates, what must happen) and stored in the same schema the real parsers produce, so the normal pipeline
(identity -> chunks -> summaries -> embeddings -> graph) runs on it unchanged.

Usage: python3 scripts/demo/generate_demo.py [--out demo]   (needs DEEPSEEK_API_KEY; ~$0.5)
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import random
import sqlite3
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

REPO = Path(__file__).resolve().parents[2]
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
ME = "Aarav"

PEOPLE = {
    # name: (platform handle per platform)
    "Rohan": {"discord": "rohan.dev", "instagram": "rohan_k"},
    "Kabir": {"discord": "kabirrr", "instagram": "kabir.singh"},
    "Meera": {"instagram": "meera.writes", "discord": "meeraaa"},
    "Ananya": {"instagram": "ananya.desai"},
    "Ishaan": {"instagram": "ishaan_m"},          # cousin
    "Priya": {"instagram": "priya.clicks"},       # photography friend
    "Dev": {"discord": "devoptics"},             # homelab buddy
    "Sana": {"instagram": "sana.k"},             # later deactivates her account
}

PERSONA = """Aarav Mehta (21) is a fictional CS student in Bengaluru, originally from Patna. He is into homelabs,
photography and indie music, texts in casual English mixed with Hindi (Hinglish) with friends, lowercase, short
bursts. Friends: Rohan (closest friend, college batchmate, dev), Kabir (school friend from Patna, funny, very
Hinglish), Meera (close friend who writes poetry), Ananya (classmate, organised), Ishaan (cousin in Delhi),
Priya (photography friend), Dev (homelab buddy he met on Discord), Sana (friend who deactivates Instagram in
mid-2025). Never include real phone numbers, emails or street addresses."""

# Storyline conversations: facts in "must" have to appear explicitly so answers can be checked.
SPECS = [
    # internship arc
    dict(platform="instagram", people=["Rohan"], start="2025-02-10 21:30", n=60,
         topic="Aarav got an internship offer email from Nimbus Labs in Bengaluru and tells Rohan.",
         must=["backend engineering internship at Nimbus Labs", "stipend is ₹60,000 per month",
               "dates 2 March to 29 May 2026", "he has not accepted yet, needs to check college project schedule"]),
    dict(platform="discord", people=["Dev"], start="2025-02-14 23:10", n=40,
         topic="Aarav asks Dev whether to accept the Nimbus Labs offer or wait for a better one.",
         must=["Dev says take it, Nimbus has good mentorship", "Aarav decides to accept by Monday 17 Feb"]),
    dict(platform="instagram", people=["Ananya"], start="2025-02-18 18:00", n=35,
         topic="Aarav tells Ananya he accepted the Nimbus Labs internship and asks about the college NOC form.",
         must=["he accepted the offer on 17 Feb", "Ananya says the NOC form must be submitted to the HOD by 25 Feb",
               "Aarav promises to send Ananya the internship letter by Friday"]),
    # goa trip arc (group chat)
    dict(platform="instagram", people=["Rohan", "Kabir", "Meera", "Ananya"], group="Goa December crew",
         start="2024-11-20 20:00", n=90,
         topic="The friends plan a December Goa trip: dates, train vs flight, budget, stay.",
         must=["trip dates 21 to 26 December 2024", "budget about ₹12,500 per person", "Ananya books a hostel in Anjuna",
               "they choose the train from Bengaluru to Madgaon"]),
    dict(platform="instagram", people=["Rohan", "Kabir", "Meera", "Ananya"], group="Goa December crew",
         start="2024-12-21 06:40", n=70,
         topic="Trip day: they miss the train and scramble.",
         must=["they missed the train because Kabir's cab got stuck in traffic near Majestic and they reached after departure",
               "they took a night bus instead and lost ₹1,800 per person", "Meera jokes about it for days"]),
    dict(platform="discord", people=["Kabir"], start="2024-12-28 22:15", n=45, hinglish=True,
         topic="After Goa, Kabir and Aarav laugh about the trip and settle money in Hinglish.",
         must=["Kabir owes Aarav ₹2,300 for the hostel", "Kabir promises to pay by 5 January", "favourite memory: Chapora fort sunset"]),
    # photography arc
    dict(platform="instagram", people=["Priya"], start="2024-07-02 19:00", n=50,
         topic="Aarav asks Priya which camera to buy as a beginner.",
         must=["Priya suggests Sony a6400 with the 16-50 kit lens", "budget around ₹75,000", "he is saving from freelance work"]),
    dict(platform="instagram", people=["Priya"], start="2024-08-16 20:30", n=40,
         topic="Aarav bought the camera and shares first photos.",
         must=["he bought a used Sony a6400 for ₹58,000 from OLX", "first photos at Cubbon Park", "Priya gives tips on shooting in RAW"]),
    # homelab arc
    dict(platform="discord", people=["Dev"], start="2024-04-05 22:00", n=60,
         topic="Aarav sets up a Raspberry Pi 5 homelab with Dev's help.",
         must=["Raspberry Pi 5 with 8GB RAM", "running Pi-hole and Jellyfin", "SD card corruption problem, Dev suggests booting from an NVMe SSD"]),
    dict(platform="discord", people=["Dev"], start="2025-05-11 21:00", n=55,
         topic="Aarav upgrades the homelab to a used office PC.",
         must=["bought a used Dell OptiPlex 7070 for ₹14,000", "installed Proxmox", "moved Jellyfin and Nextcloud to it"]),
    # meera friendship arc
    dict(platform="instagram", people=["Meera"], start="2025-06-03 23:30", n=50,
         topic="Meera feels Aarav has been ignoring her since the internship prep; tension.",
         must=["Meera says he skipped her poetry reading on 31 May", "Aarav apologises but is defensive", "they stop talking for a while"]),
    dict(platform="instagram", people=["Meera"], start="2025-09-07 19:45", n=45,
         topic="Aarav and Meera make up after three months.",
         must=["Aarav reaches out on her birthday, 7 September", "they talk about missing each other", "they plan coffee at Third Wave on Saturday"]),
    # family, exams, misc
    dict(platform="instagram", people=["Ishaan"], start="2024-11-08 22:30", n=40, hinglish=True,
         topic="Exam stress before semester exams; cousin Ishaan cheers him up.",
         must=["semester exams start 18 November 2024", "hardest subject is Operating Systems", "Ishaan suggests a study plan of 3 chapters a day"]),
    dict(platform="instagram", people=["Sana"], start="2025-03-12 21:00", n=45,
         topic="Sana and Aarav chat about music and a gig.",
         must=["they go to a Prateek Kuhad concert on 22 March 2025", "Sana says she might take a break from Instagram"]),
    dict(platform="discord", people=["Rohan"], start="2025-08-20 23:00", n=50,
         topic="Rohan and Aarav plan a side project: a habit tracker app.",
         must=["app name HabitHop", "Rohan does the Flutter frontend, Aarav the FastAPI backend",
               "Aarav promises to set up the database schema by 25 August"]),
]
FILLER_TOPICS = ["random memes and college gossip", "cricket match banter", "late night song recommendations",
                 "food delivery and hostel mess complaints", "a new web series", "assignment deadlines",
                 "weekend plans", "a funny incident in class", "laptop problems", "gym plans that never happen"]

GEN = PERSONA + """

Write a realistic {platform} {kind} between {who}, starting {start} (IST), about {n} messages total.
Situation: {topic}
These facts MUST appear explicitly in the messages: {must}
Style: natural chat, short bursts, {lang}; people send several messages in a row; time gaps of seconds to
minutes (occasionally hours). Sender names must be exactly one of: {names}.
Return JSON: {{"messages": [{{"from": "<name>", "minutes": <minutes since start>, "text": "<message>"}}]}}"""

PUBLIC = PERSONA + """

Write {n} realistic public {platform} items by Aarav and strangers, spread over 2024-2025, about: {topic}.
Each item is a post (or tweet) with 2-6 replies; include Aarav replying to strangers too. Usernames for strangers
should look like real {platform} handles (fictional).
Return JSON: {{"threads": [{{"title": "<post title or first words>", "date": "YYYY-MM-DD HH:MM",
  "posts": [{{"id": 1, "parent": null, "from": "<handle or Aarav>", "minutes": 0, "text": "..."}}]}}]}}"""


def llm(client, prompt, tries=3):
    for _ in range(tries):
        try:
            r = client.chat.completions.create(model="deepseek-v4-pro", response_format={"type": "json_object"},
                                               max_tokens=12000, extra_body={"thinking": {"type": "disabled"}},
                                               messages=[{"role": "user", "content": prompt}])
            return json.loads(r.choices[0].message.content)
        except (json.JSONDecodeError, TypeError):
            continue                     # malformed/truncated JSON: ask again
    print("  (skipped one conversation: no valid JSON after retries)")
    return {}


def main(a):
    out = REPO / a.out
    (out / "db").mkdir(parents=True, exist_ok=True)
    dbp = out / "db" / "sarthink_memory.db"
    dbp.unlink(missing_ok=True)
    db = sqlite3.connect(dbp)
    db.executescript("""
        CREATE TABLE Users (id INTEGER PRIMARY KEY AUTOINCREMENT, platform TEXT, raw_id TEXT, display_name TEXT,
                            UNIQUE(platform, raw_id));
        CREATE TABLE Threads (id INTEGER PRIMARY KEY AUTOINCREMENT, platform TEXT, platform_thread_id TEXT, title TEXT,
                              UNIQUE(platform, platform_thread_id));
        CREATE TABLE Messages (msg_id TEXT PRIMARY KEY, thread_id INTEGER, author_id INTEGER, timestamp_utc INTEGER,
                               content TEXT, parent_msg_id TEXT);
        CREATE INDEX idx_timestamp ON Messages(timestamp_utc); CREATE INDEX idx_thread ON Messages(thread_id);
        CREATE INDEX idx_author ON Messages(author_id); CREATE INDEX idx_parent ON Messages(parent_msg_id);""")
    client = OpenAI(api_key=dotenv_values(REPO / ".env")["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    rng = random.Random(42)

    def user(platform, name):
        handle = "aarav.mehta" if name == ME else PEOPLE.get(name, {}).get(platform, name)
        if platform == "instagram" and name == "Sana":
            handle, display = "instagram_deactivated_88", "Instagram User"   # deactivated later
        else:
            display = "Aarav Mehta" if name == ME else (handle if platform != "instagram" else name)
        db.execute("INSERT OR IGNORE INTO Users(platform, raw_id, display_name) VALUES (?,?,?)", (platform, handle, display))
        return db.execute("SELECT id FROM Users WHERE platform=? AND raw_id=?", (platform, handle)).fetchone()[0]

    specs = list(SPECS)
    for i in range(a.filler):
        p = rng.choice(["Rohan", "Kabir", "Meera", "Ananya", "Ishaan", "Priya", "Dev"])
        plat = rng.choice(list(PEOPLE[p]))
        day = dt.date(2024, 1, 1) + dt.timedelta(days=rng.randrange(0, 720))
        specs.append(dict(platform=plat, people=[p], start=f"{day} {rng.randrange(17, 24)}:{rng.randrange(60):02d}",
                          n=rng.randrange(20, 45), topic=rng.choice(FILLER_TOPICS), must=[], hinglish=p in ("Kabir", "Ishaan")))

    def gen(spec):
        names = [ME] + spec["people"]
        kind = "group chat called '" + spec["group"] + "'" if spec.get("group") else "private chat"
        lang = "mostly Hinglish (Roman script)" if spec.get("hinglish") else "casual English with some Hinglish"
        return spec, llm(client, GEN.format(platform=spec["platform"], kind=kind, who=", ".join(names), start=spec["start"],
                                            n=spec["n"], topic=spec["topic"], must="; ".join(spec["must"]) or "(none)",
                                            lang=lang, names=", ".join(names)))

    seq = 0
    with cf.ThreadPoolExecutor(8) as ex:
        for spec, res in ex.map(gen, specs):
            plat, names = spec["platform"], [ME] + spec["people"]
            if spec.get("group"):
                title, ptid = spec["group"], f"group_{spec['group']}"
            else:
                other = spec["people"][0]
                title = f"Direct Messages - {PEOPLE[other].get('discord', other)}" if plat == "discord" else (
                    "Instagram User" if other == "Sana" else other)
                ptid = f"dm_{plat}_{other}"
            db.execute("INSERT OR IGNORE INTO Threads(platform, platform_thread_id, title) VALUES (?,?,?)", (plat, ptid, title))
            tid = db.execute("SELECT id FROM Threads WHERE platform=? AND platform_thread_id=?", (plat, ptid)).fetchone()[0]
            t0 = dt.datetime.strptime(spec["start"], "%Y-%m-%d %H:%M").replace(tzinfo=IST)
            prev = None
            for m in res.get("messages", []):
                who = m.get("from") if m.get("from") in names else ME
                seq += 1
                mid = f"{plat}_{tid}_{seq}"
                ts = int((t0 + dt.timedelta(minutes=float(m.get("minutes") or 0))).timestamp())
                db.execute("INSERT INTO Messages VALUES (?,?,?,?,?,?)",
                           (mid, tid, user(plat, who), ts, m.get("text", ""), prev if plat == "instagram" else None))
                prev = mid

    # public threads: reddit posts and tweets with reply trees
    for plat, topic, n in [("reddit", "photography beginner questions, homelab builds and Bengaluru college life", 14),
                           ("twitter", "tech takes, homelab screenshots, cricket and student life", 25)]:
        res = llm(client, PUBLIC.format(n=n, platform=plat, topic=topic))
        for th in res.get("threads", []):
            seq += 1
            db.execute("INSERT INTO Threads(platform, platform_thread_id, title) VALUES (?,?,?)",
                       (plat, f"pub_{plat}_{seq}", th.get("title", "post")[:80] if plat == "reddit" else f"Tweet Thread {seq}"))
            tid = db.execute("SELECT last_insert_rowid()").fetchone()[0]
            t0 = dt.datetime.strptime(th.get("date", "2025-01-01 12:00"), "%Y-%m-%d %H:%M").replace(tzinfo=IST)
            ids = {}
            for p in th.get("posts", []):
                seq += 1
                mid = f"{plat}_{tid}_{seq}"
                ids[p.get("id")] = mid
                who = p.get("from") or "user"
                author = user(plat, ME) if who == ME else user(plat, who)
                ts = int((t0 + dt.timedelta(minutes=float(p.get("minutes") or 0))).timestamp())
                db.execute("INSERT INTO Messages VALUES (?,?,?,?,?,?)",
                           (mid, tid, author, ts, p.get("text", ""), ids.get(p.get("parent"))))
    db.commit()
    (out / "identity_map.json").write_text(json.dumps({
        "master_persona": "Aarav Mehta (me)",
        "aliases": {p: ["aarav.mehta"] for p in ("instagram", "discord", "reddit", "twitter")}}, indent=1))
    n = db.execute("SELECT COUNT(*) FROM Messages").fetchone()[0]
    print(f"demo archive: {n} messages, {db.execute('SELECT COUNT(*) FROM Threads').fetchone()[0]} threads, "
          f"{db.execute('SELECT COUNT(*) FROM Users').fetchone()[0]} users -> {dbp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo")
    ap.add_argument("--filler", type=int, default=45)
    main(ap.parse_args())
