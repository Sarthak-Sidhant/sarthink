"""Build the person layer: every Users row maps to exactly one Person.

- Owner ("me"): Users rows whose raw_id or display_name exactly equals (case-insensitive) an
  alias listed for *that platform* in config/identity_map.json. No substring matching.
- Everyone else: one Person per Users row, with a readable display name
  (Twitter numeric IDs are resolved via twitter_users.db and twitter_id_map.json).
- Cross-platform merges for other people are NOT automatic: same-name users on different
  platforms are written to MergeCandidates for review.

Usage: python3 scripts/index/build_identity.py
"""
import datetime as dt
import json
import re
import sqlite3
from collections import defaultdict

from common import DATA_DIR, IDENTITY_MAP, TWITTER_ID_MAP, TWITTER_USERS_DB, connect_index

SCHEMA = """
DROP TABLE IF EXISTS Persons;
DROP TABLE IF EXISTS PersonAliases;
DROP TABLE IF EXISTS MergeCandidates;
DROP TABLE IF EXISTS PersonThreads;
CREATE TABLE PersonThreads (           -- per-chat person for placeholder accounts (overrides PersonAliases)
    user_id INTEGER, thread_id INTEGER, person_id INTEGER,
    PRIMARY KEY (user_id, thread_id)
);
CREATE TABLE IF NOT EXISTS PersonNameGuesses (   -- kept across rebuilds; filled by suggest_names.py
    user_id INTEGER, thread_id INTEGER, name TEXT, evidence TEXT,
    PRIMARY KEY (user_id, thread_id)
);
CREATE TABLE Persons (
    person_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    is_me INTEGER NOT NULL DEFAULT 0,
    placeholder INTEGER NOT NULL DEFAULT 0,   -- deleted/deactivated account, one person per chat
    platform TEXT,
    message_count INTEGER DEFAULT 0
);
CREATE TABLE PersonAliases (
    user_id INTEGER PRIMARY KEY,      -- src.Users.id
    person_id INTEGER NOT NULL REFERENCES Persons(person_id),
    platform TEXT,
    name TEXT,                        -- resolved per-platform display name
    method TEXT                       -- identity_map | singleton | manual
);
CREATE INDEX idx_alias_person ON PersonAliases(person_id);
CREATE TABLE MergeCandidates (
    norm_name TEXT,
    user_ids TEXT,                    -- JSON list of src.Users.id
    platforms TEXT                    -- JSON list
);
"""

PLACEHOLDER = re.compile(r"^\[User \d+\]$")


def load_twitter_names():
    names = {}
    try:
        with open(TWITTER_ID_MAP) as f:
            for k, v in json.load(f).items():
                if not str(v).startswith("@@"):  # skip old fake animal aliases
                    names[str(k)] = str(v)
    except FileNotFoundError:
        pass
    try:
        conn = sqlite3.connect(TWITTER_USERS_DB)
        for id_str, screen_name in conn.execute(
            "SELECT id_str, screen_name FROM UserHistory ORDER BY last_seen"
        ):
            names[str(id_str)] = screen_name  # latest seen wins
        conn.close()
    except sqlite3.Error:
        pass
    return names


def display_name_for(platform, raw_id, display_name, tw_names):
    raw_id = str(raw_id or "")
    dn = str(display_name or "").strip()
    if platform == "twitter":
        if raw_id in tw_names:
            return "@" + tw_names[raw_id]
        if dn and not PLACEHOLDER.match(dn) and not dn.isdigit():
            return dn
        return f"twitter_user_{raw_id[-4:]}" if raw_id else "unknown"
    if platform == "reddit":
        return "u/" + (raw_id or dn or "unknown")
    return dn or raw_id or "unknown"


PLACEHOLDER_NAMES = {"instagram user", "deleted user", "facebook user"}


def is_placeholder(u):
    dn, rid = str(u["display_name"] or "").lower(), str(u["raw_id"] or "").lower()
    return dn in PLACEHOLDER_NAMES or rid.startswith("deleted_user") or dn.startswith("deleted_user")


def _span(t0, t1):
    if not t0:
        return "unknown dates"
    f = lambda t: dt.datetime.fromtimestamp(t, dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime("%b %Y")  # noqa: E731
    return f(t0) if f(t0) == f(t1) else f"{f(t0)} – {f(t1)}"


def placeholder_name(u, t0, t1, title):
    plat = u["platform"]
    rid = str(u["raw_id"] or "").lower()
    if plat == "reddit":
        if "room" in rid or (title or "").lower().startswith("dm"):
            return f"Deleted Reddit user (DM, {_span(t0, t1)})"
        return f"Deleted Reddit user (in “{(title or 'a thread')[:32]}”)"
    label = {"instagram": "Deactivated Instagram account", "facebook": "Deactivated Facebook account",
             "discord": "Deleted Discord user"}.get(plat, f"Deleted {plat} user")
    return f"{label} ({_span(t0, t1)})"


def build():
    with open(IDENTITY_MAP) as f:
        idmap = json.load(f)
    me_name = idmap.get("master_persona", "Me")
    aliases = {p: {str(a).lower() for a in lst} for p, lst in idmap.get("aliases", {}).items()}
    tw_names = load_twitter_names()

    conn = connect_index()
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO Persons(person_id, name, is_me) VALUES (1, ?, 1)", (me_name,))

    users = conn.execute("SELECT id, platform, raw_id, display_name FROM src.Users ORDER BY id").fetchall()

    # Strong keys (account id, twitter screen name) always identify the owner. A display-name
    # match only counts if exactly one account on that platform carries that name, since
    # display names like "Sarthak" are shared by many different people.
    strong, by_display = set(), defaultdict(list)
    for u in users:
        plat_aliases = aliases.get(u["platform"], set())
        strong_keys = {str(u["raw_id"] or "").lower()}
        if u["platform"] == "twitter" and str(u["raw_id"]) in tw_names:
            strong_keys.add(tw_names[str(u["raw_id"])].lower())
        if strong_keys & plat_aliases:
            strong.add(u["id"])
        dn = str(u["display_name"] or "").lower()
        if dn in plat_aliases:
            by_display[(u["platform"], dn)].append(u["id"])
    me_ids, ambiguous = set(strong), []
    for (platform, dn), ids in by_display.items():
        if len(ids) == 1:
            me_ids.add(ids[0])
        else:
            ambiguous.append((platform, dn, [i for i in ids if i not in strong]))

    me_matches, alias_rows, next_pid = [], [], 2
    by_norm = defaultdict(list)

    for u in users:
        name = display_name_for(u["platform"], u["raw_id"], u["display_name"], tw_names)
        if u["id"] in me_ids:
            alias_rows.append((u["id"], 1, u["platform"], name, "identity_map"))
            me_matches.append((u["id"], u["platform"], u["raw_id"], u["display_name"]))
            continue
        conn.execute("INSERT INTO Persons(person_id, name) VALUES (?, ?)", (next_pid, name))
        alias_rows.append((u["id"], next_pid, u["platform"], name, "singleton"))
        next_pid += 1
        norm = re.sub(r"[^a-z0-9]", "", name.lower().lstrip("@").removeprefix("u/"))
        if len(norm) >= 4:
            by_norm[norm].append((u["id"], u["platform"]))

    conn.executemany("INSERT INTO PersonAliases VALUES (?,?,?,?,?)", alias_rows)

    # Approved merges (DATA_DIR/person_merges.json): accounts the user confirmed are the same person across
    # platforms, e.g. [{"name": "Kabir", "accounts": [["instagram", "kabir.singh"], ["discord", "kabirrr"]]}].
    # Never automatic — MergeCandidates only suggests.
    merges_file = DATA_DIR / "person_merges.json"
    if merges_file.exists():
        by_acct = {(u["platform"], str(u["raw_id"])): u["id"] for u in users}
        pid_of = {r[0]: r[1] for r in alias_rows}
        merged = 0
        for grp in json.loads(merges_file.read_text()):
            uids = [by_acct.get((p, str(r))) for p, r in grp.get("accounts", [])]
            uids = [u for u in uids if u is not None and u not in me_ids]
            if not uids:
                continue
            keep = pid_of[uids[0]]                # a single account just gets the given display name
            for u in uids[1:]:
                gone = pid_of[u]
                conn.execute("UPDATE PersonAliases SET person_id = ?, method = 'merged' WHERE user_id = ?", (keep, u))
                conn.execute("DELETE FROM Persons WHERE person_id = ?", (gone,))
                merged += 1
            if grp.get("name"):
                conn.execute("UPDATE Persons SET name = ? WHERE person_id = ?", (grp["name"], keep))
        print(f"Approved merges applied: {merged} accounts folded into existing people")

    # Placeholder accounts ("Instagram User", "Deleted User", deleted_user_*): one person per chat with an
    # honest, distinct name. The first chat keeps the account's person id; extra chats get new ids (appended,
    # so every other person's id is unchanged).
    pid_of = {r[0]: r[1] for r in alias_rows}
    guesses = {(r[0], r[1]): r[2] for r in conn.execute("SELECT user_id, thread_id, name FROM PersonNameGuesses")}
    split = 0
    for u in users:
        if u["id"] in me_ids or not is_placeholder(u):
            continue
        chats = conn.execute("""SELECT m.thread_id, MIN(m.timestamp_utc), MAX(m.timestamp_utc), t.title
            FROM src.Messages m JOIN src.Threads t ON t.id = m.thread_id WHERE m.author_id = ?
            GROUP BY m.thread_id ORDER BY MIN(m.timestamp_utc)""", (u["id"],)).fetchall()
        for i, (tid, t0, t1, title) in enumerate(chats):
            name = guesses.get((u["id"], tid)) or placeholder_name(u, t0, t1, title)
            if i == 0:
                pid = pid_of[u["id"]]
                conn.execute("UPDATE Persons SET name = ?, placeholder = 1 WHERE person_id = ?", (name, pid))
                conn.execute("UPDATE PersonAliases SET name = ? WHERE user_id = ?", (name, u["id"]))
            else:
                pid = next_pid
                next_pid += 1
                conn.execute("INSERT INTO Persons(person_id, name, placeholder) VALUES (?,?,1)", (pid, name))
                split += 1
            conn.execute("INSERT INTO PersonThreads VALUES (?,?,?)", (u["id"], tid, pid))

    # per-person message counts and platform (used for name lookup, incl. split placeholder people)
    conn.executescript("""
        UPDATE Persons SET platform = (SELECT MIN(a.platform) FROM PersonAliases a WHERE a.person_id = Persons.person_id);
        CREATE TEMP TABLE pc AS
          SELECT COALESCE(pt.person_id, a.person_id) AS pid, COUNT(*) AS n
          FROM src.Messages m JOIN PersonAliases a ON a.user_id = m.author_id
          LEFT JOIN PersonThreads pt ON pt.user_id = m.author_id AND pt.thread_id = m.thread_id
          GROUP BY pid;
        UPDATE Persons SET message_count = COALESCE((SELECT n FROM pc WHERE pc.pid = Persons.person_id), 0);
        UPDATE Persons SET platform = (SELECT MIN(t.platform) FROM PersonThreads pt JOIN src.Threads t ON t.id = pt.thread_id
                                       WHERE pt.person_id = Persons.person_id) WHERE platform IS NULL;
    """)
    print(f"Placeholder accounts split into {split} extra per-chat people")
    cands = [
        (n, json.dumps([i for i, _ in v]), json.dumps(sorted({p for _, p in v})))
        for n, v in by_norm.items()
        if len({p for _, p in v}) > 1
    ]
    conn.executemany("INSERT INTO MergeCandidates VALUES (?,?,?)", cands)
    conn.commit()

    print(f"Persons: {next_pid - 1}  (users: {len(users)})")
    print(f"Owner '{me_name}' matched {len(me_matches)} user rows — verify these are all you:")
    for m in me_matches:
        print(f"   user_id={m[0]:<6} {m[1]:<10} raw_id={m[2]!s:<22} display={m[3]!s}")
    for platform, dn, ids in ambiguous:
        if ids:
            print(f"   AMBIGUOUS alias '{dn}' on {platform} matches {len(ids)} accounts, not assigned: user_ids={ids}")
    print(f"Cross-platform merge candidates for review: {len(cands)} (table MergeCandidates)")


if __name__ == "__main__":
    build()
