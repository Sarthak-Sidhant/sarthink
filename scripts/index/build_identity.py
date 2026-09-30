"""Build the person layer: every Users row maps to exactly one Person.

- Owner ("me"): Users rows whose raw_id or display_name exactly equals (case-insensitive) an
  alias listed for *that platform* in config/identity_map.json. No substring matching.
- Everyone else: one Person per Users row, with a readable display name
  (Twitter numeric IDs are resolved via twitter_users.db and twitter_id_map.json).
- Cross-platform merges for other people are NOT automatic: same-name users on different
  platforms are written to MergeCandidates for review.

Usage: python3 scripts/index/build_identity.py
"""
import json
import re
import sqlite3
from collections import defaultdict

from common import IDENTITY_MAP, TWITTER_ID_MAP, TWITTER_USERS_DB, connect_index

SCHEMA = """
DROP TABLE IF EXISTS Persons;
DROP TABLE IF EXISTS PersonAliases;
DROP TABLE IF EXISTS MergeCandidates;
CREATE TABLE Persons (
    person_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    is_me INTEGER NOT NULL DEFAULT 0
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


def build():
    with open(IDENTITY_MAP) as f:
        idmap = json.load(f)
    me_name = idmap.get("master_persona", "Me")
    aliases = {p: {str(a).lower() for a in lst} for p, lst in idmap.get("aliases", {}).items()}
    tw_names = load_twitter_names()

    conn = connect_index()
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO Persons(person_id, name, is_me) VALUES (1, ?, 1)", (me_name,))

    users = conn.execute("SELECT id, platform, raw_id, display_name FROM src.Users").fetchall()

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
