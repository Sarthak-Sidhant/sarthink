"""Build the account layer: every src.Users row is one account (see docs/accounts-and-people.md).

- Accounts are never merged. Grouping accounts into humans is the people layer (people.py, user-written).
- Owner accounts (is_me): Users rows whose raw_id or display_name exactly equals (case-insensitive) an alias
  listed for *that platform* in config/identity_map.json. They belong to person 1 ("me").
- A deleted/deactivated account ("Instagram User", deleted_user_*) is one account per chat, with an honest,
  distinct name, because one placeholder id covers many humans.
- account_ids are stable across rebuilds and imports (AccountIds in the user DB): notes, the profile cache,
  people and the graph's A_ nodes refer to them. New accounts get ids after the highest one.
- Same-name accounts on different platforms are written to MergeCandidates as "same person?" suggestions.

Usage: python3 scripts/index/build_identity.py
"""
import datetime as dt
import json
import re
import sqlite3
from collections import defaultdict

from common import DATA_DIR, IDENTITY_MAP, TWITTER_ID_MAP, TWITTER_USERS_DB, USER_DB, connect_index
from people import ME, PeopleStore

SCHEMA = """
DROP TABLE IF EXISTS Persons;
DROP TABLE IF EXISTS PersonAliases;
DROP TABLE IF EXISTS PersonThreads;
DROP TABLE IF EXISTS Accounts;
DROP TABLE IF EXISTS AccountUsers;
DROP TABLE IF EXISTS AccountThreads;
DROP TABLE IF EXISTS MergeCandidates;
CREATE TABLE IF NOT EXISTS PersonNameGuesses (   -- kept across rebuilds; filled by suggest_names.py
    user_id INTEGER, thread_id INTEGER, name TEXT, evidence TEXT,
    PRIMARY KEY (user_id, thread_id)
);
CREATE TABLE Accounts (
    account_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    platform TEXT,
    is_me INTEGER NOT NULL DEFAULT 0,
    placeholder INTEGER NOT NULL DEFAULT 0,   -- one chat of a deleted/deactivated account
    raw_id TEXT,
    chat TEXT NOT NULL DEFAULT '',            -- platform thread id for a placeholder account's chat, else ''
    message_count INTEGER DEFAULT 0
);
CREATE TABLE AccountUsers (                   -- the account a src.Users row writes as
    user_id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL,
    platform TEXT,
    name TEXT
);
CREATE INDEX idx_account_users ON AccountUsers(account_id);
CREATE TABLE AccountThreads (                 -- placeholder users: their account in each chat (overrides AccountUsers)
    user_id INTEGER, thread_id INTEGER, account_id INTEGER,
    PRIMARY KEY (user_id, thread_id)
);
CREATE INDEX idx_account_threads ON AccountThreads(account_id);
CREATE TABLE MergeCandidates (
    norm_name TEXT,
    account_ids TEXT,                 -- JSON list of Accounts.account_id
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


class AccountIdMap:
    """Lasting (platform, raw_id, chat) -> account_id, kept in the user DB (AccountIds) next to the people."""

    def __init__(self, store):
        self.store = store
        self.map = store.account_ids()
        self.next = max(self.map.values(), default=1) + 1      # 1 was the old merged "me"; never reused
        self.new = []

    def get(self, platform, raw_id, chat=""):
        return self.map.get((platform, str(raw_id), chat))

    def assign(self, platform, raw_id, chat=""):
        key = (platform, str(raw_id), chat)
        if key not in self.map:
            self.map[key], self.next = self.next, self.next + 1
            self.new.append((*key, self.map[key]))
        return self.map[key]

    def save(self):
        self.store.db.executemany("INSERT INTO AccountIds VALUES (?,?,?,?)", self.new)
        self.store.db.commit()
        return len(self.new)


def build(conn=None, verbose=True):
    """Rebuild the account tables. People (user DB) are not touched, apart from a one-time migration."""
    log = print if verbose else (lambda *a, **k: None)
    with open(IDENTITY_MAP) as f:
        idmap = json.load(f)
    me_name = idmap.get("master_persona", "Me")
    aliases = {p: {str(a).lower() for a in lst} for p, lst in idmap.get("aliases", {}).items()}
    tw_names = load_twitter_names()

    conn = conn or connect_index()
    have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    old_names = dict(conn.execute("SELECT person_id, name FROM Persons")) if "Persons" in have else {}
    store = PeopleStore(USER_DB, me_name, name_of=old_names.get)     # migrates contacts -> people once
    store.import_merges(DATA_DIR / "person_merges.json")
    ids = AccountIdMap(store)
    conn.executescript(SCHEMA)

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
    for (platform, dn), uids in by_display.items():
        if len(uids) == 1:
            me_ids.add(uids[0])
        else:
            ambiguous.append((platform, dn, [i for i in uids if i not in strong]))

    guesses = {(r[0], r[1]): r[2] for r in conn.execute("SELECT user_id, thread_id, name FROM PersonNameGuesses")}
    accounts, user_rows, thread_rows = [], [], []
    by_norm = defaultdict(list)
    for u in users:
        name = display_name_for(u["platform"], u["raw_id"], u["display_name"], tw_names)
        if u["id"] not in me_ids and is_placeholder(u):
            # one account per chat; the user row points at the first one, chats override (AccountThreads)
            chats = conn.execute("""SELECT m.thread_id, MIN(m.timestamp_utc), MAX(m.timestamp_utc), t.title,
                t.platform_thread_id FROM src.Messages m JOIN src.Threads t ON t.id = m.thread_id WHERE m.author_id = ?
                GROUP BY m.thread_id ORDER BY MIN(m.timestamp_utc)""", (u["id"],)).fetchall()
            first = None
            for tid, t0, t1, title, chat in chats:
                aid = ids.assign(u["platform"], u["raw_id"], chat)
                accounts.append((aid, guesses.get((u["id"], tid)) or placeholder_name(u, t0, t1, title),
                                 u["platform"], 0, 1, str(u["raw_id"]), chat))
                thread_rows.append((u["id"], tid, aid))
                first = first or aid
            if first:
                user_rows.append((u["id"], first, u["platform"], name))
            continue
        aid = ids.assign(u["platform"], u["raw_id"])
        is_me = int(u["id"] in me_ids)
        accounts.append((aid, name, u["platform"], is_me, 0, str(u["raw_id"]), ""))
        user_rows.append((u["id"], aid, u["platform"], name))
        if is_me:
            store.attach(ME, (u["platform"], str(u["raw_id"]), ""), commit=False)   # the owner holds their accounts
            continue
        norm = re.sub(r"[^a-z0-9]", "", name.lower().lstrip("@").removeprefix("u/"))
        if len(norm) >= 4:
            by_norm[norm].append((aid, u["platform"]))

    conn.executemany("INSERT INTO Accounts(account_id, name, platform, is_me, placeholder, raw_id, chat) "
                     "VALUES (?,?,?,?,?,?,?)", accounts)
    conn.executemany("INSERT INTO AccountUsers VALUES (?,?,?,?)", user_rows)
    conn.executemany("INSERT INTO AccountThreads VALUES (?,?,?)", thread_rows)
    conn.executescript("""
        DROP TABLE IF EXISTS temp.ac;
        CREATE TEMP TABLE ac (aid INTEGER PRIMARY KEY, n INTEGER);
        INSERT INTO ac
          SELECT COALESCE(at.account_id, au.account_id), COUNT(*)
          FROM src.Messages m JOIN AccountUsers au ON au.user_id = m.author_id
          LEFT JOIN AccountThreads at ON at.user_id = m.author_id AND at.thread_id = m.thread_id
          GROUP BY 1;
        UPDATE Accounts SET message_count = COALESCE((SELECT n FROM ac WHERE ac.aid = Accounts.account_id), 0);
        DROP TABLE temp.ac;
    """)
    cands = [(n, json.dumps([a for a, _ in v]), json.dumps(sorted({p for _, p in v})))
             for n, v in by_norm.items() if len({p for _, p in v}) > 1]
    conn.executemany("INSERT INTO MergeCandidates VALUES (?,?,?)", cands)
    conn.commit()
    store.db.commit()
    new = ids.save()

    n_acc = conn.execute("SELECT COUNT(*) FROM Accounts").fetchone()[0]
    n_ppl = store.db.execute("SELECT COUNT(*) FROM People").fetchone()[0]
    log(f"Accounts: {n_acc} (users: {len(users)}, {len(thread_rows)} placeholder chats); {new} new account ids")
    log(f"People: {n_ppl} (incl. me); attached accounts: {len(store.membership())}")
    log(f"Owner '{me_name}': {len(me_ids)} accounts — verify these are all you:")
    for u in users:
        if u["id"] in me_ids:
            log(f"   user_id={u['id']:<6} {u['platform']:<10} raw_id={u['raw_id']!s:<22} display={u['display_name']!s}")
    for platform, dn, uids in ambiguous:
        if uids:
            log(f"   AMBIGUOUS alias '{dn}' on {platform} matches {len(uids)} accounts, not assigned: user_ids={uids}")
    log(f"Same-name accounts across platforms (suggestions): {len(cands)}")


if __name__ == "__main__":
    build()
