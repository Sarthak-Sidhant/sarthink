"""Build the person layer: every Users row maps to exactly one Person.

- Owner ("me"): Users rows whose raw_id or display_name exactly equals (case-insensitive) an
  alias listed for *that platform* in config/identity_map.json. No substring matching.
- Everyone else: one Person per Users row, with a readable display name
  (Twitter numeric IDs are resolved via twitter_users.db and twitter_id_map.json).
- Cross-platform merges for other people are NOT automatic: same-name users on different
  platforms are written to MergeCandidates for review. Accounts the user confirmed are one human (contacts,
  kept in the user DB, see contacts.py) share one person_id.
- person_ids are stable across rebuilds and new imports: each account (and each chat of a placeholder
  account) keeps the id it got first, recorded in the user DB (PersonIds), because notes, the profile cache
  and the graph's P_ nodes refer to them. New people get new ids after the highest one.

Usage: python3 scripts/index/build_identity.py
"""
import datetime as dt
import json
import re
import sqlite3
from collections import defaultdict

from common import DATA_DIR, IDENTITY_MAP, TWITTER_ID_MAP, TWITTER_USERS_DB, USER_DB, connect_index
from contacts import ContactStore

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
    method TEXT                       -- identity_map | singleton | contact
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


class PersonIds:
    """Lasting (platform, raw_id, chat) -> person_id map, kept in the user DB next to the notes that use it.
    chat is '' for an account, or the platform thread id for one chat of a split placeholder account."""

    def __init__(self, conn):
        self.db = sqlite3.connect(USER_DB)
        self.db.execute("""CREATE TABLE IF NOT EXISTS PersonIds (
            platform TEXT, raw_id TEXT, chat TEXT NOT NULL DEFAULT '', person_id INTEGER NOT NULL,
            PRIMARY KEY (platform, raw_id, chat))""")
        self.map = {(p, r, c): pid for p, r, c, pid in self.db.execute("SELECT * FROM PersonIds")}
        if not self.map:
            self._adopt(conn)
        self.next = max(self.map.values(), default=1) + 1
        self.new = []

    def _adopt(self, conn):
        """First run: take over the ids of the existing index, so notes written against it stay correct."""
        have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"PersonAliases", "PersonThreads"} <= have:
            return
        rows = conn.execute("""SELECT u.platform, u.raw_id, '', a.person_id FROM PersonAliases a
            JOIN src.Users u ON u.id = a.user_id WHERE a.person_id != 1 AND a.method NOT IN ('merged', 'contact')
            UNION ALL
            SELECT u.platform, u.raw_id, t.platform_thread_id, pt.person_id FROM PersonThreads pt
            JOIN src.Users u ON u.id = pt.user_id JOIN src.Threads t ON t.id = pt.thread_id""").fetchall()
        self.map = {(p, str(r), c): pid for p, r, c, pid in rows}
        self.db.executemany("INSERT OR IGNORE INTO PersonIds VALUES (?,?,?,?)",
                            [(p, r, c, pid) for (p, r, c), pid in self.map.items()])
        self.db.commit()
        print(f"PersonIds: adopted {len(self.map)} ids from the existing index")

    def get(self, platform, raw_id, chat=""):
        return self.map.get((platform, str(raw_id), chat))

    def assign(self, platform, raw_id, chat="", pid=None):
        key = (platform, str(raw_id), chat)
        if key not in self.map:
            if pid is None:
                pid, self.next = self.next, self.next + 1
            self.map[key] = pid
            self.new.append((*key, pid))
        return self.map[key]

    def save(self):
        self.db.executemany("INSERT INTO PersonIds VALUES (?,?,?,?)", self.new)
        self.db.commit()
        if self.new:
            print(f"PersonIds: {len(self.new)} new ids recorded")


def build(conn=None, verbose=True):
    """Rebuild the person tables. The server passes its own connection to relink people live."""
    log = print if verbose else (lambda *a, **k: None)
    with open(IDENTITY_MAP) as f:
        idmap = json.load(f)
    me_name = idmap.get("master_persona", "Me")
    aliases = {p: {str(a).lower() for a in lst} for p, lst in idmap.get("aliases", {}).items()}
    tw_names = load_twitter_names()

    conn = conn or connect_index()
    pids = PersonIds(conn)          # before SCHEMA drops the old tables it may adopt ids from
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

    me_matches, alias_rows = [], []
    by_norm = defaultdict(list)

    for u in users:
        name = display_name_for(u["platform"], u["raw_id"], u["display_name"], tw_names)
        if u["id"] in me_ids:
            alias_rows.append((u["id"], 1, u["platform"], name, "identity_map"))
            me_matches.append((u["id"], u["platform"], u["raw_id"], u["display_name"]))
            continue
        pid = pids.assign(u["platform"], u["raw_id"])
        conn.execute("INSERT INTO Persons(person_id, name) VALUES (?, ?)", (pid, name))
        alias_rows.append((u["id"], pid, u["platform"], name, "singleton"))
        norm = re.sub(r"[^a-z0-9]", "", name.lower().lstrip("@").removeprefix("u/"))
        if len(norm) >= 4:
            by_norm[norm].append((u["id"], u["platform"]))

    conn.executemany("INSERT INTO PersonAliases VALUES (?,?,?,?,?)", alias_rows)

    # Contacts: accounts the user confirmed are one human share the contact's person_id. Never automatic —
    # MergeCandidates only suggests. A hand-written DATA_DIR/person_merges.json is imported into contacts once,
    # e.g. [{"name": "Kabir", "accounts": [["instagram", "kabir.singh"], ["discord", "kabirrr"]]}].
    store = ContactStore()
    store.import_merges(DATA_DIR / "person_merges.json", lambda p, r: pids.get(p, r))
    links, contact_names = store.links(), store.names()
    by_acct = {(u["platform"], str(u["raw_id"])): u["id"] for u in users}
    pid_of = {r[0]: r[1] for r in alias_rows}
    alias_name = {r[0]: r[3] for r in alias_rows}
    dropped, linked = set(), 0
    for (platform, raw_id, chat), cpid in links.items():
        uid = by_acct.get((platform, raw_id))
        if chat or uid is None or uid in me_ids or pid_of[uid] == cpid:
            continue                              # per-chat links are applied with the placeholder split below
        conn.execute("INSERT OR IGNORE INTO Persons(person_id, name) VALUES (?, ?)", (cpid, alias_name[uid]))
        conn.execute("UPDATE PersonAliases SET person_id = ?, method = 'contact' WHERE user_id = ?", (cpid, uid))
        dropped.add(pid_of[uid])
        pid_of[uid] = cpid
        linked += 1
    log(f"Contacts: {linked} accounts folded into {len(set(links.values()))} contacts")

    # Placeholder accounts ("Instagram User", "Deleted User", deleted_user_*): one person per chat with an
    # honest, distinct name. The account's first chat keeps the account's person id; other chats get their
    # own ids, remembered per chat in PersonIds so a newly imported chat never renumbers the others.
    guesses = {(r[0], r[1]): r[2] for r in conn.execute("SELECT user_id, thread_id, name FROM PersonNameGuesses")}
    split = 0
    for u in users:
        if u["id"] in me_ids or not is_placeholder(u):
            continue
        chats = conn.execute("""SELECT m.thread_id, MIN(m.timestamp_utc), MAX(m.timestamp_utc), t.title,
            t.platform_thread_id FROM src.Messages m JOIN src.Threads t ON t.id = m.thread_id WHERE m.author_id = ?
            GROUP BY m.thread_id ORDER BY MIN(m.timestamp_utc)""", (u["id"],)).fetchall()
        account_pid = pid_of[u["id"]]
        if account_pid in dropped or links.get((u["platform"], str(u["raw_id"]), "")):
            continue                              # the whole placeholder account was linked to a contact
        account_pid_taken = any(pids.get(u["platform"], u["raw_id"], c[4]) == account_pid for c in chats)
        account_pid_used = False
        for tid, t0, t1, title, chat in chats:
            name = guesses.get((u["id"], tid)) or placeholder_name(u, t0, t1, title)
            pid = pids.get(u["platform"], u["raw_id"], chat)
            if pid is None:
                pid = pids.assign(u["platform"], u["raw_id"], chat, None if account_pid_taken else account_pid)
                account_pid_taken = True
            cpid = links.get((u["platform"], str(u["raw_id"]), chat))
            if cpid is not None and cpid != pid:  # this chat's deleted account was linked to a contact
                conn.execute("INSERT OR IGNORE INTO Persons(person_id, name, placeholder) VALUES (?,?,1)", (cpid, name))
                conn.execute("INSERT INTO PersonThreads VALUES (?,?,?)", (u["id"], tid, cpid))
                continue
            account_pid_used |= pid == account_pid
            if pid == account_pid:
                conn.execute("UPDATE Persons SET name = ?, placeholder = 1 WHERE person_id = ?", (name, pid))
                conn.execute("UPDATE PersonAliases SET name = ? WHERE user_id = ?", (name, u["id"]))
            else:
                conn.execute("INSERT INTO Persons(person_id, name, placeholder) VALUES (?,?,1)", (pid, name))
                split += 1
            conn.execute("INSERT INTO PersonThreads VALUES (?,?,?)", (u["id"], tid, pid))
        if not account_pid_used and account_pid not in links.values():
            dropped.add(account_pid)              # every chat of the account went to contacts

    # people whose accounts all moved into contacts; then contact names (a contact with a real account is
    # no longer a placeholder)
    if dropped:
        conn.execute(f"DELETE FROM Persons WHERE person_id IN ({','.join('?' * len(dropped))}) AND person_id NOT IN "
                     """(SELECT person_id FROM PersonThreads UNION SELECT a.person_id FROM PersonAliases a
                         WHERE NOT EXISTS (SELECT 1 FROM PersonThreads x WHERE x.user_id = a.user_id))""", tuple(dropped))
    for cpid, name in contact_names.items():
        conn.execute("UPDATE Persons SET name = ? WHERE person_id = ?", (name, cpid))
    conn.execute(f"""UPDATE Persons SET placeholder = 0 WHERE person_id IN ({','.join('?' * len(links)) or 'NULL'})
        AND person_id IN (SELECT a.person_id FROM PersonAliases a
                          WHERE NOT EXISTS (SELECT 1 FROM PersonThreads x WHERE x.user_id = a.user_id))""",
                 tuple(links.values()))

    # per-person message counts and platform (used for name lookup, incl. split placeholder people)
    conn.executescript("""
        UPDATE Persons SET platform = (SELECT MIN(a.platform) FROM PersonAliases a WHERE a.person_id = Persons.person_id);
        DROP TABLE IF EXISTS temp.pc;
        CREATE TEMP TABLE pc (pid INTEGER PRIMARY KEY, n INTEGER);
        INSERT INTO pc
          SELECT COALESCE(pt.person_id, a.person_id) AS pid, COUNT(*) AS n
          FROM src.Messages m JOIN PersonAliases a ON a.user_id = m.author_id
          LEFT JOIN PersonThreads pt ON pt.user_id = m.author_id AND pt.thread_id = m.thread_id
          GROUP BY pid;
        UPDATE Persons SET message_count = COALESCE((SELECT n FROM pc WHERE pc.pid = Persons.person_id), 0);
        UPDATE Persons SET platform = (SELECT MIN(t.platform) FROM PersonThreads pt JOIN src.Threads t ON t.id = pt.thread_id
                                       WHERE pt.person_id = Persons.person_id) WHERE platform IS NULL;
    """)
    log(f"Placeholder accounts split into {split} extra per-chat people")
    cands = [
        (n, json.dumps([i for i, _ in v]), json.dumps(sorted({p for _, p in v})))
        for n, v in by_norm.items()
        if len({p for _, p in v}) > 1
    ]
    conn.executemany("INSERT INTO MergeCandidates VALUES (?,?,?)", cands)
    conn.commit()
    pids.save()

    n_persons = conn.execute("SELECT COUNT(*) FROM Persons").fetchone()[0]
    log(f"Persons: {n_persons}  (users: {len(users)})")
    log(f"Owner '{me_name}' matched {len(me_matches)} user rows — verify these are all you:")
    for m in me_matches:
        log(f"   user_id={m[0]:<6} {m[1]:<10} raw_id={m[2]!s:<22} display={m[3]!s}")
    for platform, dn, ids in ambiguous:
        if ids:
            log(f"   AMBIGUOUS alias '{dn}' on {platform} matches {len(ids)} accounts, not assigned: user_ids={ids}")
    log(f"Cross-platform merge candidates for review: {len(cands)} (table MergeCandidates)")


if __name__ == "__main__":
    build()
