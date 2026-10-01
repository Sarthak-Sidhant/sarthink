"""Contacts: the humans behind accounts, as confirmed by the user.

A contact is one person_id that several accounts point to, e.g. Rohan's Instagram and Discord, or a deactivated
Instagram DM the user recognised. Accounts are keyed like PersonIds: (platform, raw_id, chat), where chat is ''
for a whole account and the platform thread id for one chat of a deleted/deactivated account.

Contacts live in the user DB (user-written, survives rebuilds); build_identity.py applies them. A contact keeps
the person_id of its primary account (the oldest id when two people are linked), so notes stay attached.
Same-name suggestions the user turned down are remembered in ContactRejections and not offered again.
"""
import datetime as dt
import json
import sqlite3

from common import USER_DB

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def key_str(k):
    return "|".join(map(str, k))


class ContactStore:
    def __init__(self, path=USER_DB):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS Contacts (person_id INTEGER PRIMARY KEY, name TEXT, updated TEXT);
            CREATE TABLE IF NOT EXISTS ContactAccounts (
                platform TEXT, raw_id TEXT, chat TEXT NOT NULL DEFAULT '', person_id INTEGER NOT NULL,
                PRIMARY KEY (platform, raw_id, chat));
            CREATE TABLE IF NOT EXISTS ContactRejections (a TEXT, b TEXT, PRIMARY KEY (a, b));
            CREATE TABLE IF NOT EXISTS ContactImports (source TEXT PRIMARY KEY, at TEXT);
        """)

    def _now(self):
        return dt.datetime.now(IST).isoformat(timespec="minutes")

    def names(self):
        return dict(self.db.execute("SELECT person_id, name FROM Contacts WHERE name IS NOT NULL AND name != ''"))

    def links(self):
        return {(p, str(r), c): pid for p, r, c, pid in self.db.execute("SELECT * FROM ContactAccounts")}

    def import_merges(self, path, pid_for):
        """One-time import of a hand-written person_merges.json ([{"name", "accounts": [[platform, raw_id]]}]).
        pid_for(platform, raw_id) gives the account's current person id; the first account's id becomes the
        contact's. Imported once, so unlinking something listed there later is not undone by the next build."""
        if not path.exists() or self.db.execute("SELECT 1 FROM ContactImports WHERE source = ?",
                                                (path.name,)).fetchone():
            return 0
        n = 0
        for grp in json.loads(path.read_text()):
            accts = [(p, str(r)) for p, r in grp.get("accounts", [])]
            pids = [pid_for(p, r) for p, r in accts]
            if not accts or pids[0] is None:
                continue
            for (p, r), pid in zip(accts, pids):
                if pid is not None:
                    self.db.execute("INSERT OR REPLACE INTO ContactAccounts VALUES (?,?,'',?)", (p, r, pids[0]))
            self.db.execute("INSERT OR REPLACE INTO Contacts VALUES (?,?,?)", (pids[0], grp.get("name"), self._now()))
            n += 1
        self.db.execute("INSERT INTO ContactImports VALUES (?,?)", (path.name, self._now()))
        self.db.commit()
        return n

    def link(self, keep, gone, keep_accounts, gone_accounts, name=None):
        """Make `gone` part of contact `keep`: every account of both people points to keep."""
        names = self.names()
        name = name or names.get(keep) or names.get(gone)
        for k in [*keep_accounts, *gone_accounts]:
            self.db.execute("INSERT OR REPLACE INTO ContactAccounts VALUES (?,?,?,?)", (*k, keep))
        self.db.execute("UPDATE ContactAccounts SET person_id = ? WHERE person_id = ?", (keep, gone))
        self.db.execute("DELETE FROM Contacts WHERE person_id = ?", (gone,))
        self.db.execute("INSERT OR REPLACE INTO Contacts VALUES (?,?,?)", (keep, name, self._now()))
        self.db.commit()

    def unlink(self, key):
        self.db.execute("DELETE FROM ContactAccounts WHERE platform = ? AND raw_id = ? AND chat = ?", key)
        self.db.commit()

    def rename(self, pid, name, accounts):
        for k in accounts:
            self.db.execute("INSERT OR IGNORE INTO ContactAccounts VALUES (?,?,?,?)", (*k, pid))
        self.db.execute("INSERT OR REPLACE INTO Contacts VALUES (?,?,?)", (pid, name.strip() or None, self._now()))
        self.db.commit()

    def reject(self, accounts_a, accounts_b):
        for a in accounts_a:
            for b in accounts_b:
                self.db.execute("INSERT OR IGNORE INTO ContactRejections VALUES (?,?)",
                                tuple(sorted((key_str(a), key_str(b)))))
        self.db.commit()

    def rejected(self, accounts_a, accounts_b):
        pairs = [tuple(sorted((key_str(a), key_str(b)))) for a in accounts_a for b in accounts_b]
        return any(self.db.execute("SELECT 1 FROM ContactRejections WHERE a = ? AND b = ?", p).fetchone()
                   for p in pairs)


def accounts_of(conn, pid):
    """Account keys behind a person in the current index, with display info. A deleted/deactivated account
    is split per chat, so its key carries the chat; a normal account's key has chat ''."""
    rows = conn.execute("""
        SELECT u.platform, u.raw_id, t.platform_thread_id AS chat, a.name, t.title, a.method,
               (SELECT COUNT(*) FROM src.Messages m WHERE m.author_id = pt.user_id AND m.thread_id = pt.thread_id) AS n
        FROM PersonThreads pt JOIN src.Users u ON u.id = pt.user_id JOIN src.Threads t ON t.id = pt.thread_id
        JOIN PersonAliases a ON a.user_id = pt.user_id
        WHERE pt.person_id = ?
        UNION ALL
        SELECT u.platform, u.raw_id, '', a.name, NULL, a.method,
               (SELECT COUNT(*) FROM src.Messages m WHERE m.author_id = a.user_id)
        FROM PersonAliases a JOIN src.Users u ON u.id = a.user_id
        WHERE a.person_id = ? AND NOT EXISTS (SELECT 1 FROM PersonThreads x WHERE x.user_id = a.user_id)""",
                        (pid, pid)).fetchall()
    return [{"key": (r[0], str(r[1]), r[2]), "platform": r[0], "name": r[3], "chat_title": r[4], "method": r[5],
             "messages": r[6]}
            for r in rows]


def refresh_person_lists(conn):
    """After people were relinked: rewrite Chunks/Sessions.person_ids (by message volume, like build_chunks.py)
    where the set of people changed. Text and headers keep the account names until the next full rebuild."""
    conn.executescript("""
        DROP TABLE IF EXISTS temp.mp;
        CREATE TEMP TABLE mp AS
          SELECT m.msg_id, COALESCE(pt.person_id, a.person_id) AS pid
          FROM src.Messages m JOIN PersonAliases a ON a.user_id = m.author_id
          LEFT JOIN PersonThreads pt ON pt.user_id = m.author_id AND pt.thread_id = m.thread_id;
        CREATE INDEX temp.idx_mp ON mp(msg_id);
    """)
    changed = 0
    for table, key, sql in (
            ("Chunks", "chunk_id", """SELECT cm.chunk_id, mp.pid, COUNT(*) n, MIN(cm.position) o
                FROM ChunkMessages cm JOIN mp USING (msg_id) GROUP BY cm.chunk_id, mp.pid"""),
            ("Sessions", "session_id", """SELECT c.session_id, mp.pid, COUNT(DISTINCT cm.msg_id) n,
                MIN(c.start_ts * 100000 + cm.position) o
                FROM ChunkMessages cm JOIN Chunks c USING (chunk_id) JOIN mp USING (msg_id)
                GROUP BY c.session_id, mp.pid""")):
        new = {}
        for k, pid, n, o in conn.execute(sql):
            new.setdefault(k, []).append((-n, o, pid))
        old = {r[0]: json.loads(r[1]) for r in conn.execute(f"SELECT {key}, person_ids FROM {table}")}
        rows = [(json.dumps([p for _, _, p in sorted(v)]), k) for k, v in new.items()
                if k in old and set(old[k]) != {p for _, _, p in v}]
        conn.executemany(f"UPDATE {table} SET person_ids = ? WHERE {key} = ?", rows)
        changed += len(rows)
    conn.execute("DROP TABLE temp.mp")
    conn.commit()
    return changed
