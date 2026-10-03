"""People: the humans the user groups accounts into (see docs/accounts-and-people.md).

Accounts are derived from the exports (build_identity.py); people are user-written and live in the user DB
next to the notes, so a rebuild never touches them. A person groups accounts; an account belongs to at most one
person, or to none. Attaching, detaching, creating, renaming and deleting are single-row writes: nothing else
is rebuilt. Person 1 is the owner ("me") and always holds the owner's accounts.

Account keys are (platform, raw_id, chat): chat is '' for a whole account and the platform thread id for one
chat of a deleted/deactivated account (one placeholder id covers many humans). AccountIds gives every key its
stable account_id.
"""
import datetime as dt
import json
import sqlite3

from common import USER_DB

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
ME = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS AccountIds (
    platform TEXT, raw_id TEXT, chat TEXT NOT NULL DEFAULT '', account_id INTEGER NOT NULL,
    PRIMARY KEY (platform, raw_id, chat));
CREATE UNIQUE INDEX IF NOT EXISTS idx_account_ids ON AccountIds(account_id);
CREATE TABLE IF NOT EXISTS People (
    person_id INTEGER PRIMARY KEY, name TEXT NOT NULL, created TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS PersonAccounts (            -- an account belongs to at most one person
    platform TEXT, raw_id TEXT, chat TEXT NOT NULL DEFAULT '', person_id INTEGER NOT NULL,
    PRIMARY KEY (platform, raw_id, chat));
CREATE INDEX IF NOT EXISTS idx_person_accounts ON PersonAccounts(person_id);
CREATE TABLE IF NOT EXISTS PersonRejections (a TEXT, b TEXT, PRIMARY KEY (a, b));  -- "not the same human"
CREATE TABLE IF NOT EXISTS PeopleImports (source TEXT PRIMARY KEY, at TEXT);
"""


def key_str(k):
    return "|".join(map(str, k))


def now():
    return dt.datetime.now(IST).isoformat(timespec="minutes")


class PeopleStore:
    def __init__(self, path=USER_DB, me_name="Me", name_of=None):
        """name_of(account_id) -> name: used once, to name people migrated from unnamed contacts."""
        self.name_of = name_of or (lambda _a: None)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.execute("INSERT OR IGNORE INTO People VALUES (?,?,?,?)", (ME, me_name, now(), now()))  # before
        self._migrate()                                                                   # any other person
        self.db.commit()

    # ── migration from the earlier model (accounts merged into "persons"/contacts) ──────────────────────────
    def _migrate(self):
        tables = {r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "PersonIds" in tables and not self.db.execute("SELECT 1 FROM AccountIds LIMIT 1").fetchone():
            # A deleted account's first chat shared its id with the whole-account key. Accounts are per chat now,
            # so chat keys go first and keep the id; the whole-account duplicate is dropped.
            self.db.execute("""INSERT OR IGNORE INTO AccountIds
                SELECT platform, raw_id, chat, person_id FROM PersonIds ORDER BY chat = '', person_id""")
            self.db.execute("DROP TABLE PersonIds")
        if "Contacts" in tables:
            # a contact (keyed by its primary account's id) becomes a person holding the same accounts
            names = dict(self.db.execute("SELECT person_id, name FROM Contacts"))
            groups = {}
            for p, r, c, cid in self.db.execute("SELECT platform, raw_id, chat, person_id FROM ContactAccounts"):
                groups.setdefault(cid, []).append((p, r, c))
            for cid, keys in groups.items():
                name = names.get(cid) or self.name_of(cid) or "Unnamed person"
                pid = self.create(name, commit=False)
                for k in keys:
                    self.db.execute("INSERT OR REPLACE INTO PersonAccounts VALUES (?,?,?,?)", (*k, pid))
            if "ContactRejections" in tables:
                self.db.execute("INSERT OR IGNORE INTO PersonRejections SELECT a, b FROM ContactRejections")
            if "ContactImports" in tables:
                self.db.execute("INSERT OR IGNORE INTO PeopleImports SELECT source, at FROM ContactImports")
            for t in ("Contacts", "ContactAccounts", "ContactRejections", "ContactImports"):
                self.db.execute(f"DROP TABLE IF EXISTS {t}")
        if "Notes" in tables:
            cols = {r[1] for r in self.db.execute("PRAGMA table_info(Notes)")}
            if "account_id" not in cols:
                # notes so far were written against the old person ids, which are today's account ids
                self.db.execute("ALTER TABLE Notes ADD COLUMN account_id INTEGER")
                self.db.execute("UPDATE Notes SET account_id = person_id, person_id = NULL WHERE person_id IS NOT NULL")
        self.db.commit()

    # ── account ids ─────────────────────────────────────────────────────────────────────────────────────────
    def account_ids(self):
        return {(p, r, c): a for p, r, c, a in self.db.execute("SELECT * FROM AccountIds")}

    def key_of(self, account_id):
        r = self.db.execute("SELECT platform, raw_id, chat FROM AccountIds WHERE account_id = ?", (account_id,)).fetchone()
        return tuple(r) if r else None

    # ── people ──────────────────────────────────────────────────────────────────────────────────────────────
    def people(self):
        return {pid: name for pid, name in self.db.execute("SELECT person_id, name FROM People")}

    def membership(self):
        """{account key: person_id} for every attached account."""
        return {(p, r, c): pid for p, r, c, pid in self.db.execute("SELECT * FROM PersonAccounts")}

    def members(self, pid):
        return [tuple(r) for r in self.db.execute(
            "SELECT platform, raw_id, chat FROM PersonAccounts WHERE person_id = ?", (pid,))]

    def person_of(self, key):
        r = self.db.execute("SELECT person_id FROM PersonAccounts WHERE platform=? AND raw_id=? AND chat=?",
                            tuple(key)).fetchone()
        return r[0] if r else None

    def create(self, name, keys=(), commit=True):
        cur = self.db.execute("INSERT INTO People(name, created, updated) VALUES (?,?,?)",
                              ((name or "").strip() or "Unnamed person", now(), now()))
        pid = cur.lastrowid
        for k in keys:
            self.attach(pid, k, commit=False)
        if commit:
            self.db.commit()
        return pid

    def attach(self, pid, key, commit=True):
        """Put an account into a person (moving it out of any other person)."""
        self.db.execute("INSERT OR REPLACE INTO PersonAccounts VALUES (?,?,?,?)", (*key, pid))
        self.db.execute("UPDATE People SET updated = ? WHERE person_id = ?", (now(), pid))
        if commit:
            self.db.commit()

    def detach(self, key):
        self.db.execute("DELETE FROM PersonAccounts WHERE platform=? AND raw_id=? AND chat=?", tuple(key))
        self.db.commit()

    def rename(self, pid, name):
        self.db.execute("UPDATE People SET name = ?, updated = ? WHERE person_id = ?", (name.strip(), now(), pid))
        self.db.commit()

    def delete(self, pid):
        """Remove a person; their accounts stay, just without a person. The owner can't be deleted."""
        if pid == ME:
            raise ValueError("the owner can't be deleted")
        self.db.execute("DELETE FROM PersonAccounts WHERE person_id = ?", (pid,))
        self.db.execute("DELETE FROM People WHERE person_id = ?", (pid,))
        self.db.execute("UPDATE Notes SET person_id = NULL WHERE person_id = ?", (pid,)) if self._has("Notes") else None
        self.db.commit()

    def _has(self, table):
        return bool(self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone())

    # ── suggestions turned down ─────────────────────────────────────────────────────────────────────────────
    def reject(self, keys_a, keys_b):
        for a in keys_a:
            for b in keys_b:
                self.db.execute("INSERT OR IGNORE INTO PersonRejections VALUES (?,?)", tuple(sorted((key_str(a), key_str(b)))))
        self.db.commit()

    def rejected(self, keys_a, keys_b):
        return any(self.db.execute("SELECT 1 FROM PersonRejections WHERE a=? AND b=?",
                                   tuple(sorted((key_str(a), key_str(b))))).fetchone()
                   for a in keys_a for b in keys_b)

    # ── one-time import of a hand-written person_merges.json ───────────────────────────────────────────────
    def import_merges(self, path):
        """[{"name": "Kabir", "accounts": [["instagram", "kabir.singh"], ["discord", "kabirrr"]]}]: each entry becomes a
        person holding those accounts. Imported once, so later changes made in the app are not undone."""
        if not path.exists() or self.db.execute("SELECT 1 FROM PeopleImports WHERE source = ?", (path.name,)).fetchone():
            return 0
        n = 0
        for grp in json.loads(path.read_text()):
            keys = [(p, str(r), "") for p, r in grp.get("accounts", [])]
            if keys:
                self.create(grp.get("name") or keys[0][1], keys, commit=False)
                n += 1
        self.db.execute("INSERT INTO PeopleImports VALUES (?,?)", (path.name, now()))
        self.db.commit()
        return n
