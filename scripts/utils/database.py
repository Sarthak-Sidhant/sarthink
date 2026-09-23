import sqlite3
import json
import os
import logging
from datetime import datetime

class SarthinkMemoryLayer:
    def __init__(self, db_path=None, jsonl_dir=None):
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

        self.db_path = db_path if db_path else os.path.join(REPO_ROOT, 'processed_data', 'db', 'sarthink_memory.db')
        self.jsonl_dir = jsonl_dir if jsonl_dir else os.path.join(REPO_ROOT, 'processed_data', 'logs')
        self.open_jsonl_files = {}

        # In-memory lookup caches: eliminates N+1 SELECT queries during parsing.
        # Format: (platform, raw_id) -> int db_id
        self._user_cache: dict[tuple, int] = {}
        # Format: (platform, platform_thread_id) -> int db_id
        self._thread_cache: dict[tuple, int] = {}

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.jsonl_dir, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        # WAL mode: writes don't block reads; dramatically faster for parse-heavy workloads.
        self.conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL sync: safe after WAL (only syncs at checkpoints, not every commit).
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.cursor = self.conn.cursor()
        self._init_db()
        self._warm_caches()

    def _init_db(self):
        self.cursor.executescript('''
            CREATE TABLE IF NOT EXISTS Users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT,
                raw_id TEXT,
                display_name TEXT,
                UNIQUE(platform, raw_id)
            );

            CREATE TABLE IF NOT EXISTS Threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT,
                platform_thread_id TEXT,
                title TEXT,
                UNIQUE(platform, platform_thread_id)
            );

            CREATE TABLE IF NOT EXISTS Messages (
                msg_id TEXT PRIMARY KEY,
                thread_id INTEGER,
                author_id INTEGER,
                timestamp_utc INTEGER,
                content TEXT,
                parent_msg_id TEXT,
                FOREIGN KEY(thread_id) REFERENCES Threads(id),
                FOREIGN KEY(author_id) REFERENCES Users(id)
            );

            -- Retrieval indexes
            CREATE INDEX IF NOT EXISTS idx_timestamp ON Messages(timestamp_utc);
            CREATE INDEX IF NOT EXISTS idx_thread    ON Messages(thread_id);
            -- Cosmograph export does GROUP BY author_id,thread_id — needs this index
            CREATE INDEX IF NOT EXISTS idx_author    ON Messages(author_id);
            -- Thread-tree traversal (parent_msg_id self-join)
            CREATE INDEX IF NOT EXISTS idx_parent    ON Messages(parent_msg_id);
        ''')
        self.conn.commit()

    def _warm_caches(self):
        """
        Bulk-load existing DB rows into Python dicts at startup.
        All subsequent get_or_create_* calls hit the dict first, making
        the parse-loop effectively O(1) per message instead of SQL-per-message.
        """
        self.cursor.execute("SELECT platform, raw_id, id FROM Users")
        self._user_cache = {(row[0], row[1]): row[2] for row in self.cursor.fetchall()}
        self.cursor.execute("SELECT platform, platform_thread_id, id FROM Threads")
        self._thread_cache = {(row[0], row[1]): row[2] for row in self.cursor.fetchall()}
        logging.info(f"Cache warmed: {len(self._user_cache)} users, {len(self._thread_cache)} threads.")

    # ─── Entity resolution ────────────────────────────────────────────────────

    def get_or_create_user(self, platform, raw_id, display_name):
        """Returns the integer DB id for the user, creating it if needed.
        Does NOT commit immediately — caller owns the commit cadence."""
        key = (platform, str(raw_id))
        cached = self._user_cache.get(key)
        if cached is not None:
            return cached

        self.cursor.execute(
            "INSERT OR IGNORE INTO Users (platform, raw_id, display_name) VALUES (?, ?, ?)",
            (platform, str(raw_id), display_name)
        )
        if self.cursor.rowcount > 0:
            db_id = self.cursor.lastrowid
        else:
            # Row existed in DB but wasn't in our cache (e.g. first warm after partial run)
            self.cursor.execute(
                "SELECT id FROM Users WHERE platform=? AND raw_id=?", (platform, str(raw_id))
            )
            row = self.cursor.fetchone()
            db_id = row[0] if row else None

        self._user_cache[key] = db_id
        return db_id

    def get_or_create_thread(self, platform, platform_thread_id, title):
        """Returns the integer DB id for the thread, creating it if needed.
        Does NOT commit immediately — caller owns the commit cadence."""
        key = (platform, str(platform_thread_id))
        cached = self._thread_cache.get(key)
        if cached is not None:
            return cached

        self.cursor.execute(
            "INSERT OR IGNORE INTO Threads (platform, platform_thread_id, title) VALUES (?, ?, ?)",
            (platform, str(platform_thread_id), title)
        )
        if self.cursor.rowcount > 0:
            db_id = self.cursor.lastrowid
        else:
            self.cursor.execute(
                "SELECT id FROM Threads WHERE platform=? AND platform_thread_id=?",
                (platform, str(platform_thread_id))
            )
            row = self.cursor.fetchone()
            db_id = row[0] if row else None

        self._thread_cache[key] = db_id
        return db_id

    # ─── Message insertion ────────────────────────────────────────────────────

    def _write_jsonl(self, jsonl_filename, json_data):
        """Lazily open and append to a JSONL flat log."""
        if jsonl_filename not in self.open_jsonl_files:
            jsonl_path = os.path.join(self.jsonl_dir, jsonl_filename)
            self.open_jsonl_files[jsonl_filename] = open(jsonl_path, 'a', encoding='utf-8')
        self.open_jsonl_files[jsonl_filename].write(
            json.dumps(json_data, ensure_ascii=False) + '\n'
        )

    def insert_message(self, msg_id, thread_id, author_id, timestamp_utc, content,
                       parent_msg_id=None, json_data=None, jsonl_filename=None, commit_now=True):
        """Insert into SQLite and optionally write to JSONL.

        Ghost-to-real upgrade: if the row already exists as a ghost placeholder
        ([Context Missing...]) and the incoming data is real content, overwrite it.
        """
        if timestamp_utc == 0:
            timestamp_utc = None

        try:
            self.cursor.execute("""
                INSERT OR IGNORE INTO Messages (msg_id, thread_id, author_id, timestamp_utc, content, parent_msg_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (msg_id, thread_id, author_id, timestamp_utc, content, parent_msg_id))

            if self.cursor.rowcount > 0:
                # Fresh insert — write to JSONL
                if json_data and jsonl_filename:
                    self._write_jsonl(jsonl_filename, json_data)

            elif content and not content.startswith('[Context Missing'):
                # Row was ignored (already existed). If the existing row is a ghost
                # placeholder and we now have real data, upgrade it in place.
                self.cursor.execute("""
                    UPDATE Messages
                    SET thread_id    = ?,
                        author_id    = ?,
                        timestamp_utc = ?,
                        content      = ?,
                        parent_msg_id = COALESCE(?, parent_msg_id)
                    WHERE msg_id = ? AND content LIKE '[Context Missing%'
                """, (thread_id, author_id, timestamp_utc, content, parent_msg_id, msg_id))

                if self.cursor.rowcount > 0 and json_data and jsonl_filename:
                    self._write_jsonl(jsonl_filename, json_data)

            if commit_now:
                self.conn.commit()

        except sqlite3.Error as e:
            logging.error(f"DB Insert Error for {msg_id}: {e}")

    # ─── Maintenance ──────────────────────────────────────────────────────────

    def purge_platform(self, platform):
        """Delete all data for a platform so it can be re-ingested cleanly.
        Caches are refreshed after purge so subsequent get_or_create_* calls
        work correctly on the now-empty tables.
        """
        logging.info(f"Purging all '{platform}' data from DB...")
        self.cursor.execute("DELETE FROM Messages WHERE msg_id LIKE ?", (f"{platform}_%",))
        msg_count = self.cursor.rowcount
        self.cursor.execute("DELETE FROM Users WHERE platform = ?", (platform,))
        user_count = self.cursor.rowcount
        self.cursor.execute("DELETE FROM Threads WHERE platform = ?", (platform,))
        thread_count = self.cursor.rowcount
        self.conn.commit()
        # Rebuild caches to reflect the now-empty tables
        self._warm_caches()
        logging.info(
            f"Purged {msg_count} messages, {user_count} users, "
            f"{thread_count} threads for '{platform}'."
        )

    def commit(self):
        """Manual batched commit trigger."""
        self.conn.commit()

    def close(self):
        self.conn.commit()
        for f in self.open_jsonl_files.values():
            f.close()
        self.conn.close()
