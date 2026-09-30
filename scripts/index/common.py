"""Shared paths and connections for the derived index (processed_data/db/sarthink_index.db).

The index DB holds everything derived from the source DB (identity, chunks). The source DB
(sarthink_memory.db) is attached read-only as `src`, so rebuilding the index never touches it.
"""
import os
import sqlite3
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

SRC_DB = REPO_ROOT / "processed_data" / "db" / "sarthink_memory.db"
INDEX_DB = REPO_ROOT / "processed_data" / "db" / "sarthink_index.db"
TWITTER_USERS_DB = REPO_ROOT / "processed_data" / "db" / "twitter_users.db"
TWITTER_ID_MAP = REPO_ROOT / "processed_data" / "metadata" / "twitter_id_map.json"
IDENTITY_MAP = REPO_ROOT / "config" / "identity_map.json"


def connect_index():
    conn = sqlite3.connect(f"file:{INDEX_DB}", uri=True, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    if SRC_DB.exists():  # absent on remote GPU boxes, which only get the derived index
        conn.execute(f"ATTACH DATABASE 'file:{SRC_DB}?mode=ro' AS src")
    return conn


# Twitter Snowflake IDs encode creation time: (id >> 22) + epoch offset, in ms.
TWITTER_EPOCH_MS = 1288834974657


def snowflake_to_unix(tweet_id):
    try:
        n = int(tweet_id)
    except (TypeError, ValueError):
        return None
    if n < 2 ** 22 * 1000:  # pre-Snowflake IDs (before Nov 2010) carry no timestamp
        return None
    return ((n >> 22) + TWITTER_EPOCH_MS) // 1000
