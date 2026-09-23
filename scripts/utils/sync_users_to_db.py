import os
import json
import sqlite3
import logging

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
USERS_JSONL = os.path.join(REPO_ROOT, "processed_data", "context", "twitter", "users.jsonl")
CONTEXT_JSONL = os.path.join(REPO_ROOT, "processed_data", "context", "twitter", "context_cache.jsonl")
DB_PATH = os.path.join(REPO_ROOT, "processed_data", "db", "twitter_users.db")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def sync_users():
    # Ensure DB directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table with compound primary key (id_str, screen_name) to track handle history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS UserHistory (
            id_str TEXT NOT NULL,
            screen_name TEXT NOT NULL,
            name TEXT,
            description TEXT,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id_str, screen_name)
        )
    ''')

    # --- Source 1: users.jsonl (Direct Profiles) ---
    if os.path.exists(USERS_JSONL):
        logging.info(f"Harvesting direct profiles from {USERS_JSONL}...")
        count = 0
        with open(USERS_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    user = json.loads(line)
                    id_str = user.get('id_str')
                    screen_name = user.get('screen_name')
                    if id_str and screen_name:
                        cursor.execute('''
                            INSERT OR REPLACE INTO UserHistory (id_str, screen_name, name, description)
                            VALUES (?, ?, ?, ?)
                        ''', (id_str, screen_name, user.get('name'), user.get('description')))
                        count += 1
                except: continue
        logging.info(f"Imported {count} profiles.")

    # --- Source 2: context_cache.jsonl (Deep Mentions) ---
    if os.path.exists(CONTEXT_JSONL):
        logging.info(f"Deep-scanning mentions in {CONTEXT_JSONL}...")
        mentions_count = 0
        with open(CONTEXT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    data = item.get('data', {})
                    if not data: continue
                    
                    # Check entities for user_mentions
                    entities = data.get('entities', {})
                    mentions = entities.get('user_mentions', [])
                    
                    for m in mentions:
                        mid = m.get('id_str')
                        msn = m.get('screen_name')
                        mname = m.get('name')
                        if mid and msn:
                            cursor.execute('''
                                INSERT OR IGNORE INTO UserHistory (id_str, screen_name, name)
                                VALUES (?, ?, ?)
                            ''', (mid, msn, mname))
                            if cursor.rowcount > 0:
                                mentions_count += 1
                except: continue
        logging.info(f"Harvested {mentions_count} new handle-to-ID mappings from mentions.")

    conn.commit()
    conn.close()
    logging.info(f"Sync complete. Final registry at {DB_PATH}")

if __name__ == "__main__":
    sync_users()
