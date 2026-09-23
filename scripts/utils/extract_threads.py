import sqlite3
import json
import os
import logging
from pathlib import Path

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def extract_threads():
    DB_PATH = REPO_ROOT / 'processed_data' / 'db' / 'sarthink_memory.db'
    OUTPUT_PATH = REPO_ROOT / 'processed_data' / 'analysis' / 'self_authored_threads.json'

    if not DB_PATH.exists():
        logging.error(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Identify the owner's author_id from identity map or database
    identity_map_path = REPO_ROOT / 'config' / 'identity_map.json'
    twitter_handles = []
    if identity_map_path.exists():
        try:
            with open(identity_map_path, 'r', encoding='utf-8') as f:
                id_data = json.load(f)
                twitter_handles = [h.lower() for h in id_data.get('aliases', {}).get('twitter', []) if h]
        except Exception as e:
            logging.warning(f"Could not load identity_map.json: {e}")

    owner_id = None
    if twitter_handles:
        placeholders = ','.join(['?'] * len(twitter_handles))
        cursor.execute(f"SELECT id FROM Users WHERE platform='twitter' AND LOWER(raw_id) IN ({placeholders})", tuple(twitter_handles))
        row = cursor.fetchone()
        if row:
            owner_id = row['id']

    if not owner_id:
        cursor.execute("SELECT id FROM Users WHERE platform='twitter' LIMIT 1")
        row = cursor.fetchone()
        if not row:
            logging.error("Could not find Twitter user in Users table.")
            return
        owner_id = row['id']
    logging.info(f"Owner author_id identified as: {owner_id}")

    # 2. Fetch all Twitter messages
    query = """
        SELECT m.msg_id, m.parent_msg_id, m.author_id, m.content, m.timestamp_utc, u.raw_id as author_handle
        FROM Messages m
        JOIN Users u ON m.author_id = u.id
        JOIN Threads t ON m.thread_id = t.id
        WHERE t.platform = 'twitter'
    """
    cursor.execute(query)
    messages = [dict(r) for r in cursor.fetchall()]
    logging.info(f"Fetched {len(messages)} Twitter messages.")

    # 3. Build lookup maps
    msg_map = {m['msg_id']: m for m in messages}
    children_map = {}
    for m in messages:
        parent = m['parent_msg_id']
        if parent:
            if parent not in children_map:
                children_map[parent] = []
            children_map[parent].append(m['msg_id'])

    # 4. Identify chains of 3+ owner tweets
    threads_found = []
    
    for m in messages:
        if m['author_id'] == owner_id:
            parent_id = m['parent_msg_id']
            is_head = False
            if not parent_id:
                is_head = True
            else:
                parent_msg = msg_map.get(parent_id)
                if not parent_msg or parent_msg['author_id'] != owner_id:
                    is_head = True
            
            if is_head:
                chain = [m]
                current = m
                while True:
                    children = children_map.get(current['msg_id'], [])
                    owner_children = [msg_map[cid] for cid in children if msg_map[cid]['author_id'] == owner_id]
                    if not owner_children:
                        break
                    current = owner_children[0]
                    chain.append(current)

                if len(chain) >= 3:
                    thread_entry = {
                        "chain_length": len(chain),
                        "tweets": []
                    }
                    parent_of_head_id = m['parent_msg_id']
                    if parent_of_head_id:
                        parent_msg = msg_map.get(parent_of_head_id)
                        if parent_msg:
                            thread_entry["tweets"].append(parent_msg)
                    
                    thread_entry["tweets"].extend(chain)
                    threads_found.append(thread_entry)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(threads_found, f, indent=2, ensure_ascii=False)

    logging.info(f"Successfully extracted {len(threads_found)} threads. Saved to {OUTPUT_PATH}")
    conn.close()

if __name__ == "__main__":
    extract_threads()
