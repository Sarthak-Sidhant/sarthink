import os
import json
import glob
import logging
import sys
import re
from collections import deque
from datetime import datetime, timezone

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Inject utils path for database import
sys.path.append(os.path.join(REPO_ROOT, "scripts", "utils"))
from database import SarthinkMemoryLayer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

PLATFORM = "twitter"
JSONL_OUTPUT = "twitter_logs.jsonl"

# ─── Timestamp parsing ────────────────────────────────────────────────────────

def parse_twitter_timestamp(ts_str):
    if not ts_str:
        return 0
    try:
        # ISO 8601 variant used by the archive export: "2025-09-17T14:58:25.808Z"
        if ts_str.endswith('Z'):
            dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        else:
            # RFC 2822 variant from the API: "Wed Oct 10 20:19:24 +0000 2018"
            dt = datetime.strptime(ts_str, "%a %b %d %H:%M:%S +0000 %Y").replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
    except Exception:
        return 0

# ─── Archive loading helpers ─────────────────────────────────────────────────

def strip_js_wrapper(filepath):
    """Twitter archive files are JavaScript assignments like `window.YTD.tweets.part0 = [...]`.
    We strip the assignment prefix by finding the first '[' and parsing from there.
    This is safe for all Twitter archive files because the array always begins the value."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            idx = content.find('[')
            if idx != -1:
                return json.loads(content[idx:])
    except Exception as e:
        logging.error(f"Failed to read {filepath}: {e}")
    return []

def load_account_identity():
    """Read the archive owner's username, display name, and permanent numeric ID from account.js."""
    search_path = os.path.join(REPO_ROOT, "**", "account.js")
    for filepath in glob.glob(search_path, recursive=True):
        try:
            data = strip_js_wrapper(filepath)
            if data and isinstance(data, list) and len(data) > 0:
                acct = data[0].get('account', {})
                username = acct.get('username', '').strip()
                display  = acct.get('accountDisplayName', '').strip()
                id_str   = acct.get('accountId', '').strip()
                if username:
                    logging.info(f"Archive owner: @{username} ({display}) [ID: {id_str}]")
                    return username, display if display else username, id_str
        except Exception as e:
            logging.warning(f"Could not parse account.js at {filepath}: {e}")
    logging.warning("account.js not found. Falling back to hardcoded identity.")
    return "sidhant_sarthak", "Sarthak Sidhant", "1437031675557335043"

def load_twitter_id_map():
    """
    Build three lookup structures from twitter_users.db and twitter_id_map.json:

      id_map       : {handle_lower -> id_str}   — handle resolution
      name_map     : {id_str -> display_name}   — numeric ID to human name
      id_to_handle : {id_str -> handle}         — reverse of id_map, for DM participants

    Priority: DB rows (richer, include display names) supplement from static JSON.
    """
    id_map: dict[str, str]  = {}
    name_map: dict[str, str] = {}

    # --- Source 1: twitter_users.db (scraped profiles & mention history) ---
    db_path = os.path.join(REPO_ROOT, "processed_data", "db", "twitter_users.db")
    if os.path.exists(db_path):
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT screen_name, id_str, name FROM UserHistory")
            for handle, id_str, name in cursor.fetchall():
                id_map[handle.lower()] = id_str
                if name:
                    name_map[id_str] = name
            conn.close()
        except Exception as e:
            logging.error(f"Error querying twitter_users.db: {e}")

    # --- Source 2: static twitter_id_map.json ---
    map_path = os.path.join(REPO_ROOT, "processed_data", "metadata", "twitter_id_map.json")
    if os.path.exists(map_path):
        try:
            with open(map_path, 'r', encoding='utf-8') as f:
                for handle, id_str in json.load(f).items():
                    id_map[handle.lower()] = id_str
        except Exception as e:
            logging.error(f"Error loading twitter_id_map.json: {e}")

    # Build reverse map: id_str -> handle (for DM sender resolution)
    id_to_handle: dict[str, str] = {v: k for k, v in id_map.items()}

    logging.info(f"Loaded {len(id_map)} handle→ID mappings, {len(name_map)} ID→name mappings.")
    return id_map, name_map, id_to_handle

def load_archive_index():
    """
    Pre-scan ALL tweets*.js files and build a lookup dict:
        { tweet_id_str: tweet_data_dict }
    This is the ground-truth for the archive owner's own tweets (including deleted ones).
    Context cache is unreliable for the owner's deleted tweets; the local archive is not.
    """
    archive: dict[str, dict] = {}
    search_path = os.path.join(REPO_ROOT, "**", "tweets*.js")
    for filepath in glob.glob(search_path, recursive=True):
        for item in strip_js_wrapper(filepath):
            tweet = item.get('tweet', {})
            raw_id = tweet.get('id_str')
            if raw_id:
                archive[raw_id] = tweet
    logging.info(f"Archive index built: {len(archive)} tweets pre-indexed from tweets*.js")
    return archive

def load_twitter_context():
    """Load the context cache (API-fetched tweets for others in reply chains).

    Supports both the canonical .json format and the legacy .jsonl format.
    'ghost' entries indicate we tried to fetch but the tweet was deleted/private.
    """
    context_dict: dict[str, object] = {}

    # Primary: JSON format
    path = os.path.join(REPO_ROOT, "processed_data", "context", "twitter", "context_cache.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                for tid, data in json.load(f).items():
                    if data.get('status') == 'success' and 'data' in data:
                        context_dict[str(tid)] = data['data']
                    elif data.get('status') == 'error':
                        context_dict[str(tid)] = 'ghost'
            except Exception as e:
                logging.error(f"Error loading context_cache.json: {e}")

    # Legacy: JSONL format (supplement only — don't overwrite newer entries)
    path_l = os.path.join(REPO_ROOT, "processed_data", "context", "twitter", "context_cache.jsonl")
    if os.path.exists(path_l):
        with open(path_l, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                tid = str(item.get('id', ''))
                if tid and tid not in context_dict:
                    if item.get('status') == 'success' and 'data' in item:
                        context_dict[tid] = item['data']
                    elif item.get('status') == 'error':
                        context_dict[tid] = 'ghost'

    logging.info(f"Context cache loaded: {len(context_dict)} entries.")
    return context_dict

# ─── Ghost author resolution ──────────────────────────────────────────────────

def resolve_ghost_author(
    node_id: str,
    child_data: dict | None,
    twitter_id_map: dict[str, str],
    twitter_name_map: dict[str, str],
) -> tuple[str, str]:
    """
    Determine the most accurate author label for a deleted/private tweet (ghost node).

    Resolution priority:
    ──────────────────
    1. child_data.in_reply_to_user_id_str
       This is Twitter's own authoritative field: "the user ID of the person
       the child tweet is replying to" — i.e. exactly the ghost node's author.
       Only used when child_data.in_reply_to_status_id_str == node_id,
       confirming child_data is actually replying to this ghost (not a quote
       or a sibling tweet).

    2. child_data.entities.user_mentions[0]
       Fallback when in_reply_to_user_id_str is absent or zero (e.g. quote
       ghost targets). The first @mention in a reply is often the parent
       author but is not guaranteed — hence lower priority.

    3. Regex @handle scan on full_text
       Last resort; can produce false positives when multiple people are
       mentioned, but better than 'twitter_unknown'.

    Returns (ghost_id, ghost_name) strings.
    """
    if not child_data:
        return "twitter_unknown", "[Private/Deleted User]"

    # ── Priority 1: in_reply_to_user_id_str ──────────────────────────────────
    # Confirm the child is actually replying to *this* node (not quoting it).
    if child_data.get('in_reply_to_status_id_str') == node_id:
        uid = child_data.get('in_reply_to_user_id_str', '')
        if uid and uid != '0':
            real_name = twitter_name_map.get(uid)
            if real_name:
                logging.debug(f"Ghost {node_id} author resolved via in_reply_to_user_id_str: {real_name} [{uid}]")
                return uid, real_name
            # We have the numeric ID but no display name — still better than unknown
            return uid, f"[User {uid}]"

    # ── Priority 2: user_mentions[0] from entities ────────────────────────────
    entities = child_data.get('entities', {}) or {}
    mentions = entities.get('user_mentions', [])
    if mentions:
        m = mentions[0]
        ghost_id   = m.get('id_str', '')
        handle     = m.get('screen_name', '')
        real_name  = twitter_name_map.get(ghost_id) if ghost_id else None
        ghost_name = real_name or (f"@{handle}" if handle else "[Private/Deleted User]")
        if ghost_id:
            logging.debug(f"Ghost {node_id} author resolved via user_mentions: {ghost_name} [{ghost_id}]")
            return ghost_id, ghost_name

    # ── Priority 3: regex on full_text ────────────────────────────────────────
    content = child_data.get('full_text') or child_data.get('text', '')
    if content:
        handles = re.findall(r'@(\w+)', content)
        if handles:
            target_handle = handles[0]
            resolved_id = twitter_id_map.get(target_handle.lower())
            if resolved_id:
                real_name = twitter_name_map.get(resolved_id)
                ghost_name = real_name or f"@{target_handle}"
                logging.debug(f"Ghost {node_id} author resolved via regex: {ghost_name}")
                return resolved_id, ghost_name
            synthetic_id = f"twitter_handle_{target_handle.lower()}"
            return synthetic_id, f"@{target_handle}"

    return "twitter_unknown", "[Private/Deleted User]"

# ─── Parent chain insertion (iterative) ──────────────────────────────────────

def insert_parent_chain(
    db, start_node_id, context_dict, archive_index,
    twitter_id_map, twitter_name_map,
    owner_username, owner_display, owner_id,
    internal_cache, counters,
    child_node_data=None
):
    """
    Iteratively walk the parent chain from start_node_id upward, inserting
    any ancestor tweets not yet in internal_cache.

    Why iterative instead of recursive:
      A long reply thread (100+ tweets) would overflow Python's call stack at
      the default recursion limit of 1000. Converting to a deque-based loop
      gives us unlimited depth with constant stack space.

    How the chain terminates:
      - Already in internal_cache  → skip (idempotency)
      - Found in archive index     → insert + enqueue ITS parent (chain continues)
      - Found in context cache     → insert + enqueue ITS parent (chain continues)
      - Ghost (deleted/private)    → insert placeholder + STOP
        We have no data for the ghost's own parent, so we cannot continue
        walking further up the chain. The ghost is the last knowable node.
    """
    to_process: deque[tuple[str, dict | None]] = deque()
    to_process.append((start_node_id, child_node_data))

    while to_process:
        node_id, child_data = to_process.popleft()

        if not node_id or node_id in internal_cache:
            continue

        global_node_id = f"{PLATFORM}_{node_id}"
        internal_cache.add(node_id)

        # ── Tier 1: Archive index (owner's own tweets, highest fidelity) ──────
        archive_tweet = archive_index.get(node_id)
        if archive_tweet:
            raw_ts      = archive_tweet.get('created_at', '')
            utc_epoch   = parse_twitter_timestamp(raw_ts)
            content     = archive_tweet.get('full_text', '')
            thread_raw  = archive_tweet.get('conversation_id_str', node_id)
            thread_db_id = db.get_or_create_thread(PLATFORM, thread_raw, f"Tweet Thread {thread_raw}")
            author_db_id = db.get_or_create_user(PLATFORM, owner_id, owner_display)

            parent_raw       = archive_tweet.get('in_reply_to_status_id_str')
            parent_global_id = f"{PLATFORM}_{parent_raw}" if parent_raw else None

            # Enqueue this tweet's parent so we continue walking upward
            if parent_raw:
                to_process.append((parent_raw, archive_tweet))

            flat_json_entry = {
                "log_id":        global_node_id,
                "platform":      PLATFORM,
                "thread_name":   f"Tweet Thread {thread_raw}",
                "author":        owner_display,
                "author_id":     owner_id,       # always numeric ID for consistency
                "timestamp_utc": raw_ts,
                "content":       content,
                "is_reply_to":   parent_global_id,
            }
            db.insert_message(global_node_id, thread_db_id, author_db_id, utc_epoch,
                               content, parent_global_id, flat_json_entry, JSONL_OUTPUT,
                               commit_now=False)
            counters['msgs'] += 1
            continue

        # ── Tier 2: Context cache (API-fetched tweets from other users) ────────
        node_data = context_dict.get(node_id)
        if node_data and node_data != 'ghost':
            user_obj       = node_data.get('user', {})
            author_handle  = user_obj.get('screen_name', 'unknown')
            author_display = user_obj.get('name', author_handle)
            author_id      = user_obj.get('id_str', author_handle)
            author_db_id   = db.get_or_create_user(PLATFORM, author_id, author_display)

            thread_raw   = node_data.get('conversation_id_str', node_id)
            thread_db_id = db.get_or_create_thread(PLATFORM, thread_raw, f"Tweet Thread {thread_raw}")

            raw_ts    = node_data.get('tweet_created_at') or node_data.get('created_at', '')
            utc_epoch = parse_twitter_timestamp(raw_ts)
            content   = node_data.get('full_text') or node_data.get('text', '')

            parent_raw       = node_data.get('in_reply_to_status_id_str')
            parent_global_id = f"{PLATFORM}_{parent_raw}" if parent_raw else None

            # Continue up the chain using this node's own data as child context
            if parent_raw:
                to_process.append((parent_raw, node_data))

            flat_json_entry = {
                "log_id":        global_node_id,
                "platform":      PLATFORM,
                "thread_name":   f"Tweet Thread {thread_raw}",
                "author":        author_handle,
                "author_id":     author_id,
                "timestamp_utc": raw_ts,
                "content":       content,
                "is_reply_to":   parent_global_id,
            }
            db.insert_message(global_node_id, thread_db_id, author_db_id, utc_epoch,
                               content, parent_global_id, flat_json_entry, JSONL_OUTPUT,
                               commit_now=False)
            counters['msgs'] += 1
            continue

        # ── Tier 3: Ghost node ────────────────────────────────────────────────
        # The tweet is deleted/private and was not returned by the API.
        # We use child_data to infer who wrote it (reply metadata is the most
        # reliable signal; see resolve_ghost_author for full priority chain).
        # We cannot walk further up — we have no data about this node's parents.
        ghost_id, ghost_name = resolve_ghost_author(
            node_id, child_data, twitter_id_map, twitter_name_map
        )

        author_db_id = db.get_or_create_user(PLATFORM, ghost_id, ghost_name)
        thread_db_id = db.get_or_create_thread(PLATFORM, node_id, f"Ghost Thread {node_id}")
        content      = "[Context Missing: Tweet Deleted or Private]"

        flat_json_entry = {
            "log_id":        global_node_id,
            "platform":      PLATFORM,
            "thread_name":   f"Ghost Thread {node_id}",
            "author":        ghost_name,
            "author_id":     ghost_id,
            "timestamp_utc": "",
            "content":       content,
            "is_reply_to":   None,
        }
        db.insert_message(global_node_id, thread_db_id, author_db_id, 0,
                           content, None, flat_json_entry, JSONL_OUTPUT,
                           commit_now=False)
        counters['msgs'] += 1
        # Chain terminates here — we have no data to go further.

# ─── Tweet ingestion ──────────────────────────────────────────────────────────

def process_tweets(
    db, context_dict, archive_index,
    twitter_id_map, twitter_name_map,
    internal_cache, counters,
    owner_username, owner_display, owner_id
):
    search_path = os.path.join(REPO_ROOT, "**", "tweets*.js")
    tweet_files = glob.glob(search_path, recursive=True)
    logging.info(f"Found {len(tweet_files)} Tweet dataset(s).")

    for filepath in tweet_files:
        logging.info(f"Parsing tweets from {filepath}...")
        for item in strip_js_wrapper(filepath):
            tweet = item.get('tweet', {})
            if not tweet:
                continue

            raw_id = tweet.get('id_str')
            if not raw_id:
                continue

            internal_cache.add(raw_id)
            global_id = f"{PLATFORM}_{raw_id}"

            # Determine the functional parent for graph edge construction.
            # Real reply takes priority; quote falls back to quoted tweet.
            # Note: quote targets are NOT in context cache (we never fetched them),
            # so they resolve to ghost nodes — which is acceptable.
            parent_raw = tweet.get('in_reply_to_status_id_str')
            if not parent_raw:
                parent_raw = tweet.get('quoted_status_id_str')  # functionally attach quote as edge

            parent_global_id = f"{PLATFORM}_{parent_raw}" if parent_raw else None

            content = tweet.get('full_text', '')

            # Walk and insert all ancestor nodes before inserting this tweet
            if parent_raw:
                insert_parent_chain(
                    db, parent_raw, context_dict, archive_index,
                    twitter_id_map, twitter_name_map,
                    owner_username, owner_display, owner_id,
                    internal_cache, counters,
                    child_node_data=tweet
                )

            # Thread grouping
            thread_raw   = tweet.get('conversation_id_str', raw_id)
            thread_title = f"Tweet Thread {thread_raw}"
            thread_db_id = db.get_or_create_thread(PLATFORM, thread_raw, thread_title)

            # Author: always the archive owner for tweets in tweets*.js
            author_db_id = db.get_or_create_user(PLATFORM, owner_id, owner_display)
            utc_epoch    = parse_twitter_timestamp(tweet.get('created_at', ''))

            flat_json_entry = {
                "log_id":        global_id,
                "platform":      PLATFORM,
                "thread_name":   thread_title,
                "author":        owner_display,
                "author_id":     owner_id,   # numeric ID — consistent with all other entries
                "timestamp_utc": tweet.get('created_at', ''),
                "content":       content,
                "is_reply_to":   parent_global_id,
            }

            db.insert_message(global_id, thread_db_id, author_db_id, utc_epoch,
                               content, parent_global_id, flat_json_entry, JSONL_OUTPUT,
                               commit_now=False)
            counters['msgs'] += 1
            if counters['msgs'] % 1000 == 0:
                db.commit()
                logging.info(f"  {counters['msgs']} messages processed...")

# ─── DM ingestion ─────────────────────────────────────────────────────────────

def process_dms(
    db, internal_cache, counters,
    owner_username, owner_display, owner_id,
    twitter_name_map: dict[str, str],
    id_to_handle: dict[str, str]
):
    search_path = os.path.join(REPO_ROOT, "**", "direct-messages*.js")
    dm_files = glob.glob(search_path, recursive=True)
    logging.info(f"Found {len(dm_files)} Direct Message dataset(s).")

    for filepath in dm_files:
        logging.info(f"Parsing DMs from {filepath}...")
        for dm_thread in strip_js_wrapper(filepath):
            convo    = dm_thread.get('dmConversation', {})
            convo_id = convo.get('conversationId')
            if not convo_id:
                continue

            thread_db_id = db.get_or_create_thread(PLATFORM, convo_id, f"DM {convo_id}")
            messages = convo.get('messages', [])

            # Sort chronologically so previous_msg_id is always the actual predecessor
            messages_sorted = sorted(
                [m for m in messages if 'messageCreate' in m],
                key=lambda x: parse_twitter_timestamp(x['messageCreate'].get('createdAt', ''))
            )

            previous_msg_id = None

            for m in messages_sorted:
                msg    = m.get('messageCreate', {})
                raw_id = msg.get('id')
                if not raw_id:
                    continue

                global_id  = f"{PLATFORM}_{raw_id}"
                sender_raw = msg.get('senderId', 'twitter_unknown')

                if sender_raw == owner_id:
                    # Sender is the archive owner — identity is fully known
                    author_id      = owner_id
                    author_display = owner_display
                    author_handle  = owner_username
                else:
                    # Sender is a DM participant — resolve from our identity maps.
                    # twitter_name_map: id_str -> display_name (from twitter_users.db)
                    # id_to_handle:     id_str -> screen_name  (reverse of id_map)
                    author_id      = sender_raw
                    author_display = twitter_name_map.get(sender_raw) or f"[User {sender_raw}]"
                    author_handle  = id_to_handle.get(sender_raw, sender_raw)

                author_db_id = db.get_or_create_user(PLATFORM, author_id, author_display)
                utc_epoch    = parse_twitter_timestamp(msg.get('createdAt', ''))
                content      = msg.get('text', '')

                flat_json_entry = {
                    "log_id":        global_id,
                    "platform":      f"{PLATFORM}_dm",
                    "thread_name":   f"DM {convo_id}",
                    "author":        author_handle,
                    "author_id":     author_id,
                    "timestamp_utc": msg.get('createdAt', ''),
                    "content":       content,
                    "is_reply_to":   previous_msg_id,
                }

                db.insert_message(global_id, thread_db_id, author_db_id, utc_epoch,
                                   content, previous_msg_id, flat_json_entry, JSONL_OUTPUT,
                                   commit_now=False)
                counters['msgs'] += 1
                if counters['msgs'] % 1000 == 0:
                    db.commit()
                    logging.info(f"  {counters['msgs']} messages processed...")

                previous_msg_id = global_id

# ─── Entry point ──────────────────────────────────────────────────────────────

def process_twitter():
    # --- Fresh reparse: clear old twitter data so we start clean ---
    # Delete the flat JSONL log — the DB gets the platform purge below.
    jsonl_path = os.path.join(REPO_ROOT, 'processed_data', 'logs', JSONL_OUTPUT)
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)
        logging.info(f"Cleared old {JSONL_OUTPUT} for fresh reparse.")

    db           = SarthinkMemoryLayer()
    # Purge all twitter rows from the DB so ghost nodes with wrong labels
    # don't survive from a previous run.
    db.purge_platform(PLATFORM)

    internal_cache: set[str] = set()
    counters = {'msgs': 0}

    twitter_id_map, twitter_name_map, id_to_handle = load_twitter_id_map()
    context_dict  = load_twitter_context()
    archive_index = load_archive_index()

    owner_username, owner_display, owner_id = load_account_identity()

    process_tweets(
        db, context_dict, archive_index,
        twitter_id_map, twitter_name_map,
        internal_cache, counters,
        owner_username, owner_display, owner_id
    )

    process_dms(
        db, internal_cache, counters,
        owner_username, owner_display, owner_id,
        twitter_name_map, id_to_handle
    )

    db.commit()
    db.close()
    logging.info(
        f"Twitter parse complete. "
        f"Ingested {counters['msgs']} messages total."
    )

if __name__ == "__main__":
    process_twitter()
