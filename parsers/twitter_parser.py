import os
import json
import glob
import logging
from datetime import datetime, timezone
from database import SarthinkMemoryLayer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

PLATFORM = "twitter"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
JSONL_OUTPUT = "twitter_logs.jsonl"

def parse_twitter_timestamp(ts_str):
    if not ts_str:
        return 0
    try:
        # e.g., "2025-09-17T14:58:25.808Z" or "Mon Nov 25 10:00:00 +0000 2024"
        if ts_str.endswith('Z'):
            dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        else:
            # typical twitter string: "Wed Oct 10 20:19:24 +0000 2018"
            dt = datetime.strptime(ts_str, "%a %b %d %H:%M:%S +0000 %Y").replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
    except Exception:
        return 0

def strip_js_wrapper(filepath):
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
    """Read the archive owner's real username from account.js."""
    search_path = os.path.join(REPO_ROOT, "**", "account.js")
    account_files = glob.glob(search_path, recursive=True)
    for filepath in account_files:
        try:
            data = strip_js_wrapper(filepath)
            if data and isinstance(data, list) and len(data) > 0:
                acct = data[0].get('account', {})
                username = acct.get('username', '').strip()
                display = acct.get('accountDisplayName', '').strip()
                if username:
                    logging.info(f"Archive owner identified: @{username} ({display})")
                    return username, display if display else username
        except Exception as e:
            logging.warning(f"Could not parse account.js at {filepath}: {e}")
    # Hardcoded known-correct fallback
    logging.warning("account.js not found or unreadable. Falling back to known handle.")
    return "sidhant_sarthak", "Sarthak Sidhant"

def load_archive_index():
    """
    Pre-scan ALL tweets*.js files and build a lookup dict:
        { tweet_id_str: tweet_data_dict }
    This is the ground-truth for the archive owner's deleted tweets.
    Context cache is unreliable for deleted tweets; the archive is not.
    """
    archive = {}
    search_path = os.path.join(REPO_ROOT, "**", "tweets*.js")
    tweet_files = glob.glob(search_path, recursive=True)
    for filepath in tweet_files:
        for item in strip_js_wrapper(filepath):
            tweet = item.get('tweet', {})
            raw_id = tweet.get('id_str')
            if raw_id:
                archive[raw_id] = tweet
    logging.info(f"Archive index built: {len(archive)} tweets pre-indexed from tweets*.js")
    return archive

def load_context_dict():
    context_dict = {}
    path = os.path.join(REPO_ROOT, "context", "twitter", "context_cache.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                full_cache = json.load(f)
                for tid, data in full_cache.items():
                    if data.get('status') == 'success' and 'data' in data:
                        context_dict[str(tid)] = data['data']
                    elif data.get('status') == 'error':
                        context_dict[str(tid)] = 'ghost'
            except Exception as e:
                logging.error(f"Error loading context_cache.json: {e}")

    # Fallback to load older .jsonl caches if they exist
    path_l = os.path.join(REPO_ROOT, "context", "twitter", "context_cache.jsonl")
    if os.path.exists(path_l):
        with open(path_l, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                tid = str(item.get('id', ''))
                if tid and tid not in context_dict:
                    if item.get('status') == 'success' and 'data' in item:
                        context_dict[tid] = item['data']
                    elif item.get('status') == 'error':
                        context_dict[tid] = 'ghost'
    return context_dict

def insert_ghost_or_context_node(db, node_id, context_dict, archive_index, owner_username, owner_display, internal_cache, counters):
    if not node_id or node_id in internal_cache:
        return
        
    global_node_id = f"{PLATFORM}_{node_id}"
    internal_cache.add(node_id)
    
    # --- PRIORITY 1: Check archive index (user's own tweets, preserved even if deleted) ---
    archive_tweet = archive_index.get(node_id)
    node_data = context_dict.get(node_id) if not archive_tweet else None
    if archive_tweet:
        # It's the archive owner's own tweet — use their real identity
        raw_ts = archive_tweet.get('created_at', '')
        utc_epoch = parse_twitter_timestamp(raw_ts)
        content = archive_tweet.get('full_text', '')
        thread_raw = archive_tweet.get('conversation_id_str', node_id)
        thread_db_id = db.get_or_create_thread(PLATFORM, thread_raw, f"Tweet Thread {thread_raw}")
        author_db_id = db.get_or_create_user(PLATFORM, owner_username, owner_display)
        parent_raw = archive_tweet.get('in_reply_to_status_id_str')
        parent_global_id = f"{PLATFORM}_{parent_raw}" if parent_raw else None
        if parent_raw:
            insert_ghost_or_context_node(db, parent_raw, context_dict, archive_index, owner_username, owner_display, internal_cache, counters)
        flat_json_entry = {
            "log_id": global_node_id, "platform": PLATFORM,
            "thread_name": f"Tweet Thread {thread_raw}",
            "author": owner_display, "author_id": owner_username,
            "timestamp_utc": raw_ts, "content": content,
            "is_reply_to": parent_global_id
        }
        db.insert_message(global_node_id, thread_db_id, author_db_id, utc_epoch, content, parent_global_id, flat_json_entry, JSONL_OUTPUT, commit_now=False)
        counters['msgs'] += 1
        
    elif node_data and isinstance(node_data, dict):
        author = node_data.get('user', {}).get('screen_name', 'twitter_unknown')
        author_db_id = db.get_or_create_user(PLATFORM, author, author)
        
        thread_id = node_data.get('conversation_id_str', node_id)
        thread_db_id = db.get_or_create_thread(PLATFORM, thread_id, f"Tweet Thread {thread_id}")
        
        # Twitter's V2 Context Cache outputs 'tweet_created_at' instead of 'created_at'
        raw_ts = node_data.get('tweet_created_at') or node_data.get('created_at', '')
        utc_epoch = parse_twitter_timestamp(raw_ts)
        content = node_data.get('full_text') or node_data.get('text', '')
        
        parent_raw = node_data.get('in_reply_to_status_id_str')
        parent_global_id = f"{PLATFORM}_{parent_raw}" if parent_raw else None
        
        if parent_raw:
            insert_ghost_or_context_node(db, parent_raw, context_dict, archive_index, owner_username, owner_display, internal_cache, counters)
        
        flat_json_entry = {
            "log_id": global_node_id,
            "platform": PLATFORM,
            "thread_name": f"Tweet Thread {thread_id}",
            "author": author,
            "author_id": author,
            "timestamp_utc": raw_ts,
            "content": content,
            "is_reply_to": parent_global_id
        }
        
        db.insert_message(global_node_id, thread_db_id, author_db_id, utc_epoch, content, parent_global_id, flat_json_entry, JSONL_OUTPUT, commit_now=False)
        counters['msgs'] += 1
        
    else:
        # Generate Ghost Node for 404/403 or unknown
        global_ghost_id = f"{PLATFORM}_{node_id}"
        author_db_id = db.get_or_create_user(PLATFORM, "[Private/Deleted User]", "twitter_unknown")
        
        # We don't have conversation_id reliably here, so assign it as its own isolated thread
        thread_db_id = db.get_or_create_thread(PLATFORM, node_id, f"Ghost Thread {node_id}")
        
        content = "[Context Missing: Tweet Deleted or Private]"
        flat_json_entry = {
            "log_id": global_ghost_id,
            "platform": PLATFORM,
            "thread_name": f"Ghost Thread {node_id}",
            "author": "[Private/Deleted User]",
            "author_id": "twitter_unknown",
            "timestamp_utc": "",
            "content": content,
            "is_reply_to": None
        }
        
        db.insert_message(global_ghost_id, thread_db_id, author_db_id, 0, content, None, flat_json_entry, JSONL_OUTPUT, commit_now=False)
        counters['msgs'] += 1
        
def process_tweets(db, context_dict, archive_index, internal_cache, counters, owner_username, owner_display):
    # Locate all variations of tweets across context and archive folders
    search_path = os.path.join(REPO_ROOT, "**", "tweets*.js")
    tweet_files = glob.glob(search_path, recursive=True)
    logging.info(f"Found {len(tweet_files)} Tweet datasets.")
    
    for filepath in tweet_files:
        logging.info(f"Parsing tweets from {filepath}...")
        tweets_array = strip_js_wrapper(filepath)
        for item in tweets_array:
            tweet = item.get('tweet', {})
            if not tweet: continue
            
            raw_id = tweet.get('id_str')
            if not raw_id: continue
            
            internal_cache.add(raw_id)
            global_id = f"{PLATFORM}_{raw_id}"
            
            # Identify parent
            parent_raw = tweet.get('in_reply_to_status_id_str')
            parent_global_id = f"{PLATFORM}_{parent_raw}" if parent_raw else None
            
            # Reconstruct quoting structure directly mathematically, treating quote as a reply functionally in the graph
            quote_raw = tweet.get('quoted_status_id_str')
            
            # To preserve linear threading, if it replies to X, that takes precedence, else if it quotes Y, it functionally attaches to Y.
            if quote_raw and not parent_raw:
                parent_raw = quote_raw
                parent_global_id = f"{PLATFORM}_{parent_raw}"
            
            # Fetch parents if they haven't been processed
            if parent_raw:
                insert_ghost_or_context_node(db, parent_raw, context_dict, archive_index, owner_username, owner_display, internal_cache, counters)
                
            # Threads
            thread_raw = tweet.get('conversation_id_str', raw_id)
            thread_title = f"Tweet Thread {thread_raw}"
            thread_db_id = db.get_or_create_thread(PLATFORM, thread_raw, thread_title)
            
            # Author: use the real archive owner identity parsed from account.js
            author_db_id = db.get_or_create_user(PLATFORM, owner_username, owner_display)
            
            utc_epoch = parse_twitter_timestamp(tweet.get('created_at', ''))
            content = tweet.get('full_text', '')
            
            flat_json_entry = {
                "log_id": global_id,
                "platform": PLATFORM,
                "thread_name": thread_title,
                "author": owner_display,
                "author_id": owner_username,
                "timestamp_utc": tweet.get('created_at', ''),
                "content": content,
                "is_reply_to": parent_global_id
            }
            
            db.insert_message(global_id, thread_db_id, author_db_id, utc_epoch, content, parent_global_id, flat_json_entry, JSONL_OUTPUT, commit_now=False)
            counters['msgs'] += 1
            if counters['msgs'] % 1000 == 0:
                db.commit()

def process_dms(db, internal_cache, counters, owner_username, owner_display):
    search_path = os.path.join(REPO_ROOT, "**", "direct-messages*.js")
    dm_files = glob.glob(search_path, recursive=True)
    logging.info(f"Found {len(dm_files)} Direct Message datasets.")
    
    for filepath in dm_files:
        logging.info(f"Parsing DMs from {filepath}...")
        dms_array = strip_js_wrapper(filepath)
        for dm_thread in dms_array:
            convo = dm_thread.get('dmConversation', {})
            convo_id = convo.get('conversationId')
            if not convo_id: continue
            
            thread_db_id = db.get_or_create_thread(PLATFORM, convo_id, f"DM {convo_id}")
            messages = convo.get('messages', [])
            
            # Processing DMs in reverse gives correct chronological parent-child linking
            # Or we can just link to previous message sequentially
            previous_msg_id = None
            
            # sort mathematically by createdAt
            messages_sorted = sorted([m for m in messages if 'messageCreate' in m], 
                                     key=lambda x: parse_twitter_timestamp(x['messageCreate'].get('createdAt', '')))
                                     
            for m in messages_sorted:
                msg = m.get('messageCreate', {})
                raw_id = msg.get('id')
                if not raw_id: continue
                
                global_id = f"{PLATFORM}_{raw_id}"
                sender_raw = msg.get('senderId', 'twitter_unknown')
                author_db_id = db.get_or_create_user(PLATFORM, sender_raw, sender_raw)
                
                utc_epoch = parse_twitter_timestamp(msg.get('createdAt', ''))
                content = msg.get('text', '')
                
                flat_json_entry = {
                    "log_id": global_id,
                    "platform": f"{PLATFORM}_dm",
                    "thread_name": f"DM {convo_id}",
                    "author": sender_raw,
                    "author_id": sender_raw,
                    "timestamp_utc": msg.get('createdAt', ''),
                    "content": content,
                    "is_reply_to": previous_msg_id
                }
                
                db.insert_message(global_id, thread_db_id, author_db_id, utc_epoch, content, previous_msg_id, flat_json_entry, JSONL_OUTPUT, commit_now=False)
                counters['msgs'] += 1
                if counters['msgs'] % 1000 == 0:
                    db.commit()
                    
                previous_msg_id = global_id

def process_twitter():
    db = SarthinkMemoryLayer()
    counters = {'msgs': 0}
    internal_cache = set()
    
    logging.info("Resolving archive owner identity from account.js...")
    owner_username, owner_display = load_account_identity()
    
    logging.info("Loading Twitter Context Cache...")
    context_dict = load_context_dict()
    logging.info(f"Loaded {len(context_dict)} tweets from cache.")
    
    logging.info("Building archive index from tweets*.js (ground-truth for deleted tweets)...")
    archive_index = load_archive_index()
    
    process_tweets(db, context_dict, archive_index, internal_cache, counters, owner_username, owner_display)
    process_dms(db, internal_cache, counters, owner_username, owner_display)
    
    db.commit()
    db.close()
    logging.info(f"Successfully processed Twitter archive. Generated nodes/messages: {counters['msgs']}")

if __name__ == "__main__":
    process_twitter()
