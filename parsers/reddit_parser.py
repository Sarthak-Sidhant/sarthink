import os
import glob
import logging
import csv
import ijson
import gc
from datetime import datetime, timezone
from database import SarthinkMemoryLayer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

PLATFORM = "reddit"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

REDDIT_POSTS_DIR = os.path.join(REPO_ROOT, "context", "reddit", "Reddit_Context_Archive", "posts", "*.json")
REDDIT_COMMENTS_DIR = os.path.join(REPO_ROOT, "context", "reddit", "Reddit_Context_Archive", "comments", "*.json")
REDDIT_CHAT_CSV = os.path.join(REPO_ROOT, "archive", "reddit-export", "chat_history.csv")
JSONL_OUTPUT = "reddit_logs.jsonl"

def parse_reddit_timestamp(ts_str):
    if not ts_str:
        return 0
    try:
        dt = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return 0

def insert_comment_recursive(db, comment, thread_db_id, parent_global_id, thread_title, counters):
    raw_msg_id = comment.get('comment_id', 'unknown')
    if raw_msg_id == 'unknown':
        raw_msg_id = comment.get('id', 'unknown') 
    
    global_msg_id = f"{PLATFORM}_{raw_msg_id}"
    author_display = comment.get('author', 'unknown')
    
    if not author_display or author_display in ['[deleted]', '[removed]', 'unknown']:
        author_display = f"deleted_user_thread_{thread_db_id}"
        
    author_db_id = db.get_or_create_user(PLATFORM, author_display, author_display)
    utc_epoch = parse_reddit_timestamp(comment.get('created_utc', ''))
    content = comment.get('body', '')
    
    flat_json_entry = {
        "log_id": global_msg_id,
        "platform": PLATFORM,
        "thread_name": thread_title,
        "author": author_display,
        "author_id": author_display,
        "timestamp_utc": comment.get('created_utc', ''),
        "content": content,
        "is_reply_to": parent_global_id
    }
    
    if content or global_msg_id != f"{PLATFORM}_unknown":
        db.insert_message(global_msg_id, thread_db_id, author_db_id, utc_epoch, content, parent_global_id, flat_json_entry, JSONL_OUTPUT, commit_now=False)
        counters['msgs'] += 1
        if counters['msgs'] % 1000 == 0:
            db.commit()
    
    for child in comment.get('replies', []):
        if isinstance(child, dict):
            insert_comment_recursive(db, child, thread_db_id, global_msg_id, thread_title, counters)

def process_chats(db):
    if not os.path.exists(REDDIT_CHAT_CSV):
        logging.warning(f"Reddit chat file {REDDIT_CHAT_CSV} not found. Skipping chats.")
        return
        
    logging.info(f"Processing Reddit Chats from {REDDIT_CHAT_CSV}")
    counters = {'msgs': 0}
    with open(REDDIT_CHAT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_msg_id = row.get('message_id', 'unknown')
            global_msg_id = f"{PLATFORM}_{raw_msg_id}"
            
            room_url = row.get('channel_url', 'unknown_chat')
            room_id = room_url.split('/')[-1] if '/' in room_url else room_url
            channel_name = row.get('channel_name', '')
            thread_title = channel_name if channel_name else f"DM: {room_id}"
            thread_db_id = db.get_or_create_thread(PLATFORM, room_id, thread_title)
            
            author_display = row.get('username', 'unknown').replace('/u/', '')
            if not author_display or author_display in ['[deleted]', '[removed]', 'unknown']:
                author_display = f"deleted_user_room_{room_id}"
                
            author_db_id = db.get_or_create_user(PLATFORM, author_display, author_display)
            
            created_at_str = row.get('created_at', '').replace(' UTC', '')
            try:
                dt = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                utc_epoch = int(dt.timestamp())
            except Exception:
                utc_epoch = 0
                
            content = row.get('message', '')
            parent_raw = row.get('thread_parent_message_id', '')
            parent_global_id = f"{PLATFORM}_{parent_raw}" if parent_raw else None
            
            flat_json_entry = {
                "log_id": global_msg_id,
                "platform": PLATFORM,
                "thread_name": thread_title,
                "author": author_display,
                "author_id": author_display,
                "timestamp_utc": created_at_str,
                "content": content,
                "is_reply_to": parent_global_id
            }
            if content:
                db.insert_message(global_msg_id, thread_db_id, author_db_id, utc_epoch, content, parent_global_id, flat_json_entry, JSONL_OUTPUT, commit_now=False)
                counters['msgs'] += 1
                if counters['msgs'] % 1000 == 0:
                    db.commit()
    db.commit()

def process_reddit():
    db = SarthinkMemoryLayer()
    
    # 1. Process Chats
    process_chats(db)
    
    counters = {'msgs': 0}
    # 2. Process Posts
    post_files = glob.glob(REDDIT_POSTS_DIR)
    logging.info(f"Found {len(post_files)} Reddit Post Context files.")
    
    for filepath in post_files:
        logging.info(f"Streaming {os.path.basename(filepath)}...")
        try:
            with open(filepath, 'rb') as f:
                # Use ijson to extract my_post_details without loading the whole file
                post_iter = ijson.items(f, 'my_post_details')
                post = next(post_iter, {})
                
                f.seek(0)
                reply_iter = ijson.items(f, 'conversation_thread.item')
                
                if post:
                    post_raw_id = post.get('id', 'unknown')
                    global_post_id = f"{PLATFORM}_{post_raw_id}"
                    thread_title = post.get('title', f"Post {post_raw_id}")
                    subreddit = post.get('subreddit', 'unknown')
                    
                    thread_db_id = db.get_or_create_thread(PLATFORM, post_raw_id, thread_title)
                    
                    author_display = post.get('author', 'unknown')
                    if not author_display or author_display in ['[deleted]', '[removed]', 'unknown']:
                        author_display = f"deleted_user_thread_{thread_db_id}"
                        
                    author_db_id = db.get_or_create_user(PLATFORM, author_display, author_display)
                    utc_epoch = parse_reddit_timestamp(post.get('created_utc', ''))
                    content = f"[{subreddit}] {post.get('selftext', '')}"
                    
                    flat_json_entry = {
                        "log_id": global_post_id,
                        "platform": PLATFORM,
                        "thread_name": thread_title,
                        "author": author_display,
                        "author_id": author_display,
                        "timestamp_utc": post.get('created_utc', ''),
                        "content": content,
                        "is_reply_to": None
                    }
                    
                    db.insert_message(global_post_id, thread_db_id, author_db_id, utc_epoch, content, None, flat_json_entry, JSONL_OUTPUT, commit_now=False)
                    
                    for reply in reply_iter:
                        if isinstance(reply, dict):
                            insert_comment_recursive(db, reply, thread_db_id, global_post_id, thread_title, counters)
                            # Extremely important: collect garbage immediately after a heavy branch drops out of scope
                            del reply
                            
                gc.collect()
                db.commit()
        except Exception as e:
            logging.error(f"Error parsing post {filepath}: {e}")
            continue
            
    # 3. Process Comments
    comment_files = glob.glob(REDDIT_COMMENTS_DIR)
    logging.info(f"Found {len(comment_files)} Reddit Comment Context files.")
    
    for filepath in comment_files:
        try:
            with open(filepath, 'rb') as f:
                # We can load whole file if it's small, or use ijson. Comments are generally small (20KB), so either is fine. We will stream for extreme safety.
                f.seek(0)
                context_iter = ijson.items(f, 'context')
                context = next(context_iter, {})
                
                f.seek(0)
                my_comment_iter = ijson.items(f, 'my_comment_details')
                my_comment = next(my_comment_iter, {})
                
                f.seek(0)
                replies_iter = ijson.items(f, 'replies_to_my_comment')
                replies_to_my_comment = next(replies_iter, [])
                
                parent_post = context.get('parent_post')
                thread_title = "Unknown Thread"
                thread_platform_id = "unknown"
                if parent_post:
                    thread_title = parent_post.get('title', "Post context")
                    thread_platform_id = parent_post.get('id', 'unknown')
                
                thread_db_id = db.get_or_create_thread(PLATFORM, thread_platform_id, thread_title)
                last_global_id = f"{PLATFORM}_{thread_platform_id}" if thread_platform_id != "unknown" else None
                
                parent_thread = context.get('parent_thread', [])
                if isinstance(parent_thread, list):
                    for comment in reversed(parent_thread):
                        if isinstance(comment, dict):
                            insert_comment_recursive(db, comment, thread_db_id, last_global_id, thread_title, counters)
                            cid = comment.get('comment_id', comment.get('id', 'unknown'))
                            last_global_id = f"{PLATFORM}_{cid}"
                    
                if my_comment and isinstance(my_comment, dict):
                    insert_comment_recursive(db, my_comment, thread_db_id, last_global_id, thread_title, counters)
                    cid = my_comment.get('comment_id', my_comment.get('id', 'unknown'))
                    my_global_id = f"{PLATFORM}_{cid}"
                    
                    for reply in replies_to_my_comment:
                        if isinstance(reply, dict):
                            insert_comment_recursive(db, reply, thread_db_id, my_global_id, thread_title, counters)
                 
                db.commit()   
        except Exception as e:
            logging.error(f"Stream Error parsing comment {filepath}: {e}")
            continue

    db.close()
    logging.info("Successfully processed all Reddit contexts and chats into memory using OOM-safe Streaming.")

if __name__ == "__main__":
    process_reddit()
