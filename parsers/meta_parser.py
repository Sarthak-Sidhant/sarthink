import os
import json
import zipfile
import logging
from database import SarthinkMemoryLayer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def decode_meta_string(s):
    if not isinstance(s, str):
        return s
    try:
        # Meta's GDPR exports are notoriously encoded twice in latin-1 instead of utf-8
        return s.encode('latin1').decode('utf-8')
    except Exception:
        return s

def process_meta_zip(db, zip_path, platform, counters):
    logging.info(f"Scanning {platform} archive: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()
        
        # 1. Parse DMs
        msg_files = [f for f in file_list if 'messages/inbox/' in f and f.endswith('.json')]
        logging.info(f"Found {len(msg_files)} DM threads for {platform}")
        
        for mf in msg_files:
            # Reconstruct thread ID logically from filepath
            convo_id = mf.split('messages/inbox/')[1].split('/')[0]
            thread_db_id = db.get_or_create_thread(platform, convo_id, f"DM {convo_id}")
            
            data = json.loads(z.read(mf))
            messages = data.get('messages', [])
            
            # Reconstruct chronological graph sequence exactly like Twitter
            # so `previous_msg_id` models a linked list thread. 
            messages_sorted = sorted(messages, key=lambda x: x.get('timestamp_ms', 0))
            
            previous_msg_id = None
            for idx, msg in enumerate(messages_sorted):
                ts_ms = msg.get('timestamp_ms', 0)
                utc_epoch = ts_ms // 1000 if ts_ms > 0 else None
                
                content = decode_meta_string(msg.get('content', ''))
                
                # Meta usually stores shared links/memes inside 'share' array instead of content string
                if not content and 'share' in msg:
                    share = msg['share']
                    link = share.get('link', '')
                    text = decode_meta_string(share.get('share_text', ''))
                    if text or link:
                        content = f"[Attachment Shared] {text} {link}".strip()
                    else:
                        content = "[Attachment Shared]"
                        
                if not content:
                    continue
                    
                sender = decode_meta_string(msg.get('sender_name', 'unknown'))
                author_db_id = db.get_or_create_user(platform, sender, sender)
                
                # Meta does not assign globally UUIDs for DMs in zip exports. 
                # Hash algorithm: convo_id + timestamp + idx fixes this.
                raw_id = f"{convo_id}_{ts_ms}_{idx}"
                global_id = f"{platform}_{raw_id}"
                
                flat_entry = {
                    "log_id": global_id,
                    "platform": f"{platform}_dm",
                    "thread_name": f"DM {convo_id}",
                    "author": sender,
                    "author_id": sender,
                    "timestamp_utc": utc_epoch,
                    "content": content,
                    "is_reply_to": previous_msg_id
                }
                
                db.insert_message(global_id, thread_db_id, author_db_id, utc_epoch, content, previous_msg_id, flat_entry, 'meta_logs.jsonl', commit_now=False)
                counters['msgs'] += 1
                previous_msg_id = global_id
                
            if counters['msgs'] % 1000 == 0:
                db.commit()

        # 2. Parse Comments
        comment_files = [f for f in file_list if ('comments/' in f or 'comments_and_reactions/' in f or 'comments_v2/' in f) and f.endswith('.json')]
        logging.info(f"Found {len(comment_files)} Comment schemas for {platform}")
        
        for cf in comment_files:
            data = json.loads(z.read(cf))
            
            if isinstance(data, list):
                # Common Instagram structure
                for idx, c in enumerate(data):
                    smd = c.get('string_map_data', {})
                    if not smd: continue
                        
                    comp = smd.get('Comment', {})
                    content = decode_meta_string(comp.get('value', ''))
                    
                    time_data = smd.get('Time', {})
                    ts = time_data.get('timestamp', 0)
                    
                    owner_data = smd.get('Media Owner', {})
                    owner = decode_meta_string(owner_data.get('value', 'unknown_media'))
                    
                    if not content: continue
                    
                    thread_db_id = db.get_or_create_thread(platform, owner, f"Comment on {owner}")
                    
                    raw_id = f"comment_{owner}_{ts}_{idx}"
                    global_id = f"{platform}_{raw_id}"
                    
                    author = "Sarthak"  # Alias
                    author_db_id = db.get_or_create_user(platform, author, author)
                    
                    flat_entry = {
                        "log_id": global_id,
                        "platform": f"{platform}_comment",
                        "thread_name": f"Comment on {owner}",
                        "author": author,
                        "author_id": author,
                        "timestamp_utc": ts if ts > 0 else None,
                        "content": content,
                        "is_reply_to": None
                    }
                    
                    db.insert_message(global_id, thread_db_id, author_db_id, ts, content, None, flat_entry, 'meta_logs.jsonl', commit_now=False)
                    counters['msgs'] += 1
                    
            elif isinstance(data, dict):
                comments_array = data.get('comments_v2', [])
                for idx, c in enumerate(comments_array):
                    data_arr = c.get('data', [])
                    if not data_arr: continue
                    
                    comp = data_arr[0].get('comment', {})
                    content = decode_meta_string(comp.get('comment', ''))
                    if not content: continue
                    
                    ts = comp.get('timestamp', 0)
                    title = decode_meta_string(c.get('title', 'Facebook Post'))
                    
                    thread_db_id = db.get_or_create_thread(platform, title, f"Comment on: {title[:40]}")
                    raw_id = f"fb_comment_{ts}_{idx}"
                    global_id = f"{platform}_{raw_id}"
                    
                    author = decode_meta_string(comp.get('author', 'Sarthak'))
                    author_db_id = db.get_or_create_user(platform, author, author)
                    
                    flat_entry = {
                        "log_id": global_id,
                        "platform": f"{platform}_comment",
                        "thread_name": f"Comment on: {title[:40]}",
                        "author": author,
                        "author_id": author,
                        "timestamp_utc": ts if ts > 0 else None,
                        "content": content,
                        "is_reply_to": None
                    }
                    
                    db.insert_message(global_id, thread_db_id, author_db_id, ts, content, None, flat_entry, 'meta_logs.jsonl', commit_now=False)
                    counters['msgs'] += 1
                    
            db.commit()

def process_all_meta():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(SCRIPT_DIR)
    
    db = SarthinkMemoryLayer()
    counters = {'msgs': 0}
    
    archive_dir = os.path.join(REPO_ROOT, "archive")
    if not os.path.exists(archive_dir):
        logging.info(f"Archive directory {archive_dir} not found. Skipping meta parsing.")
        return
        
    archives = os.listdir(archive_dir)
    for arc in archives:
        if not arc.endswith('.zip'): continue
        
        arc_path = os.path.join(archive_dir, arc)
        if 'instagram' in arc.lower():
            process_meta_zip(db, arc_path, 'instagram', counters)
        elif 'facebook' in arc.lower():
            process_meta_zip(db, arc_path, 'facebook', counters)
            
    db.commit()
    db.close()
    
    logging.info(f"Fully processed Meta. Synchronized {counters['msgs']} messages.")

if __name__ == "__main__":
    process_all_meta()
