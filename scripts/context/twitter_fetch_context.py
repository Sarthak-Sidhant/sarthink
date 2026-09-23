import json
import requests
import time
import glob
import os
import signal
import sys
import threading
from queue import Queue, Empty
from pathlib import Path

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

# Configuration loaded from environment
API_KEY = os.getenv('TWITTER_SOCIALDATA_API_KEY')
REQUEST_INTERVAL = 0.55  # ~108 req/min (safe under 120 limit)

# Data should stay in context/ folder even if script moves
CACHE_FILE = str(REPO_ROOT / 'processed_data' / 'context' / 'twitter' / 'context_cache.json')

SAVE_INTERVAL = 20
NUM_WORKERS = 5
MAX_DEPTH = 100

# Operational state
lock = threading.Lock()
cache = {}
queued = set()  # IDs that are queued or done
stop_event = threading.Event()
save_counter = 0
last_request_time = 0

# Queue holds (tweet_id, depth) tuples
work_queue = Queue()


def load_cache():
    global cache, queued
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            queued = set(cache.keys())
            print(f"Loaded {len(cache)} entries from cache.")
        except json.JSONDecodeError:
            print("Cache file corrupted, starting fresh.")
            cache = {}
            queued = set()
    else:
        cache = {}
        queued = set()


def save_cache():
    with lock:
        cache_copy = dict(cache)
    print(f"\nSaving cache with {len(cache_copy)} entries...")
    temp_file = CACHE_FILE + ".tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(cache_copy, f, indent=4)
    os.replace(temp_file, CACHE_FILE)
    print("Cache saved.")


def signal_handler(sig, frame):
    print("\nInterrupted! Stopping...")
    stop_event.set()


def parse_tweet_files():
    search_path = os.path.join(str(REPO_ROOT), '**', 'tweets*.js')
    tweet_files = glob.glob(search_path, recursive=True)
    initial_ids = set()
    print(f"Found {len(tweet_files)} tweet files.")
    
    for file_path in tweet_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Handle potential wrapper
                idx = content.find('[')
                if idx != -1:
                    tweets = json.loads(content[idx:])
                    for item in tweets:
                        tweet = item.get('tweet', {})
                        reply_id = tweet.get('in_reply_to_status_id_str')
                        if reply_id:
                            initial_ids.add(reply_id)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            
    return initial_ids


def enqueue(tweet_id, depth=0):
    """Add to queue if not already queued/cached. Returns True if added."""
    with lock:
        if tweet_id in queued:
            return False
        queued.add(tweet_id)
    work_queue.put((tweet_id, depth))
    return True


def rate_limit():
    global last_request_time
    with lock:
        now = time.time()
        wait = REQUEST_INTERVAL - (now - last_request_time)
        last_request_time = now + max(0, wait)  # Reserve this slot
    
    if wait > 0:
        time.sleep(wait)


def worker():
    global save_counter
    
    while not stop_event.is_set():
        try:
            item = work_queue.get(timeout=0.5)
        except Empty:
            continue
        
        tweet_id, depth = item
        
        try:
            if stop_event.is_set():
                break
            
            # Check if already in cache
            with lock:
                if tweet_id in cache:
                    work_queue.task_done()
                    continue
            
            # Rate limit
            rate_limit()
            
            if stop_event.is_set():
                work_queue.task_done()
                break
            
            # Fetch
            print(f"Fetching {tweet_id} (depth {depth})...")
            url = f'https://api.socialdata.tools/twitter/tweets/{tweet_id}'
            headers = {
                'Authorization': f'Bearer {API_KEY}',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            result = None
            parent_id = None
            
            if response.status_code == 200:
                data = response.json()
                result = {"status": "success", "data": data, "status_code": 200}
                parent_id = data.get('in_reply_to_status_id_str')
                
            elif response.status_code == 429:
                print(f"Rate limited! Sleeping 60s...")
                with lock:
                    queued.discard(tweet_id)
                enqueue(tweet_id, depth)
                work_queue.task_done()
                time.sleep(60)
                continue
                
            elif response.status_code in [403, 404]:
                result = {"status": "error", "message": response.reason, "status_code": response.status_code}
            else:
                result = {"status": "error", "message": response.text, "status_code": response.status_code}
            
            # Save result
            if result:
                with lock:
                    cache[tweet_id] = result
                    save_counter += 1
                    do_save = save_counter >= SAVE_INTERVAL
                    if do_save:
                        save_counter = 0
                
                if do_save:
                    save_cache()
            
            # Queue parent if exists and not too deep
            if parent_id and depth < MAX_DEPTH:
                if enqueue(parent_id, depth + 1):
                    print(f"  -> Queued parent {parent_id} at depth {depth + 1}")
                
        except Exception as e:
            print(f"Error fetching {tweet_id}: {e}")
            with lock:
                cache[tweet_id] = {"status": "error", "message": str(e), "status_code": 0}
        
        finally:
            work_queue.task_done()


def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    load_cache()
    
    # Get initial IDs from tweet files
    initial_ids = parse_tweet_files()
    
    # Queue IDs not in cache
    queued_count = 0
    for tid in initial_ids:
        if enqueue(tid, 0):
            queued_count += 1
    
    # Also check cached items for parents we may have missed
    with lock:
        cached_items = list(cache.items())
    
    for tid, entry in cached_items:
        if entry.get('status') == 'success':
            data = entry.get('data', {})
            parent = data.get('in_reply_to_status_id_str')
            if parent:
                enqueue(parent, 1)

    print(f"Queue size: {work_queue.qsize()} items to fetch")
    print(f"Starting {NUM_WORKERS} workers...")
    
    threads = []
    for i in range(NUM_WORKERS):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)
    
    try:
        while not stop_event.is_set():
            with lock:
                cache_size = len(cache)
            queue_size = work_queue.qsize()
            
            if queue_size == 0:
                time.sleep(2)
                if work_queue.qsize() == 0:
                    print("Queue empty, finishing up...")
                    break
            
            print(f"Queue: {queue_size}, Cached: {cache_size}")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    
    stop_event.set()
    time.sleep(2)
    save_cache()
    print(f"Done! Total cached: {len(cache)}")


if __name__ == "__main__":
    main()
