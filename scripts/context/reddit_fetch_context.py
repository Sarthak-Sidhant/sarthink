import asyncio
import asyncpraw
import asyncprawcore
from asyncpraw import models
import aiohttp
import pandas as pd
import json
import os
import re
from datetime import datetime
import time 
from pathlib import Path

# Credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

POSTS_CSV_PATH = str(REPO_ROOT / "archive" / "reddit-export" / "posts.csv")
COMMENTS_CSV_PATH = str(REPO_ROOT / "archive" / "reddit-export" / "comments.csv")
# Keep context data in original folder
OUTPUT_DIRECTORY = str(REPO_ROOT / "processed_data" / "context" / "reddit" / "Reddit_Context_Archive")

CONCURRENT_REQUESTS = 25 
MAX_RETRIES = 3
INITIAL_BACKOFF_DELAY = 2
RATELIMIT_SLEEP_MULTIPLIER = 1.2

rate_limit_gate = asyncio.Event()
rate_limit_gate.set()
last_rate_limit_time = 0 # Global for thrashing detection

def create_output_directory():
    """Creates the necessary output directories."""
    print(f"[INFO] Ensuring output directory exists: {OUTPUT_DIRECTORY}")
    for subdir in ["posts", "comments"]:
        os.makedirs(os.path.join(OUTPUT_DIRECTORY, subdir), exist_ok=True)

def parse_ratelimit_sleep(error_message: str) -> int:
    """Extracts the required sleep time from a rate limit error message."""
    matches = re.search(r"(\d+)\s(minute|second)", error_message)
    if not matches: return 60
    value, unit = int(matches.group(1)), matches.group(2)
    return value * 60 if "minute" in unit else value

def get_comment_data(comment):
    """Extracts and formats data from a loaded PRAW comment object."""
    author = str(comment.author) if hasattr(comment, "author") and comment.author else "[deleted]"
    return {
        "comment_id": comment.id, "author": author, "body": comment.body,
        "score": comment.score, "created_utc": datetime.fromtimestamp(comment.created_utc).isoformat(),
        "permalink": f"https://www.reddit.com{comment.permalink}"
    }

def get_post_data(post):
    """Extracts and formats data from a loaded PRAW submission object."""
    author = str(post.author) if hasattr(post, "author") and post.author else "[deleted]"
    subreddit = str(post.subreddit) if hasattr(post, "subreddit") and post.subreddit else "[deleted]"
    return {
        "id": post.id, "title": post.title, "author": author, "subreddit": subreddit,
        "url": f"https://www.reddit.com{post.permalink}", "selftext": post.selftext,
        "score": post.score, "created_utc": datetime.fromtimestamp(post.created_utc).isoformat()
    }

async def get_replies_recursive(item):
    """Recursively fetches the comment tree using optimized traversal to avoid exponential blowups."""
    replies_data = []
    comment_forest = item.comments if isinstance(item, models.Submission) else item.replies
    await comment_forest.replace_more(limit=0) # Get top-level only without deep explosion
    for reply in comment_forest: # Iterate directly over the forest
        if isinstance(reply, models.Comment):
            reply_data = get_comment_data(reply)
            reply_data["replies"] = await get_replies_recursive(reply)
            replies_data.append(reply_data)
    return replies_data

async def process_item(session, item_id, item_type, semaphore, progress_counter, total_items):
    """A single, robust worker function to process either a post or a comment."""
    global last_rate_limit_time
    async with semaphore:
        retries = 0
        while retries <= MAX_RETRIES:
            await rate_limit_gate.wait()
            try:
                # --- Step 1: Fetch and Load the Main Item ---
                if item_type == "POST":
                    item = await session.submission(id=item_id)
                else: # COMMENT
                    item = await session.comment(id=item_id)
                await item.load()
                link = f"https://www.reddit.com{item.permalink}"

                # --- Step 2: Build the Enriched Data Structure ---
                if item_type == "POST":
                    enriched_data = {
                        "my_post_details": get_post_data(item),
                        "conversation_thread": await get_replies_recursive(item)
                    }
                    filename = os.path.join(OUTPUT_DIRECTORY, "posts", f"post_{item_id}.json")
                else: # COMMENT
                    parent_thread, parent_post_data = [], None
                    try:
                        current_object = await item.parent()
                        while not isinstance(current_object, models.Submission):
                            await current_object.load()
                            parent_thread.append(get_comment_data(current_object))
                            current_object = await current_object.parent()
                        await current_object.load()
                        parent_post_data = get_post_data(current_object)
                    except asyncprawcore.exceptions.NotFound:
                        pass # Part of the parent chain was deleted, which is fine
                    
                    # --- OPTIMIZED FETCH LOGIC ---
                    await item.replies.replace_more(limit=0)
                    replies_data = []
                    for top_level_reply in item.replies:
                        if isinstance(top_level_reply, models.Comment):
                            replies_data.append(get_comment_data(top_level_reply))
                    # --- END OF OPTIMIZED LOGIC ---
                    
                    enriched_data = {
                        "my_comment_details": get_comment_data(item),
                        "context": {"parent_post": parent_post_data, "parent_thread": parent_thread},
                        "replies_to_my_comment": replies_data
                    }
                    filename = os.path.join(OUTPUT_DIRECTORY, "comments", f"comment_{item_id}.json")
                
                # --- Step 3: Write to File and Log Success ---
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(enriched_data, f, ensure_ascii=False, indent=4)
                
                print(f"({progress_counter[0]}/{total_items}) [SUCCESS] [{item_type}] {item_id} - {link}")
                progress_counter[0] += 1
                return # Success, exit the loop

            except asyncprawcore.exceptions.TooManyRequests as e:
                current_time = time.monotonic()
                sleep_duration = parse_ratelimit_sleep(str(e))
                safe_sleep = int(sleep_duration * RATELIMIT_SLEEP_MULTIPLIER) + 1
                
                # --- ANTI-THRASHING LOGIC ---
                mandatory_cooldown = 0
                if current_time - last_rate_limit_time < 3:
                    mandatory_cooldown = 60
                    print(f"\n[THRASHING DETECTED] Enforcing mandatory {mandatory_cooldown}s cooldown.")
                
                final_sleep = max(safe_sleep, mandatory_cooldown)
                last_rate_limit_time = current_time
                
                rate_limit_gate.clear()
                print(f"\n[RATE_LIMIT] Pausing all tasks for {final_sleep // 60}m {final_sleep % 60}s.\n")
                await asyncio.sleep(final_sleep)
                rate_limit_gate.set()
                # Loop will continue, retrying this same item after the wait.
            
            except asyncprawcore.exceptions.RequestException: # Catches network errors
                retries += 1
                if retries > MAX_RETRIES:
                    print(f"({progress_counter[0]}/{total_items}) [FAIL] [{item_type}] {item_id} after {MAX_RETRIES} retries.")
                    progress_counter[0] += 1
                    return
                delay = INITIAL_BACKOFF_DELAY * (2 ** (retries - 1))
                print(f"({progress_counter[0]}/{total_items}) [RETRYING] ({retries}/{MAX_RETRIES}) [{item_type}] {item_id} due to network error. Waiting {delay}s...")
                await asyncio.sleep(delay)

            except (asyncprawcore.exceptions.Forbidden, asyncprawcore.exceptions.NotFound):
                print(f"({progress_counter[0]}/{total_items}) [SKIP] [{item_type}] {item_id} is deleted or in a private subreddit.")
                progress_counter[0] += 1
                return
            
            except Exception as e:
                print(f"({progress_counter[0]}/{total_items}) [FAIL] [{item_type}] {item_id} with unexpected error: {type(e).__name__}")
                progress_counter[0] += 1
                return

async def main():
    if not REDDIT_CLIENT_ID or "YOUR_CLIENT_ID_HERE" in REDDIT_CLIENT_ID:
        print("[ERROR] Please fill in your credentials at the top of the script or set environment variables.")
        return
    
    create_output_directory()

    # --- Resume Logic ---
    print("[INFO] Scanning for already completed items...")
    completed_post_ids = {f.replace('post_', '').replace('.json', '') for f in os.listdir(os.path.join(OUTPUT_DIRECTORY, "posts"))}
    completed_comment_ids = {f.replace('comment_', '').replace('.json', '') for f in os.listdir(os.path.join(OUTPUT_DIRECTORY, "comments"))}
    print(f"[INFO] Found {len(completed_post_ids)} completed posts and {len(completed_comment_ids)} completed comments.")
    
    # --- Load and Filter Tasks ---
    all_tasks = []
    try:
        df_posts = pd.read_csv(POSTS_CSV_PATH)
        post_ids = df_posts['id'].tolist()
        post_ids_to_process = [pid for pid in post_ids if pid not in completed_post_ids]
        print(f"[INFO] Posts: {len(post_ids)} total, {len(post_ids_to_process)} remaining.")
        all_tasks.extend([("POST", pid) for pid in post_ids_to_process])
    except FileNotFoundError:
        print(f"[WARN] Posts CSV not found at '{POSTS_CSV_PATH}'. Skipping.")

    try:
        df_comments = pd.read_csv(COMMENTS_CSV_PATH)
        comment_ids = df_comments['id'].tolist()
        comment_ids_to_process = [cid for cid in comment_ids if cid not in completed_comment_ids]
        print(f"[INFO] Comments: {len(comment_ids)} total, {len(comment_ids_to_process)} remaining.")
        all_tasks.extend([("COMMENT", cid) for cid in comment_ids_to_process])
    except FileNotFoundError:
        print(f"[WARN] Comments CSV not found at '{COMMENTS_CSV_PATH}'. Skipping.")

    total_items_to_process = len(all_tasks)
    if total_items_to_process == 0:
        print("\n[INFO] All items have been processed. Nothing to do. Exiting.")
        return
        
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    progress_counter = [1]
    
    async with aiohttp.ClientSession() as http_session:
        reddit = asyncpraw.Reddit(
            client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT, username=REDDIT_USERNAME,
            password=REDDIT_PASSWORD, requestor_kwargs={"session": http_session}
        )
        user = await reddit.user.me()
        print(f"[INFO] Authenticated as u/{user}")
        print(f"[INFO] Starting to process {total_items_to_process} items with {CONCURRENT_REQUESTS} concurrent workers...")

        tasks = [process_item(reddit, item_id, item_type, semaphore, progress_counter, total_items_to_process) for item_type, item_id in all_tasks]
        await asyncio.gather(*tasks)

    print("\n" + "="*50)
    print("Processing complete!")
    print(f"Context files saved in '{OUTPUT_DIRECTORY}' folder.")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
