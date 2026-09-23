import asyncio
import sqlite3
import os
import json
from pathlib import Path
from playwright.async_api import async_playwright

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

DB_PATH = REPO_ROOT / "processed_data" / "db" / "sarthink_memory.db"
OUTPUT_JSON = REPO_ROOT / "processed_data" / "metadata" / "twitter_id_map.json"

async def intercept_route(route):
    if route.request.resource_type in ["image", "media", "font", "stylesheet", "script"]:
        await route.abort()
    else:
        await route.continue_()

async def resolve_id_worker(user_id: str, context, semaphore: asyncio.Semaphore):
    async with semaphore:
        page = await context.new_page()
        url = f"https://x.com/i/user/{user_id}"
        
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            
            for _ in range(20):
                if "x.com/i/user" not in page.url:
                    break
                await asyncio.sleep(0.5)
                
            final_url = page.url
            username = final_url.rstrip('/').split('/')[-1] if "x.com/i/user" not in final_url else None
            return user_id, username
        except Exception as e:
            return user_id, None
        finally:
            await page.close()

async def bulk_resolve(user_ids, existing_map, max_concurrency=10):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            viewport={"width": 800, "height": 600}
        )
        
        semaphore = asyncio.Semaphore(max_concurrency)
        
        total = len(user_ids)
        print(f"Starting resolution for {total} IDs...")
        
        # We will process in chunks of 50 to continuously write to the JSON file
        chunk_size = 50
        for i in range(0, total, chunk_size):
            chunk = user_ids[i:i+chunk_size]
            tasks = [resolve_id_worker(uid, context, semaphore) for uid in chunk]
            results = await asyncio.gather(*tasks)
            
            for uid, uname in results:
                if uname:
                    existing_map[uid] = uname
                    
            print(f"[{min(i+chunk_size, total)}/{total}] Batch complete. Saving map...")
            with open(OUTPUT_JSON, "w") as f:
                json.dump(existing_map, f, indent=4)
                
        await browser.close()
        print("Scraping fully completed!")

if __name__ == "__main__":
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT raw_id FROM Users WHERE platform='twitter' AND raw_id NOT GLOB '*[^0-9]*';")
    numeric_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    # Load existing to skip repeats
    existing_map = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r") as f:
                existing_map = json.load(f)
        except Exception:
            pass
            
    filtered_ids = [uid for uid in numeric_ids if uid not in existing_map]
    
    print(f"Found {len(numeric_ids)} total numeric IDs in DB.")
    print(f"Found {len(existing_map)} already in {OUTPUT_JSON}.")
    print(f"Remaining to scrape: {len(filtered_ids)}")
    
    if filtered_ids:
        asyncio.run(bulk_resolve(filtered_ids, existing_map, max_concurrency=10))
