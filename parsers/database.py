import sqlite3
import json
import os
import logging
from datetime import datetime

class SarthinkMemoryLayer:
    def __init__(self, db_path=None, jsonl_dir=None):
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        REPO_ROOT = os.path.dirname(SCRIPT_DIR)
        
        self.db_path = db_path if db_path else os.path.join(REPO_ROOT, 'processed_data', 'sarthink_memory.db')
        self.jsonl_dir = jsonl_dir if jsonl_dir else os.path.join(REPO_ROOT, 'processed_data')
        self.open_jsonl_files = {}
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.jsonl_dir, exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._init_db()

    def _init_db(self):
        self.cursor.executescript('''
            CREATE TABLE IF NOT EXISTS Users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT,
                raw_id TEXT,
                display_name TEXT,
                UNIQUE(platform, raw_id)
            );
            
            CREATE TABLE IF NOT EXISTS Threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT,
                platform_thread_id TEXT,
                title TEXT,
                UNIQUE(platform, platform_thread_id)
            );
            
            CREATE TABLE IF NOT EXISTS Messages (
                msg_id TEXT PRIMARY KEY,
                thread_id INTEGER,
                author_id INTEGER,
                timestamp_utc INTEGER,
                content TEXT,
                parent_msg_id TEXT,
                FOREIGN KEY(thread_id) REFERENCES Threads(id),
                FOREIGN KEY(author_id) REFERENCES Users(id)
            );
            
            -- Keep things extremely fast for retrieval over time
            CREATE INDEX IF NOT EXISTS idx_timestamp ON Messages(timestamp_utc);
            CREATE INDEX IF NOT EXISTS idx_thread ON Messages(thread_id);
        ''')
        self.conn.commit()

    def get_or_create_user(self, platform, raw_id, display_name):
        self.cursor.execute("SELECT id FROM Users WHERE platform=? AND raw_id=?", (platform, raw_id))
        row = self.cursor.fetchone()
        if row: return row[0]
        
        self.cursor.execute("INSERT INTO Users (platform, raw_id, display_name) VALUES (?, ?, ?)", 
                            (platform, raw_id, display_name))
        self.conn.commit()
        return self.cursor.lastrowid

    def get_or_create_thread(self, platform, platform_thread_id, title):
        self.cursor.execute("SELECT id FROM Threads WHERE platform=? AND platform_thread_id=?", 
                            (platform, platform_thread_id))
        row = self.cursor.fetchone()
        if row: return row[0]
        
        self.cursor.execute("INSERT INTO Threads (platform, platform_thread_id, title) VALUES (?, ?, ?)", 
                            (platform, platform_thread_id, title))
        self.conn.commit()
        return self.cursor.lastrowid

    def insert_message(self, msg_id, thread_id, author_id, timestamp_utc, content, parent_msg_id=None, json_data=None, jsonl_filename=None, commit_now=True):
        """Inserts into SQLite and appends to the flat JSONL file efficiently."""
        if timestamp_utc == 0:
            timestamp_utc = None
            
        try:
            # SQLite Insert
            self.cursor.execute("""
                INSERT OR IGNORE INTO Messages (msg_id, thread_id, author_id, timestamp_utc, content, parent_msg_id) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (msg_id, thread_id, author_id, timestamp_utc, content, parent_msg_id))
            
            if self.cursor.rowcount > 0 and json_data and jsonl_filename:
                if jsonl_filename not in self.open_jsonl_files:
                    jsonl_path = os.path.join(self.jsonl_dir, jsonl_filename)
                    self.open_jsonl_files[jsonl_filename] = open(jsonl_path, 'a', encoding='utf-8')
                
                f = self.open_jsonl_files[jsonl_filename]
                f.write(json.dumps(json_data, ensure_ascii=False) + '\n')
                    
            # If the row was ignored, but the incoming data is a real message 
            # and the existing database row is just a Ghost Node placeholder, overwrite it!
            if self.cursor.rowcount == 0 and content and not content.startswith('[Context Missing'):
                self.cursor.execute("""
                    UPDATE Messages 
                    SET thread_id = ?, author_id = ?, timestamp_utc = ?, content = ?, parent_msg_id = COALESCE(?, parent_msg_id)
                    WHERE msg_id = ? AND content LIKE '[Context Missing%'
                """, (thread_id, author_id, timestamp_utc, content, parent_msg_id, msg_id))
                
                if self.cursor.rowcount > 0 and json_data and jsonl_filename:
                    if jsonl_filename not in self.open_jsonl_files:
                        jsonl_path = os.path.join(self.jsonl_dir, jsonl_filename)
                        self.open_jsonl_files[jsonl_filename] = open(jsonl_path, 'a', encoding='utf-8')
                    f = self.open_jsonl_files[jsonl_filename]
                    f.write(json.dumps(json_data, ensure_ascii=False) + '\n')
                    
            if commit_now:
                self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"DB Insert Error: {e}")

    def commit(self):
        """Manual batched commit trigger."""
        self.conn.commit()

    def close(self):
        self.conn.commit()
        for f in self.open_jsonl_files.values():
            f.close()
        self.conn.close()
