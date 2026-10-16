"""Sarthink memory agent: DeepSeek (thinking mode) + tools over the memory index.

Tools
  search_memory      hybrid semantic+keyword search with person/platform/date/me_only filters
  read_conversation  a full conversation span (session) + its summary + neighbouring sessions
  find_person        name -> person candidates
  run_sql            read-only SQL over a simple `messages` / `people` view, for counts and stats

Every search hit / conversation the agent sees is registered as a numbered source [n]. The final answer
must cite [n]; the API returns only the cited sources, each with message ids and graph nodes (P_/T_) so
the viewer can highlight them.
"""
import concurrent.futures as cf
import datetime as dt
import json
import re
import sqlite3
import threading
import time
from types import SimpleNamespace


import llm
from common import INDEX_DB, SRC_DB, USER_DB

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
MAX_STEPS = 10
SQL_ROW_LIMIT = 50

SYSTEM = """You are Sarthink, the private memory of Sarthak Sidhant. Your knowledge is his archive of
messages across WhatsApp, Twitter/X, Reddit, Discord, Instagram and Facebook (2011 to 2026). You answer his
questions about his own life, conversations and people, speaking to him as "you".

Today is {today}. Message timestamps and dates in tool results are IST (Sarthak's timezone). The short AI
summaries/context lines were written from UTC times, so for late-night messages their dates can be a day off:
when they disagree, trust the message timestamps.

How to work:
- Always look things up with tools before answering; never answer from general knowledge.
- search_memory is your main tool. Search 2-3 different phrasings when the first try is weak, and use its
  filters: `person` when a name is mentioned, `since`/`until` for time periods, `me_only=true` when the
  question is about his own conversations ("I", "my", "we"). Short keyword-style queries work well; the
  archive is in English, Hindi and Hinglish, so try Hinglish phrasings for Hinglish chats.
- Search hits are excerpts. Use read_conversation on the most promising hits before stating details, so
  you read the full context and the replies.
- Use run_sql for anything countable: who he talks to most, message counts, when something started or
  stopped, activity over time. Never estimate numbers from search results.
- Memory: recall holds consolidated facts with dates (current vs past) and support. For anything about the current
  state or history of Sarthak or a person (studies, work, where they live, relationships, possessions, health,
  plans, what changed), call recall first. Facts with checked=true were verified against the messages: when such
  current facts answer the question, ANSWER FROM THEM and cite their [n]; do not search again just to re-confirm.
  Search or read further only when the facts are missing, unchecked, disputed, or the question asks for details,
  quotes or a specific episode. Facts carry last_mentioned: for a question about NOW, a fact not mentioned for
  months (or flagged maybe_outdated) is the last known state — say so, or check for anything newer. Interpret freely when asked (patterns, how things changed), marked as such.
- For questions about a relationship or a person in general, start with person_overview, then read a few of
  the example conversations it gives for the themes you mention. Its numbers are exact; cite them as [sql].
- Use topic_timeline for "when" questions about a topic; then read the example excerpts it returns.
- People and accounts: a person (P_<id>) is a human Sarthak has grouped one or more accounts into (their
  WhatsApp + Instagram, say); an account (A_<id>) is one profile on one platform. Tools accept a name, a P_ or an
  A_ id; a person covers all of their accounts. Use find_person when a name is ambiguous.
- Pass `alternatives` to search_memory for vague questions (other phrasings, Hinglish, his own words).
- Questions like "what did X need help with / work on / talk about" can match many episodes: call
  person_overview for X first to see the range of themes, then search within the likely ones, so you do not
  lock onto the first match.
- If the evidence shows several DIFFERENT episodes that fit the question and it does not ask for all of them,
  answer the most likely one and briefly list the others (one line each with date and citation), asking
  which one he meant.

How to answer:
- Every factual claim must cite its source with [n], using the numbers shown in tool results. SQL results
  are cited as [sql] and the query summarised in words.
- Separate what Sarthak said from what others said. Give dates. Quote short phrases verbatim when useful
  (keep Hinglish as written).
- If the evidence is weak, partial or conflicting, say so plainly. If nothing relevant is found after a
  few searches, say you could not find it — never guess or invent.
- When several conversations are relevant, summarise across them rather than picking one.
- Notes: tool results may include `your_notes` — things Sarthak asked you to remember. Treat them as true and
  more authoritative than the chats; cite them as [note]. If a note contradicts the archive, say so.
- Pronouns: use a person's pronouns only if a note or the chats clearly state them; otherwise use "they".
- Cite individual numbers like [3] or [3][5]; never ranges. Never show internal ids (person_id, account_id,
  session_id, chunk/msg ids, P_/A_/T_ node ids) in the answer.
- Personal details: this is Sarthak's own archive and he is the only user. Phone numbers, addresses, emails and
  other identifiers are fine to give when they answer the question; don't pad answers with them otherwise.
- Be concise: a short direct answer first, then supporting details. Markdown is fine."""

TOOLS = [
    {"type": "function", "function": {
        "name": "recall",
        "description": "Sarthak's MEMORY: consolidated, dated facts about him, a person, or a place/organisation/"
                       "project (where someone studies, works, lives; relationships; possessions; health; plans; what "
                       "changed and when). Each fact has since/until dates, status (current/past/disputed), how many "
                       "conversations support it, and a numbered source to cite. Fast: use it FIRST for questions "
                       "about anyone's current state, history or relationships, then read sources only when a fact is "
                       "disputed, weakly supported (1 conversation) or the question needs the details.",
        "parameters": {"type": "object", "properties": {
            "about": {"type": "string", "description": "'me' for Sarthak, a person's name / P_<id> / A_<id>, or a "
                                                       "place/organisation/project name; omit to search all facts"},
            "query": {"type": "string", "description": "what you want to know, to rank the facts (e.g. 'college', "
                                                       "'where does he live', 'laptop')"},
            "include_past": {"type": "boolean", "description": "also return facts that are no longer true (history)"},
        }}}},
    {"type": "function", "function": {
        "name": "search_memory",
        "description": "Hybrid semantic + keyword search over chat excerpts. Returns numbered excerpts with date, "
                       "platform, participants and a session_id for read_conversation.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what to look for; short and specific works best"},
            "alternatives": {"type": "array", "items": {"type": "string"},
                             "description": "up to 3 other phrasings (e.g. Hinglish, synonyms, how Sarthak would "
                                            "have written it); results of all phrasings are merged"},
            "person": {"type": "string", "description": "only chats with this person or account (name, P_<id> or A_<id>)"},
            "platform": {"type": "string", "enum": ["whatsapp", "twitter", "reddit", "discord", "instagram", "facebook"]},
            "since": {"type": "string", "description": "YYYY-MM-DD"},
            "until": {"type": "string", "description": "YYYY-MM-DD"},
            "me_only": {"type": "boolean", "description": "only conversations Sarthak took part in"},
            "k": {"type": "integer", "description": "number of results, default 8, max 15"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_conversation",
        "description": "Read a whole conversation span (session) by session_id: full messages, an AI summary, "
                       "and the ids of the previous/next sessions of the same chat.",
        "parameters": {"type": "object", "properties": {
            "session_id": {"type": "string"},
        }, "required": ["session_id"]}}},
    {"type": "function", "function": {
        "name": "person_overview",
        "description": "Everything-at-a-glance for one person or account, computed over ALL conversations: exact "
                       "message counts, first/last contact, platforms, busiest months, and recurring THEMES with "
                       "example conversations as numbered sources. For a person, themes come per account "
                       "(by_account), since each account is a different place you talk. Use it for 'what do X and I talk "
                       "about', 'tell me about my friendship with X', 'how did things with X change'.",
        "parameters": {"type": "object", "properties": {
            "person": {"type": "string", "description": "name, P_<id> (person: all their accounts) or A_<id> (one account)"}},
            "required": ["person"]}}},
    {"type": "function", "function": {
        "name": "topic_timeline",
        "description": "WHEN was something talked about: month-by-month counts of the most relevant excerpts for "
                       "a topic (semantic + keyword), plus the best excerpt from each of the busiest months as "
                       "numbered sources. Use for 'when was I into X', 'when did X start/stop', 'how did X evolve'. "
                       "Counts are of top matches, not exact totals — use run_sql for exact keyword counts.",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string"},
            "person": {"type": "string"}, "platform": {"type": "string"},
            "since": {"type": "string"}, "until": {"type": "string"}, "me_only": {"type": "boolean"},
        }, "required": ["topic"]}}},
    {"type": "function", "function": {
        "name": "remember",
        "description": "Save something Sarthak EXPLICITLY asks you to remember (\"remember that…\", \"note that…\", "
                       "\"FYI X is…\"): a fact about a person (pronouns, who they are, a correction) or context about "
                       "a topic/place/project. Never save anything he did not ask you to save.",
        "parameters": {"type": "object", "properties": {
            "note": {"type": "string", "description": "the fact, as a short self-contained sentence"},
            "person": {"type": "string", "description": "the person it is about, if any (name as he wrote it)"},
            "topic": {"type": "string", "description": "the topic/context it is about, if not a person"},
            "keywords": {"type": "array", "items": {"type": "string"},
                         "description": "3-6 words that should bring this note back (names, aliases, related terms)"},
        }, "required": ["note"]}}},
    {"type": "function", "function": {
        "name": "forget",
        "description": "Remove saved notes when Sarthak asks you to forget/delete something he told you.",
        "parameters": {"type": "object", "properties": {"about": {"type": "string"}}, "required": ["about"]}}},
    {"type": "function", "function": {
        "name": "list_notes",
        "description": "List what Sarthak has asked you to remember (optionally about one person/topic).",
        "parameters": {"type": "object", "properties": {"about": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "find_person",
        "description": "Find people and accounts by (partial) name. Returns people (node P_<id>: a human with "
                       "one or more accounts) and accounts (node A_<id>: one platform profile, with the person it "
                       "belongs to, if any), each with platforms and message count.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "run_sql",
        "description": (
            "Read-only SQLite SELECT for counts and stats. Views:\n"
            "  messages(msg_id, account_id, account, person_id, person, who_id, is_me, platform, thread_id, thread, ts,"
            " day, content)\n"
            "    one row per message. account = the platform account that wrote it; person_id/person = the human it "
            "belongs to (NULL person_id if the account is in no person; then person = the account name); who_id = "
            "'P_<person_id>' or 'A_<account_id>' — group by who_id to count per human. is_me=1 for Sarthak; ts = unix "
            "seconds UTC; day = 'YYYY-MM-DD' (IST)\n"
            "  msg_counts(account_id, account, person_id, person, who_id, is_me, platform, thread_id, ts, day)\n"
            "    the same rows without msg_id/content/thread: several times faster — use it for counts and stats\n"
            "  accounts(account_id, name, platform, is_me, placeholder, message_count, person_id, person)  "
            "placeholder=1: one chat of a deleted/deactivated account (may carry a guessed name ending in '?')\n"
            "  people(person_id, name, accounts, message_count)  Sarthak is person_id 1\n"
            "  threads(thread_id, platform, title, is_dm)\n"
            "Example — who he talks with most in DMs:\n"
            "  SELECT o.person, COUNT(*) n FROM msg_counts o JOIN threads t USING(thread_id) WHERE t.is_dm=1 "
            "AND o.is_me=0 AND o.thread_id IN (SELECT thread_id FROM msg_counts WHERE is_me=1) "
            "GROUP BY o.who_id ORDER BY n DESC LIMIT 10\n"
            f"Results are capped at {SQL_ROW_LIMIT} rows; long text is truncated. Include msg_id in the SELECT when "
            "you list individual messages: each such row then gets a citable source number [n]."),
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}},
]


CITE_RE = re.compile(r"\[((?:\d+\s*(?:[-–—]\s*\d+)?\s*,?\s*)+)\]")


def cited_numbers(text):
    """Numbers cited as [3], [3, 5], [3-5] / [3–5]."""
    out = set()
    for group in CITE_RE.findall(text or ""):
        for part in re.split(r"\s*,\s*", group.strip().strip(",")):
            m = re.match(r"^(\d+)\s*[-–—]\s*(\d+)$", part)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if 0 < b - a <= 30:
                    out.update(range(a, b + 1))
            elif part.isdigit():
                out.add(int(part))
    return out


EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<![\w/])(?:\+?\d[\d\s-]{8,}\d)(?![\w/])")
ORDER_RE = re.compile(r"(?i)\b(order|invoice|tracking|awb)(\s*(?:no\.?|number|id|#)?\s*[:#]?\s*)([A-Z0-9][A-Z0-9-]{4,})")


def redact(text):
    """Deterministic safety net: mask emails and phone-like numbers in final answers (prompt rules can slip,
    e.g. inside verbatim quotes)."""
    text = EMAIL_RE.sub("[email hidden]", text or "")
    text = ORDER_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}[hidden]" if any(c.isdigit() for c in m.group(3))
                        else m.group(), text)
    return PHONE_RE.sub(lambda m: "[number hidden]" if sum(c.isdigit() for c in m.group()) >= 10 else m.group(), text)


def fmt_date(iso):
    if not iso:
        return "?"
    return dt.datetime.fromisoformat(iso).astimezone(IST).strftime("%d %b %Y")


class SqlTool:
    """Read-only connection with temp views; an authorizer blocks anything but reading."""

    def __init__(self):
        from people import PeopleStore
        PeopleStore(USER_DB).db.close()       # make sure the people tables exist before attaching read-only
        self.conn = sqlite3.connect(f"file:{INDEX_DB}?mode=ro", uri=True, check_same_thread=False)
        self.conn.execute(f"ATTACH DATABASE 'file:{SRC_DB}?mode=ro' AS src")
        self.conn.execute(f"ATTACH DATABASE 'file:{USER_DB}?mode=ro' AS u")
        self.conn.executescript("""
            CREATE TEMP VIEW accounts AS
              SELECT a.account_id, a.name, a.platform, a.is_me, a.placeholder, a.message_count,
                     pa.person_id, pe.name AS person
              FROM main.Accounts a      -- main.: the temp view "accounts" shadows the table name (SQLite is case-blind)
              LEFT JOIN u.PersonAccounts pa ON pa.platform = a.platform AND pa.raw_id = a.raw_id AND pa.chat = a.chat
              LEFT JOIN u.People pe ON pe.person_id = pa.person_id;
            CREATE TEMP VIEW people AS
              SELECT pe.person_id, pe.name, COUNT(a.account_id) AS accounts, COALESCE(SUM(a.message_count), 0) AS message_count
              FROM u.People pe
              LEFT JOIN u.PersonAccounts pa ON pa.person_id = pe.person_id
              LEFT JOIN main.Accounts a ON a.platform = pa.platform AND a.raw_id = pa.raw_id AND a.chat = pa.chat
              GROUP BY pe.person_id;
            CREATE TEMP VIEW threads AS
              SELECT id AS thread_id, platform, title,
                     CASE WHEN platform IN ('discord', 'whatsapp') OR lower(title) LIKE 'dm%'
                               OR (platform IN ('instagram','facebook') AND lower(title) NOT LIKE 'comment on%')
                               OR platform_thread_id LIKE '%:reddit.com%' THEN 1 ELSE 0 END AS is_dm
              FROM src.Threads;
        """)
        self.facts = self._facts_fresh()
        if self.facts:      # precomputed per-message account/platform/day (build_identity.py): ~3x faster stats
            self.conn.executescript("""
            CREATE TEMP VIEW msg_stats AS          -- internal: messages without the people columns (faster stats)
              SELECT f.account_id, f.is_me, f.platform, f.thread_id, t.title AS thread, f.day
              FROM main.MessageFacts f JOIN src.Threads t ON t.id = f.thread_id;
            CREATE TEMP VIEW msg_counts AS         -- for the model: messages without the text, for counting
              SELECT f.account_id, a.name AS account, pa.person_id, COALESCE(pe.name, a.name) AS person,
                     COALESCE('P_' || pa.person_id, 'A_' || f.account_id) AS who_id, f.is_me, f.platform, f.thread_id,
                     f.ts, f.day
              FROM main.MessageFacts f
              JOIN main.Accounts a ON a.account_id = f.account_id
              LEFT JOIN u.PersonAccounts pa ON pa.platform = a.platform AND pa.raw_id = a.raw_id AND pa.chat = a.chat
              LEFT JOIN u.People pe ON pe.person_id = pa.person_id;
            CREATE TEMP VIEW messages AS
              SELECT m.msg_id, f.account_id, a.name AS account, pa.person_id, COALESCE(pe.name, a.name) AS person,
                     COALESCE('P_' || pa.person_id, 'A_' || f.account_id) AS who_id, f.is_me, f.platform, f.thread_id,
                     t.title AS thread, f.ts, f.day, m.content
              FROM main.MessageFacts f
              JOIN src.Messages m ON m.rowid = f.rid
              JOIN main.Accounts a ON a.account_id = f.account_id
              LEFT JOIN u.PersonAccounts pa ON pa.platform = a.platform AND pa.raw_id = a.raw_id AND pa.chat = a.chat
              LEFT JOIN u.People pe ON pe.person_id = pa.person_id
              JOIN src.Threads t ON t.id = f.thread_id;
            """)
        else:               # older data folder, or messages imported without re-running build_identity.py
            print("[sql] MessageFacts missing or stale: run scripts/index/build_identity.py for faster stats")
            self.conn.executescript("""
            CREATE TEMP VIEW msg_stats AS          -- internal: messages without the people columns (faster stats)
              SELECT m.msg_id, a.account_id, a.is_me, t.platform, m.thread_id, t.title AS thread,
                     date(COALESCE(m.timestamp_utc, CASE WHEN substr(m.msg_id, 1, 8) = 'twitter_' AND substr(m.msg_id, 9) GLOB '[0-9]*'
                              THEN ((CAST(substr(m.msg_id, 9) AS INTEGER) >> 22) + 1288834974657) / 1000 END),
                          'unixepoch', '+330 minutes') AS day
              FROM src.Messages m
              JOIN AccountUsers au ON au.user_id = m.author_id
              LEFT JOIN AccountThreads at ON at.user_id = m.author_id AND at.thread_id = m.thread_id
              JOIN main.Accounts a ON a.account_id = COALESCE(at.account_id, au.account_id)
              JOIN src.Threads t ON t.id = m.thread_id;
            CREATE TEMP VIEW messages AS
              SELECT m.msg_id, a.account_id, a.name AS account, pa.person_id, COALESCE(pe.name, a.name) AS person,
                     COALESCE('P_' || pa.person_id, 'A_' || a.account_id) AS who_id, a.is_me, t.platform, m.thread_id,
                     t.title AS thread,
                     COALESCE(m.timestamp_utc, CASE WHEN substr(m.msg_id, 1, 8) = 'twitter_' AND substr(m.msg_id, 9) GLOB '[0-9]*'
                              THEN ((CAST(substr(m.msg_id, 9) AS INTEGER) >> 22) + 1288834974657) / 1000 END) AS ts,
                     date(COALESCE(m.timestamp_utc, CASE WHEN substr(m.msg_id, 1, 8) = 'twitter_' AND substr(m.msg_id, 9) GLOB '[0-9]*'
                              THEN ((CAST(substr(m.msg_id, 9) AS INTEGER) >> 22) + 1288834974657) / 1000 END),
                          'unixepoch', '+330 minutes') AS day, m.content
              FROM src.Messages m          -- ts computed inline (null tweet times recovered from Snowflake ids), so
                                           -- filters on thread/person use the indexes instead of scanning every message
              JOIN AccountUsers au ON au.user_id = m.author_id
              LEFT JOIN AccountThreads at ON at.user_id = m.author_id AND at.thread_id = m.thread_id
              JOIN main.Accounts a ON a.account_id = COALESCE(at.account_id, au.account_id)
              LEFT JOIN u.PersonAccounts pa ON pa.platform = a.platform AND pa.raw_id = a.raw_id AND pa.chat = a.chat
              LEFT JOIN u.People pe ON pe.person_id = pa.person_id
              JOIN src.Threads t ON t.id = m.thread_id;
            CREATE TEMP VIEW msg_counts AS
              SELECT account_id, account, person_id, person, who_id, is_me, platform, thread_id, ts, day FROM messages;
        """)
        self.conn.set_authorizer(self._authorize)
        self.lock = threading.Lock()

    def _facts_fresh(self):
        try:
            got = self.conn.execute("SELECT n, max_rid FROM main.FactsMeta").fetchone()
            now = self.conn.execute("SELECT COUNT(*), MAX(rowid) FROM src.Messages").fetchone()
            return got is not None and tuple(got) == tuple(now)
        except sqlite3.Error:
            return False

    @staticmethod
    def _authorize(action, *args):
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                   getattr(sqlite3, "SQLITE_RECURSIVE", 33)}
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

    def run(self, sql):
        with self.lock:
            return self._run(sql)

    def _run(self, sql):
        if not re.match(r"^\s*(select|with)\b", sql, re.I) or ";" in sql.strip().rstrip(";"):
            return {"error": "only a single SELECT statement is allowed"}
        deadline = time.time() + 8
        self.conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10000)
        try:
            cur = self.conn.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(SQL_ROW_LIMIT + 1)
        except sqlite3.Error as e:
            return {"error": str(e)}
        finally:
            self.conn.set_progress_handler(None, 0)
        trunc = lambda v: (v[:200] + "…") if isinstance(v, str) and len(v) > 200 else v  # noqa: E731
        return {"columns": cols, "rows": [[trunc(v) for v in r] for r in rows[:SQL_ROW_LIMIT]],
                "truncated": len(rows) > SQL_ROW_LIMIT}



from common import PROFILE_CACHE  # noqa: E402
from themes import themes as local_themes  # noqa: E402


class Profiles:
    """Exact stats (SQL) + recurring themes (local, themes.py) for a set of accounts: one account, or all
    accounts of a person. Cached on disk per scope ("A:<id>" / "P:<id>") until the accounts in it, their message
    counts or the summaries change, so attaching an account to a person refreshes that person's profile."""

    def __init__(self, sql, version="v4"):      # v4: themes per account, named by merged topics, where + when
        self.sql, self.version = sql, version
        self.cache = sqlite3.connect(PROFILE_CACHE, check_same_thread=False)
        self.cache.execute("CREATE TABLE IF NOT EXISTS ScopeProfiles (scope TEXT, version TEXT, data TEXT, "
                           "PRIMARY KEY (scope, version))")
        self.lock = threading.Lock()

    def _q(self, sql, args=()):
        with self.sql.lock:
            return self.sql.conn.execute(sql, args).fetchall()

    def _sessions(self, account_ids):
        return self._q("""SELECT s.session_id, s.start_ts, x.data,
                   (SELECT t.title FROM src.Threads t WHERE t.id = json_extract(s.thread_ids, '$[0]')) AS title
            FROM Sessions s
            JOIN SessionSummaries x ON x.session_id = s.session_id AND x.model='deepseek-flash' AND x.prompt_version='v3'
            WHERE EXISTS (SELECT 1 FROM json_each(s.account_ids) j WHERE j.value IN (SELECT value FROM json_each(?)))
            ORDER BY s.start_ts""", (json.dumps(list(account_ids)),))

    def get(self, scope, account_ids, name, themes=True):
        """themes=False (a person): totals only; a person's themes are their accounts' themes."""
        account_ids = sorted(set(account_ids))
        msgs = self._q("SELECT COALESCE(SUM(message_count), 0) FROM main.Accounts WHERE account_id IN "
                       "(SELECT value FROM json_each(?))", (json.dumps(account_ids),))[0][0]
        sessions = self._sessions(account_ids)
        fp = [account_ids, msgs, len(sessions), themes]
        with self.lock:
            row = self.cache.execute("SELECT data FROM ScopeProfiles WHERE scope=? AND version=?",
                                     (scope, self.version)).fetchone()
        if row and json.loads(row[0]).get("fingerprint") == fp:
            prof = json.loads(row[0])
            prof["name"] = name                      # names change without the data changing
            return prof
        prof = self._build(account_ids, name, sessions, themes)
        prof["fingerprint"] = fp
        with self.lock:
            self.cache.execute("INSERT OR REPLACE INTO ScopeProfiles VALUES (?,?,?)",
                               (scope, self.version, json.dumps(prof, ensure_ascii=False)))
            self.cache.commit()
        return prof

    def threads(self, account_ids):
        """Chats these accounts wrote in, looked up through the accounts (indexed), not by scanning the messages
        view. A placeholder account is one chat (AccountThreads)."""
        aj = json.dumps(list(account_ids))
        return [r[0] for r in self._q("""
            SELECT thread_id FROM AccountThreads WHERE account_id IN (SELECT value FROM json_each(?1))
            UNION
            SELECT DISTINCT m.thread_id FROM AccountUsers au JOIN src.Messages m ON m.author_id = au.user_id
            WHERE au.account_id IN (SELECT value FROM json_each(?1))
              AND NOT EXISTS (SELECT 1 FROM AccountThreads x WHERE x.user_id = au.user_id)""", (aj,))]

    def _build(self, account_ids, name, sessions, themes=True):
        threads = self.threads(account_ids)
        if not threads:
            return {"name": name, "account_ids": account_ids, "messages_total": 0, "themes": []}
        tj, aj = json.dumps(threads), json.dumps(account_ids)
        stats = self._q("""SELECT COUNT(*), SUM(is_me), SUM(account_id IN (SELECT value FROM json_each(?))), MIN(day),
                                  MAX(day), GROUP_CONCAT(DISTINCT platform)
                           FROM msg_stats WHERE thread_id IN (SELECT value FROM json_each(?))""", (aj, tj))[0]
        months = self._q("""SELECT substr(day,1,7) m, COUNT(*) n FROM msg_stats WHERE thread_id IN
                            (SELECT value FROM json_each(?)) AND day IS NOT NULL GROUP BY m ORDER BY n DESC LIMIT 4""",
                         (tj,))
        return {"name": name, "account_ids": account_ids, "platforms": stats[5], "chats": len(threads),
                "messages_total": stats[0], "messages_by_sarthak": stats[1], "messages_by_them": stats[2],
                "first_day": stats[3], "last_day": stats[4],
                "busiest_months": [{"month": m, "messages": n} for m, n in months],
                "sessions_summarized": len(sessions), "themes": self._themes(sessions) if themes else []}

    def _themes(self, sessions):
        """Recurring themes of these sessions, computed locally from their vectors and summary topics (themes.py)."""
        if not sessions:
            return []
        sids = [x[0] for x in sessions]
        chunk_rows = []
        for i in range(0, len(sids), 900):
            part = sids[i:i + 900]
            chunk_rows += self._q(f"SELECT session_id, chunk_id FROM Chunks WHERE session_id IN ({','.join('?' * len(part))})",
                                  tuple(part))
        return local_themes(sessions, chunk_rows)


from common import USER_DB as NOTES_DB  # noqa: E402


class NotesStore:
    """Things Sarthak explicitly asked Sarthink to remember. A note can be about a person (P_), one account (A_) or
    a topic. Person/account notes come back with that person or account; topic notes surface
    when one of their trigger keywords appears in the question, a search, or retrieved text. Nothing is put in
    the system prompt."""

    def __init__(self):
        self.conn = sqlite3.connect(NOTES_DB, check_same_thread=False)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS Notes (
            note_id INTEGER PRIMARY KEY, person_id INTEGER, subject TEXT, keywords TEXT, text TEXT,
            created_at TEXT, active INTEGER DEFAULT 1, account_id INTEGER)""")
        self.lock = threading.Lock()

    def add(self, text, subject, person_id=None, keywords=(), account_id=None):
        kws = sorted({k.strip().lower() for k in keywords if k and len(k.strip()) >= 3})
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO Notes(person_id, account_id, subject, keywords, text, created_at) VALUES (?,?,?,?,?,?)",
                (person_id, account_id, subject, json.dumps(kws), text,
                 dt.datetime.now(IST).isoformat(timespec="minutes")))
            self.conn.commit()
            return cur.lastrowid

    def _rows(self, where="", args=()):
        with self.lock:
            rows = self.conn.execute(f"SELECT note_id, person_id, subject, keywords, text, created_at, account_id "
                                     f"FROM Notes WHERE active=1 {where}", args).fetchall()
        return [{"note_id": r[0], "person_id": r[1], "subject": r[2], "keywords": json.loads(r[3] or "[]"),
                 "text": r[4], "saved": r[5], "account_id": r[6]} for r in rows]

    def for_nodes(self, nodes):
        """Notes about any of these P_/A_ nodes."""
        pids = [int(n[2:]) for n in nodes if n.startswith("P_") and n[2:].isdigit()]
        aids = [int(n[2:]) for n in nodes if n.startswith("A_") and n[2:].isdigit()]
        if not pids and not aids:
            return []
        return self._rows(f"AND (person_id IN ({','.join('?' * len(pids)) or 'NULL'}) "
                          f"OR account_id IN ({','.join('?' * len(aids)) or 'NULL'}))", (*pids, *aids))

    def for_text(self, text):
        """Notes whose subject or keywords occur in the text (word-boundary match)."""
        t = f" {re.sub(r'[^a-z0-9]+', ' ', (text or '').lower())} "
        out = []
        for n in self._rows():
            keys = n["keywords"] + [re.sub(r"[^a-z0-9]+", " ", n["subject"].lower()).strip()]
            if any(k and f" {re.sub(r'[^a-z0-9]+', ' ', k).strip()} " in t for k in keys):
                out.append(n)
        return out

    def search(self, about):
        a = (about or "").lower()
        return [n for n in self._rows() if not a or a in n["subject"].lower() or a in n["text"].lower()
                or any(a in k for k in n["keywords"])]

    def deactivate(self, ids):
        with self.lock:
            self.conn.executemany("UPDATE Notes SET active=0 WHERE note_id=?", [(i,) for i in ids])
            self.conn.commit()


class Trace(list):
    """Per-question tool trace that also carries an optional progress callback (for streaming UIs)."""

    def __init__(self, emit=None):
        super().__init__()
        self.emit_fn = emit
        self.notes_shown = {}          # note_id -> note, shown to the model during this question

    def emit(self, **event):
        if self.emit_fn:
            try:
                self.emit_fn(event)
            except Exception:
                pass


def describe(name, args, res):
    """Human-readable progress line (+ graph nodes to light up) for a tool call."""
    who = f" with {args['person']}" if args.get("person") else ""
    if name == "search_memory":
        nodes = [n for r in (res.get("results") or [])[:3] for n in r.get("nodes", [])] if isinstance(res, dict) else []
        return f"🔎 Searching “{args.get('query', '')}”{who}", nodes
    if name == "read_conversation":
        return f"📖 Reading: {res.get('where', 'a conversation')}", res.get("nodes", [])
    if name == "recall":
        about = args.get("about") or "everything"
        return f"🧠 Recalling what I know about {'you' if about.lower() in ('me', 'sarthak', 'myself') else about}", []
    if name == "person_overview":
        return f"👤 Looking at your history with {args.get('person', '…')}", []
    if name == "topic_timeline":
        return f"📅 Mapping when you talked about “{args.get('topic', '')}”{who}", []
    if name == "run_sql":
        return "🧮 Crunching numbers in your message database", []
    if name == "find_person":
        return f"👤 Looking up “{args.get('name', '')}”", []
    if name == "remember":
        return f"📝 Remembering: {args.get('note', '')}", []
    if name == "forget":
        return f"🗑️ Forgetting notes about {args.get('about', '')}", []
    return f"⚙️ {name}", []


PLANNER = """You plan research for a personal-memory assistant over Sarthak's chat archive (Twitter, Reddit,
Discord, Instagram, Facebook, WhatsApp; 2011-2026). Today is {today}.

Decide if the question needs ONE line of research ("simple": a specific lookup, one count, one person's
single fact) or SEVERAL independent lines worth running in parallel:
- "ambiguous": could refer to several different episodes -> one task per plausible interpretation
- "collect": asks for all/every/list of things -> split by category, platform or time period
- "compare": two or more people/things -> one task per side (plus a stats task if counts matter)
- "temporal": change over time -> split into periods, plus a stats task for activity over time
- "cross": across platforms -> one task per relevant platform
- "person": broad question about a relationship -> overview/themes task + tasks for key themes/episodes
Each task must be self-contained and specific, with optional hints (person, platform, since, until,
suggested search queries). Use 2-4 tasks; never split simple questions. Facebook is tiny (a few hundred
messages): never give it its own task — fold it into another task.

Return JSON: {{"type": "simple|ambiguous|collect|compare|temporal|cross|person", "why": "one line",
"tasks": [{{"id": 1, "goal": "...", "hints": {{"person": null, "platform": null, "since": null,
"until": null, "queries": ["..."]}}}}]}}  (tasks = [] when simple)

QUESTION: {question}"""

RESEARCHER = """

You are one RESEARCHER in a team. Research only YOUR TASK thoroughly (search several phrasings, read the most
relevant conversations, use SQL/person_overview when counts or relationships matter). Then return an
EVIDENCE BRIEF, not a polished answer: bullet points of concrete facts (who, what, when, quotes), each with its
[n] citations exactly as numbered in tool results. Include a final bullet "Not found:" listing anything you
searched for but could not find. No speculation."""

WRITER = """

You are the WRITER. Researchers have gathered evidence briefs for the question. Write the final answer for
Sarthak using ONLY facts in the briefs and database results, keeping their [n] citations (numbers are shared
across briefs; cite [sql] for database numbers). Merge overlapping facts, resolve contradictions by saying
so, lead with the direct answer, and keep it concise. If the briefs found nothing relevant, say so plainly."""

FINAL_NUDGE = ("You have used your tool budget. Do not call any more tools. Answer now using only what you "
               "have already found, with [n] citations; say clearly what you could not find.")
TOOLCALL_TEXT = re.compile(r"(<[｜|]\s*(DSML|tool[_▁ ]?call)|<\s*/?\s*function_calls?\s*>|^\s*\{\s*\"(sql|query|session_id|topic|person)\"\s*:)", re.I | re.M)

VERIFY = """You are the fact-checker for a personal-memory assistant. Below is a DRAFT answer to the user's
question and ALL the sources the assistant read, numbered [n]. Produce the final answer.

Rules:
- Check every factual claim against the sources. If it is supported by a source, make sure it cites that
  source's number (fix wrong or missing citations; a claim can cite several).
- If a claim is not supported by any source, delete it — or, if it is a reasonable inference worth keeping,
  rewrite it as clearly marked speculation ("possibly…", "my guess is…") without a citation.
- Do not add new facts. Do not change facts that are supported. Keep the structure, tone, language and
  length of the draft; keep verbatim quotes exactly as in the source.
- [sql] marks database results; keep those claims as they are.
- If deleting unsupported parts leaves nothing useful, say what could and could not be found.

Return JSON: {{"answer": "<final markdown answer>", "changes": ["short description of each change"]}}

QUESTION: {question}

DRAFT ANSWER:
{draft}

SOURCES:
{sources}"""


class Agent:
    """backend: object with search(q, k, **filters) -> list[chunk payload], session(id) -> dict,
    person(name) -> list of people/account candidates, resolve(name or P_/A_ id) -> scope dict
    {kind, id, node, name, scope, account_ids}; payloads are the API's JSON shapes."""

    def __init__(self, backend, model="deepseek-v4-pro", thinking=True, verify=False, verify_model="deepseek-flash"):
        # without a key the agent answers with a hint; stats/search/inbox elsewhere keep working
        self.client = llm.client(timeout=300)
        self.backend = backend
        self.model = model
        self.thinking = thinking
        self.verify = verify
        self.verify_model = verify_model
        # the citation check thinks only when the answer did: with thinking it catches more unsupported claims
        # (4.6% vs 6.4% of claims left, basic eval) but takes ~5x longer (median 9s vs 2s, worst 40s vs 6s).
        # The draft streams first, so in normal mode that wait is spent reading; Quick mode skips it.
        self.verify_thinking = thinking
        self.sql = SqlTool()
        self.profiles = Profiles(self.sql)
        self._lock = threading.RLock()
        self.research_model = "deepseek-v4-pro"   # flash researchers were cheaper but measurably worse
        self.notes = NotesStore()

    # ---- verifier ----------------------------------------------------------------------------
    @staticmethod
    def _source_text(s, full=None):
        body = (full or s)["text"]
        return f"[{s['n_']}] {s['header']} ({fmt_date(s['start'])})\n{body[:9000]}"

    def _verify(self, question, draft, sources, trace, usage):
        if not sources or not draft:
            return draft, []
        read = {s["session_id"]: s for s in sources.values() if s["kind"] == "conversation"}
        blocks = [self._source_text({**s, "n_": n}, read.get(s["session_id"]) if s["kind"] == "excerpt" else None)
                  for n, s in sources.items()]
        sql = [f"[sql] {x['tool']} {json.dumps(x['args'], ensure_ascii=False)}\n-> "
               f"{json.dumps(x.get('result'), ensure_ascii=False)[:3000]}"
               for x in trace if x["tool"] in ("run_sql", "person_overview") and x.get("result")]
        sql += [f"[note] about {n['subject']}: {n['text']}" for n in getattr(trace, "notes_shown", {}).values()]
        t0 = time.time()
        r = self.client.chat.completions.create(
            model=self.verify_model, response_format={"type": "json_object"}, max_tokens=16000,
            extra_body=None if self.verify_thinking else {"thinking": {"type": "disabled"}},
            messages=[{"role": "user", "content": VERIFY.format(
                question=question, draft=draft, sources="\n\n".join(blocks + sql))}])
        usage["in"] += r.usage.prompt_tokens
        usage["out"] += r.usage.completion_tokens
        try:
            out = json.loads(r.choices[0].message.content)
            answer, changes = out["answer"], out.get("changes", [])
        except (json.JSONDecodeError, KeyError, TypeError):
            return draft, ["verifier returned no valid JSON; draft kept"]
        trace.append({"tool": "verify", "args": {}, "ms": int((time.time() - t0) * 1000), "error": None,
                      "changes": changes})
        return answer, changes

    # ---- tools -------------------------------------------------------------------------------
    def _register(self, sources, key, payload):
        with self._lock:                                  # researchers run in parallel, one shared numbering
            for n, s in sources.items():
                if s["key"] == key:
                    return n
            n = len(sources) + 1
            sources[n] = {"key": key, **payload}
            return n

    def _search(self, sources, query, k=8, alternatives=None, **filters):
        k = max(1, min(int(k or 8), 15))
        filters = {f: v for f, v in filters.items() if v not in (None, "")}
        phrasings = [query] + [a for a in (alternatives or []) if isinstance(a, str) and a.strip()][:3]
        if hasattr(self.backend, "prefetch"):
            self.backend.prefetch(phrasings)          # all phrasings in one encoder request
        try:
            ranked = [self.backend.search(p, k, **filters) for p in phrasings]
        except Exception as e:
            return {"error": str(e)[:300]}
        if len(ranked) == 1:
            hits = ranked[0]
        else:                                             # reciprocal-rank fusion across phrasings
            score, by_id = {}, {}
            for lst in ranked:
                for rank, h in enumerate(lst):
                    score[h["chunk_id"]] = score.get(h["chunk_id"], 0) + 1 / (60 + rank)
                    by_id[h["chunk_id"]] = h
            hits = [by_id[c] for c in sorted(score, key=lambda c: -score[c])[:k]]
        out = []
        for h in hits:
            n = self._register(sources, ("chunk", h["chunk_id"]), {"kind": "excerpt", **h})
            out.append({"n": n, "date": fmt_date(h["start"]), "platform": h["platform"], "where": h["header"],
                        "summary": h.get("context"), "excerpt": h["text"][:700], "session_id": h["session_id"],
                        "nodes": h["nodes"]})
        return {"results": out} if out else {"results": [], "note": "nothing found; try other words or filters"}

    def _read(self, sources, session_id, limit=9000):
        try:
            s = self.backend.session(session_id)
        except Exception as e:
            return {"error": str(e)[:300]}
        n = self._register(sources, ("session", session_id), {"kind": "conversation", **s})
        text = s["text"]
        note = None
        if len(text) > limit:
            text, note = text[:limit], "truncated; the rest continues in next_session or later messages"
        summ = (s.get("summary") or {}).get("summary")
        return {"n": n, "where": s["header"], "nodes": s["nodes"], "from": fmt_date(s["start"]), "to": fmt_date(s["end"]),
                "summary": summ, "messages": text, "note": note,
                "previous_session": s.get("previous_session"), "next_session": s.get("next_session")}

    def _recall(self, sources, about=None, query=None, include_past=False, k=25):
        """Facts from the memory layer (scripts/index/build_facts.py), most relevant first, each citable."""
        import numpy as np
        c = self.sql.conn
        with self.sql.lock:
            if not c.execute("SELECT 1 FROM main.sqlite_master WHERE name = 'Facts'").fetchone():
                return {"error": "memory not built yet (scripts/index/build_facts.py); use search_memory"}
        about_l = (about or "").strip().lower()
        if about_l in ("me", "i", "myself", "sarthak", "sarthak sidhant", "you"):
            where, args, label = "e.kind = 'self'", (), "you"
        elif about:
            pid, aids = None, []
            try:
                sc = self.backend.resolve(about)
                pid = sc["id"] if sc["kind"] == "person" else None
                aids = sc.get("account_ids") or []
                label = sc.get("name") or about
            except Exception:
                label = about
            q = "e.name LIKE ? OR e.aliases LIKE ?"
            args = [f"%{about}%", f"%{about}%"]
            if pid:
                q += " OR e.person_id = ?"
                args.append(pid)
            if aids:
                q += f" OR e.account_id IN ({','.join('?' * len(aids))})"
                args += aids
            where, args = f"({q})", tuple(args)
        else:
            where, args, label = "1", (), "everything"
        with self.sql.lock:
            checks = c.execute("SELECT 1 FROM main.sqlite_master WHERE name = 'FactChecks'").fetchone()
            fresh = "last_seen" in {r[1] for r in c.execute("PRAGMA main.table_info(Facts)")}
            rows = c.execute(f"""
                SELECT f.fact_id, e.name, e.kind, f.category, f.statement, f.valid_from, f.valid_to, f.status,
                       f.confidence, f.sources, v.v, {'fc.verdict' if checks else 'NULL'},
                       {'f.last_seen, f.stale' if fresh else 'NULL, NULL'}
                FROM main.Facts f JOIN main.Entities e USING(entity_id)
                LEFT JOIN main.FactVectors v USING(fact_id)
                {'LEFT JOIN main.FactChecks fc USING(fact_id)' if checks else ''}
                WHERE {where} {'' if include_past else "AND f.status != 'past'"}""", args).fetchall()
        with self.sql.lock:     # facts Sarthak marked wrong are never used
            try:
                wrong = {r[0] for r in c.execute("SELECT fact_id FROM u.FactFeedback WHERE verdict = 'wrong'")}
            except sqlite3.Error:
                wrong = set()
        rows = [r for r in rows if r[0] not in wrong]
        if not rows:
            return {"about": label, "facts": [], "note": "nothing in memory; try search_memory"}
        if query and any(r[10] for r in rows):
            try:
                qv = np.asarray(self.backend.encode(query), dtype=np.float32)
                score = [float(np.frombuffer(r[10], dtype=np.float16).astype(np.float32) @ qv) if r[10] else -1
                         for r in rows]
            except Exception:
                words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
                score = [sum(w in r[4].lower() for w in words) for r in rows]
            order = sorted(range(len(rows)), key=lambda i: (-score[i], -(rows[i][8] or 0)))
        else:
            order = sorted(range(len(rows)), key=lambda i: (rows[i][7] == "past", -(rows[i][8] or 0)))
        out = []
        for i in order[:k]:
            fid, ename, kind, cat, st, vf, vt, status, conf, srcs, _, checked, last_seen, stale = rows[i]
            n = None
            for sid in json.loads(srcs or "[]")[:1]:
                try:
                    sess = self.backend.session(sid)
                    n = self._register(sources, ("session", sid), {"kind": "conversation", **sess})
                except Exception:
                    pass
            out.append({"n": n, "about": ename if kind != "self" else "Sarthak", "category": cat, "fact": st,
                        "since": vf, "until": vt, "status": status, "conversations": conf,
                        "checked": bool(checked), "last_mentioned": last_seen,
                        **({"maybe_outdated": True} if stale else {})})
        return {"about": label, "facts": out, "more": max(0, len(rows) - k)}

    def _overview(self, sources, person):
        try:
            sc = self.backend.resolve(str(person))          # person (all accounts) or one account
            is_person = sc["kind"] == "person"
            prof = self.profiles.get(sc["scope"], sc["account_ids"], sc["name"], themes=not is_person)
        except Exception as e:
            return {"error": str(e)[:300]}
        out = {k: v for k, v in prof.items() if k not in ("themes", "fingerprint", "account_ids")}
        out["node"] = sc["node"]
        if not is_person:
            out["themes"] = self._themes_with_examples(sources, prof.get("themes", []))
            return out
        # a person: totals over all their accounts; themes belong to each account
        out["by_account"] = []
        for a in sc.get("accounts", []):
            if not a.get("messages"):
                continue
            ap = self.profiles.get(f"A:{a['account_id']}", [a["account_id"]], a["name"])
            out["by_account"].append({"account": a["name"], "platform": a["platform"],
                                      "messages_total": ap.get("messages_total"),
                                      "messages_by_them": ap.get("messages_by_them"),
                                      "first_day": ap.get("first_day"), "last_day": ap.get("last_day"),
                                      "themes": self._themes_with_examples(sources, ap.get("themes", [])[:6])})
        return out

    def _themes_with_examples(self, sources, themes):
        out = []
        for t in themes:
            examples = []
            for sid in t["session_ids"][-3:]:          # most recent examples
                try:
                    sess = self.backend.session(sid)
                except Exception:
                    continue
                n = self._register(sources, ("session", sid), {"kind": "conversation", **sess})
                examples.append({"n": n, "date": fmt_date(sess["start"]), "session_id": sid})
            out.append({"theme": t["theme"], "where_when": t["description"], "example": t.get("example"),
                        "sessions": t["sessions"], "examples": examples})
        return out

    def _remember(self, note, person=None, topic=None, keywords=None):
        pid, aid, subject = None, None, topic or "general"
        if person:
            try:
                sc = self.backend.resolve(person)           # a person if one matches, else one account
            except Exception:
                return {"error": f"no person or account matching {person!r}; ask Sarthak who he means"}
            subject = sc["name"]
            pid, aid = (sc["id"], None) if sc["kind"] == "person" else (None, sc["id"])
        kws = list(keywords or []) + ([person] if person else []) + ([topic] if topic else [])
        nid = self.notes.add(note, subject, pid, kws, account_id=aid)
        return {"saved": True, "about": subject, "note": note, "id": nid}

    def _attach_notes(self, name, args, res, trace):
        """Add not-yet-shown relevant notes to a tool result: person notes for people in it, topic notes whose
        keywords appear in the query or retrieved text."""
        if not isinstance(res, dict) or "error" in res or name in ("remember", "forget", "list_notes"):
            return
        nodes, text = set(), " ".join(str(v) for v in args.values())
        for r in res.get("results", []) if isinstance(res.get("results"), list) else []:
            nodes.update(r.get("nodes", []))
            text += " " + r.get("where", "") + " " + (r.get("excerpt") or "")[:400]
        nodes.update(res.get("nodes", []) if isinstance(res.get("nodes"), list) else [])
        if isinstance(res.get("node"), str):
            nodes.add(res["node"])
        for c in res.get("candidates", []) if isinstance(res.get("candidates"), list) else []:
            nodes.update([c.get("node"), c.get("person_node")] if c.get("person_node") else [c.get("node")])
        text += " " + res.get("where", "") + " " + (res.get("messages") or "")[:2000]
        fresh = [n for n in self.notes.for_nodes({n for n in nodes if n}) + self.notes.for_text(text)
                 if n["note_id"] not in trace.notes_shown]
        if fresh:
            for n in fresh:
                trace.notes_shown[n["note_id"]] = n
            res["your_notes"] = [{"about": n["subject"], "note": n["text"]} for n in fresh]

    def _question_notes(self, question, trace):
        found = [n for n in self.notes.for_text(question) if n["note_id"] not in trace.notes_shown]
        # person notes whose person's name appears in the question
        for n in self.notes._rows("AND (person_id IS NOT NULL OR account_id IS NOT NULL)"):
            first = re.sub(r"[^a-z0-9]+", " ", n["subject"].lower()).strip().split(" ")[0]
            if first and len(first) >= 3 and re.search(rf"\b{re.escape(first)}\b", question.lower()) \
                    and n["note_id"] not in trace.notes_shown and n not in found:
                found.append(n)
        for n in found:
            trace.notes_shown[n["note_id"]] = n
        if not found:
            return ""
        return "\n\n(Notes I asked you to remember that may be relevant:\n" + "\n".join(
            f"- about {n['subject']}: {n['text']}" for n in found) + ")"

    def _cite_rows(self, sources, res):
        """Rows that carry a msg_id become citable: register the chunk containing that message."""
        i = res["columns"].index("msg_id")
        res["columns"] = ["n"] + res["columns"]
        rows = []
        for row in res["rows"][:25]:
            n = None
            try:
                ch = self.backend.chunk_for_message(row[i])
                if ch:
                    n = self._register(sources, ("chunk", ch["chunk_id"]), {"kind": "excerpt", **ch})
            except Exception:
                pass
            rows.append([n] + list(row))
        res["rows"] = rows + [[None] + list(r) for r in res["rows"][25:]]
        return res

    def _timeline(self, sources, topic, **filters):
        filters = {f: v for f, v in filters.items() if v not in (None, "")}
        try:
            t = self.backend.timeline(topic, **filters)
        except Exception as e:
            return {"error": str(e)[:300]}
        examples = {}
        for month, h in t["examples"].items():
            n = self._register(sources, ("chunk", h["chunk_id"]), {"kind": "excerpt", **h})
            examples[month] = {"n": n, "where": h["header"], "excerpt": h["text"][:400]}
        return {"months": t["months"], "busiest": t["busiest"], "examples": examples,
                "note": "counts are of the top ~120 matches, a relative signal, not exact totals"}

    def _call(self, name, args, sources, trace, label=None):
        t0 = time.time()
        if name == "search_memory":
            res = self._search(sources, **args)
        elif name == "read_conversation":
            res = self._read(sources, args.get("session_id", ""), limit=6000 if label else 9000)
        elif name == "person_overview":
            res = self._overview(sources, args.get("person", ""))
        elif name == "remember":
            res = self._remember(**args)
        elif name == "forget":
            found = self.notes.search(args.get("about", ""))
            self.notes.deactivate([n["note_id"] for n in found])
            res = {"forgotten": [n["text"] for n in found]} if found else {"forgotten": [], "note": "no matching notes"}
        elif name == "list_notes":
            res = {"notes": [{k: n[k] for k in ("subject", "text", "saved")} for n in self.notes.search(args.get("about"))]}
        elif name == "recall":
            res = self._recall(sources, args.get("about"), args.get("query"), bool(args.get("include_past")))
        elif name == "find_person":
            res = {"candidates": self.backend.person(args.get("name", ""))[:8]}
        elif name == "run_sql":
            res = self.sql.run(args.get("sql", ""))
            if "rows" in res and "msg_id" in res["columns"]:
                res = self._cite_rows(sources, res)
        elif name == "topic_timeline":
            res = self._timeline(sources, **args)
        else:
            res = {"error": f"unknown tool {name}"}
        entry = {"tool": name, "args": args, "ms": int((time.time() - t0) * 1000),
                 "error": res.get("error") if isinstance(res, dict) else None, "by": label or "agent"}
        if name == "run_sql" and "rows" in res:
            entry["result"] = {"columns": res["columns"], "rows": res["rows"][:40]}
        if name == "person_overview" and "error" not in res:
            brief = lambda ts: [{k: t[k] for k in ("theme", "where_when", "sessions")} for t in ts]  # noqa: E731
            entry["result"] = {k: v for k, v in res.items() if k not in ("themes", "by_account")} | {
                "themes": brief(res.get("themes", []))} | ({"by_account": [
                    {k: v for k, v in a.items() if k != "themes"} | {"themes": brief(a["themes"])}
                    for a in res["by_account"]]} if "by_account" in res else {})
        trace.append(entry)
        if isinstance(trace, Trace):
            self._attach_notes(name, args, res, trace)
        if isinstance(trace, Trace) and isinstance(res, dict) and "error" not in res:
            text, nodes = describe(name, args, res)
            trace.emit(type="step", text=text, nodes=nodes, by=label or "agent")
        if isinstance(res, dict):                     # graph node ids are for the UI only, not the model
            res.pop("nodes", None)
            for r in res.get("results", []) if isinstance(res.get("results"), list) else []:
                r.pop("nodes", None)
        return json.dumps(res, ensure_ascii=False)

    # ---- loop --------------------------------------------------------------------------------
    def _chat(self, trace, live, **kw):
        """One model call. live=True (the main agent with a streaming UI): stream it, sending answer text to the UI
        as it is written ({"type": "delta"}); if the reply turns out to be tool calls after some text, the UI is
        told to drop that text ({"type": "reset"}). Returns (message, usage) shaped like a non-streamed reply."""
        if not live:
            r = self.client.chat.completions.create(**kw)
            return r.choices[0].message, r.usage
        content, reasoning, calls, usage, shown = [], [], {}, None, False
        for ch in self.client.chat.completions.create(stream=True, stream_options={"include_usage": True}, **kw):
            if ch.usage:
                usage = ch.usage
            if not ch.choices:
                continue
            d = ch.choices[0].delta
            if getattr(d, "reasoning_content", None):
                reasoning.append(d.reasoning_content)
            for tc in d.tool_calls or []:
                c = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                c["id"] = tc.id or c["id"]
                if tc.function:
                    c["name"] += tc.function.name or ""
                    c["args"] += tc.function.arguments or ""
            if d.content:
                content.append(d.content)
                if not calls:
                    trace.emit(type="delta", text=d.content)
                    shown = True
        if shown and calls:
            trace.emit(type="reset")
        fn = lambda c: SimpleNamespace(name=c["name"], arguments=c["args"])  # noqa: E731
        msg = SimpleNamespace(content="".join(content), reasoning_content="".join(reasoning) or None,
                              tool_calls=[SimpleNamespace(id=c["id"], function=fn(c)) for _, c in sorted(calls.items())]
                              or None)
        return msg, usage

    def _loop(self, messages, sources, trace, usage, max_steps=MAX_STEPS, label=None, model=None):
        """Tool-calling loop; returns the model's final text."""
        extra = None if self.thinking else {"thinking": {"type": "disabled"}}
        live = label is None and isinstance(trace, Trace) and trace.emit_fn is not None
        for step in range(max_steps + 1):
            final_round = step == max_steps
            if final_round:
                messages.append({"role": "user", "content": FINAL_NUDGE})
            m, u = self._chat(trace, live,
                              model=model or self.model, messages=messages, tools=TOOLS, extra_body=extra,
                              max_tokens=8000, tool_choice="none" if final_round else "auto")
            with self._lock:
                usage["in"] += u.prompt_tokens if u else 0
                usage["out"] += u.completion_tokens if u else 0
                usage["cached"] = usage.get("cached", 0) + (getattr(u, "prompt_cache_hit_tokens", 0) or 0)
            msg = {"role": "assistant", "content": m.content or ""}
            rc = getattr(m, "reasoning_content", None)
            if rc:
                msg["reasoning_content"] = rc   # thinking mode: pass reasoning back within the tool loop
            if m.tool_calls and not final_round:
                msg["tool_calls"] = [{"id": tc.id, "type": "function",
                                      "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                     for tc in m.tool_calls]
                messages.append(msg)
                for tc in m.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": self._call(tc.function.name, args, sources, trace, label)})
                continue
            text = m.content or ""
            if final_round and TOOLCALL_TEXT.search(text):   # model wrote a tool call as text: ask once more
                trace.emit(type="reset")
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": FINAL_NUDGE + " Plain prose only."})
                r = self.client.chat.completions.create(
                    model=model or self.model, messages=messages, extra_body=extra, max_tokens=8000)
                text = r.choices[0].message.content or ""
                if TOOLCALL_TEXT.search(text):
                    text = ""
            return text
        return ""

    # ---- multi-agent: planner -> parallel researchers -> writer -------------------------------
    def _plan(self, question, today, usage):
        r = self.client.chat.completions.create(
            model="deepseek-flash", response_format={"type": "json_object"}, max_tokens=2000,
            extra_body={"thinking": {"type": "disabled"}},
            messages=[{"role": "user", "content": PLANNER.format(today=today, question=question)}])
        with self._lock:
            usage["in"] += r.usage.prompt_tokens
            usage["out"] += r.usage.completion_tokens
        try:
            plan = json.loads(r.choices[0].message.content)
            tasks = [t for t in plan.get("tasks", []) if isinstance(t, dict) and t.get("goal")][:4]
            return {"type": plan.get("type", "simple"), "tasks": tasks, "why": plan.get("why", "")}
        except (json.JSONDecodeError, AttributeError):
            return {"type": "simple", "tasks": [], "why": "planner output unreadable"}

    def _research(self, question, task, today, sources, trace, usage):
        hints = task.get("hints") or {}
        messages = [{"role": "system", "content": SYSTEM.format(today=today) + RESEARCHER},
                    {"role": "user", "content": f"Overall question: {question}\n\nYOUR TASK: {task['goal']}\n"
                                                f"Hints (optional): {json.dumps(hints, ensure_ascii=False)}"}]
        return self._loop(messages, sources, trace, usage, max_steps=5, label=f"researcher {task.get('id', '?')}",
                          model=self.research_model)

    def _write(self, question, today, briefs, trace, usage):
        sql = [f"[sql] {x['tool']} {json.dumps(x['args'], ensure_ascii=False)} -> "
               f"{json.dumps(x.get('result'), ensure_ascii=False)[:1500]}"
               for x in trace if x["tool"] in ("run_sql", "person_overview") and x.get("result")]
        body = "\n\n".join(f"### Brief {i + 1}: {goal}\n{text}" for i, (goal, text) in enumerate(briefs))
        messages = [{"role": "system", "content": SYSTEM.format(today=today) + WRITER},
                    {"role": "user", "content": f"QUESTION: {question}\n\nRESEARCH BRIEFS:\n{body}\n\n"
                                                f"DATABASE RESULTS:\n" + ("\n".join(sql) or "(none)")}]
        extra = None if self.thinking else {"thinking": {"type": "disabled"}}
        r = self.client.chat.completions.create(model=self.model, messages=messages, extra_body=extra,
                                                max_tokens=8000)
        with self._lock:
            usage["in"] += r.usage.prompt_tokens
            usage["out"] += r.usage.completion_tokens
        return r.choices[0].message.content or ""

    def ask(self, question, history=None, multi=True, emit=None):
        t_start = time.time()
        if not self.client:
            return {"question": question, "mode": "disabled", "plan": {}, "draft": None, "verifier_changes": [],
                    "answer": "The memory assistant needs an API key: add `OPENROUTER_API_KEY=...` (or `DEEPSEEK_API_KEY=...`) to `.env` "
                              "and restart. Meanwhile you can explore the graph, click people for their dossier, "
                              "and use the 📥 Inbox and 📅 On this day buttons.",
                    "sources": [], "invalid_citations": [], "steps": [], "model": None, "thinking": None,
                    "tokens": {"in": 0, "out": 0}, "seconds": 0}
        today = dt.datetime.now(IST).strftime("%d %b %Y")
        sources, trace, usage = {}, Trace(emit), {"in": 0, "out": 0}
        if multi and not history:
            trace.emit(type="step", text="🧭 Planning the research", nodes=[])
        plan = self._plan(question, today, usage) if multi and not history else {"type": "simple", "tasks": []}
        trace.append({"tool": "plan", "args": {"type": plan["type"], "tasks": [t["goal"] for t in plan["tasks"]]},
                      "ms": int((time.time() - t_start) * 1000), "error": None})

        if len(plan["tasks"]) >= 2:
            with cf.ThreadPoolExecutor(len(plan["tasks"])) as ex:
                futs = [ex.submit(self._research, question, t, today, sources, trace, usage) for t in plan["tasks"]]
                briefs = [(t["goal"], f.result()) for t, f in zip(plan["tasks"], futs)]
            trace.emit(type="step", text="✍️ Writing the answer from all findings", nodes=[])
            answer = self._write(question, today, briefs, trace, usage)
            mode = "multi"
        else:
            messages = [{"role": "system", "content": SYSTEM.format(today=today)}]
            for turn in (history or [])[-6:]:
                messages.append({"role": turn["role"], "content": turn["content"]})
            messages.append({"role": "user", "content": question + self._question_notes(question, trace)})
            answer = self._loop(messages, sources, trace, usage)
            mode = "single"

        draft, changes = answer, []
        if self.verify:
            trace.emit(type="checking")          # the streamed draft stays readable while it is checked
            trace.emit(type="step", text="✅ Double-checking every citation", nodes=[])
            answer, changes = self._verify(question, draft, sources, trace, usage)

        answer = redact(answer)
        nums = cited_numbers(answer)
        cited = sorted(n for n in nums if n in sources)
        bad = sorted(n for n in nums if n not in sources and n < 1000)   # [2020] is a year, not a cite
        read = {s["session_id"]: s for s in sources.values() if s["kind"] == "conversation"}
        out_sources = []
        for n in cited:
            s = sources[n]
            full = read.get(s["session_id"]) if s["kind"] == "excerpt" else None
            # an excerpt whose whole conversation was read: the answer may quote anywhere in that conversation
            out_sources.append({
                "n": n, "kind": s["kind"], "read_full": bool(full) or s["kind"] == "conversation",
                "header": s["header"], "platform": s["platform"],
                "date": fmt_date(s["start"]), "session_id": s["session_id"], "chunk_id": s.get("chunk_id"),
                "snippet": ((s.get("summary") or {}).get("summary") if s["kind"] == "conversation"
                            else s.get("context")) or s["text"][:300],
                "nodes": sorted(set(s["nodes"]) | set((full or {}).get("nodes", []))),
                "message_ids": (full or s).get("message_ids", []),
                "excerpt_message_ids": s.get("message_ids", []) if full else None,
                "people": s.get("people", []),
            })
        return {"question": question, "answer": answer, "mode": mode, "plan": plan,
                "draft": draft if self.verify else None, "verifier_changes": changes, "sources": out_sources,
                "invalid_citations": bad, "steps": trace, "model": self.model, "thinking": self.thinking,
                "tokens": usage, "seconds": round(time.time() - t_start, 1)}
