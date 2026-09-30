"""Keyword index (SQLite FTS5) over chunks: header, text, LLM context and session keywords.

Catches what embeddings miss: exact names, handles, amounts, rare words, Hinglish spellings.
Usage: python3 scripts/index/build_fts.py
"""
import json

from common import connect_index


def main(ctx_model="deepseek-flash", ctx_version="v3"):
    conn = connect_index()
    conn.executescript("""
        DROP TABLE IF EXISTS ChunksFTS;
        CREATE VIRTUAL TABLE ChunksFTS USING fts5(
            chunk_id UNINDEXED, header, text, context, keywords,
            tokenize = 'unicode61 remove_diacritics 2'
        );
    """)
    kw = {}
    for r in conn.execute("SELECT session_id, data FROM SessionSummaries WHERE model=? AND prompt_version=?",
                          (ctx_model, ctx_version)):
        d = json.loads(r["data"])
        kw[r["session_id"]] = " ".join(d.get("keywords") or []) + " " + " ".join(d.get("topics") or [])
    rows = conn.execute("""
        SELECT c.chunk_id, c.session_id, c.header, c.text, cc.context FROM Chunks c
        LEFT JOIN ChunkContexts cc ON cc.chunk_id=c.chunk_id AND cc.model=? AND cc.prompt_version=?""",
                        (ctx_model, ctx_version)).fetchall()
    conn.executemany("INSERT INTO ChunksFTS VALUES (?,?,?,?,?)",
                     [(r["chunk_id"], r["header"], r["text"], r["context"] or "", kw.get(r["session_id"], ""))
                      for r in rows])
    conn.commit()
    print(f"FTS rows: {len(rows)} (with session keywords: {sum(1 for r in rows if r['session_id'] in kw)})")


if __name__ == "__main__":
    main()
