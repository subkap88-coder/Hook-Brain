import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hookbrain.db")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS scans (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                hook_text      TEXT    NOT NULL,
                brain_data     TEXT    NOT NULL,
                viral_score    REAL,
                mechanic       TEXT,
                parent_scan_id INTEGER,
                created_at     TEXT    NOT NULL
            );
        """)


def save_scan(hook_text, data, parent_scan_id=None, mechanic=None):
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO scans
               (hook_text, brain_data, viral_score, mechanic, parent_scan_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                hook_text,
                json.dumps(data),
                data.get("viral", {}).get("viral_score"),
                mechanic,
                parent_scan_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cur.lastrowid


def get_history(limit=60):
    with _conn() as c:
        rows = c.execute(
            """SELECT id, hook_text, viral_score, mechanic, parent_scan_id, created_at
               FROM scans
               ORDER BY created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_scan(scan_id):
    with _conn() as c:
        row = c.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    if not row:
        return None
    data = dict(row)
    try:
        data["brain_data"] = json.loads(data["brain_data"])
    except Exception:
        pass
    return data
