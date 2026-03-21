"""
SQLite persistence for scan results.

Public API:
    init_db()                          — create table if not exists
    save_scan_result(counts, errors, image_path) → int  (row id)
    get_latest_result()                → dict | None
    get_recent_results(limit)          → list[dict]
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import configs.config as config

_DB_PATH = Path(config.DB_PATH)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS scan_results (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT    NOT NULL,
    count_0    INTEGER NOT NULL DEFAULT 0,
    count_1    INTEGER NOT NULL DEFAULT 0,
    count_2    INTEGER NOT NULL DEFAULT 0,
    count_3    INTEGER NOT NULL DEFAULT 0,
    count_4    INTEGER NOT NULL DEFAULT 0,
    errors     TEXT    NOT NULL DEFAULT '[]',
    image_path TEXT
)
"""


def init_db() -> None:
    """Create the scan_results table if it does not already exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as con:
        con.execute(_CREATE_SQL)


def save_scan_result(counts: dict, errors: list, image_path) -> int:
    """Insert one scan result row and return the new row id."""
    with sqlite3.connect(_DB_PATH) as con:
        cur = con.execute(
            """INSERT INTO scan_results
               (timestamp, count_0, count_1, count_2, count_3, count_4, errors, image_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(timespec="seconds"),
                int(counts.get(0, 0)),
                int(counts.get(1, 0)),
                int(counts.get(2, 0)),
                int(counts.get(3, 0)),
                int(counts.get(4, 0)),
                json.dumps(list(errors)),
                str(image_path) if image_path else None,
            ),
        )
        return cur.lastrowid


def get_latest_result() -> dict | None:
    """Return the most recent scan as a dict, or None if no rows exist."""
    with sqlite3.connect(_DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT * FROM scan_results ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return _row_to_dict(row) if row else None


def get_recent_results(limit: int = 20) -> list[dict]:
    """Return the most recent `limit` scans, newest first."""
    with sqlite3.connect(_DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM scan_results ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["errors"] = json.loads(d["errors"])
    d["counts"] = {i: d.pop(f"count_{i}") for i in range(5)}
    return d
