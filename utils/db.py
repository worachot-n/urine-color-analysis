"""
SQLite persistence for scan results.

Public API:
    init_db()                                        — create tables if not exist
    save_scan_result(counts, errors, image_path)     → int  (row id)
    save_slot_results(scan_id, slot_assignments)     — insert per-slot rows
    get_latest_result()                              → dict | None
    get_recent_results(limit)                        → list[dict]
    get_slot_results(start, end)                     → list[dict]
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import configs.config as config

_DB_PATH = Path(config.DB_PATH)

_CREATE_SCAN_SQL = """
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

_CREATE_SLOT_SQL = """
CREATE TABLE IF NOT EXISTS slot_results (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id   INTEGER NOT NULL REFERENCES scan_results(id),
    timestamp TEXT    NOT NULL,
    slot_id   TEXT    NOT NULL,
    level     INTEGER NOT NULL
)
"""


def init_db() -> None:
    """Create tables if they do not already exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as con:
        con.execute(_CREATE_SCAN_SQL)
        con.execute(_CREATE_SLOT_SQL)


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


def save_slot_results(scan_id: int, slot_assignments: dict) -> None:
    """Insert one row per slot from a slot_assignments dict."""
    ts = datetime.now().isoformat(timespec="seconds")
    rows = [
        (scan_id, ts, slot_id, int(data["level"]))
        for slot_id, data in slot_assignments.items()
        if data.get("level") is not None
    ]
    if not rows:
        return
    with sqlite3.connect(_DB_PATH) as con:
        con.executemany(
            "INSERT INTO slot_results (scan_id, timestamp, slot_id, level) VALUES (?,?,?,?)",
            rows,
        )


def get_slot_results(start: str = "", end: str = "") -> list[dict]:
    """Return slot-level rows filtered by date range (ISO date strings, inclusive)."""
    sql    = "SELECT timestamp, slot_id, level FROM slot_results"
    params: list = []
    if start:
        sql += " WHERE timestamp >= ?"
        params.append(start)
    if end:
        sql += (" AND" if start else " WHERE") + " timestamp <= ?"
        params.append(end + "T23:59:59")
    sql += " ORDER BY timestamp ASC"
    with sqlite3.connect(_DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


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
