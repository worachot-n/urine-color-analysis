"""
SQLite offline backup — persists scan data locally.

Tables:
  scan_summary  — one row per scan (mirrors Summary sheet tab + synced flag)
  scan_detail   — one row per bottle per scan (mirrors Detail sheet tab)
  scan_images   — image locations
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from loguru import logger


def _connect(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    """Create all tables if they do not exist."""
    with _connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS scan_summary (
                scan_id          TEXT PRIMARY KEY,
                created_at       TEXT NOT NULL,
                detected_count   INTEGER NOT NULL DEFAULT 0,
                total_assigned   INTEGER NOT NULL DEFAULT 0,
                missing_slots    TEXT NOT NULL DEFAULT '',
                L0               INTEGER NOT NULL DEFAULT 0,
                L1               INTEGER NOT NULL DEFAULT 0,
                L2               INTEGER NOT NULL DEFAULT 0,
                L3               INTEGER NOT NULL DEFAULT 0,
                L4               INTEGER NOT NULL DEFAULT 0,
                synced           INTEGER NOT NULL DEFAULT 0,
                grid_rows        INTEGER,
                grid_cols        INTEGER,
                wb_offset        TEXT,
                master_path      TEXT,
                weights_json     TEXT
            );

            CREATE TABLE IF NOT EXISTS scan_detail (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id          TEXT NOT NULL,
                created_at       TEXT NOT NULL,
                is_reference     INTEGER NOT NULL DEFAULT 0,
                ref_level        INTEGER,
                slot_id          TEXT NOT NULL,
                cell_index       INTEGER,
                row              INTEGER,
                col              INTEGER,
                L                REAL,
                a                REAL,
                b                REAL,
                hex              TEXT,
                path_idx         REAL,
                path_delta       REAL,
                hist_delta       REAL,
                delta_e          REAL,
                color_level      INTEGER,
                is_forced        INTEGER NOT NULL DEFAULT 0,
                is_achromatic    INTEGER NOT NULL DEFAULT 0,
                detected         INTEGER NOT NULL DEFAULT 0,
                confident        INTEGER NOT NULL DEFAULT 0,
                combined         REAL,
                soft_penalty     INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_detail_scan ON scan_detail(scan_id);

            CREATE TABLE IF NOT EXISTS scan_images (
                scan_id          TEXT PRIMARY KEY,
                local_path       TEXT NOT NULL,
                drive_file_id    TEXT,
                synced           INTEGER NOT NULL DEFAULT 0,
                created_at       TEXT NOT NULL
            );
        """)

        # Migrations for existing DBs — each ALTER is silently skipped if column exists
        _migrate(conn, "scan_summary", [
            "ADD COLUMN grid_rows    INTEGER",
            "ADD COLUMN grid_cols    INTEGER",
            "ADD COLUMN wb_offset    TEXT",
            "ADD COLUMN master_path  TEXT",
            "ADD COLUMN weights_json TEXT",
        ])
        _migrate(conn, "scan_detail", [
            "ADD COLUMN combined     REAL",
            "ADD COLUMN soft_penalty INTEGER NOT NULL DEFAULT 0",
        ])
        # Drop the old JSON blob column if SQLite ≥ 3.35 supports it
        try:
            conn.execute("ALTER TABLE scan_summary DROP COLUMN scan_json")
        except Exception:
            pass

    logger.debug("sqlite: DB initialised at {}", db_path)


def _migrate(conn: sqlite3.Connection, table: str, statements: list[str]) -> None:
    for stmt in statements:
        try:
            conn.execute(f"ALTER TABLE {table} {stmt}")
        except Exception:
            pass  # column already exists


def save_scan(scan_result: dict, db_path: str | Path) -> None:
    """Persist summary + all detail rows for one scan. Idempotent on scan_id."""
    sid       = scan_result["scan_id"]
    ts        = scan_result.get("timestamp", "")
    summary   = scan_result.get("summary", {})
    grid_cols = scan_result.get("grid_cols", 15)

    wb_offset   = scan_result.get("wb_offset")
    master_path = scan_result.get("master_path")
    weights     = scan_result.get("weights")

    detail_rows: list[tuple] = []

    def _row_col(ci: int | None) -> tuple[int | None, int | None]:
        if ci is None:
            return None, None
        return (ci - 1) // grid_cols + 1, (ci - 1) % grid_cols + 1

    for slot_id, d in scan_result.get("slots", {}).items():
        ci  = d.get("cell_index")
        lab = d.get("lab") or [None, None, None]
        r, c = _row_col(ci)
        detail_rows.append((
            sid, ts, 0, None, slot_id, ci, r, c,
            lab[0], lab[1], lab[2], d.get("hex"),
            d.get("concentration_index"), d.get("path_distance"),
            d.get("hist_bhatt"), d.get("delta_e"),
            d.get("color_level"),
            1 if d.get("best_fit")    else 0,
            1 if d.get("force_l0")   else 0,
            1 if d.get("detected")   else 0,
            1 if d.get("confident")  else 0,
            d.get("combined"),
            1 if d.get("soft_penalty") else 0,
        ))

    for ref_level_str, refs in scan_result.get("reference_labs", {}).items():
        ref_level = int(ref_level_str)
        for i, ref in enumerate(refs):
            lab = ref.get("lab") or [None, None, None]
            ci  = ref.get("cell_index")
            r, c = _row_col(ci)
            detail_rows.append((
                sid, ts, 1, ref_level, f"REF_L{ref_level}_{i}", ci, r, c,
                lab[0], lab[1], lab[2], ref.get("hex"),
                None, None, None, None, ref_level,
                0, 0, 1, 1,
                None, 0,  # combined, soft_penalty — not applicable for references
            ))

    try:
        with _connect(db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO scan_summary
                   (scan_id, created_at, detected_count, total_assigned,
                    missing_slots, L0, L1, L2, L3, L4, synced,
                    grid_rows, grid_cols, wb_offset, master_path, weights_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,0, ?,?,?,?,?)""",
                (
                    sid, ts,
                    scan_result.get("detected_count", 0),
                    scan_result.get("total_assigned", 0),
                    ",".join(scan_result.get("missing_slots", [])),
                    summary.get("L0", 0), summary.get("L1", 0),
                    summary.get("L2", 0), summary.get("L3", 0),
                    summary.get("L4", 0),
                    scan_result.get("grid_rows"),
                    grid_cols,
                    json.dumps(wb_offset)   if wb_offset   is not None else None,
                    json.dumps(master_path) if master_path is not None else None,
                    json.dumps(weights)     if weights     is not None else None,
                ),
            )
            conn.executemany(
                """INSERT INTO scan_detail
                   (scan_id, created_at, is_reference, ref_level, slot_id,
                    cell_index, row, col, L, a, b, hex,
                    path_idx, path_delta, hist_delta, delta_e, color_level,
                    is_forced, is_achromatic, detected, confident,
                    combined, soft_penalty)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                detail_rows,
            )
        logger.debug("sqlite: saved scan {} ({} detail rows)", sid, len(detail_rows))
    except Exception as e:
        logger.warning("sqlite: save_scan failed for {}: {}", sid, e)


def save_image_path(
    scan_id: str,
    local_path: str | Path,
    created_at: str,
    db_path: str | Path,
) -> None:
    """Record the local image backup path for a scan."""
    try:
        with _connect(db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO scan_images
                   (scan_id, local_path, drive_file_id, synced, created_at)
                   VALUES (?,?,NULL,0,?)""",
                (scan_id, str(local_path), created_at),
            )
    except Exception as e:
        logger.warning("sqlite: save_image_path failed for {}: {}", scan_id, e)


def mark_scan_synced(scan_id: str, db_path: str | Path) -> None:
    """Mark a scan's Sheets data as successfully uploaded."""
    try:
        with _connect(db_path) as conn:
            conn.execute(
                "UPDATE scan_summary SET synced=1 WHERE scan_id=?", (scan_id,)
            )
    except Exception as e:
        logger.warning("sqlite: mark_scan_synced failed for {}: {}", scan_id, e)


def mark_image_synced(
    scan_id: str,
    drive_file_id: str | None,
    db_path: str | Path,
) -> None:
    """Mark a scan's image as successfully uploaded to Google Drive."""
    try:
        with _connect(db_path) as conn:
            conn.execute(
                "UPDATE scan_images SET synced=1, drive_file_id=? WHERE scan_id=?",
                (drive_file_id, scan_id),
            )
    except Exception as e:
        logger.warning("sqlite: mark_image_synced failed for {}: {}", scan_id, e)


def get_pending_scans(db_path: str | Path) -> list[str]:
    """Return scan_ids whose Sheets data has not been uploaded yet."""
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                "SELECT scan_id FROM scan_summary WHERE synced=0 ORDER BY created_at ASC"
            ).fetchall()
        return [r["scan_id"] for r in rows]
    except Exception as e:
        logger.warning("sqlite: get_pending_scans failed: {}", e)
        return []


def get_drive_file_id(scan_id: str, db_path: str | Path) -> str | None:
    """Return the Drive file_id for a scan if it was previously uploaded."""
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT drive_file_id FROM scan_images WHERE scan_id=?", (scan_id,)
            ).fetchone()
        return row["drive_file_id"] if row else None
    except Exception as e:
        logger.warning("sqlite: get_drive_file_id failed for {}: {}", scan_id, e)
        return None


def get_pending_images(db_path: str | Path) -> list[dict]:
    """Return image records whose Drive upload has not completed yet."""
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                "SELECT scan_id, local_path FROM scan_images WHERE synced=0 ORDER BY created_at ASC"
            ).fetchall()
        return [{"scan_id": r["scan_id"], "local_path": r["local_path"]} for r in rows]
    except Exception as e:
        logger.warning("sqlite: get_pending_images failed: {}", e)
        return []


def count_pending(db_path: str | Path) -> dict[str, int]:
    """Return counts of unsynced scans and images for status reporting."""
    try:
        with _connect(db_path) as conn:
            scans  = conn.execute("SELECT COUNT(*) FROM scan_summary WHERE synced=0").fetchone()[0]
            images = conn.execute("SELECT COUNT(*) FROM scan_images  WHERE synced=0").fetchone()[0]
        return {"pending_scans": scans, "pending_images": images}
    except Exception as e:
        logger.warning("sqlite: count_pending failed: {}", e)
        return {"pending_scans": 0, "pending_images": 0}


def load_scan(scan_id: str, db_path: str | Path) -> dict | None:
    """Reconstruct the full scan result dict from structured tables."""
    try:
        with _connect(db_path) as conn:
            s = conn.execute(
                "SELECT * FROM scan_summary WHERE scan_id=?", (scan_id,)
            ).fetchone()
            if s is None:
                return None
            details = conn.execute(
                "SELECT * FROM scan_detail WHERE scan_id=? ORDER BY id ASC", (scan_id,)
            ).fetchall()

        slots: dict = {}
        reference_labs: dict = {}

        for d in details:
            lab = [d["L"], d["a"], d["b"]] if d["L"] is not None else None
            if d["is_reference"]:
                level_key = str(d["ref_level"])
                reference_labs.setdefault(level_key, []).append({
                    "lab":        lab,
                    "hex":        d["hex"],
                    "cell_index": d["cell_index"],
                })
            else:
                slots[d["slot_id"]] = {
                    "cell_index":          d["cell_index"],
                    "detected":            bool(d["detected"]),
                    "color_level":         d["color_level"],
                    "concentration_index": d["path_idx"],
                    "path_distance":       d["path_delta"],
                    "hist_bhatt":          d["hist_delta"],
                    "delta_e":             d["delta_e"],
                    "combined":            d["combined"],
                    "confident":           bool(d["confident"]),
                    "best_fit":            bool(d["is_forced"]),
                    "force_l0":            bool(d["is_achromatic"]),
                    "soft_penalty":        bool(d["soft_penalty"]),
                    "lab":                 lab,
                    "hex":                 d["hex"],
                }

        def _maybe_json(val):
            return json.loads(val) if val else None

        return {
            "status":         "success",
            "scan_id":        s["scan_id"],
            "timestamp":      s["created_at"],
            "detected_count": s["detected_count"],
            "total_assigned": s["total_assigned"],
            "missing_slots":  [x for x in s["missing_slots"].split(",") if x],
            "summary": {
                "L0": s["L0"], "L1": s["L1"], "L2": s["L2"],
                "L3": s["L3"], "L4": s["L4"],
            },
            "reference_labs": reference_labs,
            "master_path":    _maybe_json(s["master_path"]),
            "wb_offset":      _maybe_json(s["wb_offset"]),
            "weights":        _maybe_json(s["weights_json"]) or {},
            "grid_rows":      s["grid_rows"],
            "grid_cols":      s["grid_cols"],
            "image_url":      f"/img/{s['scan_id']}.jpg",
            "slots":          slots,
        }
    except Exception as e:
        logger.warning("sqlite: load_scan failed for {}: {}", scan_id, e)
        return None


def load_all_scans(db_path: str | Path) -> list[dict]:
    """Return lightweight scan summary dicts for all scans, newest first."""
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                """SELECT scan_id, created_at, detected_count, total_assigned,
                          missing_slots, L0, L1, L2, L3, L4
                   FROM scan_summary ORDER BY created_at DESC"""
            ).fetchall()
        result = []
        for r in rows:
            result.append({
                "scan_id":        r["scan_id"],
                "timestamp":      r["created_at"],
                "detected_count": r["detected_count"],
                "total_assigned": r["total_assigned"],
                "missing_slots":  [x for x in r["missing_slots"].split(",") if x],
                "summary": {
                    "L0": r["L0"], "L1": r["L1"], "L2": r["L2"],
                    "L3": r["L3"], "L4": r["L4"],
                },
            })
        return result
    except Exception as e:
        logger.warning("sqlite: load_all_scans failed: {}", e)
        return []
