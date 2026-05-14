"""
FastAPI server — V3 web layer.

Routes:
    POST /analyze           — Pi + browser upload (same endpoint)
    GET  /upload            — Upload form page
    GET  /results           — Scan history (newest first)
    GET  /result/{scan_id}  — Scan detail page
    GET  /settings          — Slot assignment page
    GET  /api/slots         — Return slot_config.json as JSON
    POST /api/slots         — Save slot config
    POST /api/sync          — Trigger manual sync of pending offline data to Google
    GET  /auto_grid         — Grid debug page (CV only, no YOLO, no letterbox)
    POST /api/auto_grid     — Run full-res HoughCircles + KDE grid reconstruction
    GET  /health            — {"status": "ok"}
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from server.slot_config import (
    SlotConfig,
    load_slot_config,
    save_slot_config,
)
from server.pipeline import run_pipeline, _load_yolo
from app.shared.config import cfg as _app_cfg
from utils.auto_grid_detector import detect_grid_full, draw_grid_lines

_HERE        = Path(__file__).parent
_RESULTS_DIR = _HERE / "static" / "results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_API_KEY = os.environ.get("API_KEY", "test")

# ---------------------------------------------------------------------------
# Slot config cache — loaded once, invalidated when POST /api/slots saves
# ---------------------------------------------------------------------------
_slot_cfg_cache: SlotConfig | None = None


def _get_slot_cfg() -> SlotConfig:
    global _slot_cfg_cache
    if _slot_cfg_cache is None:
        _slot_cfg_cache = load_slot_config()
    return _slot_cfg_cache


# ---------------------------------------------------------------------------
# Telegram notifications — fire-and-forget, silently skipped if not configured
# ---------------------------------------------------------------------------
try:
    from bot.telegram_bot import send_scan_report as _telegram_send
    _TELEGRAM_AVAILABLE = True
except ImportError:
    _TELEGRAM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Try importing Google integrations — silently disabled if not configured
# ---------------------------------------------------------------------------
try:
    from server.integrations.sheets import (
        write_slot_config_to_sheet,
        append_detail_to_sheet,
        append_summary_to_sheet,
    )
    _GOOGLE_AVAILABLE = True
except ImportError:
    _GOOGLE_AVAILABLE = False

# ---------------------------------------------------------------------------
# SQLite backup — always available (stdlib sqlite3, no extra dependencies)
# ---------------------------------------------------------------------------
from server.integrations.sqlite_backup import (
    init_db,
    save_scan,
    mark_scan_synced,
    get_pending_scans,
    count_pending,
)

# ---------------------------------------------------------------------------
# Config — parsed once at module load; never re-read per request
# ---------------------------------------------------------------------------

def _load_toml() -> dict:
    try:
        import tomllib
        cfg_path = Path(__file__).parent.parent / "configs" / "config.toml"
        with open(cfg_path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


_TOML = _load_toml()


def _google_cfg() -> dict:
    return _TOML.get("google", {})


def _sqlite_cfg() -> dict:
    return _TOML.get("sqlite", {})


def _db_path() -> Path:
    raw = _sqlite_cfg().get("db_path", "logs/urine_analysis.db")
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).parent.parent / p
    return p


def _img_backup_dir() -> Path:
    raw = _TOML.get("debug", {}).get("img_dir", "logs/img/")
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).parent.parent / p
    p.mkdir(parents=True, exist_ok=True)
    return p


# Precompute at startup — avoids repeated lookups inside hot paths
_DB_PATH       = _db_path()
_IMG_BACKUP    = _img_backup_dir()


# ---------------------------------------------------------------------------
# Cloud + Telegram — runs in a thread pool; never blocks the HTTP response
# ---------------------------------------------------------------------------

def _cloud_and_notify(
    scan_id: str,
    scan_result: dict,
    img_backup: Path,
    db: Path,
) -> None:
    """
    Blocking Sheets sync and Telegram notification.
    Always called via asyncio.to_thread() so the event loop is never stalled.
    """
    if _GOOGLE_AVAILABLE:
        gcfg        = _google_cfg()
        sa_file     = gcfg.get("service_account_file", "credentials.json")
        spreadsheet = gcfg.get("spreadsheet_id", "")
        detail_tab  = gcfg.get("detail_tab",  "Detail")
        summary_tab = gcfg.get("summary_tab", "Summary")

        if spreadsheet:
            try:
                server_url = _app_cfg.server_url.rstrip("/")
                img_url = f"{server_url}/static/results/{scan_id}.jpg"
                append_detail_to_sheet(scan_result, spreadsheet, detail_tab, sa_file)
                append_summary_to_sheet(scan_result, spreadsheet, summary_tab, sa_file, image_url=img_url)
                mark_scan_synced(scan_id, db)
            except Exception as e:
                logger.warning("analyze: Sheets append failed (will retry on next sync): {}", e)

    if _TELEGRAM_AVAILABLE:
        try:
            _telegram_send(
                count=scan_result["detected_count"],
                color_summary=scan_result.get("summary", {}),
                image_path=img_backup,
            )
        except Exception as e:
            logger.warning("analyze: Telegram notification failed: {}", e)


# ---------------------------------------------------------------------------
# Background sync — runs on startup and can be triggered via /api/sync
# ---------------------------------------------------------------------------

def _sync_pending_sync() -> dict[str, int]:
    """Blocking Sheets sync — must be called via asyncio.to_thread()."""
    if not _GOOGLE_AVAILABLE:
        return {"synced_scans": 0}

    gcfg        = _google_cfg()
    sa_file     = gcfg.get("service_account_file", "credentials.json")
    spreadsheet = gcfg.get("spreadsheet_id", "")
    detail_tab  = gcfg.get("detail_tab",  "Detail")
    summary_tab = gcfg.get("summary_tab", "Summary")
    db          = _DB_PATH

    if not spreadsheet:
        return {"synced_scans": 0}

    synced_scans = 0
    server_url   = _app_cfg.server_url.rstrip("/")

    for scan_id in get_pending_scans(db):
        json_path = _RESULTS_DIR / f"{scan_id}.json"
        if not json_path.exists():
            logger.warning("sync: scan JSON not found for {}, skipping Sheets sync", scan_id)
            continue
        try:
            scan_result = json.loads(json_path.read_text())
            img_url = f"{server_url}/static/results/{scan_id}.jpg"
            append_detail_to_sheet(scan_result, spreadsheet, detail_tab, sa_file)
            append_summary_to_sheet(scan_result, spreadsheet, summary_tab, sa_file, image_url=img_url)
            mark_scan_synced(scan_id, db)
            synced_scans += 1
        except Exception as e:
            logger.warning("sync: Sheets sync failed for {}: {}", scan_id, e)

    if synced_scans:
        logger.success("sync: synced {} scans to Sheets", synced_scans)
    return {"synced_scans": synced_scans}


async def _sync_pending() -> dict[str, int]:
    """Upload any locally-saved scans that haven't reached Google yet."""
    return await asyncio.to_thread(_sync_pending_sync)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    init_db(_DB_PATH)
    asyncio.create_task(asyncio.to_thread(_load_yolo))   # pre-warm; eliminates ~1.2s on first /analyze
    asyncio.create_task(_sync_pending())
    yield


app = FastAPI(title="Urine Color Analyzer V3", lifespan=_lifespan)

app.mount(
    "/static",
    StaticFiles(directory=str(_HERE / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(_HERE / "templates"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_scan(scan_id: str) -> dict | None:
    p = _RESULTS_DIR / f"{scan_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _all_scans_sorted() -> list[dict]:
    """Return all scan result dicts sorted newest-first."""
    scans = []
    for f in _RESULTS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            scans.append(data)
        except Exception:
            pass
    scans.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return scans


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "upload.html")


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse(request, "upload.html")


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    scans = await asyncio.to_thread(_all_scans_sorted)
    return templates.TemplateResponse(request, "results.html", {"scans": scans})


@app.get("/result/{scan_id}", response_class=HTMLResponse)
async def result_detail_page(request: Request, scan_id: str):
    scan = await asyncio.to_thread(_load_scan, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    return templates.TemplateResponse(request, "result.html", {"scan": scan})


@app.get("/auto_grid", response_class=HTMLResponse)
async def auto_grid_page(request: Request):
    return templates.TemplateResponse(request, "auto_grid.html", {"active_page": "auto_grid"})


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    slot_cfg = _get_slot_cfg()
    return templates.TemplateResponse(
        request, "settings.html",
        {"rows": slot_cfg.rows, "cols": slot_cfg.cols},
    )


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/sync")
async def api_sync():
    """Manually trigger sync of pending offline scans to Google Sheets."""
    pending = count_pending(_DB_PATH)
    result  = await _sync_pending()
    return {
        "status": "ok",
        "pending_before": pending,
        **result,
    }


@app.get("/api/slots")
async def get_slots():
    return JSONResponse(content=_get_slot_cfg().to_dict())


@app.post("/api/slots")
async def post_slots(request: Request):
    try:
        body = await request.json()
        rows = int(body.get("rows", 13))
        cols = int(body.get("cols", 15))
        from server.slot_config import CellConfig
        cells = {}
        for k, v in body.get("cells", {}).items():
            is_ref = bool(v.get("is_reference", False))
            is_wb  = bool(v.get("is_white_reference", False))
            if is_ref and is_wb:
                raise HTTPException(
                    status_code=400,
                    detail=f"cell {k}: is_reference and is_white_reference cannot both be true",
                )
            cells[int(k)] = CellConfig(
                slot_id=v["slot_id"],
                is_reference=is_ref,
                ref_level=v.get("ref_level"),
                is_white_reference=is_wb,
            )
        cfg = SlotConfig(rows=rows, cols=cols, cells=cells)
        save_slot_config(cfg)

        global _slot_cfg_cache
        _slot_cfg_cache = cfg   # update cache immediately

        # Mirror to Google Sheets (fire-and-forget)
        if _GOOGLE_AVAILABLE:
            gcfg = _google_cfg()
            sid = gcfg.get("spreadsheet_id", "")
            tab = gcfg.get("slots_tab", "SlotAssignment")
            sa_file = gcfg.get("service_account_file", "credentials.json")
            if sid:
                try:
                    write_slot_config_to_sheet(cfg, sid, tab, sa_file)
                except Exception as e:
                    logger.warning("slots: Google Sheets sync failed: {}", e)

        logger.info("slots: saved {} assigned cells", len(cells))
        return {"status": "ok", "cells": len(cells)}
    except Exception as e:
        logger.error("slots: save failed: {}", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auto_grid")
async def api_auto_grid(file: UploadFile = File(...)):
    """
    Classical-CV-only grid debug endpoint.

    Runs HoughCircles + grid reconstruction on the full-resolution image.
    No YOLO, no letterboxing — letterboxing is a YOLO requirement and distorts
    circle geometry at non-square aspect ratios.
    Slot config always read from local slot_config.json (web never reads Sheets).
    """
    jpeg_bytes = await file.read()
    if not jpeg_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    slot_cfg = _get_slot_cfg()
    if not slot_cfg.cells:
        raise HTTPException(
            status_code=400,
            detail="No slot config available. Visit /settings to assign slots first.",
        )

    # Decode image
    arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    roi = frame

    # Full-resolution classical CV: HoughCircles → KDE peaks → grid reconstruction
    # Run in thread — CPU-bound OpenCV work must not block the event loop
    try:
        grid_result = await asyncio.to_thread(detect_grid_full, roi, slot_cfg.rows, slot_cfg.cols)
    except Exception as e:
        logger.exception("auto_grid: detect_grid_full failed")
        raise HTTPException(status_code=500, detail=f"Grid detection error: {e}")

    grid_pts = grid_result["grid_pts"]           # (rows*cols, 2) in roi coords
    canvas   = grid_result["result_image"].copy()

    # Overlay slot labels at each assigned cell's grid position
    n_pts = len(grid_pts)
    for cell_idx, cell in sorted(slot_cfg.cells.items()):
        if cell_idx < 1 or cell_idx > n_pts:
            continue
        gx    = int(grid_pts[cell_idx - 1, 0])
        gy    = int(grid_pts[cell_idx - 1, 1])
        label = cell.slot_id
        color = (0, 220, 255) if cell.is_reference else (255, 255, 255)
        cv2.putText(canvas, label, (gx - 13, gy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (gx - 13, gy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    # Scale down 50% for browser display (full-res output is very large)
    h, w = canvas.shape[:2]
    canvas_display = cv2.resize(canvas, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    ok, enc = cv2.imencode(".jpg", canvas_display, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(enc.tobytes()).decode() if ok else ""

    logger.success(
        "auto_grid (full-res CV): detected={} reconstructed={} grid={}×{}",
        grid_result["detected_count"],
        grid_result["reconstructed_count"],
        slot_cfg.rows, slot_cfg.cols,
    )
    return JSONResponse(content={
        "image_b64":             img_b64,
        "detected_count":        grid_result["detected_count"],
        "reconstructed_count":   grid_result["reconstructed_count"],
        "total_cells":           slot_cfg.rows * slot_cfg.cols,
        "assigned_cells":        len(slot_cfg.cells),
        "grid_angle_deg":        round(grid_result.get("grid_angle_deg", 0.0), 2),
    })


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    x_auth_token: str | None = Header(default=None, alias="x-auth-token"),
):
    if x_auth_token != _API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    jpeg_bytes = await file.read()
    if not jpeg_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    slot_cfg = _get_slot_cfg()
    if not slot_cfg.cells:
        raise HTTPException(
            status_code=400,
            detail="No slots configured. Visit /settings to assign slots first.",
        )

    try:
        # Run CPU-bound pipeline in a thread — keeps the event loop free
        scan_result, annotated_jpeg = await asyncio.to_thread(
            run_pipeline, jpeg_bytes, slot_cfg
        )
    except Exception as e:
        logger.exception("analyze: pipeline error")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # Persist result files locally — run in a thread to avoid blocking the event loop
    scan_id    = scan_result["scan_id"]
    img_backup = _IMG_BACKUP / f"{scan_id}.jpg"

    def _save_local():
        (_RESULTS_DIR / f"{scan_id}.jpg").write_bytes(annotated_jpeg)
        (_RESULTS_DIR / f"{scan_id}.json").write_text(
            json.dumps(scan_result, ensure_ascii=False, indent=2)
        )
        img_backup.write_bytes(annotated_jpeg)
        save_scan(scan_result, _DB_PATH)

    await asyncio.to_thread(_save_local)

    # Sheets + Telegram — offloaded to a thread; response returns immediately
    asyncio.create_task(asyncio.to_thread(
        _cloud_and_notify, scan_id, scan_result, img_backup, _DB_PATH
    ))

    logger.success(
        "analyze: scan_id={} detected={}/{} missing={}",
        scan_id,
        scan_result["detected_count"],
        scan_result["total_assigned"],
        scan_result["missing_slots"],
    )
    return JSONResponse(content=scan_result)
