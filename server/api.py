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
    GET  /auto_grid         — Grid debug page (CV only, no YOLO)
    POST /api/auto_grid     — Run HoughCircles + grid reconstruction only
    GET  /health            — {"status": "ok"}
"""

from __future__ import annotations

import json
import os
from pathlib import Path

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
from server.pipeline import run_pipeline

_HERE        = Path(__file__).parent
_RESULTS_DIR = _HERE / "static" / "results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_API_KEY = os.environ.get("API_KEY", "test")

# ---------------------------------------------------------------------------
# Try importing Google integrations — silently disabled if not configured
# ---------------------------------------------------------------------------
try:
    from server.integrations.sheets import (
        write_slot_config_to_sheet,
        append_result_to_sheet,
        read_slot_config_from_sheet,
    )
    from server.integrations.drive import upload_image as drive_upload_image
    _GOOGLE_AVAILABLE = True
except ImportError:
    _GOOGLE_AVAILABLE = False

def _google_cfg():
    """Load Google config from config.toml; return empty dict if missing."""
    try:
        import tomllib
        cfg_path = Path(__file__).parent.parent / "configs" / "config.toml"
        with open(cfg_path, "rb") as f:
            return tomllib.load(f).get("google", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Urine Color Analyzer V3")

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
    scans = _all_scans_sorted()
    return templates.TemplateResponse(request, "results.html", {"scans": scans})


@app.get("/result/{scan_id}", response_class=HTMLResponse)
async def result_detail_page(request: Request, scan_id: str):
    scan = _load_scan(scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    return templates.TemplateResponse(request, "result.html", {"scan": scan})


@app.get("/auto_grid", response_class=HTMLResponse)
async def auto_grid_page(request: Request):
    return templates.TemplateResponse(request, "auto_grid.html", {"active_page": "auto_grid"})


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    slot_cfg = load_slot_config()
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


@app.get("/api/slots")
async def get_slots():
    cfg = load_slot_config()
    return JSONResponse(content=cfg.to_dict())


@app.post("/api/slots")
async def post_slots(request: Request):
    try:
        body = await request.json()
        rows = int(body.get("rows", 13))
        cols = int(body.get("cols", 15))
        from server.slot_config import CellConfig
        cells = {}
        for k, v in body.get("cells", {}).items():
            cells[int(k)] = CellConfig(
                slot_id=v["slot_id"],
                is_reference=bool(v["is_reference"]),
                ref_level=v.get("ref_level"),
            )
        cfg = SlotConfig(rows=rows, cols=cols, cells=cells)
        save_slot_config(cfg)

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

    Runs ONLY HoughCircles + grid reconstruction (no YOLO, no color analysis).
    Returns the annotated grid image from detect_grid() with slot labels overlaid,
    so the user can visually verify that circle detection and grid spacing are correct.
    Slot config (rows, cols, cell assignments) is loaded from Google Sheets.
    """
    import base64
    import tomllib
    import cv2
    import numpy as np
    from utils.grid_circle_detector import detect_grid
    from app.shared.processor import crop_sample_roi, letterbox_white_padding

    jpeg_bytes = await file.read()
    if not jpeg_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Load slot config from Google Sheets (provides rows, cols, slot assignments)
    slot_cfg = None
    if _GOOGLE_AVAILABLE:
        gcfg = _google_cfg()
        sid  = gcfg.get("spreadsheet_id", "")
        tab  = gcfg.get("slots_tab", "SlotAssignment")
        sa   = gcfg.get("service_account_file", "credentials.json")
        if sid:
            slot_cfg = read_slot_config_from_sheet(sid, tab, sa)

    if slot_cfg is None or not slot_cfg.cells:
        raise HTTPException(
            status_code=400,
            detail="Could not load slot config from Google Sheets. Check spreadsheet_id and service account in config.toml.",
        )

    # Decode image
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    # Same ROI crop + letterbox as the main pipeline so results are comparable
    _sroi = tomllib.load(open(Path(__file__).parent.parent / "configs" / "config.toml", "rb")).get("sample_roi", {})
    roi, _, _ = crop_sample_roi(
        frame,
        int(_sroi.get("top", 0)), int(_sroi.get("bottom", 0)),
        int(_sroi.get("left", 0)), int(_sroi.get("right", 0)),
    )
    padded, _scale, _pad_x, _pad_y = letterbox_white_padding(roi, 640)

    # ── Classical CV only: HoughCircles → spacing → RANSAC → reconstruct grid ──
    try:
        grid_result = detect_grid(padded, slot_cfg.rows, slot_cfg.cols)
    except Exception as e:
        logger.exception("auto_grid: detect_grid failed")
        raise HTTPException(status_code=500, detail=f"Grid detection error: {e}")

    grid_pts = grid_result["grid_pts"]              # (rows*cols, 2) in 640-space
    canvas   = grid_result["result_image"].copy()   # already has circles + midpoint grid lines

    # Overlay slot labels at each assigned cell's detected/reconstructed position
    n_pts = len(grid_pts)
    for cell_idx, cell in sorted(slot_cfg.cells.items()):
        if cell_idx < 1 or cell_idx > n_pts:
            continue
        gx = int(grid_pts[cell_idx - 1, 0])
        gy = int(grid_pts[cell_idx - 1, 1])
        label = cell.slot_id
        # Yellow for reference cells, white for sample cells
        color = (0, 220, 255) if cell.is_reference else (255, 255, 255)
        cv2.putText(canvas, label, (gx - 13, gy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (gx - 13, gy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    ok, enc = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(enc.tobytes()).decode() if ok else ""

    logger.success(
        "auto_grid (CV only): detected={} reconstructed={} grid={}×{}",
        grid_result["detected_count"],
        grid_result["reconstructed_count"],
        slot_cfg.rows, slot_cfg.cols,
    )
    return JSONResponse(content={
        "image_b64":        img_b64,
        "detected_count":   grid_result["detected_count"],
        "reconstructed_count": grid_result["reconstructed_count"],
        "total_cells":      slot_cfg.rows * slot_cfg.cols,
        "assigned_cells":   len(slot_cfg.cells),
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

    slot_cfg = load_slot_config()
    if not slot_cfg.cells:
        raise HTTPException(
            status_code=400,
            detail="No slots configured. Visit /settings to assign slots first.",
        )

    try:
        scan_result, annotated_jpeg = run_pipeline(jpeg_bytes, slot_cfg)
    except Exception as e:
        logger.exception("analyze: pipeline error")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # Persist result files
    scan_id = scan_result["scan_id"]
    (_RESULTS_DIR / f"{scan_id}.jpg").write_bytes(annotated_jpeg)
    (_RESULTS_DIR / f"{scan_id}.json").write_text(
        json.dumps(scan_result, ensure_ascii=False, indent=2)
    )

    # Google Drive + Sheets (fire-and-forget)
    if _GOOGLE_AVAILABLE:
        gcfg = _google_cfg()
        sa_file = gcfg.get("service_account_file", "credentials.json")

        folder_id = gcfg.get("drive_folder_id", "")
        if folder_id:
            try:
                drive_upload_image(
                    annotated_jpeg,
                    f"{scan_id}.jpg",
                    folder_id,
                    sa_file,
                )
            except Exception as e:
                logger.warning("analyze: Drive upload failed: {}", e)

        spreadsheet_id = gcfg.get("spreadsheet_id", "")
        results_tab = gcfg.get("results_tab", "Results")
        if spreadsheet_id:
            try:
                append_result_to_sheet(
                    scan_result, spreadsheet_id, results_tab, sa_file
                )
            except Exception as e:
                logger.warning("analyze: Sheets append failed: {}", e)

    logger.success(
        "analyze: scan_id={} detected={}/{} missing={}",
        scan_id,
        scan_result["detected_count"],
        scan_result["total_assigned"],
        scan_result["missing_slots"],
    )
    return JSONResponse(content=scan_result)
