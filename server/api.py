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
    GET  /auto_grid         — Grid debug page (CV only, no YOLO, no letterbox)
    POST /api/auto_grid     — Run full-res HoughCircles + KDE grid reconstruction
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

    Runs HoughCircles + grid reconstruction on the full-resolution ROI image.
    No YOLO, no letterboxing — letterboxing is a YOLO requirement and distorts
    circle geometry at non-square aspect ratios.
    Slot config loaded from Google Sheets.
    """
    import base64
    import tomllib
    import cv2
    import numpy as np
    from utils.auto_grid_detector import detect_grid_full
    from app.shared.processor import crop_sample_roi

    jpeg_bytes = await file.read()
    if not jpeg_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Grid dimensions come from the local slot_config.json (always correct rows/cols).
    # Cell assignments come from Google Sheets (user-edited via /settings → Save).
    local_cfg = load_slot_config()

    sheets_cfg = None
    if _GOOGLE_AVAILABLE:
        gcfg = _google_cfg()
        sid  = gcfg.get("spreadsheet_id", "")
        tab  = gcfg.get("slots_tab", "SlotAssignment")
        sa   = gcfg.get("service_account_file", "credentials.json")
        if sid:
            sheets_cfg = read_slot_config_from_sheet(sid, tab, sa)

    # Prefer Sheets assignments; fall back to local if Sheets unavailable
    if sheets_cfg is not None and sheets_cfg.cells:
        from server.slot_config import SlotConfig
        slot_cfg = SlotConfig(
            rows=local_cfg.rows,
            cols=local_cfg.cols,
            cells=sheets_cfg.cells,
        )
    elif local_cfg.cells:
        slot_cfg = local_cfg
        logger.warning("auto_grid: Google Sheets unavailable — using local slot_config.json")
    else:
        raise HTTPException(
            status_code=400,
            detail="No slot config available. Visit /settings to assign slots first.",
        )

    # Decode image
    arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    # ROI crop — NO letterbox (letterbox is YOLO-only).
    # Use top=0 so the reference row at the top of the tray is included in the
    # debug view. Normal pipeline uses top=220 to exclude it from sample analysis.
    _sroi = tomllib.load(
        open(Path(__file__).parent.parent / "configs" / "config.toml", "rb")
    ).get("sample_roi", {})
    roi, _, _ = crop_sample_roi(
        frame,
        0,
        int(_sroi.get("bottom", 0)),
        int(_sroi.get("left", 0)),
        int(_sroi.get("right", 0)),
    )

    # Full-resolution classical CV: HoughCircles → KDE peaks → grid reconstruction
    try:
        grid_result = detect_grid_full(roi, slot_cfg.rows, slot_cfg.cols)
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
