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
    from server.integrations.sheets import write_slot_config_to_sheet, append_result_to_sheet
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
            creds_file = gcfg.get("credentials_file", "client_secrets.json")
            token_file = gcfg.get("token_file", "token.json")
            if sid:
                try:
                    write_slot_config_to_sheet(cfg, sid, tab, creds_file, token_file)
                except Exception as e:
                    logger.warning("slots: Google Sheets sync failed: {}", e)

        logger.info("slots: saved {} assigned cells", len(cells))
        return {"status": "ok", "cells": len(cells)}
    except Exception as e:
        logger.error("slots: save failed: {}", e)
        raise HTTPException(status_code=400, detail=str(e))


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
        creds_file = gcfg.get("credentials_file", "client_secrets.json")
        token_file = gcfg.get("token_file", "token.json")

        folder_id = gcfg.get("drive_folder_id", "")
        if folder_id:
            try:
                drive_upload_image(
                    annotated_jpeg,
                    f"{scan_id}.jpg",
                    folder_id,
                    creds_file,
                    token_file,
                )
            except Exception as e:
                logger.warning("analyze: Drive upload failed: {}", e)

        spreadsheet_id = gcfg.get("spreadsheet_id", "")
        results_tab = gcfg.get("results_tab", "Results")
        if spreadsheet_id:
            try:
                append_result_to_sheet(
                    scan_result, spreadsheet_id, results_tab, creds_file, token_file
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
