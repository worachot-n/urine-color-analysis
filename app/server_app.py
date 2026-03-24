"""
Server-side FastAPI application — Ubuntu / any Linux host.

Responsibilities:
  - Receive 4608 × 2592 JPEG from Pi via POST /analyze
  - Validate X-Auth-Token header
  - Run YOLO inference (PyTorch, singleton loaded at startup)
  - Perform slot-based color analysis if grid_config.json is present
  - Detect duplicate-slot and wrong-color-zone errors
  - Persist result to SQLite + save annotated JPEG to static/captures/
  - Return JSON: {status, count, color_summary, errors, timestamp, image_id}
  - Serve /dashboard (Jinja2), /settings (Jinja2), /health, and /static files

Install:
    uv sync --extra server --extra common

Run:
    uv run main.py --role server
"""

from __future__ import annotations

import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request, Header, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from sqlalchemy import create_engine, Column, Integer, String, DateTime, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.shared.config import cfg
from app.shared.processor import (
    letterbox_white_padding,
    scale_coordinates,
    find_slot_conflicts,
    validate_color_zones,
)


# ─── Database ────────────────────────────────────────────────────────────────

class _Base(DeclarativeBase):
    pass


class AnalysisResult(_Base):
    __tablename__ = "results"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    timestamp   = Column(DateTime, default=datetime.utcnow)
    count       = Column(Integer)
    color_json  = Column(String)             # JSON string of {color: count} summary
    image_path  = Column(String)             # Relative URL path served under /static/
    errors_json = Column(String, nullable=True)  # JSON: {duplicate_slots, wrong_color_slots}


def _make_engine():
    db_path = Path(cfg.database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})


_engine = _make_engine()
_Session = sessionmaker(bind=_engine, autoflush=False)


# ─── YOLO inference singleton ─────────────────────────────────────────────────

class _YoloInference:
    """
    Wraps a YOLOv8 PyTorch model (.pt).  Loaded once at app startup.

    detect(img_bytes) → (count, color_summary, annotated_bgr, errors)
    """

    def __init__(self):
        from ultralytics import YOLO

        model_path = cfg.model_path
        logger.info("Loading YOLO model: {}", model_path)
        self._model = YOLO(model_path, task="detect")
        self._input_size = cfg.model_input_size
        logger.success("YOLO model loaded — {}", model_path)

    # ------------------------------------------------------------------

    def detect(self, img_bytes: bytes) -> tuple[int, dict, np.ndarray, dict]:
        """
        Run inference on raw JPEG bytes.

        Returns:
            count          — number of unique slots occupied (or raw box count if no grid)
            color_summary  — {f"L{i}": count} per level (zeros if no grid calibration)
            annotated      — BGR image with bounding-box overlays
            errors         — {duplicate_slots: [...], wrong_color_slots: [...]}
        """
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None — invalid image data")

        padded, scale, pad_x, pad_y = letterbox_white_padding(img, self._input_size)

        results = self._model.predict(
            source=padded,
            imgsz=self._input_size,
            conf=cfg.model_conf,
            iou=cfg.model_iou,
            device="cpu",
            verbose=False,
        )

        boxes_640: list[list[float]] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes_640.append([x1, y1, x2, y2, float(box.conf[0]), int(box.cls[0])])

        boxes_orig = scale_coordinates(boxes_640, scale, pad_x, pad_y)

        # ── Slot-based color analysis (requires grid_config.json) ─────────
        color_summary: dict[str, int] = {f"L{i}": 0 for i in range(5)}
        sample_hits:      dict[str, dict] = {}
        classified_levels: dict[str, int]  = {}
        errors: dict = {}

        try:
            from utils.grid import GridConfig
            grid_cfg = GridConfig()
            if grid_cfg.slot_data:
                color_summary, sample_hits, classified_levels = _run_color_analysis(
                    img, boxes_orig, grid_cfg
                )
                count = len(sample_hits)
                errors = {
                    "duplicate_slots":  find_slot_conflicts(sample_hits),
                    "wrong_color_slots": validate_color_zones(sample_hits, classified_levels),
                }
            else:
                count = len(boxes_orig)
        except Exception as exc:
            logger.warning("Color analysis / slot mapping skipped: {}", exc)
            count = len(boxes_orig)

        # ── Draw bounding boxes on a copy of the original image ───────────
        annotated = img.copy()
        for box in boxes_orig:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            conf = box[4]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 3)
            cv2.putText(
                annotated, f"{conf:.2f}",
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0), 2, cv2.LINE_AA,
            )

        return count, color_summary, annotated, errors


def _run_color_analysis(
    img: np.ndarray,
    boxes_orig: list,
    grid_cfg,
) -> tuple[dict[str, int], dict[str, dict], dict[str, int]]:
    """
    Map detected boxes onto grid slots, classify colors using K-means
    nearest-centroid assignment in CIE Lab space.

    Centroid priority
    -----------------
    1. color.json  — 5 pre-calibrated centroids, each = mean Lab of the 3
                     reference bottles for that class (build_kmeans_centroids).
    2. Live frame  — dynamically sample the reference row in the current image
                     (build_reference_baseline), used when color.json is absent.
    3. Fallback    — return empty dicts if neither source yields a baseline.

    Returns:
        summary           — {"L0"…"L4": count} per color level
        sample_hits       — {slot_id: {"cx": int, "cy": int}} for each assigned box
        classified_levels — {slot_id: level_int} for color-zone validation
    """
    from utils.color_analysis import (
        build_kmeans_centroids,
        build_reference_baseline,
        classify_sample,
        extract_bottle_color,
    )
    from configs.config import COLOR_JSON_FILE

    summary:           dict[str, int]  = {f"L{i}": 0 for i in range(5)}
    sample_hits:       dict[str, dict] = {}
    classified_levels: dict[str, int]  = {}

    # ── Step 1: build K-means centroids ──────────────────────────────────
    # Primary: load from color.json (3 reference bottles → mean per class)
    baseline = build_kmeans_centroids(COLOR_JSON_FILE)

    if not baseline:
        # Fallback: sample reference row live from this frame
        logger.debug("color.json centroids unavailable — falling back to live reference extraction")
        ref_positions = grid_cfg.get_reference_positions()
        if ref_positions:
            baseline = build_reference_baseline(img, ref_positions)

    if not baseline:
        logger.warning("No color baseline available — skipping color classification")
        return summary, sample_hits, classified_levels

    logger.debug("Color baseline loaded: {} levels", len(baseline))

    # ── Step 2: assign each detected box to a slot + classify ────────────
    for box in boxes_orig:
        cx     = int((box[0] + box[2]) / 2)
        cy     = int((box[1] + box[3]) / 2)
        radius = int((box[2] - box[0]) / 2)

        slot_id, _ = grid_cfg.find_slot_for_circle(cx, cy, radius)
        if slot_id is None or slot_id in grid_cfg.reference_slots:
            continue

        # Deduplicate: keep first (highest-confidence YOLO box) per slot
        if slot_id in sample_hits:
            continue

        sample_hits[slot_id] = {"cx": cx, "cy": cy}

        # ── Step 3: extract median Lab from bottle center (inner crop) ────
        lab = extract_bottle_color(img, cx, cy, radius)
        if lab is None:
            continue

        # ── Step 4: nearest-centroid assignment (≡ K-means assign step) ──
        level, delta_e, confident = classify_sample(lab, baseline)
        if level is not None:
            classified_levels[slot_id] = level
            summary[f"L{level}"] = summary.get(f"L{level}", 0) + 1
            logger.debug(
                "Slot {} → L{} (ΔE={:.1f}, confident={})",
                slot_id, level, delta_e, confident,
            )

    return summary, sample_hits, classified_levels


# ─── FastAPI application ──────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────
    _Base.metadata.create_all(_engine)
    # Migrate existing DBs that pre-date the errors_json column
    with _engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE results ADD COLUMN errors_json TEXT"))
            conn.commit()
        except Exception:
            pass  # column already exists
    Path(cfg.captures_dir).mkdir(parents=True, exist_ok=True)
    app.state.yolo = _YoloInference()
    logger.success("Server startup complete — listening on {}:{}", cfg.server_host, cfg.server_port)
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Server shutting down")


app = FastAPI(title="Urine Analysis API", version="0.3.0", lifespan=_lifespan)

# Static files (captured annotated images)
_static_dir = Path(cfg.captures_dir)
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir.parent.parent)), name="static")

# Jinja2 templates
_templates_dir = Path(__file__).parent / "web" / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))
templates.env.filters["fromjson"] = json.loads


# ─── Auth dependency ──────────────────────────────────────────────────────────

async def _require_auth(x_auth_token: str = Header(..., alias="X-Auth-Token")):
    if x_auth_token != cfg.api_key:
        logger.warning("Rejected request — invalid X-Auth-Token")
        raise HTTPException(status_code=401, detail="Invalid API key")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/analyze", dependencies=[Depends(_require_auth)])
async def analyze(file: UploadFile = File(...)):
    """
    Accept a JPEG from the Pi, run YOLO inference + slot-based analysis, persist, return JSON.
    """
    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        count, color_summary, annotated, errors = app.state.yolo.detect(img_bytes)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}")

    # Save annotated image
    image_id = str(uuid.uuid4())
    captures = Path(cfg.captures_dir)
    captures.mkdir(parents=True, exist_ok=True)
    img_filename = f"{image_id}.jpg"
    img_path = captures / img_filename
    cv2.imwrite(str(img_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    static_url = f"/static/captures/{img_filename}"

    # Persist to SQLite
    with _Session() as session:
        row = AnalysisResult(
            timestamp=datetime.utcnow(),
            count=count,
            color_json=json.dumps(color_summary),
            errors_json=json.dumps(errors),
            image_path=static_url,
        )
        session.add(row)
        session.commit()

    now = datetime.utcnow()
    logger.success(
        "Analysis done — count={}, colors={}, dups={}, wc={}, id={}",
        count,
        color_summary,
        len(errors.get("duplicate_slots", [])),
        len(errors.get("wrong_color_slots", [])),
        image_id,
    )

    # ── Telegram notification (non-blocking, non-critical) ────────────────
    try:
        from bot.telegram_bot import send_scan_report
        send_scan_report(
            count=count,
            color_summary=color_summary,
            image_path=img_path,
            timestamp=now,
        )
    except Exception as exc:
        logger.warning("Telegram notification failed: {}", exc)

    return {
        "status":               "success",
        "total_physical_count": count,   # total unique occupied slots (for TM1637)
        "count":                count,   # legacy alias
        "summary":              color_summary,  # per-level counts keyed L0–L4
        "color_summary":        color_summary,  # legacy alias
        "errors":               errors,
        "timestamp":            now.isoformat(),
        "image_id":             image_id,
    }


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    with _Session() as session:
        results = (
            session.query(AnalysisResult)
            .order_by(AnalysisResult.timestamp.desc())
            .limit(50)
            .all()
        )
        total_scans = session.execute(text("SELECT COUNT(*) FROM results")).scalar() or 0

    latest_colors: dict[str, int] = {f"L{i}": 0 for i in range(5)}
    latest_errors: dict           = {}
    latest_image:  str | None     = None
    thai_timestamp: str           = "—"

    if results:
        r = results[0]
        latest_image = r.image_path
        try:
            latest_colors = json.loads(r.color_json) if r.color_json else latest_colors
        except Exception:
            pass
        try:
            latest_errors = json.loads(r.errors_json) if r.errors_json else {}
        except Exception:
            pass
        ts = r.timestamp
        thai_year = ts.year + 543
        thai_timestamp = f"{ts.day}/{ts.month}/{thai_year} เวลา {ts.strftime('%H%M')}"

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "results":        results,
            "total_scans":    total_scans,
            "latest_image":   latest_image,
            "latest_colors":  latest_colors,   # {L0..L4: count}
            "latest_errors":  latest_errors,   # {duplicate_slots, wrong_color_slots}
            "thai_timestamp": thai_timestamp,
        },
    )


# ─── Trend analysis ───────────────────────────────────────────────────────────

@app.get("/trend", response_class=HTMLResponse)
async def trend_page(request: Request):
    return templates.TemplateResponse(request=request, name="trend.html", context={})


@app.get("/api/trend")
async def api_trend(from_date: str = None, to_date: str = None):
    """
    Return historical scan records as JSON for the trend chart.

    from_date / to_date: Thai Buddhist date string "D/M/YYYY" (e.g. "1/1/2569").
    """
    def _parse_thai(s: str) -> datetime | None:
        try:
            d, m, y = s.strip().split("/")
            return datetime(int(y) - 543, int(m), int(d))
        except Exception:
            return None

    with _Session() as session:
        query = session.query(AnalysisResult).order_by(AnalysisResult.timestamp.asc())

        if from_date:
            dt = _parse_thai(from_date)
            if dt:
                query = query.filter(AnalysisResult.timestamp >= dt)
        if to_date:
            dt = _parse_thai(to_date)
            if dt:
                # include the whole day
                query = query.filter(AnalysisResult.timestamp < dt + timedelta(days=1))

        rows = query.limit(500).all()

    records = []
    for r in rows:
        colors: dict = {}
        errors: dict = {}
        try:
            colors = json.loads(r.color_json) if r.color_json else {}
        except Exception:
            pass
        try:
            errors = json.loads(r.errors_json) if r.errors_json else {}
        except Exception:
            pass

        ts = r.timestamp
        thai_year = ts.year + 543
        records.append({
            "id":         r.id,
            "timestamp":  ts.isoformat(),
            "thai_date":  f"{ts.day}/{ts.month}/{thai_year}",
            "count":      r.count or 0,
            "colors":     colors,
            "n_errors":   (
                len(errors.get("duplicate_slots",  [])) +
                len(errors.get("wrong_color_slots", []))
            ),
        })

    return {"data": records}


# ─── Settings ─────────────────────────────────────────────────────────────────

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    grid_path  = Path("grid_config.json")
    color_path = Path("color.json")

    grid_date   = None
    color_date  = None
    color_data: dict = {"baseline": {}, "bottles": {}}

    if grid_path.exists():
        try:
            d = json.loads(grid_path.read_text())
            grid_date = d.get("system_metadata", {}).get("calibration_date", "unknown")
        except Exception:
            grid_date = "invalid"

    if color_path.exists():
        try:
            d = json.loads(color_path.read_text())
            color_date = d.get("calibration_date", "unknown")
            color_data = d
        except Exception:
            color_date = "invalid"

    # Latest capture image URL
    latest_capture: str | None = None
    captures_dir = Path(cfg.captures_dir)
    if captures_dir.exists():
        imgs = sorted(captures_dir.glob("*.jpg"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if imgs:
            latest_capture = f"/static/captures/{imgs[0].name}"

    # Thai timestamp for grid calibration date
    thai_grid_date = "ยังไม่ได้ตั้งค่า"
    if grid_date and grid_date not in ("unknown", "invalid"):
        try:
            from datetime import datetime as _dt
            gd = _dt.strptime(grid_date, "%Y-%m-%d")
            thai_grid_date = f"{gd.day}/{gd.month}/{gd.year + 543}"
        except Exception:
            thai_grid_date = grid_date

    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={
            "grid_date":      grid_date,
            "color_date":     color_date,
            "color_data":     color_data,
            "latest_capture": latest_capture,
            "thai_grid_date": thai_grid_date,
        },
    )


@app.get("/api/latest-capture")
async def api_latest_capture():
    """Return the URL of the most recent captured image."""
    captures_dir = Path(cfg.captures_dir)
    if captures_dir.exists():
        imgs = sorted(captures_dir.glob("*.jpg"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if imgs:
            return {"url": f"/static/captures/{imgs[0].name}"}
    return {"url": None}


@app.get("/api/settings/grid-corners")
async def api_grid_corners():
    """Return the 4 corner coordinates from the current grid_config.json."""
    try:
        d = json.loads(Path("grid_config.json").read_text())
        corners = d.get("system_metadata", {}).get("corners", None)
        calibration_date = d.get("system_metadata", {}).get("calibration_date", None)
        return {"corners": corners, "calibration_date": calibration_date}
    except Exception:
        return {"corners": None, "calibration_date": None}


@app.post("/settings/grid")
async def settings_grid(
    file: UploadFile = File(...),
    corners: str = Form(...),
):
    """
    Receive a calibration image + 4 corner coordinates, compute grid polygons,
    save grid_config.json.

    corners: JSON string of [[x,y], [x,y], [x,y], [x,y]] — TL, TR, BR, BL
             in original image pixel coordinates.
    """
    try:
        corner_list = json.loads(corners)
        if len(corner_list) != 4:
            raise ValueError("Exactly 4 corners required")

        from utils.calibration import _corners_to_grid_pts, compute_slot_polygons_from_grid, compute_sample_roi

        grid_pts = _corners_to_grid_pts(corner_list)
        reference_slots, slot_data = compute_slot_polygons_from_grid(grid_pts)
        sample_roi = compute_sample_roi(grid_pts)

        # Convert numpy arrays to serializable lists
        def _to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _to_list(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_list(i) for i in obj]
            return obj

        calibration_date = datetime.utcnow().strftime("%Y-%m-%d")
        grid_config = {
            "system_metadata": {
                "project_name":    "Urine Color Analysis",
                "grid_dimensions": "16x14 lines",
                "calibration_date": calibration_date,
                "corners":         corner_list,
                "grid_pts":        _to_list(grid_pts),
                "sample_roi":      sample_roi,
            },
            "reference_row": {
                "slots": _to_list(reference_slots),
            },
            "main_grid": {
                "slot_data": _to_list(slot_data),
            },
        }

        Path("grid_config.json").write_text(json.dumps(grid_config, indent=2))
        logger.success("grid_config.json updated via web UI — date={}", calibration_date)
        return {"status": "ok", "calibration_date": calibration_date}

    except Exception as exc:
        logger.exception("settings/grid failed")
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/settings/colors")
async def settings_colors(file: UploadFile = File(...)):
    """
    Receive an image containing the reference-row bottles, extract Lab colors
    for all 5 levels, save color.json.

    Requires grid_config.json to already exist (to know reference positions).
    """
    try:
        from utils.grid import GridConfig
        from utils.color_analysis import build_reference_baseline, extract_bottle_color

        grid_cfg = GridConfig()
        ref_positions = grid_cfg.get_reference_positions()
        if not ref_positions:
            raise ValueError("grid_config.json has no reference positions — calibrate grid first")

        img_bytes = await file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")

        baseline = build_reference_baseline(img, ref_positions)
        if not baseline:
            raise ValueError("Could not extract any reference colors from the image")

        def _lab_to_hex(lab_tuple):
            L, a, b = [int(round(v)) for v in lab_tuple]
            lab_pixel = np.array([[[L, a, b]]], dtype=np.uint8)
            bgr = cv2.cvtColor(lab_pixel, cv2.COLOR_Lab2BGR)[0][0]
            return "#{:02x}{:02x}{:02x}".format(int(bgr[2]), int(bgr[1]), int(bgr[0]))

        calibration_date = datetime.utcnow().strftime("%Y-%m-%d")
        color_data: dict = {"calibration_date": calibration_date, "baseline": {}}

        for level, lab_tuple in sorted(baseline.items()):
            color_data["baseline"][str(level)] = {
                "lab": list(lab_tuple),
                "hex": _lab_to_hex(lab_tuple),
            }

        Path("color.json").write_text(json.dumps(color_data, indent=2))
        logger.success("color.json updated via web UI — {} levels extracted", len(baseline))

        # Return swatches for preview in the browser
        swatches = [
            {"level": lvl, "hex": info["hex"], "lab": info["lab"]}
            for lvl, info in color_data["baseline"].items()
        ]
        return {"status": "ok", "calibration_date": calibration_date, "swatches": swatches}

    except Exception as exc:
        logger.exception("settings/colors failed")
        raise HTTPException(status_code=400, detail=str(exc))


# ─── Runner (called from main.py) ────────────────────────────────────────────

def run_server():
    import uvicorn
    uvicorn.run(
        "app.server_app:app",
        host=cfg.server_host,
        port=cfg.server_port,
        reload=False,
        log_level="info",
    )
