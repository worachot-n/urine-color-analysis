"""
Server-side FastAPI application — Ubuntu / any Linux host.

Responsibilities:
  - Receive 4608 × 2592 JPEG from Pi via POST /analyze
  - Validate X-Auth-Token header
  - Run YOLO inference (OpenVINO, singleton loaded at startup)
  - Perform full color analysis if grid_config.json is present
  - Persist result to SQLite + save annotated JPEG to static/captures/
  - Return JSON: {status, count, color_summary, timestamp, image_id}
  - Serve /dashboard (Jinja2), /health, and /static files

Install:
    uv sync --extra server --extra common

Run:
    uv run main.py --role server
"""

from __future__ import annotations

import uuid
import json
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Header, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from sqlalchemy import create_engine, Column, Integer, String, DateTime, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.shared.config import cfg
from app.shared.processor import letterbox_white_padding, scale_coordinates


# ─── Database ────────────────────────────────────────────────────────────────

class _Base(DeclarativeBase):
    pass


class AnalysisResult(_Base):
    __tablename__ = "results"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    timestamp  = Column(DateTime, default=datetime.utcnow)
    count      = Column(Integer)
    color_json = Column(String)   # JSON string of {color: count} summary
    image_path = Column(String)   # Relative URL path served under /static/


def _make_engine():
    db_path = Path(cfg.database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})


_engine = _make_engine()
_Session = sessionmaker(bind=_engine, autoflush=False)


# ─── YOLO inference singleton ─────────────────────────────────────────────────

class _YoloInference:
    """
    Wraps a YOLOv8-OpenVINO model.  Loaded once at app startup.

    detect(img_bytes) → (count, color_summary, annotated_bgr)
    """

    def __init__(self):
        from ultralytics import YOLO

        model_path = cfg.model_path

        # Inject OpenVINO CACHE_DIR + LATENCY so compilation is cached to disk.
        _orig_core_init = None
        try:
            import openvino as ov
            _orig_core_init = ov.Core.__init__
            _cache = str(Path(model_path).resolve().parent.parent / "model_cache")
            Path(_cache).mkdir(parents=True, exist_ok=True)

            def _patched(self_core, *a, **kw):
                _orig_core_init(self_core, *a, **kw)
                try:
                    self_core.set_property("CPU", {
                        "CACHE_DIR": _cache,
                        "PERFORMANCE_HINT": "LATENCY",
                    })
                except Exception:
                    pass

            ov.Core.__init__ = _patched
            logger.info("OpenVINO CACHE_DIR={} injected", _cache)
        except ImportError:
            pass

        logger.info("Loading YOLO model: {}", model_path)
        self._model = YOLO(model_path, task="detect")
        self._input_size = cfg.model_input_size

        # Restore ov.Core.__init__
        if _orig_core_init is not None:
            try:
                import openvino as ov
                ov.Core.__init__ = _orig_core_init
            except ImportError:
                pass

        # Warmup — triggers OpenVINO JIT compilation and writes blob to cache
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model(dummy, imgsz=self._input_size, device="cpu", verbose=False)
        logger.success("YOLO model ready — warmup complete (device=cpu)")

    # ------------------------------------------------------------------

    def detect(self, img_bytes: bytes) -> tuple[int, dict, np.ndarray]:
        """
        Run inference on raw JPEG bytes.

        Returns:
            count          — number of bottles detected
            color_summary  — {color_label: count} (empty if no grid calibration)
            annotated      — BGR image with bounding-box overlays
        """
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None — invalid image data")

        padded, scale, pad_x, pad_y = letterbox_white_padding(img, self._input_size)

        results = self._model(
            padded,
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
        count = len(boxes_orig)

        # ── Color analysis (requires grid_config.json on the server) ──────
        color_summary: dict[str, int] = {}
        try:
            from utils.grid import GridConfig
            from utils.color_analysis import (
                build_reference_baseline,
                classify_sample,
                extract_bottle_color,
            )
            grid_cfg = GridConfig()
            if grid_cfg.slot_data:
                color_summary = _run_color_analysis(img, boxes_orig, grid_cfg)
        except Exception as exc:
            logger.warning("Color analysis skipped: {}", exc)

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

        return count, color_summary, annotated


def _run_color_analysis(img: np.ndarray, boxes_orig: list, grid_cfg) -> dict[str, int]:
    """
    Map detected boxes onto grid slots, compare to reference baseline,
    and return a {level_label: count} summary.
    """
    from utils.color_analysis import (
        build_reference_baseline,
        classify_sample,
        extract_bottle_color,
    )

    # Build reference baseline from reference-row slots
    ref_positions = grid_cfg.get_reference_positions()
    if not ref_positions:
        return {}

    baseline = build_reference_baseline(img, ref_positions, grid_cfg)
    if not baseline:
        return {}

    summary: dict[str, int] = {}
    for box in boxes_orig:
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        slot_id = grid_cfg.find_slot(cx, cy)
        if slot_id is None or slot_id in grid_cfg.reference_slots:
            continue
        color = extract_bottle_color(img, cx, cy, int((box[2] - box[0]) / 2))
        level = classify_sample(color, baseline)
        label = f"L{level}" if level is not None else "unknown"
        summary[label] = summary.get(label, 0) + 1

    return summary


# ─── FastAPI application ──────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────
    _Base.metadata.create_all(_engine)
    Path(cfg.captures_dir).mkdir(parents=True, exist_ok=True)
    app.state.yolo = _YoloInference()
    logger.success("Server startup complete — listening on {}:{}", cfg.server_host, cfg.server_port)
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Server shutting down")


app = FastAPI(title="Urine Analysis API", version="0.2.0", lifespan=_lifespan)

# Static files (captured annotated images)
_static_dir = Path(cfg.captures_dir)
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir.parent.parent)), name="static")

# Jinja2 templates
_templates_dir = Path(__file__).parent / "web" / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))


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
    Accept a JPEG from the Pi, run YOLO inference + color analysis, persist, return JSON.
    """
    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        count, color_summary, annotated = app.state.yolo.detect(img_bytes)
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
            image_path=static_url,
        )
        session.add(row)
        session.commit()

    now = datetime.utcnow()
    logger.success("Analysis done — count={}, colors={}, id={}", count, color_summary, image_id)

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
        "status": "success",
        "count": count,
        "color_summary": color_summary,
        "timestamp": now.isoformat(),
        "image_id": image_id,
    }


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

    latest_count = results[0].count if results else 0
    avg_count = (
        round(sum(r.count for r in results) / len(results), 1) if results else 0
    )
    latest_image = results[0].image_path if results else None

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "results": results,
            "total_scans": total_scans,
            "latest_count": latest_count,
            "avg_count": avg_count,
            "latest_image": latest_image,
            "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        },
    )


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
