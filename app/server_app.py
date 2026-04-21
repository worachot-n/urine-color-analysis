"""
Server-side FastAPI application — Ubuntu / any Linux host (V2).

Pipeline per scan:
  • Active tray selected via Tray Management UI
  • Grid loaded from trays.grid_json (calibrated once via /settings, bilinear interpolation)
  • YOLO bottle detection → slot matching (195 calibrated centres)
  • Live CIE Lab colour baseline from reference row of each capture
  • Wrong-colour-zone validation
  • Dashboard: interactive rows×cols grid

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
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, JSON, String,
    create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from app.shared.config import cfg
from app.shared.processor import (
    crop_sample_roi,
    letterbox_white_padding,
    scale_coordinates,
    save_visual_log,
    generate_visual_report,
)


# ─── Database models (PostgreSQL) ─────────────────────────────────────────────

class _Base(DeclarativeBase):
    pass


class Tray(_Base):
    """Physical plaswood plate registry — one row per physical tray."""
    __tablename__ = "trays"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    tray_name      = Column(String, nullable=True)           # human-readable display name
    is_active      = Column(Boolean, default=False, nullable=False)  # only one active at a time
    rows           = Column(Integer, default=13)
    cols           = Column(Integer, default=15)
    total_slots    = Column(Integer, default=195)
    dimension_info = Column(String, default="13x15")
    created_at     = Column(DateTime, default=datetime.utcnow)
    layout_json    = Column(JSON, nullable=True)   # {"1": "A01", "2": "A02", ...}
    grid_json      = Column(JSON, nullable=True)   # {calibration_date, corners, grid_pts, sample_centres, ref_centres, grid_spacing}

    sessions       = relationship("ScanSession", back_populates="tray")


class ScanSession(_Base):
    """One row per scan event — primary source for dashboard queries."""
    __tablename__ = "scan_sessions"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    tray_id              = Column(Integer, ForeignKey("trays.id"), nullable=False, index=True)
    scanned_at           = Column(DateTime, default=datetime.utcnow, index=True)
    image_raw_path       = Column(String, nullable=True)
    image_annotated_path = Column(String, nullable=True)
    color_0              = Column(Integer, default=0)
    color_1              = Column(Integer, default=0)
    color_2              = Column(Integer, default=0)
    color_3              = Column(Integer, default=0)
    color_4              = Column(Integer, default=0)
    error_count          = Column(Integer, default=0)
    is_clean             = Column(Boolean, default=True)

    tray                 = relationship("Tray", back_populates="sessions")
    slots                = relationship("TestSlot", back_populates="session")


class TestSlot(_Base):
    """195 rows per scan session — granular slot results for grid UI + identity linking."""
    __tablename__ = "test_slots"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    session_id     = Column(Integer, ForeignKey("scan_sessions.id"), nullable=False, index=True)
    position_index = Column(Integer, nullable=False)   # 1-195 row-major
    color_result   = Column(Integer, nullable=True)    # 0-4; None = empty slot
    is_error       = Column(Boolean, default=False)
    person_id      = Column(Integer, ForeignKey("people.id"), nullable=True)

    session        = relationship("ScanSession", back_populates="slots")
    person         = relationship("People", back_populates="slots")


class People(_Base):
    """Personnel registry — pre-built for future identity integration."""
    __tablename__ = "people"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    full_name    = Column(String, nullable=False)
    personnel_id = Column(String, unique=True, index=True)
    department   = Column(String, nullable=True)

    slots        = relationship("TestSlot", back_populates="person")


def _make_engine():
    db_url = cfg.database_url
    if db_url.startswith("sqlite"):
        db_path = Path(db_url.replace("sqlite:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(db_url, connect_args={"check_same_thread": False})
    return create_engine(db_url)


_engine  = _make_engine()
_Session = sessionmaker(bind=_engine, autoflush=False)


# ─── YOLO inference singleton ─────────────────────────────────────────────────

class _YoloInference:
    """Wraps a YOLOv8 PyTorch model. Loaded once at startup."""

    def __init__(self):
        from ultralytics import YOLO
        logger.info("Loading YOLO model: {}", cfg.model_path)
        self._model      = YOLO(cfg.model_path, task="detect")
        self._input_size = cfg.model_input_size
        logger.success("YOLO model loaded — {}", cfg.model_path)

    def detect_raw(self, img_bytes: bytes) -> "tuple[np.ndarray, list[list[float]]]":
        """
        Decode JPEG, letterbox full image to 640×640, run YOLO.
        Returns (full_image, boxes_in_full_image_space).
        No ROI crop — matches training pre-processing (full frame).
        boxes format: [[x1, y1, x2, y2, conf, cls], ...]

        Letterbox math:
            gain  = min(640/W, 640/H)   (uniform scale that fits the image)
            pad_x = (640 - W*gain) / 2  (horizontal white bar half-width)
            pad_y = (640 - H*gain) / 2  (vertical white bar half-height)
        Inverse:
            X_orig = (X_640 - pad_x) / gain
            Y_orig = (Y_640 - pad_y) / gain
        Boxes whose center lies inside the white padding zone are discarded as
        false positives before coordinate inversion.
        """
        nparr = np.frombuffer(img_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None — invalid image data")

        h_img, w_img = img.shape[:2]
        padded, scale, pad_x, pad_y = letterbox_white_padding(img, self._input_size)

        # Bounds of actual image content inside the 640×640 frame
        content_x1 = pad_x
        content_y1 = pad_y
        content_x2 = pad_x + int(round(w_img * scale))
        content_y2 = pad_y + int(round(h_img * scale))

        results = self._model.predict(
            source=padded,
            imgsz=self._input_size,
            conf=cfg.model_conf,
            iou=cfg.model_iou,
            max_det=cfg.model_max_det,
            device="cpu",
            verbose=False,
        )

        # Step 1 — collect boxes, drop only padding-zone false positives (640-space)
        boxes_640: list[list[float]] = []
        n_padding_rejected = 0
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx_640 = (x1 + x2) / 2
                cy_640 = (y1 + y2) / 2
                if not (content_x1 <= cx_640 <= content_x2 and
                        content_y1 <= cy_640 <= content_y2):
                    n_padding_rejected += 1
                    continue
                boxes_640.append([x1, y1, x2, y2, float(box.conf[0]), int(box.cls[0])])

        # Step 2 — invert letterbox transform → original image pixel space
        boxes_full = scale_coordinates(boxes_640, scale, pad_x, pad_y)

        # Step 3 — size filter in ORIGINAL image space (140–200 px per side).
        # Valid bottles must satisfy: box_min < width < box_max AND box_min < height < box_max.
        # Detections outside this range are noise (shadows, reflections, grid edges).
        box_min = cfg.model_box_min_px   # 140 px in original image
        box_max = cfg.model_box_max_px   # 200 px in original image
        boxes_orig: list[list[float]] = []
        n_size_rejected = 0
        for box in boxes_full:
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            if bw <= box_min or bh <= box_min or bw >= box_max or bh >= box_max:
                n_size_rejected += 1
                continue
            boxes_orig.append(box)

        logger.info(
            "YOLO raw={} pad_drop={} size_drop={} kept={} "
            "(image {}×{}, scale={:.4f}, pad_x={}, pad_y={}, "
            "size_filter={}–{}px original)",
            sum(len(r.boxes) for r in results),
            n_padding_rejected, n_size_rejected, len(boxes_orig),
            w_img, h_img, scale, pad_x, pad_y, box_min, box_max,
        )
        return img, boxes_orig


# ─── V2 analysis helpers ──────────────────────────────────────────────────────

def _normalize_layout_map(layout_json, cols: int = 15) -> dict[str, str]:
    """Convert layout_json (dict or 2D list) → {str(position_index): label}.

    The DB stores layout_json as a 2D list [row_idx][col_idx]; the annotation
    layer needs it as a flat dict keyed by position_index string.
    """
    if not layout_json:
        return {}
    if isinstance(layout_json, dict):
        return {str(k): str(v) for k, v in layout_json.items()}
    if isinstance(layout_json, list):
        result: dict[str, str] = {}
        for row_idx, row in enumerate(layout_json):
            if isinstance(row, list):
                for col_idx, label in enumerate(row):
                    pos = row_idx * cols + col_idx + 1
                    result[str(pos)] = str(label)
        return result
    return {}


def _grid_pts_to_result(grid_pts_list):
    """Convert [[14][17][2]] bilinear grid → GridDetectionResult (same shape as U-Net output)."""
    from utils.grid_detector import GridDetectionResult
    gp = np.array(grid_pts_list)   # shape (14, 17, 2)

    sample_centres = [
        (int(gp[hi][vi][0]), int(gp[hi][vi][1]))
        for hi in range(1, 14)
        for vi in range(0, 15)
    ]

    ref_centres = []
    for vi in range(0, 15):
        xs = [gp[r][c][0] for r in (0, 1) for c in (vi, vi + 1)]
        ys = [gp[r][c][1] for r in (0, 1) for c in (vi, vi + 1)]
        ref_centres.append((int(sum(xs) / 4), int(sum(ys) / 4)))

    if len(sample_centres) >= 2:
        row1 = np.array(sample_centres[:15], dtype=np.float32)
        diffs = np.linalg.norm(np.diff(row1, axis=0), axis=1)
        grid_spacing = float(np.median(diffs)) if len(diffs) else 50.0
    else:
        grid_spacing = 50.0

    return GridDetectionResult(sample_centres, ref_centres, grid_spacing)



def _expected_level_from_position(position_index: int) -> int:
    """
    Derive expected colour level (0-4) from position_index (1-195).

    Grid layout: 13 rows × 15 cols, 3 cols per level.
      cols 1-3  → L0, cols 4-6  → L1, cols 7-9  → L2,
      cols 10-12→ L3, cols 13-15→ L4
    """
    col_idx     = (position_index - 1) % 15          # 0-14
    return min(col_idx // 3, 4)


def _match_detections_to_slots(
    boxes_orig: list[list[float]],
    slot_centers: list[tuple[int, int]],
) -> dict[int, dict]:
    """
    Greedy nearest-slot matching.  Each YOLO box claims the closest slot center
    whose distance < half the minimum inter-slot spacing.

    Returns:
        {position_index: {"cx": int, "cy": int, "radius": int, "box": [x1,y1,x2,y2]}}
    """
    if not slot_centers or not boxes_orig:
        return {}

    centers = np.array(slot_centers, dtype=np.float32)   # (N, 2)

    # Compute matching threshold: half the minimum inter-slot spacing.
    # Use pairwise distances between ALL slot pairs (not just consecutive) to
    # find the true minimum spacing, which prevents cross-slot false claims.
    if len(centers) > 1:
        # Use only nearest-neighbor distances (cheap: sort each row, take col 1)
        from scipy.spatial import cKDTree
        tree = cKDTree(centers)
        nn_dists, _ = tree.query(centers, k=2)   # k=2: self + nearest neighbour
        min_spacing = float(np.median(nn_dists[:, 1]))
    else:
        min_spacing = 50.0
    threshold = min_spacing * 0.75   # allow up to 75% of nearest-slot spacing
    logger.info("Slot matching: {} boxes vs {} slots  threshold={:.0f}px",
                len(boxes_orig), len(slot_centers), threshold)

    claimed: set[int] = set()
    # Sort boxes by descending confidence so higher-confidence detections win ties
    sorted_boxes = sorted(boxes_orig, key=lambda b: -b[4])

    hits: dict[int, dict] = {}
    for box in sorted_boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        box_pt = np.array([[cx, cy]], dtype=np.float32)

        dists_to_centers = np.linalg.norm(centers - box_pt, axis=1)
        nearest_idx      = int(np.argmin(dists_to_centers))
        min_dist         = float(dists_to_centers[nearest_idx])

        if min_dist > threshold:
            logger.debug("Box [{},{},{},{}] dropped — nearest slot dist={:.0f}px > {:.0f}px",
                         x1, y1, x2, y2, min_dist, threshold)
            continue
        if nearest_idx in claimed:
            logger.debug("Box [{},{},{},{}] dropped — slot {} already claimed",
                         x1, y1, x2, y2, nearest_idx + 1)
            continue

        claimed.add(nearest_idx)
        position_index = nearest_idx + 1
        radius         = max((x2 - x1) // 2, (y2 - y1) // 2)
        hits[position_index] = {
            "cx": cx, "cy": cy, "radius": radius,
            "box": [x1, y1, x2, y2],
        }

    logger.info("Slot matching: {} / {} boxes matched to slots",
                len(hits), len(boxes_orig))
    return hits


def _run_color_analysis_v2(
    img: np.ndarray,
    slot_hits: dict[int, dict],
    baseline: dict[int, tuple[float, float, float]] | None = None,
) -> dict[int, int]:
    """
    Classify the colour of each occupied slot.

    Returns:
        {position_index: level_int}
    """
    from utils.color_analysis import (
        build_kmeans_centroids,
        classify_sample,
        extract_bottle_color,
    )

    if baseline is None:
        from configs.config import COLOR_JSON_FILE
        baseline = build_kmeans_centroids(COLOR_JSON_FILE)

    if not baseline:
        logger.warning("No colour baseline available — skipping colour classification")
        return {}

    classified: dict[int, int] = {}
    for pos_idx, hit in slot_hits.items():
        lab = extract_bottle_color(img, hit["cx"], hit["cy"], hit["radius"])
        if lab is None:
            continue
        level, delta_e, confident = classify_sample(lab, baseline)
        if level is not None:
            classified[pos_idx] = level
            logger.debug("Slot {} → L{} (ΔE={:.1f}, confident={})",
                         pos_idx, level, delta_e, confident)

    return classified


def _build_ref_positions(
    ref_centres: list[tuple[int, int]],
    grid_spacing: float,
) -> dict[int, list[tuple[int, int, int]]]:
    """Map 15 ref centres → {level: [(cx, cy, radius), ...]} for build_reference_baseline.

    Centres are ordered left→right (cols 1-15); 3 per level:
        i=0,1,2 → level 0 … i=12,13,14 → level 4
    """
    radius = max(1, int(grid_spacing / 2))
    positions: dict[int, list[tuple[int, int, int]]] = {}
    for i, (cx, cy) in enumerate(ref_centres):
        positions.setdefault(i // 3, []).append((cx, cy, radius))
    return positions


def _build_slot_rows(
    slot_hits: dict[int, dict],
    classified: dict[int, int],
) -> list[dict]:
    """
    Build the 195-element list of slot dicts for DB + JSON response.

    Each dict: {position_index, color_result, is_error}
    """
    rows: list[dict] = []
    for pos_idx in range(1, 196):
        if pos_idx not in slot_hits:
            rows.append({"position_index": pos_idx, "color_result": None, "is_error": False})
            continue

        level    = classified.get(pos_idx)
        expected = _expected_level_from_position(pos_idx)
        is_error = (level is not None) and (level != expected)
        rows.append({
            "position_index": pos_idx,
            "color_result":   level,
            "is_error":       is_error,
        })
    return rows


# ─── Telegram helpers ─────────────────────────────────────────────────────────

def _build_thai_caption(
    now: datetime,
    tray_name: str,
    count: int,
    color_counts: dict[int, int],
    error_count: int,
    server_url: str,
) -> str:
    thai_year = now.year + 543
    date_str  = f"{now.day}/{now.month}/{thai_year}"
    time_str  = now.strftime("%H:%M:%S")

    lines = [
        "📊 รายงานการตรวจสีปัสสาวะ (V2)",
        f"🗓 วันที่: {date_str} | ⏱ เวลา: {time_str}",
        f"🔖 ถาด: {tray_name}",
        "",
        f"✅ ตรวจพบขวดทั้งหมด: {count} ขวด",
        "🎨 สรุปผลตามกลุ่มสี:",
    ]
    for i in range(5):
        lines.append(f"  • สี {i}: {color_counts.get(i, 0)} นาย")

    if error_count:
        lines.append(f"\n⚠️ พบข้อผิดพลาด: {error_count} ช่อง")

    dashboard_url = server_url.rstrip("/") + "/dashboard"
    lines.extend(["", f"🔗 ดูรายละเอียดเพิ่มเติม: {dashboard_url}"])
    return "\n".join(lines)


def _send_telegram_report(
    img_bytes: bytes,
    caption: str,
    token: str,
    chat_id: str,
) -> None:
    import requests as _req
    url  = f"https://api.telegram.org/bot{token}/sendPhoto"
    resp = _req.post(
        url,
        data={"chat_id": chat_id, "caption": caption},
        files={"photo": ("visual_report.jpg", img_bytes, "image/jpeg")},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Telegram API returned {resp.status_code}: {resp.text[:300]}")
    logger.debug("Telegram report sent to chat_id={}", chat_id)


# ─── Maintenance helpers ──────────────────────────────────────────────────────

def _run_migrations() -> None:
    """Idempotent ALTER TABLE migrations — safe on repeated startup."""
    is_pg    = not cfg.database_url.startswith("sqlite")
    json_t   = "JSONB" if is_pg else "TEXT"
    bool_t   = "BOOLEAN DEFAULT FALSE" if is_pg else "INTEGER DEFAULT 0"

    migrations = [
        ("trays", "layout_json",    json_t),
        ("trays", "grid_json",      json_t),
        ("trays", "tray_name",      "VARCHAR"),
        ("trays", "is_active",      bool_t),
        ("trays", "rows",           "INTEGER DEFAULT 13"),
        ("trays", "cols",           "INTEGER DEFAULT 15"),
    ]
    for table, col, col_type in migrations:
        try:
            with _engine.connect() as conn:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))
                conn.commit()
                logger.info("Migration: added {}.{}", table, col)
        except Exception:
            pass   # column already exists


def _generate_default_layout(rows: int, cols: int) -> dict[str, str]:
    """Build default {"1": "A01", ...} mapping for a rows×cols tray."""
    return {
        str((r - 1) * cols + c): f"{chr(64 + r)}{c:02d}"
        for r in range(1, rows + 1)
        for c in range(1, cols + 1)
    }


def _cleanup_old_logs(log_dir: Path, max_age_days: int = 30) -> None:
    if not log_dir.exists():
        return
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    deleted = 0
    for jpg in log_dir.glob("URINE_SCAN_*.jpg"):
        try:
            mtime = datetime.utcfromtimestamp(jpg.stat().st_mtime)
            if mtime < cutoff:
                jpg.unlink(missing_ok=True)
                deleted += 1
        except Exception:
            pass
    if deleted:
        logger.info("Cleanup: removed {} visual log(s) older than {} days", deleted, max_age_days)


# ─── FastAPI lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Startup
    _Base.metadata.create_all(_engine)
    _run_migrations()
    Path(cfg.captures_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.visual_log_dir).mkdir(parents=True, exist_ok=True)
    _cleanup_old_logs(Path(cfg.visual_log_dir), max_age_days=30)
    app.state.yolo = _YoloInference()
    logger.success("Server V2 startup complete — {}:{}", cfg.server_host, cfg.server_port)
    yield
    logger.info("Server shutting down")


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Urine Analysis API", version="2.0.0", lifespan=_lifespan)

_static_dir = Path(cfg.captures_dir)
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir.parent)), name="static")

_assets_dir = Path(__file__).resolve().parent.parent / "assets"
if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")

_templates_dir = Path(__file__).parent / "web" / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))
templates.env.filters["fromjson"] = json.loads


# ─── Auth ─────────────────────────────────────────────────────────────────────

async def _require_auth(x_auth_token: str = Header(..., alias="X-Auth-Token")):
    if x_auth_token != cfg.api_key:
        logger.warning("Rejected request — invalid X-Auth-Token")
        raise HTTPException(status_code=401, detail="Invalid API key")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


@app.post("/analyze", dependencies=[Depends(_require_auth)])
async def analyze(file: UploadFile = File(...)):
    """
    V2 analysis pipeline (active-tray driven):
      Requires an active tray selected via the Tray Management UI.
      Phase B/C: Load calibrated grid from DB (manual corner calibration)
      Phase D: YOLO → slot matching → live colour baseline from reference row
      Phase E: wrong-colour-zone validation
      DB: insert ScanSession + TestSlot rows linked to the active tray
    """
    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Require an active tray
    with _Session() as session:
        active_tray = session.query(Tray).filter_by(is_active=True).first()
        if active_tray is None:
            raise HTTPException(
                status_code=400,
                detail="กรุณาเลือกถาดก่อนสแกน (No active tray selected)",
            )
        active_tray_id = active_tray.id
        tray_grid_json = active_tray.grid_json
        layout_map: dict[str, str] = _normalize_layout_map(
            active_tray.layout_json, cols=active_tray.cols or 15
        )
        tray_name = active_tray.tray_name or f"Tray-{active_tray.id}"

    if not tray_grid_json:
        raise HTTPException(
            status_code=400,
            detail="กรุณาปรับเทียบกริดก่อนสแกน (Grid not calibrated for this tray)",
        )

    now      = datetime.utcnow()
    image_id = str(uuid.uuid4())
    ts_str   = now.strftime("%Y%m%d_%H%M%S")

    captures   = Path(cfg.captures_dir)
    captures.mkdir(parents=True, exist_ok=True)
    thumb_path = captures / f"{image_id}.jpg"
    static_url = f"/static/captures/{image_id}.jpg"

    log_dir  = Path(cfg.visual_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"URINE_SCAN_{ts_str}.jpg"

    try:
        nparr    = np.frombuffer(img_bytes, np.uint8)
        img_full = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_full is None:
            raise ValueError("Invalid image data")

        # Load grid from DB calibration
        grid_result  = _grid_pts_to_result(tray_grid_json["grid_pts"])

        slot_centers = grid_result.sample_centres

        # Build live colour baseline from reference row
        from utils.color_analysis import build_reference_baseline
        ref_positions = _build_ref_positions(grid_result.ref_centres, grid_result.grid_spacing)
        live_baseline = build_reference_baseline(img_full, ref_positions) if ref_positions else {}
        if not live_baseline:
            logger.warning("/analyze: reference row extraction failed — falling back to color.json")

        # Phase D — YOLO raw detection
        _, boxes_orig = app.state.yolo.detect_raw(img_bytes)
        slot_hits     = _match_detections_to_slots(boxes_orig, slot_centers)
        classified    = _run_color_analysis_v2(img_full, slot_hits, baseline=live_baseline or None)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("V2 inference failed")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}")

    # Phase E — wrong-colour-zone validation (per position_index)
    slot_rows   = _build_slot_rows(slot_hits, classified)
    error_slots = [r for r in slot_rows if r["is_error"]]
    count       = len(slot_hits)
    error_count = len(error_slots)

    color_counts: dict[int, int] = {i: 0 for i in range(5)}
    for lvl in classified.values():
        if lvl is not None:
            color_counts[lvl] = color_counts.get(lvl, 0) + 1

    # Build validation_results for annotation (error slots get labels)
    validation_results_compat: dict = {
        "slots": {
            str(r["position_index"]): {
                "cx":          slot_hits[r["position_index"]]["cx"] if r["position_index"] in slot_hits else 0,
                "cy":          slot_hits[r["position_index"]]["cy"] if r["position_index"] in slot_hits else 0,
                "radius":      slot_hits[r["position_index"]]["radius"] if r["position_index"] in slot_hits else 30,
                "level":       r["color_result"],
                "ok":          not r["is_error"],
                "wrong_color": r["is_error"],
                "duplicate":   False,
            }
            for r in slot_rows if r["color_result"] is not None
        },
        "duplicate_slots":     [],
        "wrong_color_slots":   [
            {"slot_id":    str(r["position_index"]),
             "slot_label": layout_map.get(str(r["position_index"]),
                           f"R{(r['position_index']-1)//15+1:02d}C{(r['position_index']-1)%15+1:02d}"),
             "expected":   _expected_level_from_position(r["position_index"]),
             "actual":     r["color_result"]}
            for r in error_slots
        ],
        "raw_detection_count": len(boxes_orig),
    }

    # Save raw image to captures_dir — used by /api/latest-capture for grid calibration.
    cv2.imwrite(str(thumb_path), img_full, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # Annotated image — calibrated bilinear grid + detections (dashboard + Telegram)
    from app.shared.processor import _render_annotated_canvas
    _raw_grid_pts = tray_grid_json.get("grid_pts")
    annotated_canvas = _render_annotated_canvas(
        img_full, validation_results_compat, None, slot_centers, layout_map,
        grid_pts=_raw_grid_pts,
    )
    annotated_path = captures / f"{image_id}_annotated.jpg"
    annotated_url  = f"/static/captures/{image_id}_annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated_canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # Visual log (best-effort)
    try:
        save_visual_log(
            image=img_full,
            validation_results=validation_results_compat,
            grid_cfg=None,
            filepath=log_path,
            slot_centers=slot_centers,
            layout_map=layout_map,
            grid_pts=_raw_grid_pts,
        )
    except Exception as exc:
        logger.warning("save_visual_log failed (non-critical): {}", exc)

    # DB write — ScanSession + 195 TestSlots linked to active tray
    with _Session() as session:
        scan = ScanSession(
            tray_id=active_tray_id,
            scanned_at=now,
            image_raw_path=static_url,
            image_annotated_path=str(annotated_path),
            color_0=color_counts[0],
            color_1=color_counts[1],
            color_2=color_counts[2],
            color_3=color_counts[3],
            color_4=color_counts[4],
            error_count=error_count,
            is_clean=(error_count == 0),
        )
        session.add(scan)
        session.flush()

        session.bulk_insert_mappings(TestSlot, [
            {
                "session_id":     scan.id,
                "position_index": r["position_index"],
                "color_result":   r["color_result"],
                "is_error":       r["is_error"],
                "person_id":      None,
            }
            for r in slot_rows
        ])
        session.commit()
        session_id = scan.id

    logger.success(
        "Analysis done — tray_id={}, count={}, errors={}, session_id={}",
        active_tray_id, count, error_count, session_id,
    )

    # Telegram (non-blocking, non-critical)
    if cfg.telegram_token and cfg.telegram_chat_id:
        try:
            caption = _build_thai_caption(
                now=now, tray_name=tray_name, count=count,
                color_counts=color_counts, error_count=error_count,
                server_url=cfg.server_url,
            )
            report_bytes = (annotated_path.read_bytes()
                            if annotated_path.exists()
                            else (log_path.read_bytes() if log_path.exists() else None))
            if report_bytes:
                _send_telegram_report(
                    img_bytes=report_bytes,
                    caption=caption,
                    token=cfg.telegram_token,
                    chat_id=cfg.telegram_chat_id,
                )
        except Exception as exc:
            logger.warning("Telegram report failed (non-critical): {}", exc)

    # Build error coordinate strings for LCD client
    error_coords = [
        "R{:02d}C{:02d}".format(
            (r["position_index"] - 1) // 15 + 1,
            (r["position_index"] - 1) % 15  + 1,
        )
        for r in error_slots
    ]

    return {
        "status":       "success",
        "session_id":   session_id,
        "tray_id":      active_tray_id,
        "tray_name":    tray_name,
        "count":        count,
        "total_physical_count": count,
        "summary": {f"L{i}": color_counts[i] for i in range(5)},
        "is_clean":      error_count == 0,
        "error_count":   error_count,
        "errors": {
            "duplicate_slots":  [],
            "wrong_color_slots": error_coords,
        },
        "slots":         slot_rows,
        "timestamp":     now.isoformat(),
        "image_id":      image_id,
    }


# ─── Dashboard ────────────────────────────────────────────────────────────────

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    with _Session() as session:
        latest_session = (
            session.query(ScanSession)
            .order_by(ScanSession.scanned_at.desc())
            .first()
        )
        total_scans  = session.query(ScanSession).count()
        active_tray  = session.query(Tray).filter_by(is_active=True).first()

    latest_colors          = {f"L{i}": 0 for i in range(5)}
    latest_image: str | None = None
    thai_timestamp           = "—"
    active_tray_name         = active_tray.tray_name if active_tray else None
    active_tray_id_ctx       = active_tray.id if active_tray else None

    if latest_session:
        latest_image  = latest_session.image_raw_path
        latest_colors = {
            "L0": latest_session.color_0,
            "L1": latest_session.color_1,
            "L2": latest_session.color_2,
            "L3": latest_session.color_3,
            "L4": latest_session.color_4,
        }
        ts = latest_session.scanned_at
        thai_year      = ts.year + 543
        thai_timestamp = f"{ts.day}/{ts.month}/{thai_year} เวลา {ts.strftime('%H%M')}"

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "total_scans":      total_scans,
            "latest_image":     latest_image,
            "latest_colors":    latest_colors,
            "thai_timestamp":   thai_timestamp,
            "active_tray_name": active_tray_name,
            "active_tray_id":   active_tray_id_ctx,
            "session_id":       latest_session.id if latest_session else None,
        },
    )


@app.get("/api/latest")
async def api_latest():
    """
    Return the most recent scan session with all 195 slot results.
    Used by the interactive dashboard grid.
    """
    with _Session() as session:
        scan = (
            session.query(ScanSession)
            .order_by(ScanSession.scanned_at.desc())
            .first()
        )
        if scan is None:
            return {"session": None, "tray": None, "slots": []}

        tray_data = {
            "id":        scan.tray.id,
            "tray_name": scan.tray.tray_name,
            "is_active": scan.tray.is_active,
            "rows":      scan.tray.rows or 13,
            "cols":      scan.tray.cols or 15,
        } if scan.tray else None

        session_data = {
            "id":         scan.id,
            "scanned_at": scan.scanned_at.isoformat(),
            "color_0":    scan.color_0,
            "color_1":    scan.color_1,
            "color_2":    scan.color_2,
            "color_3":    scan.color_3,
            "color_4":    scan.color_4,
            "error_count": scan.error_count,
            "is_clean":   scan.is_clean,
        }

        slots_data = [
            {
                "position_index": s.position_index,
                "color_result":   s.color_result,
                "is_error":       s.is_error,
            }
            for s in sorted(scan.slots, key=lambda x: x.position_index)
        ]

    return {"session": session_data, "tray": tray_data, "slots": slots_data}


# ─── Tray management page ─────────────────────────────────────────────────────

@app.get("/trays", response_class=HTMLResponse)
async def trays_page(request: Request):
    with _Session() as session:
        trays = session.query(Tray).order_by(Tray.created_at.desc()).all()
        tray_list = [
            {
                "id":        t.id,
                "tray_name": t.tray_name or f"Tray-{t.id}",
                "rows":      t.rows or 13,
                "cols":      t.cols or 15,
                "is_active": bool(t.is_active),
                "created_at": t.created_at.strftime("%d/%m/%Y") if t.created_at else "—",
                "session_count": len(t.sessions),
            }
            for t in trays
        ]
    return templates.TemplateResponse(
        request=request,
        name="trays.html",
        context={"trays": tray_list},
    )


# ─── Tray CRUD API ────────────────────────────────────────────────────────────

@app.get("/api/trays")
async def api_trays_list():
    with _Session() as session:
        trays = session.query(Tray).order_by(Tray.created_at.desc()).all()
        return [
            {
                "id":        t.id,
                "tray_name": t.tray_name or f"Tray-{t.id}",
                "rows":      t.rows or 13,
                "cols":      t.cols or 15,
                "is_active": bool(t.is_active),
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in trays
        ]


@app.post("/api/trays")
async def api_trays_create(body: dict):
    tray_name = str(body.get("tray_name", "")).strip() or None
    rows      = int(body.get("rows", 13))
    cols      = int(body.get("cols", 15))
    layout    = body.get("layout_json") or _generate_default_layout(rows, cols)

    with _Session() as session:
        tray = Tray(
            tray_name=tray_name,
            rows=rows,
            cols=cols,
            total_slots=rows * cols,
            dimension_info=f"{rows}x{cols}",
            layout_json=layout,
            is_active=False,
        )
        session.add(tray)
        session.commit()
        session.refresh(tray)
        result = {
            "id": tray.id,
            "tray_name": tray.tray_name or f"Tray-{tray.id}",
            "rows": tray.rows,
            "cols": tray.cols,
            "is_active": bool(tray.is_active),
        }
    logger.success("Tray created — id={}, name={}", result["id"], result["tray_name"])
    return result


@app.patch("/api/trays/{tray_id}")
async def api_trays_update(tray_id: int, body: dict):
    with _Session() as session:
        tray = session.query(Tray).filter_by(id=tray_id).first()
        if tray is None:
            raise HTTPException(status_code=404, detail="Tray not found")

        if "tray_name" in body:
            tray.tray_name = str(body["tray_name"]).strip() or None
        if "rows" in body or "cols" in body:
            tray.rows = int(body.get("rows", tray.rows or 13))
            tray.cols = int(body.get("cols", tray.cols or 15))
            tray.total_slots  = tray.rows * tray.cols
            tray.dimension_info = f"{tray.rows}x{tray.cols}"
        if "layout_json" in body:
            tray.layout_json = body["layout_json"] or _generate_default_layout(
                tray.rows or 13, tray.cols or 15
            )

        session.commit()
        result = {
            "id": tray.id,
            "tray_name": tray.tray_name or f"Tray-{tray.id}",
            "rows": tray.rows,
            "cols": tray.cols,
            "is_active": bool(tray.is_active),
        }
    logger.info("Tray updated — id={}", tray_id)
    return result


@app.post("/api/trays/{tray_id}/activate")
async def api_trays_activate(tray_id: int):
    with _Session() as session:
        target = session.query(Tray).filter_by(id=tray_id).first()
        if target is None:
            raise HTTPException(status_code=404, detail="Tray not found")
        # Deactivate all, then activate the selected one
        session.query(Tray).filter(Tray.is_active == True).update(  # noqa: E712
            {"is_active": False}, synchronize_session=False
        )
        target.is_active = True
        session.commit()
        result = {
            "id":        target.id,
            "tray_name": target.tray_name or f"Tray-{target.id}",
            "is_active": True,
        }
    logger.success("Active tray set — id={}, name={}", result["id"], result["tray_name"])
    return result


@app.delete("/api/trays/{tray_id}")
async def api_trays_delete(tray_id: int):
    with _Session() as session:
        tray = session.query(Tray).filter_by(id=tray_id).first()
        if tray is None:
            raise HTTPException(status_code=404, detail="Tray not found")
        if tray.is_active:
            raise HTTPException(status_code=409, detail="Cannot delete the active tray")
        session.delete(tray)
        session.commit()
    logger.info("Tray deleted — id={}", tray_id)
    return {"deleted": tray_id}


# ─── Trend ────────────────────────────────────────────────────────────────────

@app.get("/trend", response_class=HTMLResponse)
async def trend_page(request: Request):
    return templates.TemplateResponse(request=request, name="trend.html", context={})


@app.get("/api/trend")
async def api_trend(from_date: str = None, to_date: str = None):
    def _parse_thai(s: str) -> datetime | None:
        try:
            d, m, y = s.strip().split("/")
            return datetime(int(y) - 543, int(m), int(d))
        except Exception:
            return None

    with _Session() as session:
        query = session.query(ScanSession).order_by(ScanSession.scanned_at.asc())
        if from_date:
            dt = _parse_thai(from_date)
            if dt:
                query = query.filter(ScanSession.scanned_at >= dt)
        if to_date:
            dt = _parse_thai(to_date)
            if dt:
                query = query.filter(ScanSession.scanned_at < dt + timedelta(days=1))
        rows = query.limit(500).all()

    records = []
    for r in rows:
        ts        = r.scanned_at
        thai_year = ts.year + 543
        records.append({
            "id":        r.id,
            "timestamp": ts.isoformat(),
            "thai_date": f"{ts.day}/{ts.month}/{thai_year}",
            "count":     (r.color_0 + r.color_1 + r.color_2 + r.color_3 + r.color_4),
            "colors":    {"L0": r.color_0, "L1": r.color_1, "L2": r.color_2,
                          "L3": r.color_3, "L4": r.color_4},
            "n_errors":  r.error_count,
        })

    return {"data": records}


@app.get("/api/sessions")
async def api_sessions(tray_id: int = None, from_date: str = None, to_date: str = None,
                       limit: int = 50, offset: int = 0):
    """Paginated scan history, optionally filtered by tray_id and date range."""
    def _parse_thai(s: str) -> datetime | None:
        try:
            d, m, y = s.strip().split("/")
            return datetime(int(y) - 543, int(m), int(d))
        except Exception:
            return None

    with _Session() as session:
        query = session.query(ScanSession).order_by(ScanSession.scanned_at.desc())
        if tray_id:
            query = query.filter(ScanSession.tray_id == tray_id)
        if from_date:
            dt = _parse_thai(from_date)
            if dt:
                query = query.filter(ScanSession.scanned_at >= dt)
        if to_date:
            dt = _parse_thai(to_date)
            if dt:
                query = query.filter(ScanSession.scanned_at < dt + timedelta(days=1))
        total = query.count()
        rows  = query.offset(offset).limit(limit).all()

    return {
        "total": total,
        "data": [
            {
                "id":          r.id,
                "tray_id":     r.tray_id,
                "tray_name":   r.tray.tray_name if r.tray else None,
                "scanned_at":  r.scanned_at.isoformat(),
                "count":       r.color_0 + r.color_1 + r.color_2 + r.color_3 + r.color_4,
                "error_count": r.error_count,
                "is_clean":    r.is_clean,
            }
            for r in rows
        ],
    }


# ─── Settings ─────────────────────────────────────────────────────────────────

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    with _Session() as session:
        tray      = session.query(Tray).filter_by(is_active=True).first()
        last_scan = session.query(ScanSession).order_by(ScanSession.scanned_at.desc()).first()
        tray_name = tray.tray_name if tray else None
        grid_date = tray.grid_json.get("calibration_date") if tray and tray.grid_json else None

    latest_capture = None
    captures_dir = Path(cfg.captures_dir)
    if captures_dir.exists():
        imgs = sorted(captures_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        if imgs:
            latest_capture = f"/static/captures/{imgs[0].name}"

    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={
            "active_tray":    tray_name,
            "grid_date":      grid_date,
            "last_scan_at":   last_scan.scanned_at.isoformat() if last_scan else None,
            "latest_capture": latest_capture,
        },
    )


@app.get("/api/latest-capture")
async def api_latest_capture():
    captures_dir = Path(cfg.captures_dir)
    if captures_dir.exists():
        imgs = sorted(captures_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        if imgs:
            return {"url": f"/static/captures/{imgs[0].name}"}
    return {"url": None}


@app.post("/api/upload-capture")
async def api_upload_capture(file: UploadFile = File(...)):
    """Save an uploaded image to captures_dir so it becomes the latest capture."""
    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    captures_dir = Path(cfg.captures_dir)
    captures_dir.mkdir(parents=True, exist_ok=True)

    image_id = str(uuid.uuid4())
    dest = captures_dir / f"{image_id}.jpg"

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    url = f"/static/captures/{image_id}.jpg"
    logger.info("Capture uploaded: {}", dest.name)
    return {"url": url}


@app.get("/api/settings/grid-corners")
async def api_grid_corners():
    with _Session() as session:
        tray = session.query(Tray).filter_by(is_active=True).first()
        if tray and tray.grid_json:
            gj = tray.grid_json
            return {
                "corners":          gj.get("corners"),
                "calibration_date": gj.get("calibration_date"),
                "grid_pts":         gj.get("grid_pts"),
                "image_width":      gj.get("image_width", 0),
                "image_height":     gj.get("image_height", 0),
            }
    return {"corners": None, "calibration_date": None, "grid_pts": None,
            "image_width": 0, "image_height": 0}


@app.post("/settings/grid")
async def settings_grid(
    file: UploadFile = File(None),
    corners: str = Form(...),
    grid_pts: str = Form(None),
    img_width: int = Form(0),
    img_height: int = Form(0),
):
    try:
        corner_list = json.loads(corners)
        if len(corner_list) != 4:
            raise ValueError("Exactly 4 corners required")

        if grid_pts:
            gp_list = json.loads(grid_pts)
            gp_np   = np.array(gp_list, dtype=np.float64)
        else:
            from utils.calibration import _corners_to_grid_pts
            gp_np   = _corners_to_grid_pts(corner_list)
            gp_list = gp_np.tolist()

        result           = _grid_pts_to_result(gp_list)
        calibration_date = datetime.utcnow().strftime("%Y-%m-%d")
        new_grid_json    = {
            "calibration_date": calibration_date,
            "corners":          corner_list,
            "grid_pts":         gp_list,
            "sample_centres":   result.sample_centres,
            "ref_centres":      result.ref_centres,
            "grid_spacing":     result.grid_spacing,
            "image_width":      img_width,
            "image_height":     img_height,
        }

        with _Session() as session:
            tray = session.query(Tray).filter_by(is_active=True).first()
            if tray is None:
                raise HTTPException(status_code=400, detail="No active tray — select a tray first")
            tray.grid_json = new_grid_json
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(tray, "grid_json")
            session.commit()
            tray_id = tray.id

        logger.success("Grid calibrated — tray_id={}, date={}", tray_id, calibration_date)
        return {"status": "ok", "calibration_date": calibration_date}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("settings/grid failed")
        raise HTTPException(status_code=400, detail=str(exc))


# ─── Upload page (full pipeline, no auth) ────────────────────────────────────

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse(request=request, name="upload.html", context={})


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """
    Web upload — identical to /analyze but no auth header required.
    Saves ScanSession + TestSlot rows and sends Telegram report.
    """
    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    with _Session() as session:
        active_tray = session.query(Tray).filter_by(is_active=True).first()
        if active_tray is None:
            raise HTTPException(
                status_code=400,
                detail="กรุณาเลือกถาดก่อนสแกน (No active tray selected)",
            )
        active_tray_id = active_tray.id
        tray_grid_json = active_tray.grid_json
        layout_map: dict[str, str] = _normalize_layout_map(
            active_tray.layout_json, cols=active_tray.cols or 15
        )
        tray_name = active_tray.tray_name or f"Tray-{active_tray.id}"

    if not tray_grid_json:
        raise HTTPException(
            status_code=400,
            detail="กรุณาปรับเทียบกริดก่อนสแกน (Grid not calibrated for this tray)",
        )

    now      = datetime.utcnow()
    image_id = str(uuid.uuid4())
    ts_str   = now.strftime("%Y%m%d_%H%M%S")

    captures = Path(cfg.captures_dir)
    captures.mkdir(parents=True, exist_ok=True)
    thumb_path = captures / f"{image_id}.jpg"
    static_url = f"/static/captures/{image_id}.jpg"

    log_dir  = Path(cfg.visual_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"URINE_SCAN_{ts_str}.jpg"

    try:
        nparr    = np.frombuffer(img_bytes, np.uint8)
        img_full = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_full is None:
            raise ValueError("Invalid image data")

        grid_result  = _grid_pts_to_result(tray_grid_json["grid_pts"])

        slot_centers = grid_result.sample_centres

        from utils.color_analysis import build_reference_baseline
        ref_positions = _build_ref_positions(grid_result.ref_centres, grid_result.grid_spacing)
        live_baseline = build_reference_baseline(img_full, ref_positions) if ref_positions else {}
        if not live_baseline:
            logger.warning("/api/upload: reference row extraction failed — falling back to color.json")

        _, boxes_orig = app.state.yolo.detect_raw(img_bytes)
        slot_hits     = _match_detections_to_slots(boxes_orig, slot_centers)
        classified    = _run_color_analysis_v2(img_full, slot_hits, baseline=live_baseline or None)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload inference failed")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}")

    slot_rows   = _build_slot_rows(slot_hits, classified)
    error_slots = [r for r in slot_rows if r["is_error"]]
    count       = len(slot_hits)
    error_count = len(error_slots)

    color_counts: dict[int, int] = {i: 0 for i in range(5)}
    for lvl in classified.values():
        if lvl is not None:
            color_counts[lvl] = color_counts.get(lvl, 0) + 1

    validation_results_compat: dict = {
        "slots": {
            str(r["position_index"]): {
                "cx":          slot_hits[r["position_index"]]["cx"] if r["position_index"] in slot_hits else 0,
                "cy":          slot_hits[r["position_index"]]["cy"] if r["position_index"] in slot_hits else 0,
                "radius":      slot_hits[r["position_index"]]["radius"] if r["position_index"] in slot_hits else 30,
                "level":       r["color_result"],
                "ok":          not r["is_error"],
                "wrong_color": r["is_error"],
                "duplicate":   False,
            }
            for r in slot_rows if r["color_result"] is not None
        },
        "duplicate_slots":     [],
        "wrong_color_slots":   [
            {"slot_id":    str(r["position_index"]),
             "slot_label": layout_map.get(str(r["position_index"]),
                           f"R{(r['position_index']-1)//15+1:02d}C{(r['position_index']-1)%15+1:02d}"),
             "expected":   _expected_level_from_position(r["position_index"]),
             "actual":     r["color_result"]}
            for r in error_slots
        ],
        "raw_detection_count": len(boxes_orig),
    }

    # Raw image → captures_dir (used by /api/latest-capture for grid calibration)
    cv2.imwrite(str(thumb_path), img_full, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # Annotated image — calibrated bilinear grid + detections (UI + Telegram)
    from app.shared.processor import _render_annotated_canvas
    _raw_grid_pts    = tray_grid_json.get("grid_pts")
    annotated_canvas = _render_annotated_canvas(
        img_full, validation_results_compat, None, slot_centers, layout_map,
        grid_pts=_raw_grid_pts,
    )
    annotated_path   = captures / f"{image_id}_annotated.jpg"
    annotated_url    = f"/static/captures/{image_id}_annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated_canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])

    try:
        save_visual_log(
            image=img_full,
            validation_results=validation_results_compat,
            grid_cfg=None,
            filepath=log_path,
            slot_centers=slot_centers,
            layout_map=layout_map,
            grid_pts=_raw_grid_pts,
        )
    except Exception as exc:
        logger.warning("save_visual_log failed (non-critical): {}", exc)

    with _Session() as session:
        scan = ScanSession(
            tray_id=active_tray_id,
            scanned_at=now,
            image_raw_path=static_url,
            image_annotated_path=str(annotated_path),
            color_0=color_counts[0],
            color_1=color_counts[1],
            color_2=color_counts[2],
            color_3=color_counts[3],
            color_4=color_counts[4],
            error_count=error_count,
            is_clean=(error_count == 0),
        )
        session.add(scan)
        session.flush()

        session.bulk_insert_mappings(TestSlot, [
            {
                "session_id":     scan.id,
                "position_index": r["position_index"],
                "color_result":   r["color_result"],
                "is_error":       r["is_error"],
                "person_id":      None,
            }
            for r in slot_rows
        ])
        session.commit()
        session_id = scan.id

    logger.success(
        "Upload done — tray_id={}, count={}, errors={}, session_id={}",
        active_tray_id, count, error_count, session_id,
    )

    if cfg.telegram_token and cfg.telegram_chat_id:
        try:
            caption = _build_thai_caption(
                now=now, tray_name=tray_name, count=count,
                color_counts=color_counts, error_count=error_count,
                server_url=cfg.server_url,
            )
            report_bytes = (annotated_path.read_bytes()
                            if annotated_path.exists()
                            else (log_path.read_bytes() if log_path.exists() else None))
            if report_bytes:
                _send_telegram_report(
                    img_bytes=report_bytes,
                    caption=caption,
                    token=cfg.telegram_token,
                    chat_id=cfg.telegram_chat_id,
                )
        except Exception as exc:
            logger.warning("Telegram report failed (non-critical): {}", exc)

    wrong_color_items = [
        {
            "slot_id":  "R{:02d}C{:02d}".format(
                (r["position_index"] - 1) // 15 + 1,
                (r["position_index"] - 1) % 15  + 1,
            ),
            "expected": _expected_level_from_position(r["position_index"]),
            "actual":   r["color_result"],
        }
        for r in error_slots
    ]

    return {
        "status":    "success",
        "session_id": session_id,
        "tray_name": tray_name,
        "count":     count,
        "summary":   {f"L{i}": color_counts[i] for i in range(5)},
        "is_clean":  error_count == 0,
        "errors": {
            "duplicate_slots":   [],
            "wrong_color_slots": wrong_color_items,
        },
        "slots":     slot_rows,
        "image_url": annotated_url,
        "timestamp": now.isoformat(),
    }


# ─── Test page ────────────────────────────────────────────────────────────────

@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    return templates.TemplateResponse(request=request, name="test.html", context={})


@app.post("/api/test-upload")
async def api_test_upload(file: UploadFile = File(...)):
    """
    Manual test — full V2 pipeline without DB write or Telegram.
    """
    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    with _Session() as session:
        tray = session.query(Tray).filter_by(is_active=True).first()
        test_grid_json = tray.grid_json if tray else None
    if not test_grid_json:
        raise HTTPException(status_code=400, detail="กรุณาปรับเทียบกริดก่อนทดสอบ (Grid not calibrated)")

    test_dir = Path(cfg.captures_dir).parent / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        import io as _io
        from PIL import Image as _PILImage
        _pil_img  = _PILImage.open(_io.BytesIO(img_bytes))
        _rgb_path = test_dir / "test_rgb.jpg"
        _pil_img.save(str(_rgb_path), format="JPEG", quality=95)
    except Exception as _exc:
        logger.warning("Could not save test_rgb.jpg: {}", _exc)

    image_id = str(uuid.uuid4())
    log_path  = test_dir / f"{image_id}.jpg"

    try:
        nparr    = np.frombuffer(img_bytes, np.uint8)
        img_full = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_full is None:
            raise ValueError("Invalid image data")

        grid_result  = _grid_pts_to_result(test_grid_json["grid_pts"])

        slot_centers = grid_result.sample_centres

        from utils.color_analysis import build_reference_baseline
        ref_positions = _build_ref_positions(grid_result.ref_centres, grid_result.grid_spacing)
        live_baseline = build_reference_baseline(img_full, ref_positions) if ref_positions else {}
        if not live_baseline:
            logger.warning("/api/test-upload: reference row extraction failed — falling back to color.json")

        _, boxes_orig = app.state.yolo.detect_raw(img_bytes)
        slot_hits     = _match_detections_to_slots(boxes_orig, slot_centers)
        classified    = _run_color_analysis_v2(img_full, slot_hits, baseline=live_baseline or None)

    except Exception as exc:
        logger.exception("Test inference failed")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}")

    slot_rows   = _build_slot_rows(slot_hits, classified)
    error_slots = [r for r in slot_rows if r["is_error"]]
    count       = len(slot_hits)
    color_counts: dict[int, int] = {i: 0 for i in range(5)}
    for lvl in classified.values():
        if lvl is not None:
            color_counts[lvl] = color_counts.get(lvl, 0) + 1

    # Save annotated visual log
    try:
        validation_results_compat = {
            "slots": {
                str(r["position_index"]): {
                    "cx":          slot_hits[r["position_index"]]["cx"] if r["position_index"] in slot_hits else 0,
                    "cy":          slot_hits[r["position_index"]]["cy"] if r["position_index"] in slot_hits else 0,
                    "radius":      slot_hits[r["position_index"]]["radius"] if r["position_index"] in slot_hits else 30,
                    "level":       r["color_result"],
                    "ok":          not r["is_error"],
                    "wrong_color": r["is_error"],
                    "duplicate":   False,
                }
                for r in slot_rows if r["color_result"] is not None
            },
            "duplicate_slots":   [],
            "wrong_color_slots": [{"slot_id": str(r["position_index"]),
                                   "expected": _expected_level_from_position(r["position_index"]),
                                   "actual": r["color_result"]}
                                  for r in error_slots],
            "raw_detection_count": len(boxes_orig),
        }
        save_visual_log(
            image=img_full,
            validation_results=validation_results_compat,
            grid_cfg=None,
            filepath=log_path,
            slot_centers=slot_centers,
            layout_map={},
            grid_pts=test_grid_json.get("grid_pts"),
        )
    except Exception as exc:
        logger.warning("Test visual log failed: {}", exc)

    image_url = f"/static/test/{image_id}.jpg"
    logger.info("Test upload — count={}, errors={}", count, len(error_slots))

    wrong_color_items = [
        {
            "slot_id":  "R{:02d}C{:02d}".format(
                (r["position_index"] - 1) // 15 + 1,
                (r["position_index"] - 1) % 15  + 1,
            ),
            "expected": _expected_level_from_position(r["position_index"]),
            "actual":   r["color_result"],
        }
        for r in error_slots
    ]

    return {
        "status":    "success",
        "count":     count,
        "summary":   {f"L{i}": color_counts[i] for i in range(5)},
        "is_clean":  len(error_slots) == 0,
        "errors": {
            "duplicate_slots":  [],
            "wrong_color_slots": wrong_color_items,
        },
        "slots":     slot_rows,
        "image_url": image_url,
    }


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_server():
    import uvicorn
    uvicorn.run(
        "app.server_app:app",
        host=cfg.server_host,
        port=cfg.server_port,
        reload=False,
        log_level="info",
    )
