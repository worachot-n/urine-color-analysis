"""
Shared CV pipeline — called identically by /analyze regardless of whether
the request comes from the Raspberry Pi camera or a browser file upload.

Public API:
    run_pipeline(jpeg_bytes, slot_cfg, rows, cols) -> (scan_result, annotated_jpeg)
    lab_to_hex(L_cv, a_cv, b_cv)                 -> "#rrggbb"
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from utils.auto_grid_detector import detect_grid_full, draw_grid_lines
from utils.color_analysis import (
    extract_bottle_color,
    extract_bottle_features,
    compute_white_balance_offset,
    build_reference_set,
    build_reference_histograms,
    build_master_path,
    classify_sample_path,
    is_achromatic,
    filter_reference_outliers,
    W_CHROMA,
    W_HIST,
    W_LAB_L,
    W_LAB_A,
    W_LAB_B,
    CHROMA_SCALE,
    MAX_COMBINED_SCORE,
    COMBINED_MARGIN,
    MAX_PATH_DISTANCE,
    PATH_CONFIDENT_DISTANCE,
    SOFT_PENALTY_PATH_DE_THRESHOLD,
    SOFT_PENALTY_FACTOR,
    SATURATION_L0_THRESHOLD,
    SATURATION_HIST_WEIGHT,
)
from app.shared.processor import crop_sample_roi, letterbox_white_padding
from server.slot_config import (
    SlotConfig,
    active_cell_indices,
    reference_cells,
    sample_cells,
    white_reference_cells,
)
import tomllib
_cfg = tomllib.load(open(Path(__file__).parent.parent / "configs" / "config.toml", "rb"))
_y    = _cfg["yolo"]
_sroi = _cfg.get("sample_roi", {})
_ca   = _cfg.get("color_analysis", {})
_wb   = _cfg.get("white_balance", {})
YOLO_MODEL_PATH: str       = _y["model_path"]
YOLO_CONF_THRESHOLD: float = float(_y["conf_threshold"])
YOLO_IOU_THRESHOLD: float  = float(_y["iou_threshold"])
YOLO_IMGSZ: int            = int(_y["imgsz"])
CONFIDENCE_MARGIN: float   = float(_ca.get("confidence_margin", 3.0))
MAX_DELTA_E: float         = float(_ca.get("max_delta_e", 18.0))
WB_ENABLED: bool           = bool(_wb.get("enabled", True))
SAMPLE_ROI_TOP: int        = int(_sroi.get("top", 0))
SAMPLE_ROI_BOTTOM: int     = int(_sroi.get("bottom", 0))
SAMPLE_ROI_LEFT: int       = int(_sroi.get("left", 0))
SAMPLE_ROI_RIGHT: int      = int(_sroi.get("right", 0))


# ---------------------------------------------------------------------------
# YOLO model — loaded once at first call
# ---------------------------------------------------------------------------

_yolo_model = None


def _load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
        model_path = Path(YOLO_MODEL_PATH)
        if not model_path.exists():
            logger.warning("pipeline: YOLO model not found at {} — detection disabled", YOLO_MODEL_PATH)
            return None
        _yolo_model = YOLO(str(model_path), task="detect")
        logger.info("pipeline: YOLO model loaded from {}", YOLO_MODEL_PATH)
    except Exception as e:
        logger.error("pipeline: YOLO load failed: {}", e)
    return _yolo_model


# ---------------------------------------------------------------------------
# Lab ↔ Hex conversion
# ---------------------------------------------------------------------------

def lab_to_hex(L_cv: float, a_cv: float, b_cv: float) -> str:
    """
    Convert OpenCV 8-bit CIE Lab to '#rrggbb'.

    OpenCV encoding: L ∈ [0,255], a/b ∈ [0,255] (a=128, b=128 = neutral).
    """
    # OpenCV 8-bit → standard Lab
    L = L_cv * 100.0 / 255.0
    a = a_cv - 128.0
    b_std = b_cv - 128.0

    # Lab → XYZ (D65)
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b_std / 200.0

    delta = 6.0 / 29.0
    delta3 = delta ** 3

    def f_inv(t: float) -> float:
        return t ** 3 if t > delta else 3 * delta ** 2 * (t - 4.0 / 29.0)

    X = 0.95047 * f_inv(fx)
    Y = 1.00000 * f_inv(fy)
    Z = 1.08883 * f_inv(fz)

    # XYZ → linear sRGB
    r_lin =  3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b_lin =  0.0557 * X - 0.2040 * Y + 1.0570 * Z

    def gamma(c: float) -> int:
        c = max(0.0, min(1.0, c))
        if c <= 0.0031308:
            v = 12.92 * c
        else:
            v = 1.055 * c ** (1.0 / 2.4) - 0.055
        return max(0, min(255, round(v * 255)))

    return f"#{gamma(r_lin):02x}{gamma(g_lin):02x}{gamma(b_lin):02x}"


# ---------------------------------------------------------------------------
# Slot assignment
# ---------------------------------------------------------------------------

def _assign_bottles_to_slots(
    yolo_boxes: list,
    grid_pts_full: np.ndarray,
    active_cells: set[int],
    max_dist: float,
) -> dict[int, dict]:
    """
    Match YOLO detections to the nearest active grid cell.

    yolo_boxes: list of [cx, cy, w, h, conf, cls] in full-image coords
    grid_pts_full: (n_cells, 2) array, 0-based; cell_index 1-based → [cell_idx-1]
    max_dist: maximum allowed distance (px) for a match

    Returns {cell_index: {"cx", "cy", "w", "h", "conf"}}
    """
    hits: dict[int, dict] = {}
    active_list = sorted(active_cells)

    for box in yolo_boxes:
        cx, cy = float(box[0]), float(box[1])
        w, h   = float(box[2]), float(box[3])
        conf   = float(box[4]) if len(box) > 4 else 1.0

        best_dist = max_dist
        best_cell: Optional[int] = None

        for cell_idx in active_list:
            gx, gy = float(grid_pts_full[cell_idx - 1, 0]), float(grid_pts_full[cell_idx - 1, 1])
            dist = np.hypot(cx - gx, cy - gy)
            if dist < best_dist:
                best_dist = dist
                best_cell = cell_idx

        if best_cell is not None and best_cell not in hits:
            hits[best_cell] = {"cx": cx, "cy": cy, "w": w, "h": h, "conf": conf}

    return hits


# ---------------------------------------------------------------------------
# Annotated image rendering
# ---------------------------------------------------------------------------

_LEVEL_COLORS_BGR = [
    (0, 240, 255),   # L0 — yellow
    (0, 200, 255),   # L1 — orange-ish
    (0, 140, 255),   # L2 — orange
    (50, 80, 200),   # L3 — red
    (40, 20, 180),   # L4 — dark red
]
_COLOR_MISSING = (0, 60, 220)   # red for missing assigned slots


def _render_annotated(
    frame_full: np.ndarray,
    slot_hits: dict[int, dict],
    result_slots: dict[str, dict],
    grid_pts_full: np.ndarray,
    slot_cfg: SlotConfig,
    radius_full: float,
    cell_to_slotid: dict[int, str],
) -> bytes:
    """
    Draw annotations on frame_full (in-place on a copy) and return JPEG bytes.

    Layers:
      1. Full grid lines (matches /auto_grid page rendering)
      2. Green circle + slot_id for detected bottles
      3. Red circle + slot_id for missing assigned sample cells
    """
    canvas = frame_full.copy()
    h, w = canvas.shape[:2]
    sc = max(w, h) / 1920.0
    lw = max(1, int(sc * 1.5))
    r  = max(10, int(radius_full))

    # ── 1. Full grid lines (same as /auto_grid page) ─────────────────────
    draw_grid_lines(canvas, grid_pts_full, slot_cfg.rows, slot_cfg.cols, radius_full)

    # ── 2. Detected bottles ───────────────────────────────────────────────
    fsc = max(0.5, sc * 0.7)
    ftk = max(1, int(sc * 1.5))
    ref_cells_map = reference_cells(slot_cfg)

    for cell_idx, hit in slot_hits.items():
        cx, cy = int(hit["cx"]), int(hit["cy"])
        slot_id = cell_to_slotid.get(cell_idx, "")

        slot_data = result_slots.get(slot_id, {})
        level = slot_data.get("color_level")
        color = _LEVEL_COLORS_BGR[level] if level is not None else (0, 220, 0)

        cv2.circle(canvas, (cx, cy), r, color, lw + 1)
        if slot_id:
            cv2.putText(canvas, slot_id, (cx - r, cy - r - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, fsc, (0, 0, 0), ftk + 2, cv2.LINE_AA)
            cv2.putText(canvas, slot_id, (cx - r, cy - r - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, fsc, (255, 255, 255), ftk, cv2.LINE_AA)

    # ── 3. Missing sample slots ───────────────────────────────────────────
    sample_map = sample_cells(slot_cfg)
    for cell_idx, slot_id in sample_map.items():
        if cell_idx not in slot_hits:
            gx, gy = int(grid_pts_full[cell_idx - 1, 0]), int(grid_pts_full[cell_idx - 1, 1])
            cv2.circle(canvas, (gx, gy), r, _COLOR_MISSING, lw + 1)
            cv2.putText(canvas, slot_id, (gx - r, gy - r - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, fsc, (0, 0, 0), ftk + 2, cv2.LINE_AA)
            cv2.putText(canvas, slot_id, (gx - r, gy - r - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, fsc, (0, 60, 220), ftk, cv2.LINE_AA)

    # Scale down for storage (50%)
    out = cv2.resize(canvas, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes() if ok else b""


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(jpeg_bytes: bytes, slot_cfg: SlotConfig) -> tuple[dict, bytes]:
    """
    Full CV pipeline from raw JPEG bytes to (scan_result dict, annotated JPEG bytes).

    Steps:
      1. Decode JPEG
      2. crop_sample_roi
      3. letterbox_white_padding(640) for grid detection
      4. detect_grid → grid_pts in 640-space → convert to full-image coords
      5. YOLO detection on letterboxed image → boxes → scale to full-image coords
      6. Assign YOLO boxes to active slots
      7. Extract Lab from reference cells → build_reference_baseline
      8. Extract Lab from detected sample cells → classify_sample
      9. Build scan_result dict
     10. Render annotated JPEG
    """
    # ── 1. Decode ────────────────────────────────────────────────────────
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame_full = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame_full is None:
        raise ValueError("Failed to decode image")

    # ── 2. Crop ROI — top=0 to include reference row (matches /auto_grid) ─
    roi, roi_x1, roi_y1 = crop_sample_roi(
        frame_full, 0, SAMPLE_ROI_BOTTOM, SAMPLE_ROI_LEFT, SAMPLE_ROI_RIGHT
    )

    # ── 3. Grid detection — full-resolution ROI (same as /auto_grid) ─────
    grid_result = detect_grid_full(roi, slot_cfg.rows, slot_cfg.cols)
    grid_pts_roi = grid_result["grid_pts"]   # (rows*cols, 2) in ROI coords

    # Convert ROI → full-image coords (simple offset; no letterbox math)
    grid_pts_full = grid_pts_roi.copy()
    grid_pts_full[:, 0] += roi_x1
    grid_pts_full[:, 1] += roi_y1

    # Radius from actual HoughCircles detections (median of detected circle radii)
    radius_full = float(grid_result["avg_radius_px"])

    # ── 4. Letterbox ROI for YOLO (grid detection no longer uses this) ───
    padded, scale, pad_x, pad_y = letterbox_white_padding(roi, 640)

    # ── 5. YOLO detection ─────────────────────────────────────────────────
    yolo_boxes: list = []
    model = _load_yolo()
    if model is not None:
        try:
            results = model(
                padded,
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONF_THRESHOLD,
                iou=YOLO_IOU_THRESHOLD,
                verbose=False,
            )
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    # Convert from padded-640 to full-image coords
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    bw = x2 - x1
                    bh = y2 - y1
                    cx_f = (cx - pad_x) / scale + roi_x1
                    cy_f = (cy - pad_y) / scale + roi_y1
                    yolo_boxes.append([cx_f, cy_f, bw / scale, bh / scale, conf, 0])
            logger.info("pipeline: YOLO detected {} boxes", len(yolo_boxes))
        except Exception as e:
            logger.error("pipeline: YOLO inference failed: {}", e)

    # ── 6. Assign YOLO boxes to active slots ─────────────────────────────
    active = active_cell_indices(slot_cfg)
    max_dist = radius_full * 2.0
    slot_hits = _assign_bottles_to_slots(yolo_boxes, grid_pts_full, active, max_dist)
    logger.info("pipeline: {} bottles assigned to slots", len(slot_hits))

    # ── 7a. White-balance offset (from is_white_reference cells) ─────────
    wb_offset = None
    if WB_ENABLED:
        wb_cells = white_reference_cells(slot_cfg)
        if wb_cells:
            wb_positions = [
                (
                    float(grid_pts_full[i - 1, 0]),
                    float(grid_pts_full[i - 1, 1]),
                    radius_full,
                )
                for i in wb_cells
            ]
            wb_offset = compute_white_balance_offset(frame_full, wb_positions)
        else:
            logger.info("pipeline: no white-reference cells configured — WB disabled")

    # ── 7b. Reference Lab set + per-level histogram templates ─────────────
    ref_map = reference_cells(slot_cfg)   # {cell_idx: ref_level}
    ref_positions: dict[int, list] = {}
    for cell_idx, ref_level in ref_map.items():
        cx = float(grid_pts_full[cell_idx - 1, 0])
        cy = float(grid_pts_full[cell_idx - 1, 1])
        ref_positions.setdefault(ref_level, []).append((cx, cy, radius_full))

    # Outlier filter (b*-based 2σ) BEFORE building either ref structure, so
    # both ref_set and ref_hists see the same cleaned positions.
    pre_counts = {lvl: len(ps) for lvl, ps in ref_positions.items()}
    ref_positions = filter_reference_outliers(frame_full, ref_positions)
    ref_outliers_dropped = {
        lvl: pre_counts[lvl] - len(ref_positions.get(lvl, []))
        for lvl in pre_counts
        if pre_counts[lvl] - len(ref_positions.get(lvl, [])) > 0
    }

    ref_set = build_reference_set(frame_full, ref_positions)
    ref_hists = build_reference_histograms(frame_full, ref_positions, wb_offset=wb_offset)
    logger.info("pipeline: reference set levels: {} | hist templates: {} | outliers dropped: {}",
                list(ref_set.keys()), list(ref_hists.keys()), ref_outliers_dropped)

    # Build the Master Color Path from per-level Lab centroids (mean of each
    # level's individual references). Used by classify_sample_path below.
    level_centroids = {
        lvl: tuple(float(v) for v in np.mean(np.array(refs), axis=0))
        for lvl, refs in ref_set.items()
        if refs
    }
    master_path = build_master_path(level_centroids)
    if master_path is not None:
        logger.info(
            "pipeline: master color path — levels: {} ({} segments)",
            master_path["levels"], len(master_path["segments"]),
        )

    # Reference Labs → hex (for scatter plot and Sheets)
    reference_labs: dict[str, list] = {}
    for ref_level, positions in ref_positions.items():
        level_key = str(ref_level)
        reference_labs[level_key] = []
        for cx, cy, r in positions:
            lab = extract_bottle_color(frame_full, cx, cy, r)
            if lab:
                reference_labs[level_key].append({
                    "lab": [round(v, 2) for v in lab],
                    "hex": lab_to_hex(*lab),
                })

    # ── 8. Classify sample bottles (hybrid: chroma ΔE + histogram) ───────
    sample_map = sample_cells(slot_cfg)   # {cell_idx: slot_id}
    cell_to_slotid: dict[int, str] = sample_map.copy()

    result_slots: dict[str, dict] = {}

    def _round(v, n=3):
        if v is None or not np.isfinite(v):
            return None
        return round(float(v), n)

    for cell_idx, slot_id in sample_map.items():
        hit = slot_hits.get(cell_idx)

        if hit:
            lab, sample_hist = extract_bottle_features(
                frame_full, hit["cx"], hit["cy"], radius_full,
                wb_offset=wb_offset,
            )
        else:
            lab, sample_hist = None, None

        force_l0 = False
        soft_penalty_active = False
        concentration_index = None
        path_distance = None

        if lab is not None and hit is not None and is_achromatic(
            frame_full, hit["cx"], hit["cy"], radius_full,
        ):
            # Near-neutral sample → force-classify L0 (clear) before hue noise
            # can flip it to L1/L2. Skip the path projection entirely.
            level     = 0
            delta_e   = 0.0
            hist_b    = 0.0
            combined  = 0.0
            confident = True
            hex_color = lab_to_hex(*lab)
            force_l0  = True
            concentration_index = 0.0
            path_distance       = 0.0
        elif lab is not None and master_path is not None:
            cls = classify_sample_path(
                lab, sample_hist, master_path, ref_hists,
                weight_chroma=W_CHROMA,
                weight_hist=W_HIST,
                chroma_scale=CHROMA_SCALE,
                max_score=MAX_COMBINED_SCORE,
                max_path_distance=MAX_PATH_DISTANCE,
                path_confident_distance=PATH_CONFIDENT_DISTANCE,
                soft_penalty_path_de_threshold=SOFT_PENALTY_PATH_DE_THRESHOLD,
                soft_penalty_factor=SOFT_PENALTY_FACTOR,
            )
            level     = cls["level"]
            delta_e   = cls["chroma_de"]
            hist_b    = cls["hist_bhatt"]
            combined  = cls["combined"]
            confident = cls["confident"]
            hex_color = lab_to_hex(*lab)
            concentration_index = cls["concentration_index"]
            path_distance       = cls["path_distance"]
            soft_penalty_active = cls["soft_penalty"]
        else:
            level, delta_e, hist_b, combined, confident, hex_color = (
                None, None, None, None, False, None
            )

        result_slots[slot_id] = {
            "cell_index": cell_idx,
            "detected":   hit is not None,
            "color_level": level,
            "concentration_index": _round(concentration_index, 2),
            "path_distance":       _round(path_distance, 2),
            "delta_e":    _round(delta_e, 2),
            "hist_bhatt": _round(hist_b, 3),
            "combined":   _round(combined, 3),
            "confident":  confident,
            "force_l0":   force_l0,
            "soft_penalty": soft_penalty_active,
            "lab":        [round(v, 2) for v in lab] if lab else None,
            "hex":        hex_color,
        }

    # ── 9. Build scan result ──────────────────────────────────────────────
    scan_id = uuid.uuid4().hex[:8]
    summary = {f"L{lvl}": 0 for lvl in range(5)}
    for d in result_slots.values():
        if d["color_level"] is not None:
            summary[f"L{d['color_level']}"] += 1

    missing_slots = [sid for sid, d in result_slots.items() if not d["detected"]]

    scan_result = {
        "status":          "success",
        "scan_id":         scan_id,
        "detected_count":  sum(1 for d in result_slots.values() if d["detected"]),
        "total_assigned":  len(sample_map),
        "missing_slots":   missing_slots,
        "summary":         summary,
        "reference_labs":  reference_labs,
        "max_delta_e":     MAX_DELTA_E,
        "wb_offset":       (
            [round(wb_offset[0], 2), round(wb_offset[1], 2)]
            if wb_offset is not None else None
        ),
        "weights": {
            "chroma":       W_CHROMA,
            "hist":         W_HIST,
            "chroma_scale": CHROMA_SCALE,
            "max_score":    MAX_COMBINED_SCORE,
            "lab_L":        W_LAB_L,
            "lab_a":        W_LAB_A,
            "lab_b":        W_LAB_B,
        },
        "saturation_l0_threshold": SATURATION_L0_THRESHOLD,
        "ref_outliers_dropped":    ref_outliers_dropped,
        "max_path_distance":       MAX_PATH_DISTANCE,
        "path_confident_distance": PATH_CONFIDENT_DISTANCE,
        "saturation_hist_weight":  SATURATION_HIST_WEIGHT,
        "master_path": (
            {
                "levels": master_path["levels"],
                # Lab points in OpenCV-Lab encoding (L=0..255, a=b=0..255 with 128=neutral).
                # Front-end converts to standard Lab on display.
                "points": [[float(v) for v in p] for p in master_path["points"].tolist()],
            }
            if master_path is not None else None
        ),
        "slots":           result_slots,
        "image_url":       f"/static/results/{scan_id}.jpg",
        "timestamp":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # ── 10. Annotated image ───────────────────────────────────────────────
    annotated_jpeg = _render_annotated(
        frame_full, slot_hits, result_slots, grid_pts_full,
        slot_cfg, radius_full, cell_to_slotid,
    )

    return scan_result, annotated_jpeg
