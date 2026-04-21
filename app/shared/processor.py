"""
Shared image-processing utilities used by the server for pre- and post-processing.

letterbox_white_padding()
    Resize any image to target_size × target_size using white (255, 255, 255) fill.
    White padding matches the Roboflow "Fit" pre-processing used during training,
    preventing false-contrast edges that black padding would introduce near bottles.

scale_coordinates()
    Inverse-transform bounding boxes from the 640 × 640 letterboxed space back to
    the original camera coordinate system (e.g. 4608 × 2592).

save_visual_log()
    Draw full diagnostic overlay (grid polygons, slot labels, K-means results,
    error markers) on the original image and save as a high-quality JPEG.
    Uses Pillow for Thai text rendering; falls back to ASCII if no TTF font found.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def crop_sample_roi(
    img: np.ndarray,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
) -> tuple[np.ndarray, int, int]:
    """
    Crop a fixed-margin region of interest from the full frame.

    Called before letterbox_white_padding() so YOLO sees only sample bottles —
    not the reference row (top margin) or the dead-zone column (left margin).

    Args:
        img:    Full-resolution BGR image (H × W × 3).
        top:    Pixels to trim from the top edge (skips reference row).
        bottom: Pixels to trim from the bottom edge.
        left:   Pixels to trim from the left edge (skips dead-zone column ZZ).
        right:  Pixels to trim from the right edge.
                If (width − right) ≤ left the right crop is ignored (safe fallback).

    Returns:
        roi  — Cropped BGR sub-image.
        x1   — Left origin in full-image pixel coordinates.
        y1   — Top origin in full-image pixel coordinates.
    """
    h, w = img.shape[:2]
    y1 = max(0, top)
    y2 = (h - bottom) if bottom > 0 else h
    y2 = max(y1 + 1, y2)
    x1 = max(0, left)
    x2 = (w - right) if right > 0 else w
    x2 = x2 if x2 > x1 else w   # safe fallback if right margin is too large
    return img[y1:y2, x1:x2], x1, y1


def letterbox_white_padding(
    img: np.ndarray,
    target_size: int = 640,
) -> tuple[np.ndarray, float, int, int]:
    """
    Resize *img* to *target_size* × *target_size* with white padding.

    Args:
        img:         BGR image (H × W × 3, uint8).
        target_size: Square output size in pixels (default 640).

    Returns:
        padded   — target_size × target_size BGR image, uint8
        scale    — uniform scale factor applied (new_dim = original_dim * scale)
        pad_x    — pixels of padding added to the left
        pad_y    — pixels of padding added to the top
    """
    h, w = img.shape[:2]
    scale = target_size / max(w, h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_x = (target_size - nw) // 2
    pad_y = (target_size - nh) // 2

    padded = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    padded[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

    return padded, scale, pad_x, pad_y


def scale_coordinates(
    boxes: list[list[float]],
    scale: float,
    pad_x: int,
    pad_y: int,
    roi_x1: int = 0,
    roi_y1: int = 0,
) -> list[list[float]]:
    """
    Map bounding boxes from 640 × 640 letterboxed space back to full-image space.

    Inverse letterbox transform:
        X_orig = (X_640 - pad_x) / scale + roi_x1
        Y_orig = (Y_640 - pad_y) / scale + roi_y1

    where scale = min(640/W, 640/H), pad_x/pad_y are the half-widths of the
    white padding bars returned by letterbox_white_padding().

    Args:
        boxes:   List of [x1, y1, x2, y2, conf, cls, ...] in letterboxed coords.
        scale:   The *scale* value returned by letterbox_white_padding().
        pad_x:   The *pad_x* value returned by letterbox_white_padding().
        pad_y:   The *pad_y* value returned by letterbox_white_padding().
        roi_x1:  Left origin of an upstream ROI crop in full-image pixels (0 = none).
        roi_y1:  Top origin of an upstream ROI crop in full-image pixels (0 = none).

    Returns:
        Same list structure with x1/y1/x2/y2 in full-image coordinates.
        conf, cls, and any extra fields are passed through unchanged.
        Coordinates are clamped to ≥ 0 so partial-padding boxes don't produce
        negative values.
    """
    out = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        ox1 = max(0.0, (x1 - pad_x) / scale) + roi_x1
        oy1 = max(0.0, (y1 - pad_y) / scale) + roi_y1
        ox2 = max(0.0, (x2 - pad_x) / scale) + roi_x1
        oy2 = max(0.0, (y2 - pad_y) / scale) + roi_y1
        out.append([ox1, oy1, ox2, oy2, *box[4:]])
    return out


# ─── Visual logging ───────────────────────────────────────────────────────────

# BGR drawing constants
_C_GRID_LINE  = (100, 100, 100)   # thin grey — calibrated bilinear grid
_C_BOX        = (0,   220,   0)   # bright green — detection bounding box
_C_LABEL      = (0,   255, 255)   # bright yellow — slot coordinate label
_C_LABEL_DARK = (0,     0,   0)   # black — label shadow


def _pos_to_label(pos_key: str) -> str:
    """
    Convert a position_index string (1-195) to grid coordinate label.

    Format: Row letter (A–M) + two-digit column (01–15).
    Examples: "1" → "A01",  "15" → "A15",  "195" → "M15"
    """
    try:
        idx = int(pos_key)
        row = (idx - 1) // 15 + 1   # 1-13
        col = (idx - 1) % 15  + 1   # 1-15
        return f"{chr(64 + row)}{col:02d}"
    except (ValueError, TypeError):
        return str(pos_key)


def _draw_calibrated_grid(
    canvas: np.ndarray,
    grid_pts: list,
    color: tuple = _C_GRID_LINE,
    thickness: int = 1,
) -> None:
    """
    Draw the bilinear calibration grid onto *canvas* in-place.

    grid_pts is the [NH][NV][2] array stored in grid_json — identical structure
    to the JS gridPtsArr used in the /settings page canvas.  NH horizontal
    polylines and NV vertical polylines are drawn, producing the same visual as
    the calibration preview.

    Matches the settings page JS logic exactly:
        for r in 0..NH: polyline over gp[r][0..NV]
        for c in 0..NV: polyline over gp[0..NH][c]
    """
    gp = np.array(grid_pts, dtype=np.float32)   # (NH, NV, 2)
    if gp.ndim != 3 or gp.shape[2] != 2:
        return
    nh, nv = gp.shape[:2]

    # Horizontal lines — one polyline per row
    for r in range(nh):
        pts = gp[r, :, :].astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness)

    # Vertical lines — one polyline per column
    for c in range(nv):
        pts = gp[:, c, :].astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness)


def _render_annotated_canvas(
    image: np.ndarray,
    validation_results: dict,
    grid_cfg,                                          # unused — kept for call-site compat
    slot_centers: "list[tuple[int,int]] | None" = None,
    layout_map: "dict[str,str] | None" = None,
    grid_pts: "list | None" = None,
) -> np.ndarray:
    """
    Draw annotation layers onto *image* and return a 50%-scaled canvas.

    Shared by save_visual_log() and generate_visual_report() (Telegram).
    Never modifies the original array.

    Layers (bottom → top):
      1. Calibrated bilinear grid from grid_pts (mirrors /settings page exactly).
         Falls back to slot-centre-derived lines when grid_pts is unavailable.
      2. Green bounding rectangle per detected bottle.
      3. Coordinate label "A01"–"M15" in bright yellow, only on error slots.
    """
    canvas = image.copy()
    h, w   = canvas.shape[:2]
    slots  = validation_results.get("slots", {})

    sc  = max(w, h) / 1920.0
    lw  = max(1, int(sc * 1.5))    # grid line width
    blw = max(3, int(sc * 4))      # bounding box line width
    fsc = max(0.8, sc * 0.85)      # cv2 font scale
    ftk = max(2, int(sc * 2))      # cv2 font thickness

    # ── 1. Calibrated grid ────────────────────────────────────────────────
    if grid_pts is not None:
        # Primary: bilinear grid from /settings calibration (same as canvas overlay)
        _draw_calibrated_grid(canvas, grid_pts, color=_C_GRID_LINE, thickness=lw)
    elif slot_centers and len(slot_centers) == 195:
        # Fallback: approximate grid from slot centres (no zone dividers)
        centers = np.array(slot_centers, dtype=np.float32)
        row_ys  = [int(centers[r * 15:(r + 1) * 15, 1].mean()) for r in range(13)]
        col_xs  = [int(centers[c::15, 0].mean()) for c in range(15)]
        row_half = max((row_ys[1] - row_ys[0]) // 2, 20) if len(row_ys) > 1 else 20
        col_half = max((col_xs[1] - col_xs[0]) // 2, 20) if len(col_xs) > 1 else 20
        gx0 = col_xs[0] - col_half
        gx1 = col_xs[-1] + col_half
        gy0 = row_ys[0] - row_half
        gy1 = row_ys[-1] + row_half
        h_lines = ([row_ys[0] - row_half] +
                   [(row_ys[i] + row_ys[i + 1]) // 2 for i in range(len(row_ys) - 1)] +
                   [row_ys[-1] + row_half])
        v_lines = ([col_xs[0] - col_half] +
                   [(col_xs[i] + col_xs[i + 1]) // 2 for i in range(len(col_xs) - 1)] +
                   [col_xs[-1] + col_half])
        for y in h_lines:
            cv2.line(canvas, (gx0, y), (gx1, y), _C_GRID_LINE, lw)
        for x in v_lines:
            cv2.line(canvas, (x, gy0), (x, gy1), _C_GRID_LINE, lw)

    # ── 2. Bounding boxes (all detections) + labels (error slots only) ────
    for pos_key, hit in slots.items():
        cx = hit.get("cx", 0)
        cy = hit.get("cy", 0)
        r  = hit.get("radius", max(30, int(sc * 40)))

        x1, y1 = cx - r, cy - r
        x2, y2 = cx + r, cy + r

        cv2.rectangle(canvas, (x1, y1), (x2, y2), _C_BOX, blw)

        # ── 3. Label only error slots ─────────────────────────────────────
        if not (hit.get("wrong_color", False) or hit.get("duplicate", False)):
            continue

        label = (layout_map.get(pos_key) if layout_map else None) or _pos_to_label(pos_key)
        lbl_x = x1
        lbl_y = max(int(fsc * 24) + 4, y1 - 6)

        cv2.putText(canvas, label, (lbl_x + 2, lbl_y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fsc, _C_LABEL_DARK, ftk + 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (lbl_x, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, fsc, _C_LABEL, ftk, cv2.LINE_AA)

    return cv2.resize(canvas, (w // 2, h // 2), interpolation=cv2.INTER_AREA)


def save_visual_log(
    image: np.ndarray,
    validation_results: dict,
    grid_cfg,
    filepath: "str | Path",
    jpeg_quality: int = 90,
    slot_centers: "list[tuple[int,int]] | None" = None,
    layout_map: "dict[str,str] | None" = None,
    grid_pts: "list | None" = None,
) -> None:
    """Render the annotated diagnostic image and save to *filepath* as JPEG."""
    if image is None or image.size == 0:
        return
    out = _render_annotated_canvas(
        image, validation_results, grid_cfg, slot_centers, layout_map, grid_pts
    )
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(filepath), out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])


def generate_visual_report(
    image: np.ndarray,
    validation_results: dict,
    grid_cfg=None,
    jpeg_quality: int = 90,
    slot_centers: "list[tuple[int,int]] | None" = None,
    layout_map: "dict[str,str] | None" = None,
    grid_pts: "list | None" = None,
) -> bytes:
    """Render the annotated diagnostic image and return raw JPEG bytes (for Telegram)."""
    if image is None or image.size == 0:
        return b""
    out = _render_annotated_canvas(
        image, validation_results, grid_cfg, slot_centers, layout_map, grid_pts
    )
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return buf.tobytes() if ok else b""
