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
_C_GRID_LINE  = (100, 100, 100)   # thin grey — 13×15 grid lines
_C_BOX        = (0,   220,   0)   # bright green — detection bounding box
_C_LABEL      = (0,   255, 255)   # bright yellow — slot coordinate label
_C_LABEL_DARK = (0,     0,   0)   # black — label shadow

# Zone-divider BGR colors (4 thick vertical lines creating 5 color zones)
# Ordered: L0/L1, L1/L2, L2/L3, L3/L4
_C_ZONE_DIVIDERS = [
    (255, 255,   0),   # Cyan    — L0 / L1 boundary (after col 3)
    (255,   0, 255),   # Magenta — L1 / L2 boundary (after col 6)
    (0,   255, 255),   # Yellow  — L2 / L3 boundary (after col 9)
    (0,   140, 255),   # Orange  — L3 / L4 boundary (after col 12)
]

# Zone-divider positions: divider i sits between col (i+1)*3 and (i+1)*3+1 (1-based)
_ZONE_DIVIDER_AFTER_COLS = [3, 6, 9, 12]   # zero-based col index of the LEFT col


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


def _build_grid_lines(
    slot_centers: list[tuple[int, int]],
) -> tuple[list[int], list[int], int, int, int, int]:
    """
    Derive 14 horizontal and 16 vertical grid line positions from 195 slot centres.

    Returns:
        h_lines  — sorted list of 14 y-coordinates (top border … bottom border)
        v_lines  — sorted list of 16 x-coordinates (left border … right border)
        x_min, y_min, x_max, y_max  — grid bounding box
    """
    centers = np.array(slot_centers)              # (195, 2)

    # Row y-centres: mean y across 15 columns for each of 13 rows
    row_ys = [int(centers[r * 15:(r + 1) * 15, 1].mean()) for r in range(13)]
    # Col x-centres: mean x across 13 rows for each of 15 columns
    col_xs = [int(centers[c::15, 0].mean()) for c in range(15)]

    # Half-spacings for border lines
    row_half = max((row_ys[1] - row_ys[0]) // 2, 20) if len(row_ys) > 1 else 20
    col_half = max((col_xs[1] - col_xs[0]) // 2, 20) if len(col_xs) > 1 else 20

    h_lines = [row_ys[0] - row_half]
    for i in range(len(row_ys) - 1):
        h_lines.append((row_ys[i] + row_ys[i + 1]) // 2)
    h_lines.append(row_ys[-1] + row_half)

    v_lines = [col_xs[0] - col_half]
    for i in range(len(col_xs) - 1):
        v_lines.append((col_xs[i] + col_xs[i + 1]) // 2)
    v_lines.append(col_xs[-1] + col_half)

    return (
        h_lines, v_lines,
        v_lines[0], h_lines[0], v_lines[-1], h_lines[-1],
    )


def _render_annotated_canvas(
    image: np.ndarray,
    validation_results: dict,
    grid_cfg,
    slot_centers: "list[tuple[int,int]] | None" = None,
    layout_map: "dict[str,str] | None" = None,
) -> np.ndarray:
    """
    Draw annotation layers onto *image* and return a 50%-scaled canvas.

    Shared by save_visual_log() and generate_visual_report() (Telegram).
    Never modifies the original array.

    Layers (bottom → top):
      1. 13×15 grid — thin grey lines derived from slot_centers (if provided)
      2. Zone dividers — 4 thick colored vertical lines (every 3 columns)
      3. Green bounding rectangle per detected bottle
      4. Coordinate label "A01"–"M15" in bright yellow, above each box
    """
    canvas = image.copy()
    h, w   = canvas.shape[:2]
    slots  = validation_results.get("slots", {})

    sc  = max(w, h) / 1920.0
    lw  = max(1, int(sc * 1.5))    # thin grid line width
    blw = max(3, int(sc * 4))      # bounding box line width
    zlw = max(6, int(sc * 8))      # zone divider line width
    fsc = max(0.8, sc * 0.85)      # cv2 font scale
    ftk = max(2, int(sc * 2))      # cv2 font thickness

    # ── 1. 13×15 grid lines ───────────────────────────────────────────────
    if slot_centers and len(slot_centers) == 195:
        h_lines, v_lines, gx0, gy0, gx1, gy1 = _build_grid_lines(slot_centers)

        # Horizontal lines
        for y in h_lines:
            cv2.line(canvas, (gx0, y), (gx1, y), _C_GRID_LINE, lw)

        # Vertical lines (thin)
        for x in v_lines:
            cv2.line(canvas, (x, gy0), (x, gy1), _C_GRID_LINE, lw)

        # ── 2. Zone dividers (thick colored) ─────────────────────────────
        # v_lines[0] = left border, v_lines[1..15] = between-col lines, v_lines[16] = right border
        # Divider after col C (0-based): v_lines[C + 1]
        for div_idx, after_col in enumerate(_ZONE_DIVIDER_AFTER_COLS):
            line_idx = after_col + 1           # index into v_lines (1-based between-col)
            if line_idx < len(v_lines):
                x = v_lines[line_idx]
                color = _C_ZONE_DIVIDERS[div_idx]
                cv2.line(canvas, (x, gy0), (x, gy1), color, zlw)

    # ── 3. Bounding boxes (all detections) + labels (error slots only) ────
    for pos_key, hit in slots.items():
        cx = hit.get("cx", 0)
        cy = hit.get("cy", 0)
        r  = hit.get("radius", max(30, int(sc * 40)))

        x1, y1 = cx - r, cy - r
        x2, y2 = cx + r, cy + r

        # Green bounding rectangle for every detected bottle
        cv2.rectangle(canvas, (x1, y1), (x2, y2), _C_BOX, blw)

        # ── 4. Label only error slots ─────────────────────────────────────
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
) -> None:
    """
    Render the annotated diagnostic image and save to *filepath* as JPEG.

    slot_centers: optional list of 195 (x, y) tuples from GridDetector (V2).
    layout_map: optional {position_index_str: label} from tray.layout_json.
    """
    if image is None or image.size == 0:
        return
    out = _render_annotated_canvas(image, validation_results, grid_cfg, slot_centers, layout_map)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(filepath), out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])


def generate_visual_report(
    image: np.ndarray,
    validation_results: dict,
    grid_cfg=None,
    jpeg_quality: int = 90,
    slot_centers: "list[tuple[int,int]] | None" = None,
    layout_map: "dict[str,str] | None" = None,
) -> bytes:
    """
    Render the annotated diagnostic image and return raw JPEG bytes (for Telegram).

    slot_centers: optional list of 195 (x, y) tuples from GridDetector (V2).
    layout_map: optional {position_index_str: label} from tray.layout_json.
    """
    if image is None or image.size == 0:
        return b""
    out = _render_annotated_canvas(image, validation_results, grid_cfg, slot_centers, layout_map)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return buf.tobytes() if ok else b""
