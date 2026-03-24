"""
Shared image-processing utilities used by the server for pre- and post-processing.

letterbox_white_padding()
    Resize any image to target_size × target_size using white (255, 255, 255) fill.
    White padding matches the Roboflow "Fit" pre-processing used during training,
    preventing false-contrast edges that black padding would introduce near bottles.

scale_coordinates()
    Inverse-transform bounding boxes from the 640 × 640 letterboxed space back to
    the original camera coordinate system (e.g. 4608 × 2592).

find_slot_conflicts(sample_hits)
    Detect physical-slot duplicates: same base slot ID (e.g. "A11") detected in
    more than one color-level zone (e.g. both "A11_0" and "A11_2").

validate_color_zones(sample_hits, classified_levels)
    Detect wrong-color-zone placements: bottle's classified color level does not
    match the expected level encoded in its slot_id suffix.

save_visual_log()
    Draw full diagnostic overlay (grid polygons, slot labels, K-means results,
    error markers) on the original image and save as a high-quality JPEG.
    Uses Pillow for Thai text rendering; falls back to ASCII if no TTF font found.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


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
) -> list[list[float]]:
    """
    Map bounding boxes from 640 × 640 padded space back to the original image space.

    Args:
        boxes:  List of [x1, y1, x2, y2, conf, cls, ...] in letterboxed coordinates.
        scale:  The *scale* value returned by letterbox_white_padding().
        pad_x:  The *pad_x* value returned by letterbox_white_padding().
        pad_y:  The *pad_y* value returned by letterbox_white_padding().

    Returns:
        Same list structure with x1/y1/x2/y2 in original image coordinates.
        conf, cls, and any extra fields are passed through unchanged.
    """
    out = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        ox1 = (x1 - pad_x) / scale
        oy1 = (y1 - pad_y) / scale
        ox2 = (x2 - pad_x) / scale
        oy2 = (y2 - pad_y) / scale
        out.append([ox1, oy1, ox2, oy2, *box[4:]])
    return out


def find_slot_conflicts(sample_hits: dict) -> list[dict]:
    """
    Detect physical-slot duplicates across color-level zones.

    A "physical slot" is the base part of a slot_id before the underscore
    (e.g. "A11" from "A11_1").  If the same base appears in more than one
    detected slot, that means a single physical slot was assigned to multiple
    level zones — a Duplicate/Overlap Error.

    Args:
        sample_hits: {slot_id: any} mapping of detected slots.

    Returns:
        List of conflict dicts: [{"base": "A11", "slots": ["A11_0", "A11_2"]}, ...]
    """
    groups: dict[str, list[str]] = {}
    for slot_id in sample_hits:
        base = slot_id.split("_")[0]
        groups.setdefault(base, []).append(slot_id)
    return [{"base": b, "slots": s} for b, s in groups.items() if len(s) > 1]


def validate_color_zones(
    sample_hits: dict,
    classified_levels: dict[str, int],
) -> list[dict]:
    """
    Detect Wrong Color Zone placements.

    For each detected slot, compare the expected color level encoded in the
    slot_id suffix (e.g. level 1 in "A11_1") against the level the color
    analysis actually classified.  A mismatch means the bottle is in the wrong
    zone of the tray.

    Args:
        sample_hits:       {slot_id: any} mapping of detected slots.
        classified_levels: {slot_id: level_int} from color analysis.

    Returns:
        List of mismatch dicts:
        [{"slot_id": "A11_1", "expected": 1, "actual": 3}, ...]
    """
    from utils.grid import parse_slot_id

    mismatches = []
    for slot_id in sample_hits:
        actual = classified_levels.get(slot_id)
        if actual is None:
            continue
        expected = parse_slot_id(slot_id)["expected_level"]
        if actual != expected:
            mismatches.append({"slot_id": slot_id, "expected": expected, "actual": actual})
    return mismatches


# ─── Visual logging ───────────────────────────────────────────────────────────

# BGR constants for cv2 geometric drawing
_C_GRID     = (160, 160, 160)   # grey  — sample-slot grid outlines
_C_REF_LINE = (0,   200, 255)   # amber — reference-row outlines
_C_BOX      = (0,   220,   0)   # bright green — detection bounding box (always)
_C_RING_OK  = (0,   255,   0)   # bright green ring — validation OK
_C_RING_ERR = (0,     0, 255)   # bright red ring   — any validation error

# RGB constants for PIL text (R, G, B)
_T_WHITE  = (255, 255, 255)
_T_RED    = (220,   0,   0)
_T_AMBER  = (255, 200,   0)
_T_SHADOW = (  0,   0,   0)

# Thai TTF font candidates (tried in order, first match wins)
_THAI_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/thai-tlwg/Waree.ttf",
    "/usr/share/fonts/truetype/thai-tlwg/Loma.ttf",
    "/usr/share/fonts/truetype/thai-tlwg/Sarabun.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThaiUI-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


def _get_pil_font(size: int):
    """Return a PIL ImageFont at *size* px, or None if Pillow is unavailable."""
    try:
        from PIL import ImageFont
        for path in _THAI_FONT_CANDIDATES:
            if Path(path).exists():
                return ImageFont.truetype(path, size)
        return ImageFont.load_default()
    except ImportError:
        return None


def _render_annotated_canvas(
    image: np.ndarray,
    validation_results: dict,
    grid_cfg,
) -> np.ndarray:
    """
    Draw all annotation layers onto *image* and return a 50%-scaled canvas.

    Shared by save_visual_log() (saves to disk) and generate_visual_report()
    (returns bytes for Telegram).  Never modifies the original array.

    Layers (bottom → top):
      1. Sample-slot polygon outlines — grey, thin
      2. Reference-row polygon outlines — amber, slightly thicker
      3. Green bounding box per bottle (detection result, always green)
      4. Color-coded ring outside the box:
            Bright green → OK   (correct colour, no duplicate)
            Bright red   → FAIL (wrong colour and/or duplicate)
      5. Text labels — one PIL pass for Thai glyph support:
            Line 1: "{slot_id}: สี {N}"      white when OK, red when FAIL
            Line 2: "Error: สีผิด/วางซ้ำ"   red, only on FAIL
      6. Reference-row banner: "แถวอ้างอิง K-means (5 ระดับสี)"
    """
    canvas = image.copy()
    h, w   = canvas.shape[:2]
    slots  = validation_results.get("slots", {})

    sc     = max(w, h) / 1920.0
    lw     = max(2, int(sc * 2))
    blw    = max(4, int(sc * 4))
    rlw    = max(6, int(sc * 7))
    rm     = max(12, int(sc * 16))
    fsc    = max(1.0, sc * 0.9)
    ftk    = max(2,   int(sc * 2))
    pil_sm = max(28,  int(sc * 32))
    pil_lg = max(44,  int(sc * 52))

    # ── 1. Sample-slot grid outlines ──────────────────────────────────────
    if grid_cfg is not None:
        for info in grid_cfg.slot_data.values():
            pts = info["coords"].reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(canvas, [pts], True, _C_GRID, lw)

    # ── 2. Reference-row outlines ─────────────────────────────────────────
    if grid_cfg is not None:
        for info in grid_cfg.reference_slots.values():
            pts = info["coords"].reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(canvas, [pts], True, _C_REF_LINE, lw + 2)

    # ── 3 & 4. Bounding boxes + validation rings ──────────────────────────
    for hit in slots.values():
        cx = hit["cx"]
        cy = hit["cy"]
        r  = hit.get("radius", max(30, int(sc * 40)))
        cv2.rectangle(canvas, (cx - r, cy - r), (cx + r, cy + r), _C_BOX, blw)
        ring_col = _C_RING_OK if hit.get("ok", True) else _C_RING_ERR
        cv2.circle(canvas, (cx, cy), r + rm, ring_col, rlw)

    # ── 5. Text labels — single PIL pass ─────────────────────────────────
    font_sm = _get_pil_font(pil_sm)
    use_pil = font_sm is not None

    if use_pil:
        try:
            from PIL import Image as _PI, ImageDraw as _PD
            font_lg = _get_pil_font(pil_lg)
            pil_img = _PI.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw    = _PD.Draw(pil_img)

            for slot_id, hit in slots.items():
                cx     = hit["cx"]
                cy     = hit["cy"]
                r      = hit.get("radius", max(30, int(sc * 40)))
                level  = hit.get("level")
                is_ok  = hit.get("ok", True)
                is_dup = hit.get("duplicate", False)
                is_wc  = hit.get("wrong_color", False)

                text_col = _T_WHITE if is_ok else _T_RED
                lbl1     = (f"{slot_id}: สี {level}"
                            if level is not None else f"{slot_id}: สี ?")
                tx = cx - r
                ty = max(0, cy - r - rm - pil_sm * 2 - 8)
                draw.text((tx + 2, ty + 2), lbl1, font=font_sm, fill=_T_SHADOW)
                draw.text((tx,     ty),     lbl1, font=font_sm, fill=text_col)

                if not is_ok:
                    err_str = ("Error: วางซ้ำ+สีผิด" if (is_dup and is_wc) else
                               "Error: วางซ้ำ"        if is_dup else
                               "Error: สีผิด")
                    ey = ty + pil_sm + 4
                    draw.text((tx + 2, ey + 2), err_str, font=font_sm, fill=_T_SHADOW)
                    draw.text((tx,     ey),     err_str, font=font_sm, fill=_T_RED)

            # ── 6. Reference-row banner ──────────────────────────────────
            if grid_cfg is not None and grid_cfg.reference_slots:
                ref_ys = [int(np.min(info["coords"][:, 1]))
                          for info in grid_cfg.reference_slots.values()]
                by = max(0, min(ref_ys) - pil_lg - 8)
                banner = "แถวอ้างอิง K-means (5 ระดับสี)"
                draw.text((12, by + 2), banner, font=font_lg, fill=_T_SHADOW)
                draw.text((10, by),     banner, font=font_lg, fill=_T_AMBER)

            canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        except Exception:
            pass  # fall through to ASCII path

    if not use_pil:
        # cv2 ASCII fallback (no Thai)
        for slot_id, hit in slots.items():
            cx     = hit["cx"]
            cy     = hit["cy"]
            r      = hit.get("radius", max(30, int(sc * 40)))
            level  = hit.get("level")
            is_ok  = hit.get("ok", True)
            is_dup = hit.get("duplicate", False)
            is_wc  = hit.get("wrong_color", False)
            col    = _C_BOX if is_ok else (0, 0, 220)
            lbl1   = f"{slot_id}: L{level}" if level is not None else f"{slot_id}: L?"
            tx = cx - r
            ty = max(20, cy - r - rm - int(fsc * 30) * 2)
            cv2.putText(canvas, lbl1, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fsc, col, ftk, cv2.LINE_AA)
            if not is_ok:
                err_str = ("DUP+ERR" if (is_dup and is_wc) else
                           "DUP"     if is_dup else "ERR")
                cv2.putText(canvas, f"Error: {err_str}", (tx, ty + int(fsc * 34)),
                            cv2.FONT_HERSHEY_SIMPLEX, fsc, (0, 0, 220), ftk, cv2.LINE_AA)
        if grid_cfg is not None and grid_cfg.reference_slots:
            ref_ys = [int(np.min(info["coords"][:, 1]))
                      for info in grid_cfg.reference_slots.values()]
            by = max(40, min(ref_ys) - int(fsc * 42))
            cv2.putText(canvas, "K-means Ref Row (5 levels)", (10, by),
                        cv2.FONT_HERSHEY_SIMPLEX, fsc * 1.1, _C_REF_LINE, ftk + 1, cv2.LINE_AA)

    return cv2.resize(canvas, (w // 2, h // 2), interpolation=cv2.INTER_AREA)


def save_visual_log(
    image: np.ndarray,
    validation_results: dict,
    grid_cfg,
    filepath: "str | Path",
    jpeg_quality: int = 90,
) -> None:
    """
    Render the annotated diagnostic image and save to *filepath* as JPEG.

    Calls _render_annotated_canvas() then writes the 50%-scaled result.
    Parent directories are created automatically.
    """
    if image is None or image.size == 0:
        return
    out = _render_annotated_canvas(image, validation_results, grid_cfg)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(filepath), out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])


def generate_visual_report(
    image: np.ndarray,
    validation_results: dict,
    grid_cfg=None,
    jpeg_quality: int = 90,
) -> bytes:
    """
    Render the annotated diagnostic image and return it as raw JPEG bytes.

    Intended for in-memory delivery (e.g. Telegram sendPhoto) without writing
    to disk.  Shares the same drawing pipeline as save_visual_log() via the
    common _render_annotated_canvas() helper.

    Args:
        image:              Original full-resolution BGR frame.
        validation_results: Same structure accepted by save_visual_log().
        grid_cfg:           GridConfig instance, or None.
        jpeg_quality:       0–100 (default 90).

    Returns:
        Raw JPEG bytes ready to pass to requests.post(files=...) or
        telegram sendPhoto multipart upload.
    """
    if image is None or image.size == 0:
        return b""
    out = _render_annotated_canvas(image, validation_results, grid_cfg)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return buf.tobytes() if ok else b""
