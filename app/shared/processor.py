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
"""

from __future__ import annotations

import cv2
import numpy as np


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
