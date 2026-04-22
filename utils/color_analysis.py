"""
Color analysis engine: CIE Lab color space + Delta E classification.

Core design principles (from CLAUDE.md):
- ALL classification uses CIE Lab Delta E — never raw RGB or HSV.
- NEVER trust absolute color values. Always compare RELATIVE to the
  reference row captured in the same frame under the same lighting.
- The reference row self-calibrates every scan cycle.

Public API:
    delta_e_cie76(lab1, lab2)           -> float
    extract_bottle_color(frame, cx, cy, radius, inner_crop_px) -> (L, a, b) | None
    build_reference_baseline(frame, reference_positions)       -> {level: (L,a,b)}
    classify_sample(sample_lab, baseline, threshold)           -> (level, delta_e, confident)
"""

import json
import cv2
import numpy as np
from pathlib import Path
from loguru import logger

from configs.config import INNER_CROP_PX, CONFIDENCE_MARGIN, \
                           GLARE_L_THRESHOLD, GLARE_MIN_VALID_PX, REF_INNER_CROP_PX

# OpenCV 8-bit Lab representation of pure white (L=255, a=128, b=128)
WHITE_LAB = (255.0, 128.0, 128.0)


# ---------------------------------------------------------------------------
# Delta E
# ---------------------------------------------------------------------------

def delta_e_cie76(lab1, lab2):
    """
    CIE76 Delta E between two CIE Lab colors.

    Args:
        lab1: (L, a, b) tuple
        lab2: (L, a, b) tuple

    Returns:
        Perceptual color distance (float). 0 = identical.
    """
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2))))


# ---------------------------------------------------------------------------
# Color extraction
# ---------------------------------------------------------------------------

def extract_bottle_color(frame, cx, cy, radius, inner_crop_px=INNER_CROP_PX):
    """
    Extract the median CIE Lab color from the center of a detected bottle cap.

    An inner crop is applied (shrinking by inner_crop_px on each side) to
    exclude cap edges and focus on the liquid color beneath the cap.

    Args:
        frame:          BGR image (numpy array, full resolution)
        cx, cy:         Circle center in image pixel coordinates
        radius:         Circle radius in pixels
        inner_crop_px:  Pixels to shrink from each side before sampling

    Returns:
        (L, a, b) tuple of median CIE Lab values, or None if the crop is invalid.
    """
    r = int(radius)
    x1 = int(cx) - r + inner_crop_px
    y1 = int(cy) - r + inner_crop_px
    x2 = int(cx) + r - inner_crop_px
    y2 = int(cy) + r - inner_crop_px

    # Clamp to image bounds
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Convert BGR → CIE Lab
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)

    # Exclude glare pixels (over-exposed hotspot at cap center)
    non_glare = lab[:, :, 0] < GLARE_L_THRESHOLD
    if non_glare.sum() >= GLARE_MIN_VALID_PX:
        L = float(np.median(lab[:, :, 0][non_glare]))
        a = float(np.median(lab[:, :, 1][non_glare]))
        b = float(np.median(lab[:, :, 2][non_glare]))
    else:
        L = float(np.median(lab[:, :, 0]))
        a = float(np.median(lab[:, :, 1]))
        b = float(np.median(lab[:, :, 2]))

    return (L, a, b)


# ---------------------------------------------------------------------------
# Reference baseline
# ---------------------------------------------------------------------------

def build_reference_baseline(frame, reference_positions):
    """
    Sample all 15 reference bottles and compute per-level live Lab standards.

    Called once per scan cycle BEFORE classifying any sample bottle.
    The returned baseline is specific to the current frame's lighting.

    Args:
        frame:                BGR image (full frame)
        reference_positions:  dict {level (int): [(cx, cy, radius), ...]}
                              Each level has REFS_PER_LEVEL (3) positions.

    Returns:
        dict {level: (L, a, b)} — median Lab value per color level.
        Only levels with at least one valid sample are included.
    """
    baseline = {}

    for level, positions in reference_positions.items():
        labs = []
        for cx, cy, radius in positions:
            lab = extract_bottle_color(frame, cx, cy, radius,
                                       inner_crop_px=REF_INNER_CROP_PX)
            if lab is not None:
                labs.append(lab)

        if not labs:
            logger.warning("Baseline L{}: no valid samples extracted (all ref positions failed)", level)
            continue

        L = float(np.mean([v[0] for v in labs]))
        a = float(np.mean([v[1] for v in labs]))
        b = float(np.mean([v[2] for v in labs]))
        baseline[level] = (L, a, b)
        logger.info("Baseline L{}: Lab=({:.1f}, {:.1f}, {:.1f}) from {} ref(s)", level, L, a, b, len(labs))

    return baseline


# ---------------------------------------------------------------------------
# Static baseline (from color.json)
# ---------------------------------------------------------------------------

def load_static_baseline(path="color.json"):
    """
    Load pre-saved reference Lab colors from color.json.

    Returns the same dict shape as build_reference_baseline() so it can be
    used as a drop-in replacement when color.json exists.

    Args:
        path: path to color.json (default "color.json")

    Returns:
        dict {level (int): (L, a, b)} in OpenCV 8-bit Lab encoding,
        or empty dict if the file is missing or malformed.
    """
    try:
        data = json.loads(Path(path).read_text())
        return {
            int(lvl): tuple(float(v) for v in info["lab"])
            for lvl, info in data["baseline"].items()
        }
    except Exception:
        return {}


def build_kmeans_centroids(path="color.json"):
    """
    Compute K-means initial centroids (k=5) from the 3 individual reference
    bottle Lab values stored in color.json under the "bottles" key.

    Each centroid is the mean of the 3 reference bottles for that class,
    computed directly from the raw Lab measurements rather than from the
    pre-averaged "baseline" key.  This makes the centroid origin explicit
    and independent of how the file was generated.

    Centroid assignment uses nearest-centroid (minimum Delta E CIE76 in Lab
    space), which is equivalent to the assignment step of K-means with fixed
    centroids — no iteration needed because the reference set is already the
    ground truth.

    Args:
        path: path to color.json (default "color.json")

    Returns:
        dict {level (int): (L, a, b)} — one centroid per color class (0–4).
        Returns empty dict if the file is missing, malformed, or has no
        "bottles" key with valid Lab values.
    """
    try:
        data = json.loads(Path(path).read_text())
        bottles = data.get("bottles", {})
        if not bottles:
            return {}

        centroids: dict[int, tuple[float, float, float]] = {}
        for cls_str, refs in bottles.items():
            cls = int(cls_str)
            labs = [
                (float(b["lab"][0]), float(b["lab"][1]), float(b["lab"][2]))
                for b in refs
                if "lab" in b and len(b["lab"]) == 3
            ]
            if not labs:
                continue
            L = float(np.mean([v[0] for v in labs]))
            a = float(np.mean([v[1] for v in labs]))
            b = float(np.mean([v[2] for v in labs]))
            centroids[cls] = (L, a, b)

        return centroids
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_sample(sample_lab, baseline, margin=CONFIDENCE_MARGIN):
    """
    Classify a sample bottle by nearest-reference matching (minimum Delta E).

    Confidence uses a margin-of-victory check rather than a fixed threshold:
    confident when (second_best_ΔE − best_ΔE) > margin. This is relative to
    the current frame's lighting — the system always assigns the nearest level,
    and margin only controls how unambiguous the match is.

    Args:
        sample_lab: (L, a, b) of the sample bottle
        baseline:   dict {level: (L, a, b)} from build_reference_baseline
        margin:     Min gap between best and second-best ΔE for confidence

    Returns:
        (level, delta_e, confident)
        - level:     int 0-4 — level with the absolute minimum Delta E
        - delta_e:   float — distance to the closest standard
        - confident: bool — True if margin of victory exceeds threshold
    """
    if not baseline or sample_lab is None:
        return None, float('inf'), False

    deltas = {
        level: delta_e_cie76(sample_lab, ref_lab)
        for level, ref_lab in baseline.items()
    }

    sorted_levels = sorted(deltas, key=lambda lvl: deltas[lvl])
    best_level = sorted_levels[0]
    best_delta = deltas[best_level]

    if len(sorted_levels) >= 2:
        second_best_delta = deltas[sorted_levels[1]]
        confident = (second_best_delta - best_delta) > margin
    else:
        confident = True   # only one reference level available

    return best_level, best_delta, confident
