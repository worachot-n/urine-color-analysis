"""
Color analysis engine: CIE Lab color space + Delta E classification.

Core design principles (from CLAUDE.md):
- ALL classification uses CIE Lab Delta E — never raw RGB or HSV.
- NEVER trust absolute color values. Always compare RELATIVE to the
  reference row captured in the same frame under the same lighting.
- The reference row self-calibrates every scan cycle.

Public API:
    delta_e_cie76(lab1, lab2)           -> float
    extract_bottle_color(frame, cx, cy, radius, outer_crop_px, inner_crop_px) -> (L, a, b) | None
    build_reference_baseline(frame, reference_positions)       -> {level: (L,a,b)}
    classify_sample(sample_lab, baseline, threshold)           -> (level, delta_e, confident)
"""

import json
import cv2
import numpy as np
from pathlib import Path
from loguru import logger

import tomllib
_cfg = tomllib.load(open(Path(__file__).parent.parent / "configs" / "config.toml", "rb"))
_ca = _cfg.get("color_analysis", {})
_ip = _cfg.get("image_processing", {})
OUTER_CROP_PX: int       = int(_ca.get("outer_crop_px", 15))
INNER_CROP_PX: int       = int(_ca.get("inner_crop_px", 10))
CONFIDENCE_MARGIN: float = float(_ca.get("confidence_margin", 3.0))
MAX_DELTA_E: float       = float(_ca.get("max_delta_e", 18.0))
GLARE_L_THRESHOLD: float = float(_ip.get("glare_l_threshold", 220))
GLARE_MIN_VALID_PX: int  = int(_ip.get("glare_min_valid_px", 10))

# OpenCV 8-bit Lab representation of pure white (L=255, a=128, b=128)
WHITE_LAB = (255.0, 128.0, 128.0)


# ---------------------------------------------------------------------------
# Delta E
# ---------------------------------------------------------------------------

def delta_e_cie76(lab1, lab2):
    """
    CIE76 Delta E between two CIE Lab colors (full 3D Euclidean distance).

    Args:
        lab1: (L, a, b) tuple
        lab2: (L, a, b) tuple

    Returns:
        Perceptual color distance (float). 0 = identical.
    """
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2))))


def delta_e_chroma(lab1, lab2):
    """
    Chromaticity-only Delta E — Euclidean distance in the (a*, b*) plane.

    Ignores L* (lightness). Used for urine-color level classification because
    concentration manifests in hue (yellow → amber → red) while lightness varies
    with bottle fill level and lighting drift. Robust to those nuisances.
    """
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    return float(np.hypot(da, db))


# ---------------------------------------------------------------------------
# Color extraction
# ---------------------------------------------------------------------------

def extract_bottle_color(frame, cx, cy, radius,
                          outer_crop_px=OUTER_CROP_PX,
                          inner_crop_px=INNER_CROP_PX):
    """
    Extract the median CIE Lab color from an annular ring on a detected bottle cap.

    Sampling region = ring between inner_crop_px (from center) and
    (radius - outer_crop_px) (from center outward). This avoids:
      - outer_crop_px: plastic cap ring / edge artifacts
      - inner_crop_px: specular glare hot-spot at cap center

    Args:
        frame:          BGR image (numpy array, full resolution)
        cx, cy:         Circle center in image pixel coordinates
        radius:         Circle radius in pixels
        outer_crop_px:  Pixels to shrink from bottle edge (outer ring boundary)
        inner_crop_px:  Pixels from center to exclude (inner ring boundary)

    Returns:
        (L, a, b) tuple of median CIE Lab values, or None if the ring is invalid.
    """
    r = int(radius)
    # Bounding box for the outer boundary
    x1 = max(0, int(cx) - r + outer_crop_px)
    y1 = max(0, int(cy) - r + outer_crop_px)
    x2 = min(frame.shape[1], int(cx) + r - outer_crop_px)
    y2 = min(frame.shape[0], int(cy) + r - outer_crop_px)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Annular mask: exclude pixels closer than inner_crop_px to the bottle center
    h_crop, w_crop = crop.shape[:2]
    Y, X = np.ogrid[:h_crop, :w_crop]
    dist = np.sqrt((X - (int(cx) - x1)) ** 2 + (Y - (int(cy) - y1)) ** 2)
    ring_mask = dist >= inner_crop_px  # True = in the usable ring

    if ring_mask.sum() < GLARE_MIN_VALID_PX:
        return None

    # Convert BGR → CIE Lab
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)

    # Combine ring mask with glare filter
    non_glare = lab[:, :, 0] < GLARE_L_THRESHOLD
    combined  = ring_mask & non_glare
    mask = combined if combined.sum() >= GLARE_MIN_VALID_PX else ring_mask

    L = float(np.median(lab[:, :, 0][mask]))
    a = float(np.median(lab[:, :, 1][mask]))
    b = float(np.median(lab[:, :, 2][mask]))

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
            lab = extract_bottle_color(frame, cx, cy, radius)
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

def classify_sample(sample_lab, baseline, margin=CONFIDENCE_MARGIN, max_delta_e=MAX_DELTA_E):
    """
    Classify a sample bottle by nearest-reference matching in chromaticity space.

    Distance: chromaticity-only (a*, b*) — ignores L* so the result is robust
    to bottle-fill and lighting variations that don't reflect concentration.

    Rejection: if the best ΔE_chroma exceeds `max_delta_e`, the sample is
    reported as out-of-range (level=None) instead of being forced to a noisy
    nearest neighbour.

    Confidence: still uses margin-of-victory (`second_best − best > margin`)
    on top of the rejection check.

    Returns:
        (level, delta_e, confident)
        - level:     int 0-4 if classified, or None if out of range
        - delta_e:   float — chromaticity distance to the closest reference
        - confident: bool — True only when both within range AND margin met
    """
    if not baseline or sample_lab is None:
        return None, float('inf'), False

    deltas = {
        level: delta_e_chroma(sample_lab, ref_lab)
        for level, ref_lab in baseline.items()
    }

    sorted_levels = sorted(deltas, key=lambda lvl: deltas[lvl])
    best_level = sorted_levels[0]
    best_delta = deltas[best_level]

    # Absolute-distance rejection: every reference is too far → no honest match
    if best_delta > max_delta_e:
        return None, best_delta, False

    if len(sorted_levels) >= 2:
        second_best_delta = deltas[sorted_levels[1]]
        confident = (second_best_delta - best_delta) > margin
    else:
        confident = True   # only one reference level available

    return best_level, best_delta, confident
