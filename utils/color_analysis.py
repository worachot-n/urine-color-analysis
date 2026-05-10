"""
Color analysis engine: CIE Lab color space + hybrid Delta E / histogram classification.

Core design principles (from CLAUDE.md):
- ALL classification uses CIE Lab — never raw RGB or HSV.
- References self-calibrate every scan cycle.

Public API:
    delta_e_cie76(lab1, lab2)            -> float
    delta_e_chroma(lab1, lab2)           -> float
    extract_bottle_color(frame, cx, cy, radius, ...)  -> (L, a, b) | None
    extract_bottle_features(frame, cx, cy, radius, ..., wb_offset)
        -> ((L, a, b) | None, hist | None)
    build_reference_baseline(frame, reference_positions) -> {level: (L,a,b)}
    build_reference_set(frame, reference_positions)      -> {level: [(L,a,b), ...]}
    build_reference_histograms(frame, ref_positions, ..., wb_offset) -> {level: hist}
    compute_white_balance_offset(frame, white_positions, ...) -> (Δa, Δb) | None
    classify_sample(sample_lab, baseline, ...)           -> (level, ΔE, confident)
    classify_sample_nn(sample_lab, refs, ...)            -> (level, ΔE, confident)
    classify_sample_hybrid(sample_lab, sample_hist,
                           ref_set, ref_hists, ...)      -> dict
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
GLARE_L_THRESHOLD: float  = float(_ip.get("glare_l_threshold", 230))
SHADOW_L_THRESHOLD: float = float(_ip.get("shadow_l_threshold", 51))
GLARE_MIN_VALID_PX: int   = int(_ip.get("glare_min_valid_px", 10))

# Histogram + hybrid classifier
HIST_BINS_A: int          = int(_ca.get("hist_bins_a", 16))
HIST_BINS_B: int          = int(_ca.get("hist_bins_b", 16))
_hra = _ca.get("hist_range_a", [80, 180])
_hrb = _ca.get("hist_range_b", [80, 180])
HIST_RANGE_A: tuple[float, float] = (float(_hra[0]), float(_hra[1]))
HIST_RANGE_B: tuple[float, float] = (float(_hrb[0]), float(_hrb[1]))
W_CHROMA: float           = float(_ca.get("weight_chroma", 0.8))
W_HIST: float             = float(_ca.get("weight_hist", 0.2))
CHROMA_SCALE: float       = float(_ca.get("chroma_scale", 18.0))
MAX_COMBINED_SCORE: float = float(_ca.get("max_combined_score", 1.5))
COMBINED_MARGIN: float    = float(_ca.get("combined_margin", 0.10))
LOW_CONFIDENCE_SCORE_THRESHOLD: float = float(_ca.get("low_confidence_score_threshold", 0.6))
REFERENCE_MIN_SATURATION: float       = float(_ca.get("reference_min_saturation", 0.15))

# 1D Trajectory projection
MAX_PATH_DISTANCE: float              = float(_ca.get("max_path_distance", 12.0))
PATH_CONFIDENT_DISTANCE: float        = float(_ca.get("path_confident_distance", 4.0))
SOFT_PENALTY_PATH_DE_THRESHOLD: float = float(_ca.get("soft_penalty_path_de_threshold", 5.0))
SOFT_PENALTY_FACTOR: float            = float(_ca.get("soft_penalty_factor", 0.5))
SATURATION_HIST_WEIGHT: bool          = bool(_ca.get("saturation_hist_weight", True))

# Weighted ΔE — ellipsoid acceptance zones (defaults: b* dominates, a* down-weighted)
W_LAB_L: float                  = float(_ca.get("weight_lab_L", 0.5))
W_LAB_A: float                  = float(_ca.get("weight_lab_a", 0.1))
W_LAB_B: float                  = float(_ca.get("weight_lab_b", 1.0))
SATURATION_L0_THRESHOLD: float  = float(_ca.get("saturation_l0_threshold", 0.05))

if abs((W_CHROMA + W_HIST) - 1.0) > 1e-6:
    logger.warning(
        "color_analysis: weight_chroma + weight_hist = {:.3f} ≠ 1.0 — hybrid score will not be in [0, 1]",
        W_CHROMA + W_HIST,
    )

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


def delta_e_weighted(lab1, lab2, w_L=None, w_a=None, w_b=None):
    """
    Weighted Lab Δ — sqrt(w_L·ΔL² + w_a·Δa² + w_b·Δb²).

    Ellipsoid acceptance zones in 3-D Lab space. Defaults from config:
      w_b = 1.0 — b* (yellow) is the discrimination axis
      w_L = 0.5 — lightness contributes weakly (fill / camera bias is partial signal)
      w_a = 0.1 — a* (red-green) is mostly noise (skin, edge tints, hue drift)

    Substituting defaults: a 20-unit a* mismatch contributes the same as a
    ~6.3-unit b* mismatch — exactly the "tolerate hue, sweat the yellow" intent
    behind the user's spec.
    """
    if w_L is None: w_L = W_LAB_L
    if w_a is None: w_a = W_LAB_A
    if w_b is None: w_b = W_LAB_B
    dL = lab1[0] - lab2[0]
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    return float(np.sqrt(w_L*dL*dL + w_a*da*da + w_b*db*db))


# ---------------------------------------------------------------------------
# Color extraction
# ---------------------------------------------------------------------------

def _ring_pixels_lab(frame, cx, cy, radius,
                     outer_crop_px=OUTER_CROP_PX,
                     inner_crop_px=INNER_CROP_PX,
                     return_hsv=False):
    """
    Return the annular-ring CIE Lab pixels of a bottle cap as an (N, 3) array,
    or None if the ring contains too few valid pixels.

    Strict intensity gating: pixels with `L > GLARE_L_THRESHOLD` (specular
    highlights) or `L < SHADOW_L_THRESHOLD` (deep shadows / cap rim / dark
    container edges) are dropped before the array is returned. If gating
    leaves fewer than GLARE_MIN_VALID_PX pixels, the ring mask is used
    unfiltered as a fallback so well-lit-but-dim bottles still extract.

    Args:
        return_hsv: if True, also return the matching (N, 3) HSV pixels in
                    OpenCV-HSV encoding. Used by saturation-weighted
                    histograms in extract_bottle_features().

    Returns:
        Lab pixels as (N, 3) uint8, or None.
        If return_hsv: (lab_pixels, hsv_pixels) tuple, or (None, None).
    """
    r = int(radius)
    x1 = max(0, int(cx) - r + outer_crop_px)
    y1 = max(0, int(cy) - r + outer_crop_px)
    x2 = min(frame.shape[1], int(cx) + r - outer_crop_px)
    y2 = min(frame.shape[0], int(cy) + r - outer_crop_px)
    if x2 <= x1 or y2 <= y1:
        return (None, None) if return_hsv else None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return (None, None) if return_hsv else None

    h_crop, w_crop = crop.shape[:2]
    Y, X = np.ogrid[:h_crop, :w_crop]
    dist = np.sqrt((X - (int(cx) - x1)) ** 2 + (Y - (int(cy) - y1)) ** 2)
    ring_mask = dist >= inner_crop_px
    if ring_mask.sum() < GLARE_MIN_VALID_PX:
        return (None, None) if return_hsv else None

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0]
    intensity_ok = (L < GLARE_L_THRESHOLD) & (L > SHADOW_L_THRESHOLD)
    combined = ring_mask & intensity_ok
    mask = combined if combined.sum() >= GLARE_MIN_VALID_PX else ring_mask

    lab_pixels = lab[mask]
    if return_hsv:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return lab_pixels, hsv[mask]
    return lab_pixels


def extract_bottle_color(frame, cx, cy, radius,
                          outer_crop_px=OUTER_CROP_PX,
                          inner_crop_px=INNER_CROP_PX):
    """
    Extract the median CIE Lab color from an annular ring on a detected bottle cap.

    Sampling region = ring between inner_crop_px and (radius - outer_crop_px).
    Avoids the plastic cap ring at the edge and the specular glare hot-spot at
    the centre. See _ring_pixels_lab() for the underlying mask logic.

    Returns:
        (L, a, b) tuple of OpenCV-Lab medians, or None if the ring is invalid.
    """
    pixels = _ring_pixels_lab(frame, cx, cy, radius, outer_crop_px, inner_crop_px)
    if pixels is None:
        return None
    L = float(np.median(pixels[:, 0]))
    a = float(np.median(pixels[:, 1]))
    b = float(np.median(pixels[:, 2]))
    return (L, a, b)


def extract_bottle_features(frame, cx, cy, radius,
                             outer_crop_px=OUTER_CROP_PX,
                             inner_crop_px=INNER_CROP_PX,
                             wb_offset=None,
                             min_saturation=0.0):
    """
    Extract BOTH the raw median Lab and a WB-corrected (a*, b*) 2-D histogram
    of the annular ring in one pass.

    Args:
        wb_offset:      optional (Δa, Δb) shift to apply to ring pixels before
                        histogramming, in OpenCV-Lab units (Δa = a_wb − 128).
                        When None, the histogram is computed in raw Lab space.
        min_saturation: hard cutoff on HSV saturation, in [0, 1]. Pixels with
                        S/255 below this are dropped from BOTH the median and
                        the histogram. Default 0 (no cutoff). Used by reference
                        templates to keep them "pure" (no grey/glare pixels).
                        On empty result, falls back to the unfiltered ring.

    Returns:
        (raw_lab, hist) where:
            raw_lab: (L, a, b) tuple of medians on the UNSHIFTED ring pixels,
                     or None if the ring is invalid.
            hist:    np.ndarray, shape (HIST_BINS_A, HIST_BINS_B), float32,
                     normalised to sum to 1. Same None condition as raw_lab.

    The median is intentionally computed on the unshifted pixels so the
    displayed hex / scatter-plot dot reflects the actual measured colour. The
    histogram is shifted so that, across scans, all bottles live in a common
    WB-corrected (a*, b*) space — useful for histogram comparison.

    When `SATURATION_HIST_WEIGHT` is True, each pixel's contribution to the
    histogram is weighted by its HSV saturation (S/255). This suppresses
    near-grey ring pixels (plastic container artefacts, glare residuals) and
    lets saturated liquid pixels dominate.
    """
    need_hsv = SATURATION_HIST_WEIGHT or (min_saturation > 0.0)
    if need_hsv:
        ring = _ring_pixels_lab(
            frame, cx, cy, radius, outer_crop_px, inner_crop_px, return_hsv=True,
        )
        if ring is None or ring[0] is None:
            return None, None
        pixels, hsv_pixels = ring
    else:
        pixels = _ring_pixels_lab(frame, cx, cy, radius, outer_crop_px, inner_crop_px)
        if pixels is None:
            return None, None
        hsv_pixels = None

    # Hard saturation cutoff (templates only): drop pixels whose HSV S is below
    # `min_saturation`. If too few pixels survive, revert to the full ring so
    # we don't lose the bottle entirely on a uniformly desaturated reference.
    sat_filtered = False
    if min_saturation > 0.0 and hsv_pixels is not None:
        sat_norm = hsv_pixels[:, 1].astype(np.float32) / 255.0
        keep_mask = sat_norm >= float(min_saturation)
        if int(keep_mask.sum()) >= GLARE_MIN_VALID_PX:
            pixels = pixels[keep_mask]
            hsv_pixels = hsv_pixels[keep_mask]
            sat_filtered = True
        else:
            logger.warning(
                "extract_bottle_features: only {}/{} pixels passed S≥{:.2f} cutoff "
                "at ({},{}) — falling back to full ring",
                int(keep_mask.sum()), len(sat_norm), min_saturation, int(cx), int(cy),
            )

    weights = None
    if SATURATION_HIST_WEIGHT and hsv_pixels is not None:
        weights = hsv_pixels[:, 1].astype(np.float32) / 255.0

    L = float(np.median(pixels[:, 0]))
    a = float(np.median(pixels[:, 1]))
    b = float(np.median(pixels[:, 2]))

    a_chan = pixels[:, 1].astype(np.float32)
    b_chan = pixels[:, 2].astype(np.float32)
    if wb_offset is not None:
        a_chan = a_chan - float(wb_offset[0])
        b_chan = b_chan - float(wb_offset[1])

    hist, _, _ = np.histogram2d(
        a_chan, b_chan,
        bins=[HIST_BINS_A, HIST_BINS_B],
        range=[list(HIST_RANGE_A), list(HIST_RANGE_B)],
        weights=weights,
    )
    hist = hist.astype(np.float32)
    s = float(hist.sum())
    if s > 0.0:
        hist /= s
    elif weights is not None:
        # All saturation weights were zero (extremely desaturated bottle).
        # Fall back to unweighted histogram so we still emit something usable.
        hist, _, _ = np.histogram2d(
            a_chan, b_chan,
            bins=[HIST_BINS_A, HIST_BINS_B],
            range=[list(HIST_RANGE_A), list(HIST_RANGE_B)],
        )
        hist = hist.astype(np.float32)
        s = float(hist.sum())
        if s > 0.0:
            hist /= s
    return (L, a, b), hist


def is_achromatic(frame, cx, cy, radius,
                  threshold=None,
                  outer_crop_px=OUTER_CROP_PX,
                  inner_crop_px=INNER_CROP_PX):
    """
    True iff the bottle ring is near-neutral (very low HSV saturation).

    Use as a pre-classification short-circuit so "clear" / very pale samples
    don't get routed through unstable hue-based metrics. Returns False on
    extraction failure (caller falls through to the regular classifier).

    threshold: median S cutoff in [0, 1]. None → SATURATION_L0_THRESHOLD.
    """
    thr = SATURATION_L0_THRESHOLD if threshold is None else float(threshold)

    # Re-fetch the same ring crop in BGR so we can convert to HSV (the existing
    # _ring_pixels_lab returns Lab already; we need raw BGR pixels for HSV).
    r = int(radius)
    x1 = max(0, int(cx) - r + outer_crop_px)
    y1 = max(0, int(cy) - r + outer_crop_px)
    x2 = min(frame.shape[1], int(cx) + r - outer_crop_px)
    y2 = min(frame.shape[0], int(cy) + r - outer_crop_px)
    if x2 <= x1 or y2 <= y1:
        return False
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    h_crop, w_crop = crop.shape[:2]
    Y, X = np.ogrid[:h_crop, :w_crop]
    dist = np.sqrt((X - (int(cx) - x1)) ** 2 + (Y - (int(cy) - y1)) ** 2)
    ring_mask = dist >= inner_crop_px
    if ring_mask.sum() < GLARE_MIN_VALID_PX:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s_pixels = hsv[:, :, 1][ring_mask]
    if s_pixels.size == 0:
        return False
    median_s = float(np.median(s_pixels)) / 255.0
    return median_s < thr


def compute_white_balance_offset(frame, white_positions,
                                 outer_crop_px=OUTER_CROP_PX,
                                 inner_crop_px=INNER_CROP_PX):
    """
    Median (Δa, Δb) measured at user-designated neutral patches.

    Δa = a_median − 128, Δb = b_median − 128 (so a perfectly neutral patch
    yields Δa = Δb = 0). Pass the returned tuple as `wb_offset` to
    extract_bottle_features() to align all bottles to a common WB space.

    Args:
        frame:           full BGR image
        white_positions: iterable of (cx, cy, radius) for each WB cell

    Returns:
        (Δa, Δb) floats, or None if no valid WB cell was sampled.
    """
    a_vals: list[float] = []
    b_vals: list[float] = []
    for cx, cy, radius in white_positions:
        pixels = _ring_pixels_lab(frame, cx, cy, radius, outer_crop_px, inner_crop_px)
        if pixels is None:
            continue
        a_vals.append(float(np.median(pixels[:, 1])))
        b_vals.append(float(np.median(pixels[:, 2])))
    if not a_vals:
        return None
    delta_a = float(np.median(a_vals)) - 128.0
    delta_b = float(np.median(b_vals)) - 128.0
    logger.info("white_balance: Δa={:+.2f}, Δb={:+.2f} from {} patch(es)",
                delta_a, delta_b, len(a_vals))
    return (delta_a, delta_b)


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


def filter_reference_outliers(frame, reference_positions,
                                sigma=2.0, min_n=4,
                                outer_crop_px=OUTER_CROP_PX,
                                inner_crop_px=INNER_CROP_PX):
    """
    Drop reference positions whose b* deviates more than `sigma` standard
    deviations from the level mean. Levels with fewer than `min_n` references
    are passed through unchanged (σ is unstable for tiny samples). If filtering
    would empty a level, the closest-to-mean reference is retained.

    Args:
        frame:               BGR image (full frame)
        reference_positions: dict {level: [(cx, cy, radius), ...]}
        sigma:               threshold in σ units (default 2.0)
        min_n:               minimum N for σ-filter to apply (default 4)

    Returns:
        dict same shape as reference_positions, with outliers removed.
        Per-level INFO log records kept/dropped counts.
    """
    out: dict[int, list] = {}
    for level, positions in reference_positions.items():
        if len(positions) < min_n:
            out[level] = list(positions)
            continue
        b_vals = []
        labs = []
        for cx, cy, radius in positions:
            lab = extract_bottle_color(frame, cx, cy, radius,
                                        outer_crop_px=outer_crop_px,
                                        inner_crop_px=inner_crop_px)
            labs.append(lab)
            b_vals.append(lab[2] if lab is not None else None)

        valid = [(p, b) for p, b in zip(positions, b_vals) if b is not None]
        if not valid:
            out[level] = list(positions)
            continue

        b_arr = np.array([b for _, b in valid], dtype=np.float64)
        mean_b = float(b_arr.mean())
        std_b  = float(b_arr.std())
        if std_b < 1e-6:
            out[level] = list(positions)
            continue

        kept = [p for p, b in valid if abs(b - mean_b) <= sigma * std_b]
        if not kept:
            # Keep the single closest-to-mean ref so the level isn't lost.
            closest = min(valid, key=lambda pb: abs(pb[1] - mean_b))[0]
            kept = [closest]

        dropped = len(positions) - len(kept)
        if dropped:
            logger.info(
                "Reference outlier filter L{}: kept {}/{} (mean b*={:.1f}, σ={:.1f}, ±{:.1f}σ)",
                level, len(kept), len(positions), mean_b, std_b, sigma,
            )
        out[level] = kept
    return out


# ---------------------------------------------------------------------------
# Master Color Path (1D Trajectory Projection)
# ---------------------------------------------------------------------------

def _weighted_dot(u, v, w_L=None, w_a=None, w_b=None):
    """Weighted Lab dot product ⟨u, v⟩_W = w_L·u_L·v_L + w_a·u_a·v_a + w_b·u_b·v_b."""
    if w_L is None: w_L = W_LAB_L
    if w_a is None: w_a = W_LAB_A
    if w_b is None: w_b = W_LAB_B
    return w_L * u[0] * v[0] + w_a * u[1] * v[1] + w_b * u[2] * v[2]


def build_master_path(level_centroids):
    """
    Build a polyline through per-level Lab centroids, sorted by level number.

    Each adjacent pair of available levels forms one segment. Missing levels
    are skipped — e.g. if only L0, L2, L4 have references, the polyline is
    L0→L2→L4 and a sample projecting halfway between L0 and L2 reports
    concentration_index = 1.0 (i.e. L1) by linear interpolation.

    Args:
        level_centroids: dict {level: (L, a, b)} — per-level mean Lab from
                         build_reference_baseline / build_reference_set.

    Returns:
        dict {
          "levels":   list[int]            sorted level numbers
          "points":   np.ndarray (N, 3)    Lab centroid per level
          "segments": list[dict]           each: level_i, level_j, point_i,
                                           point_j, vec, vec_dot_w
        }
        Empty list of segments when only a single centroid is available.
        None when level_centroids is empty.
    """
    if not level_centroids:
        return None
    levels = sorted(int(k) for k in level_centroids.keys())
    pts = np.array([level_centroids[lvl] for lvl in levels], dtype=np.float64)
    segments = []
    for i in range(len(levels) - 1):
        p_i, p_j = pts[i], pts[i + 1]
        v = p_j - p_i
        v_dot_w = _weighted_dot(v, v)
        segments.append({
            "level_i": levels[i],
            "level_j": levels[i + 1],
            "point_i": p_i,
            "point_j": p_j,
            "vec":     v,
            "vec_dot_w": float(v_dot_w),
        })
    return {"levels": levels, "points": pts, "segments": segments}


def project_onto_path(lab, master_path):
    """
    Project a sample Lab onto the master color path under the weighted-Lab
    metric (`delta_e_weighted`). Returns the closest segment's projection.

    The "concentration index" is a continuous scalar that makes "halfway
    between L1 and L2" expressible as 1.5 — useful for transitional samples.

    Returns:
        dict {
          "concentration_index": float    interpolated level along path
          "path_distance":       float    weighted ⊥ distance from path
          "projected_lab":       np.ndarray (3,)
          "nearest_segment":     int      segment index (or -1 for single-pt path)
        }
        None when master_path is None or empty.
    """
    if master_path is None or master_path.get("points") is None:
        return None
    sample = np.asarray(lab, dtype=np.float64)
    levels = master_path["levels"]
    points = master_path["points"]
    segments = master_path["segments"]

    if not segments:
        # Single-point path
        diff = sample - points[0]
        d = float(np.sqrt(_weighted_dot(diff, diff)))
        return {
            "concentration_index": float(levels[0]),
            "path_distance":       d,
            "projected_lab":       points[0].copy(),
            "nearest_segment":     -1,
        }

    best = None
    for idx, seg in enumerate(segments):
        diff = sample - seg["point_i"]
        if seg["vec_dot_w"] < 1e-12:
            t = 0.0
        else:
            t = float(_weighted_dot(diff, seg["vec"]) / seg["vec_dot_w"])
            t = max(0.0, min(1.0, t))
        proj = seg["point_i"] + t * seg["vec"]
        perp = sample - proj
        d = float(np.sqrt(_weighted_dot(perp, perp)))
        if best is None or d < best["d"]:
            index = float(seg["level_i"]) + t * float(seg["level_j"] - seg["level_i"])
            best = {"d": d, "t": t, "proj": proj, "index": index, "seg_idx": idx}

    return {
        "concentration_index": best["index"],
        "path_distance":       best["d"],
        "projected_lab":       best["proj"],
        "nearest_segment":     best["seg_idx"],
    }


def build_reference_set(frame, reference_positions):
    """
    Like build_reference_baseline, but returns the RAW per-bottle Lab values
    instead of averaging them per level. Used by k-NN classification so that
    each individual reference bottle remains a distinct anchor point.

    Args:
        frame:               BGR image (full frame)
        reference_positions: dict {level: [(cx, cy, radius), ...]}

    Returns:
        dict {level: [(L, a, b), (L, a, b), ...]} — every individual valid
        reference Lab. Levels with no valid samples are omitted.
    """
    out = {}
    for level, positions in reference_positions.items():
        labs = []
        for cx, cy, radius in positions:
            lab = extract_bottle_color(frame, cx, cy, radius)
            if lab is not None:
                labs.append(lab)
        if labs:
            out[level] = labs
            logger.info("Reference set L{}: {} valid bottle(s)", level, len(labs))
        else:
            logger.warning("Reference set L{}: no valid samples extracted", level)
    return out


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


def classify_sample_nn(sample_lab, references_per_level,
                       margin=CONFIDENCE_MARGIN, max_delta_e=MAX_DELTA_E):
    """
    Classify a sample by its NEAREST INDIVIDUAL REFERENCE in chromaticity space.

    Per-level distance = min ΔE_chroma over that level's references — i.e.
    each individual reference bottle anchors its own neighbourhood, and the
    level whose closest reference is nearest wins. This is k=1 nearest-neighbour
    over labelled references.

    Why nearest-individual instead of nearest-centroid: when references for
    one level fall at multiple chroma points (e.g. scattered L4 bottles due
    to physical variation or lighting), the per-level mean may not match any
    real bottle. k=1 NN preserves the multi-modal information embedded in
    the reference set so a sample close to *any* L4 bottle classifies as L4.

    Args:
        sample_lab:           (L, a, b) tuple of the sample
        references_per_level: dict {level: [(L,a,b), (L,a,b), ...]}
                              from build_reference_set
        margin:               Min gap between best and second-best level for confidence
        max_delta_e:          Reject classification if best > max_delta_e

    Returns:
        (level, delta_e, confident)
        - level:     int 0-4 if classified, None if best ΔE > max_delta_e
        - delta_e:   chromaticity distance to the closest reference
        - confident: True only when within range AND margin met
    """
    if not references_per_level or sample_lab is None:
        return None, float('inf'), False

    deltas = {
        level: min(delta_e_chroma(sample_lab, ref) for ref in refs)
        for level, refs in references_per_level.items()
        if refs
    }
    if not deltas:
        return None, float('inf'), False

    sorted_levels = sorted(deltas, key=deltas.get)
    best_level = sorted_levels[0]
    best_delta = deltas[best_level]

    if best_delta > max_delta_e:
        return None, best_delta, False

    confident = True
    if len(sorted_levels) >= 2:
        confident = (deltas[sorted_levels[1]] - best_delta) > margin

    return best_level, best_delta, confident


# ---------------------------------------------------------------------------
# Histogram template + hybrid classifier
# ---------------------------------------------------------------------------

def build_reference_histograms(frame, reference_positions, wb_offset=None,
                                outer_crop_px=OUTER_CROP_PX,
                                inner_crop_px=INNER_CROP_PX,
                                min_saturation=None):
    """
    Build one (a*, b*) template histogram per reference level by pooling all
    ring pixels of that level's bottles.

    Pooling is performed by summing per-bottle histograms and renormalising,
    which is equivalent to histogramming the union of all ring pixels — but
    keeps each bottle weighted equally regardless of how many valid pixels it
    contributed.

    `min_saturation` (None → REFERENCE_MIN_SATURATION default) is a hard
    cutoff: pixels with HSV S/255 below this are dropped before histogramming,
    keeping the per-level template "pure" (no plastic/glare contamination).

    Returns:
        dict {level: np.ndarray (HIST_BINS_A, HIST_BINS_B) float32 sum-to-1}.
        Levels with no valid bottles are omitted.
    """
    if min_saturation is None:
        min_saturation = REFERENCE_MIN_SATURATION
    out: dict[int, np.ndarray] = {}
    for level, positions in reference_positions.items():
        accum = np.zeros((HIST_BINS_A, HIST_BINS_B), dtype=np.float32)
        n = 0
        for cx, cy, radius in positions:
            _, hist = extract_bottle_features(
                frame, cx, cy, radius,
                outer_crop_px=outer_crop_px,
                inner_crop_px=inner_crop_px,
                wb_offset=wb_offset,
                min_saturation=min_saturation,
            )
            if hist is None:
                continue
            accum += hist
            n += 1
        if n == 0:
            logger.warning("Reference hist L{}: no valid bottles", level)
            continue
        s = float(accum.sum())
        if s > 0.0:
            accum /= s
        out[level] = accum
    return out


def _bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    """Bhattacharyya distance ∈ [0, 1] (0 = identical histograms)."""
    return float(cv2.compareHist(
        h1.astype(np.float32),
        h2.astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA,
    ))


def classify_sample_hybrid(sample_lab, sample_hist,
                            references_per_level, reference_hists,
                            weight_chroma: float = W_CHROMA,
                            weight_hist: float = W_HIST,
                            chroma_scale: float = CHROMA_SCALE,
                            max_score: float = MAX_COMBINED_SCORE,
                            margin: float = COMBINED_MARGIN,
                            low_confidence_score: float = LOW_CONFIDENCE_SCORE_THRESHOLD):
    """
    Hybrid classifier — combines weighted Lab ΔE k-NN and Bhattacharyya
    histogram distance into one weighted score per level. Picks argmin.

    Force-Match: always returns the argmin level when at least one level has
    refs/hist data. Hard rejection (`level=None`) only fires when no level can
    be scored — a setup error, not a per-bottle issue. High-score classifications
    are flagged via `best_fit=True` for the UI.

    Returns:
        dict {
          "level":      int | None,    # only None when no levels available
          "chroma_de":  float,
          "hist_bhatt": float,         # 0 = identical, 1 = no overlap
          "combined":   float,
          "confident":  bool,
          "best_fit":   bool,          # True when combined > max_score / low_confidence threshold
          "per_level":  {lvl: {"chroma_de", "hist_bhatt", "combined"}},
        }
    """
    empty = {
        "level": None, "chroma_de": float("inf"), "hist_bhatt": 1.0,
        "combined": float("inf"), "confident": False, "best_fit": False,
        "per_level": {},
    }
    if sample_lab is None and sample_hist is None:
        return empty

    levels = set(references_per_level.keys()) | set(reference_hists.keys())
    if not levels:
        return empty

    per_level: dict[int, dict] = {}
    for lvl in levels:
        refs = references_per_level.get(lvl) or []
        if refs and sample_lab is not None:
            chroma_de = min(delta_e_weighted(sample_lab, r) for r in refs)
        else:
            chroma_de = float("inf")

        tpl = reference_hists.get(lvl)
        if tpl is not None and sample_hist is not None:
            hist_bhatt = _bhattacharyya(sample_hist, tpl)
        else:
            hist_bhatt = 1.0

        chroma_term = min(chroma_de / chroma_scale, 1.0) if np.isfinite(chroma_de) else 1.0
        combined = weight_chroma * chroma_term + weight_hist * hist_bhatt

        per_level[lvl] = {
            "chroma_de": chroma_de,
            "hist_bhatt": hist_bhatt,
            "combined": combined,
        }

    sorted_levels = sorted(per_level, key=lambda k: per_level[k]["combined"])
    best = sorted_levels[0]
    best_score = per_level[best]["combined"]
    best_chroma = per_level[best]["chroma_de"]
    best_hist   = per_level[best]["hist_bhatt"]

    # Force-Match: always return the argmin level. The strict `combined > max_score
    # → None` branch is gone; the score itself surfaces in the UI via best_fit.
    if not np.isfinite(best_score):
        return {
            "level": None,
            "chroma_de": best_chroma, "hist_bhatt": best_hist,
            "combined": best_score, "confident": False, "best_fit": False,
            "per_level": per_level,
        }

    margin_ok = True
    if len(sorted_levels) >= 2:
        margin_ok = (per_level[sorted_levels[1]]["combined"] - best_score) > margin
    confident = margin_ok and (best_score < low_confidence_score)
    best_fit  = (best_score > max_score) or (best_score >= low_confidence_score and not confident)

    return {
        "level": best,
        "chroma_de": best_chroma,
        "hist_bhatt": best_hist,
        "combined": best_score,
        "confident": confident,
        "best_fit":  best_fit,
        "per_level": per_level,
    }


def classify_sample_path(sample_lab, sample_hist, master_path, reference_hists,
                          weight_chroma: float = W_CHROMA,
                          weight_hist: float = W_HIST,
                          chroma_scale: float = CHROMA_SCALE,
                          max_score: float = MAX_COMBINED_SCORE,
                          max_path_distance: float = MAX_PATH_DISTANCE,
                          path_confident_distance: float = PATH_CONFIDENT_DISTANCE,
                          soft_penalty_path_de_threshold: float = SOFT_PENALTY_PATH_DE_THRESHOLD,
                          soft_penalty_factor: float = SOFT_PENALTY_FACTOR,
                          low_confidence_score: float = LOW_CONFIDENCE_SCORE_THRESHOLD):
    """
    Classify a sample by 1-D projection onto the Master Color Path.

    The path is the polyline through per-level Lab centroids; the sample is
    projected (under the weighted-Lab metric) onto the closest segment, and
    its level is `round(concentration_index)`.

    **Force-Match**: every detected sample is assigned its nearest level along
    the path. The strict-rejection branch (`level=None` when path_distance or
    combined exceed thresholds) is gone — high-error samples instead surface
    `best_fit=True` for the UI to badge them as "best fit only". Rejection
    only happens when the projection itself is unavailable (no master_path /
    no references configured).

    Soft-penalty histogram weighting:
        path_distance < soft_penalty_path_de_threshold (default 5.0)
        → halve the histogram weight (default factor 0.5).

    Returns:
        dict with both legacy fields (`chroma_de`, `hist_bhatt`, `combined`,
        `confident`, `level`) for UI/Sheets compatibility, plus path-specific
        fields (`concentration_index`, `path_distance`, `soft_penalty`,
        `best_fit`).
    """
    empty = {
        "level": None, "concentration_index": None,
        "path_distance": float("inf"), "chroma_de": float("inf"),
        "hist_bhatt": 1.0, "combined": float("inf"),
        "confident": False, "soft_penalty": False, "best_fit": False,
    }
    if sample_lab is None or master_path is None:
        return empty

    proj = project_onto_path(sample_lab, master_path)
    if proj is None:
        return empty

    path_distance = proj["path_distance"]
    index         = proj["concentration_index"]
    rounded_level = int(round(index)) if np.isfinite(index) else None

    # Histogram comparison against the rounded level's template, falling back
    # to the closest available level if the rounded one has no template.
    hist_bhatt = 1.0
    if sample_hist is not None and reference_hists:
        tpl = reference_hists.get(rounded_level) if rounded_level is not None else None
        if tpl is None:
            avail = sorted(reference_hists.keys(), key=lambda k: abs(k - index))
            if avail:
                tpl = reference_hists[avail[0]]
        if tpl is not None:
            hist_bhatt = _bhattacharyya(sample_hist, tpl)

    chroma_term = min(path_distance / chroma_scale, 1.0) if np.isfinite(path_distance) else 1.0
    soft_penalty_active = path_distance < soft_penalty_path_de_threshold
    h_weight = weight_hist * (soft_penalty_factor if soft_penalty_active else 1.0)
    combined = weight_chroma * chroma_term + h_weight * hist_bhatt

    # Force-Match: the level is always the rounded concentration index. Confidence
    # tiers (confident / best-fit / borderline) surface the score quality to the UI
    # without changing the assigned level.
    near_path = path_distance < path_confident_distance
    low_score = combined < low_confidence_score
    confident = near_path and low_score
    best_fit  = (
        path_distance > max_path_distance
        or combined > max_score
        or (combined >= low_confidence_score and not confident)
    )

    return {
        "level": rounded_level, "concentration_index": index,
        "path_distance": path_distance, "chroma_de": path_distance,
        "hist_bhatt": hist_bhatt, "combined": combined,
        "confident": confident, "best_fit": best_fit,
        "soft_penalty": soft_penalty_active,
    }
