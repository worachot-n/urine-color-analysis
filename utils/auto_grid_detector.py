"""
Robust grid detector for bottle trays — classical CV only, no deep learning.

Designed to run on the FULL-RESOLUTION image without any letterbox/resize
step (letterboxing is a YOLO requirement and distorts circle geometry at
non-square aspect ratios).

Pipeline
--------
1. preprocess()            — grayscale + CLAHE normalisation
2. detect_circles()        — HoughCircles with image-adaptive parameters
3. find_grid_positions()   — Gaussian KDE projection + scipy peak detection
                             for each axis independently
4. fit_uniform_grid()      — extend partial detections to full n_rows × n_cols
5. build_grid_pts()        — Cartesian product of row × col positions
6. assign_detections()     — snap each detected circle to its nearest cell
7. draw_grid()             — midpoint grid lines + circle markers

Public API
----------
    detect_grid_full(image, n_rows, n_cols, config=None) -> dict
    AutoGridConfig — all tunable parameters in one dataclass
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

try:
    from scipy.signal import find_peaks as _scipy_find_peaks
    from scipy.ndimage import gaussian_filter1d as _scipy_gaussian
    _SCIPY = True
except ImportError:  # pragma: no cover
    _SCIPY = False
    logger.warning("auto_grid: scipy not found — using built-in peak finder")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AutoGridConfig:
    """All tunable parameters for the grid detector."""

    # --- HoughCircles ---
    hough_dp:     float = 1.2   # inverse accumulator resolution
    hough_param1: int   = 60    # Canny high threshold
    hough_param2: int   = 22    # accumulator threshold (lower → more permissive)

    # Fractions of the expected circle spacing
    min_dist_frac:  float = 0.55   # min_dist   = est_spacing × frac
    radius_min_frac: float = 0.22  # min_radius = est_spacing × frac
    radius_max_frac: float = 0.52  # max_radius = est_spacing × frac

    # --- Projection KDE ---
    kde_bw_frac:        float = 0.30  # KDE bandwidth = est_spacing × frac
    peak_min_h_frac:    float = 0.04  # min peak height = max_density × frac
    peak_min_dist_frac: float = 0.60  # min separation between peaks = est_spacing × frac

    # --- Grid snap ---
    snap_tol_frac: float = 0.45   # max distance to snap = est_spacing × frac

    # --- Visualisation (BGR) ---
    color_detected:      tuple = (0, 220,  0)    # green  — HoughCircles hit
    color_reconstructed: tuple = (0, 140, 255)   # orange — inferred position
    color_h_lines:       tuple = (255, 220, 50)  # yellow — horizontal midpoint lines
    color_v_lines:       tuple = (50,  200, 255) # cyan   — vertical midpoint lines
    circle_thickness:    int   = 2
    line_thickness:      int   = 2


# ---------------------------------------------------------------------------
# Step 1: Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(image: np.ndarray) -> np.ndarray:
    """
    CLAHE on the Lab L-channel then convert to blurred greyscale.

    CLAHE normalises white caps, yellow caps, and dark backgrounds to
    similar luminance contrast before Canny / HoughCircles.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


# ---------------------------------------------------------------------------
# Step 2: Circle detection
# ---------------------------------------------------------------------------

def _detect_circles(
    gray: np.ndarray,
    config: AutoGridConfig,
    est_spacing: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    HoughCircles with parameters derived from the expected grid spacing.

    Returns:
        centers — (N, 2) float32 array of (x, y) positions
        radii   — (N,)   float32 array
    """
    min_r = max(5,  int(est_spacing * config.radius_min_frac))
    max_r = max(15, int(est_spacing * config.radius_max_frac))
    min_d = max(8,  int(est_spacing * config.min_dist_frac))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=config.hough_dp,
        minDist=min_d,
        param1=config.hough_param1,
        param2=config.hough_param2,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        logger.warning("auto_grid: HoughCircles found no circles")
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    data = circles[0]
    logger.info("auto_grid: HoughCircles found {} circles (r=[{:.0f},{:.0f}])",
                len(data), data[:, 2].min(), data[:, 2].max())
    return data[:, :2].astype(np.float32), data[:, 2].astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3: 1-D axis-position estimation via projection KDE
# ---------------------------------------------------------------------------

def _kde_1d(values: np.ndarray, size: int, bandwidth: float) -> np.ndarray:
    """Gaussian KDE of `values` evaluated on the integer grid [0, size)."""
    x = np.arange(size, dtype=np.float64)
    density = np.zeros(size, dtype=np.float64)
    for v in values:
        density += np.exp(-0.5 * ((x - float(v)) / bandwidth) ** 2)
    return density


def _find_peaks_scipy(density: np.ndarray, min_dist: int, min_height: float) -> np.ndarray:
    peaks, _ = _scipy_find_peaks(density, distance=max(1, min_dist), height=min_height)
    return peaks.astype(float)


def _find_peaks_builtin(density: np.ndarray, min_dist: int, min_height: float) -> np.ndarray:
    """Simple local-maximum finder without scipy."""
    n = len(density)
    peaks = []
    i = 1
    while i < n - 1:
        if density[i] >= min_height and density[i] >= density[i - 1] and density[i] >= density[i + 1]:
            # Find the true local max within the next min_dist window
            end = min(n, i + min_dist)
            local_max = i + int(np.argmax(density[i:end]))
            peaks.append(float(local_max))
            i = end
        else:
            i += 1
    return np.array(peaks)


def _estimate_fundamental_spacing(diffs: np.ndarray, est_spacing: float) -> float:
    """
    Given inter-peak gaps (may be 1×, 2×, 3× the true spacing due to missing
    circles), return the fundamental grid period.
    """
    min_sp = est_spacing * 0.35
    valid = diffs[(diffs >= min_sp) & (diffs <= est_spacing * 3.6)]
    if len(valid) == 0:
        return est_spacing

    sp = float(np.median(valid))

    # De-aliasing: check if sp/2 or sp/3 is better supported
    for divisor in (2, 3):
        candidate = sp / divisor
        if candidate < min_sp:
            continue
        near = np.minimum(valid % candidate, candidate - valid % candidate)
        tol  = 0.25 * candidate
        n_fit = int(np.sum(near < tol))
        n_old = int(np.sum(np.abs(valid - sp) < 0.25 * sp))
        if n_fit >= max(1, n_old * 0.5):
            sp = candidate
            break

    return max(sp, min_sp)


def _extend_to_full_grid(
    detected: np.ndarray,
    n_total: int,
    est_spacing: float,
) -> np.ndarray:
    """
    Fit a uniform n_total-point grid to a partial set of detected positions.

    Uses RANSAC-style origin voting: each detected peak votes for
    `peak_position mod spacing`, and the median vote is the grid origin.
    The grid is then centred on the detected peak cluster.
    """
    detected = np.sort(detected)

    if len(detected) >= 2:
        spacing = _estimate_fundamental_spacing(np.diff(detected), est_spacing)
    else:
        spacing = est_spacing

    # Origin voting (circular median)
    votes = detected % spacing
    votes[votes > spacing * 0.5] -= spacing
    origin_frac = float(np.median(votes))
    if origin_frac < 0:
        origin_frac += spacing

    # Centre the n_total grid on the detected cluster
    centre_detected = float(np.mean(detected))
    centre_idx = (n_total - 1) / 2.0
    start_i = int(round((centre_detected - origin_frac) / spacing - centre_idx))

    return np.array(
        [origin_frac + (start_i + i) * spacing for i in range(n_total)],
        dtype=np.float32,
    )


def _find_grid_positions(
    coords: np.ndarray,
    n_expected: int,
    axis_size: int,
    est_spacing: float,
    config: AutoGridConfig,
) -> np.ndarray:
    """
    Find the `n_expected` grid positions along a single axis.

    1. Build a Gaussian KDE histogram of the detected coordinates.
    2. Find peaks (scipy or built-in fallback).
    3. If enough peaks found, return the n_expected strongest.
    4. If fewer peaks, extend a fitted regular grid to n_expected.
    5. If no circles at all, fall back to a uniform grid across the image.

    Returns sorted float32 array of length n_expected.
    """
    if len(coords) < 2:
        return np.linspace(
            est_spacing * 0.5,
            est_spacing * 0.5 + (n_expected - 1) * est_spacing,
            n_expected,
            dtype=np.float32,
        )

    bw       = max(2.0, est_spacing * config.kde_bw_frac)
    min_sep  = max(3,   int(est_spacing * config.peak_min_dist_frac))
    density  = _kde_1d(coords, axis_size, bw)

    min_h = float(density.max()) * config.peak_min_h_frac

    if _SCIPY:
        peak_pos = _find_peaks_scipy(density, min_sep, min_h)
    else:
        peak_pos = _find_peaks_builtin(density, min_sep, min_h)

    if len(peak_pos) == 0:
        # No peaks at all — uniform fallback across detected range
        lo, hi = float(coords.min()), float(coords.max())
        return np.linspace(lo, hi, n_expected, dtype=np.float32)

    peak_heights = density[peak_pos.astype(int)]

    if len(peak_pos) >= n_expected:
        # More peaks than needed — keep the n_expected tallest
        top = np.argsort(peak_heights)[-n_expected:]
        return np.sort(peak_pos[top]).astype(np.float32)

    # Fewer peaks than expected — extend the fitted regular grid
    return _extend_to_full_grid(peak_pos, n_expected, est_spacing)


# ---------------------------------------------------------------------------
# Step 4: Grid points and assignment
# ---------------------------------------------------------------------------

def _build_grid_pts(
    row_positions: np.ndarray,
    col_positions: np.ndarray,
) -> np.ndarray:
    """Return (n_rows × n_cols, 2) float32 Cartesian product."""
    pts = np.array(
        [[cx, ry] for ry in row_positions for cx in col_positions],
        dtype=np.float32,
    )
    return pts


def _assign_detections(
    centers: np.ndarray,
    grid_pts: np.ndarray,
    snap_dist: float,
) -> np.ndarray:
    """
    For each detected circle centre, find the nearest grid cell and mark it.
    Returns bool array `is_detected` of length len(grid_pts).
    """
    is_detected = np.zeros(len(grid_pts), dtype=bool)
    if len(centers) == 0:
        return is_detected
    for cx, cy in centers:
        dists = np.hypot(grid_pts[:, 0] - cx, grid_pts[:, 1] - cy)
        nearest = int(np.argmin(dists))
        if dists[nearest] <= snap_dist:
            is_detected[nearest] = True
    return is_detected


# ---------------------------------------------------------------------------
# Step 5: Annotation
# ---------------------------------------------------------------------------

def _draw_grid(
    image: np.ndarray,
    grid_pts: np.ndarray,
    is_detected: np.ndarray,
    row_positions: np.ndarray,
    col_positions: np.ndarray,
    avg_radius: float,
    config: AutoGridConfig,
) -> np.ndarray:
    """
    Draw the annotated grid on a copy of `image`.

    Grid lines are drawn at the midpoint between consecutive rows/columns.
    The outermost lines are drawn half-spacing OUTSIDE the edge circles so
    the grid cell is visually larger than the circle inside it.

    Green circles  = detected by HoughCircles
    Orange circles = reconstructed (inferred from grid fit)
    """
    canvas = image.copy()
    n_rows = len(row_positions)
    n_cols = len(col_positions)
    h, w   = canvas.shape[:2]
    r      = max(5, int(avg_radius))
    lw     = max(1, config.line_thickness)

    # ── Horizontal lines ─────────────────────────────────────────────────
    dy_top    = (row_positions[1] - row_positions[0])  * 0.5 if n_rows > 1 else avg_radius
    dy_bottom = (row_positions[-1] - row_positions[-2]) * 0.5 if n_rows > 1 else avg_radius
    dx_left   = (col_positions[1] - col_positions[0])  * 0.5 if n_cols > 1 else avg_radius
    dx_right  = (col_positions[-1] - col_positions[-2]) * 0.5 if n_cols > 1 else avg_radius

    h_ys = (
        [row_positions[0] - dy_top]
        + [(row_positions[i] + row_positions[i + 1]) / 2.0 for i in range(n_rows - 1)]
        + [row_positions[-1] + dy_bottom]
    )
    x0 = int(round(col_positions[0]  - dx_left))
    x1 = int(round(col_positions[-1] + dx_right))
    for y in h_ys:
        cv2.line(canvas, (x0, int(round(y))), (x1, int(round(y))),
                 config.color_h_lines, lw, cv2.LINE_AA)

    # ── Vertical lines ───────────────────────────────────────────────────
    v_xs = (
        [col_positions[0] - dx_left]
        + [(col_positions[i] + col_positions[i + 1]) / 2.0 for i in range(n_cols - 1)]
        + [col_positions[-1] + dx_right]
    )
    y0 = int(round(row_positions[0]  - dy_top))
    y1 = int(round(row_positions[-1] + dy_bottom))
    for x in v_xs:
        cv2.line(canvas, (int(round(x)), y0), (int(round(x)), y1),
                 config.color_v_lines, lw, cv2.LINE_AA)

    # ── Circle markers ───────────────────────────────────────────────────
    for pt, detected in zip(grid_pts, is_detected):
        cx_i = int(round(pt[0]))
        cy_i = int(round(pt[1]))
        color = config.color_detected if detected else config.color_reconstructed
        cv2.circle(canvas, (cx_i, cy_i), r, color, config.circle_thickness, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_grid_full(
    image: np.ndarray,
    n_rows: int,
    n_cols: int,
    config: Optional[AutoGridConfig] = None,
) -> dict:
    """
    Full grid-detection pipeline on an arbitrary-resolution BGR image.

    No letterboxing.  No YOLO.  Pure classical CV.

    Args:
        image:  Full-resolution (or ROI-cropped) BGR image.
        n_rows: Expected number of grid rows (from slot config).
        n_cols: Expected number of grid columns (from slot config).
        config: Optional tuning config; sensible defaults if None.

    Returns:
        {
          "grid_pts":            (n_rows*n_cols, 2) float32 — cell centres
          "is_detected":         (n_rows*n_cols,) bool
          "result_image":        BGR annotated image (same resolution as input)
          "detected_count":      int
          "reconstructed_count": int
          "col_positions":       (n_cols,) float32
          "row_positions":       (n_rows,) float32
          "avg_radius_px":       float
        }
    """
    if config is None:
        config = AutoGridConfig()

    h, w = image.shape[:2]
    est_dx = w / n_cols
    est_dy = h / n_rows
    est_spacing = (est_dx + est_dy) / 2.0

    logger.info(
        "auto_grid: image={}×{}  grid={}×{}  est_spacing={:.1f}px",
        w, h, n_cols, n_rows, est_spacing,
    )

    # ── 1. Preprocess + detect circles ────────────────────────────────────
    gray    = _preprocess(image)
    centers, radii = _detect_circles(gray, config, est_spacing)

    avg_radius = float(np.median(radii)) if len(radii) > 0 else est_spacing * 0.38

    # ── 2. Projection KDE → axis positions ───────────────────────────────
    if len(centers) >= 3:
        col_positions = _find_grid_positions(
            centers[:, 0], n_cols, w, est_dx, config
        )
        row_positions = _find_grid_positions(
            centers[:, 1], n_rows, h, est_dy, config
        )
    else:
        logger.warning(
            "auto_grid: only {} circles detected — falling back to uniform grid",
            len(centers),
        )
        col_positions = np.linspace(est_dx * 0.5, w - est_dx * 0.5, n_cols, dtype=np.float32)
        row_positions = np.linspace(est_dy * 0.5, h - est_dy * 0.5, n_rows, dtype=np.float32)

    # ── 3. Build grid_pts ─────────────────────────────────────────────────
    grid_pts = _build_grid_pts(row_positions, col_positions)

    # ── 4. Assign detected circles to cells ───────────────────────────────
    snap_dist   = min(est_dx, est_dy) * config.snap_tol_frac
    is_detected = _assign_detections(centers, grid_pts, snap_dist)

    detected_count      = int(is_detected.sum())
    reconstructed_count = len(grid_pts) - detected_count

    logger.info(
        "auto_grid: detected={} reconstructed={} avg_r={:.1f}px",
        detected_count, reconstructed_count, avg_radius,
    )

    # ── 5. Draw annotated image ───────────────────────────────────────────
    result_image = _draw_grid(
        image, grid_pts, is_detected,
        row_positions, col_positions, avg_radius, config,
    )

    return {
        "grid_pts":            grid_pts,
        "is_detected":         is_detected,
        "result_image":        result_image,
        "detected_count":      detected_count,
        "reconstructed_count": reconstructed_count,
        "col_positions":       col_positions,
        "row_positions":       row_positions,
        "avg_radius_px":       avg_radius,
    }
