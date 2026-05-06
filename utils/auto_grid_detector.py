"""
Robust grid detector for bottle trays — classical CV only, no deep learning.

Designed to run on the FULL-RESOLUTION image without any letterbox/resize
step (letterboxing is a YOLO requirement and distorts circle geometry at
non-square aspect ratios).

Pipeline
--------
1. preprocess()              — grayscale + CLAHE normalisation
2. detect_circles()          — HoughCircles with image-adaptive parameters
3. estimate_grid_angle()     — estimate tray rotation from row line-fits
4. find_grid_positions()     — Gaussian KDE projection + scipy peak detection
                               on rotation-corrected centres
5. fit_uniform_grid()        — extend partial detections to full n_rows × n_cols
6. build_grid_pts()          — Cartesian product, then rotate back to image space
7. assign_detections()       — snap each detected circle to its nearest cell
8. draw_grid()               — rotated midpoint grid lines + circle markers

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
    min_dist_frac:   float = 0.55   # min_dist   = est_spacing × frac
    radius_min_frac: float = 0.22   # min_radius = est_spacing × frac
    radius_max_frac: float = 0.52   # max_radius = est_spacing × frac

    # --- Projection KDE ---
    kde_bw_frac:        float = 0.30  # KDE bandwidth = est_spacing × frac
    peak_min_h_frac:    float = 0.04  # min peak height = max_density × frac
    peak_min_dist_frac: float = 0.60  # min separation between peaks = est_spacing × frac

    # --- Grid snap ---
    snap_tol_frac: float = 0.45   # max distance to snap = est_spacing × frac

    # --- Rotation correction ---
    # Minimum rotation angle (degrees) to bother correcting
    min_correct_angle_deg: float = 0.20

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
    """CLAHE on the Lab L-channel then convert to blurred greyscale."""
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
# Step 3: Rotation estimation and correction
# ---------------------------------------------------------------------------

def _rotate_points(pts: np.ndarray, angle_rad: float, origin: np.ndarray) -> np.ndarray:
    """Rotate 2-D points around `origin` by `angle_rad` radians (CCW positive)."""
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return ((pts.astype(np.float64) - origin) @ R.T + origin).astype(np.float32)


def _estimate_grid_angle(centers: np.ndarray, est_dy: float) -> float:
    """
    Estimate the tray rotation angle (radians) from detected circle centres.

    Groups circles into approximate rows by Y proximity (± est_dy/2),
    fits a line per row, and returns the median row slope angle.
    """
    if len(centers) < 6:
        return 0.0

    sorted_pts = centers[np.argsort(centers[:, 1])]

    # Group into rows
    rows: list[list] = []
    cur_row: list = [sorted_pts[0].tolist()]
    cur_mean_y = float(sorted_pts[0][1])
    for pt in sorted_pts[1:]:
        if abs(float(pt[1]) - cur_mean_y) < est_dy * 0.5:
            cur_row.append(pt.tolist())
            cur_mean_y = float(np.mean([p[1] for p in cur_row]))
        else:
            rows.append(cur_row)
            cur_row = [pt.tolist()]
            cur_mean_y = float(pt[1])
    rows.append(cur_row)

    angles = []
    for row_pts_list in rows:
        if len(row_pts_list) < 3:
            continue
        row_np = np.array(row_pts_list, dtype=np.float64)
        x = row_np[:, 0]
        y = row_np[:, 1]
        A = np.vstack([x, np.ones_like(x)]).T
        result = np.linalg.lstsq(A, y, rcond=None)
        m = float(result[0][0])
        angles.append(float(np.arctan(m)))

    if len(angles) < 2:
        return 0.0

    return float(np.median(angles))


# ---------------------------------------------------------------------------
# Step 4: 1-D axis-position estimation via projection KDE
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
        lo, hi = float(coords.min()), float(coords.max())
        return np.linspace(lo, hi, n_expected, dtype=np.float32)

    peak_heights = density[peak_pos.astype(int)]

    if len(peak_pos) >= n_expected:
        top = np.argsort(peak_heights)[-n_expected:]
        return np.sort(peak_pos[top]).astype(np.float32)

    return _extend_to_full_grid(peak_pos, n_expected, est_spacing)


# ---------------------------------------------------------------------------
# Step 5: Grid points and assignment
# ---------------------------------------------------------------------------

def _build_grid_pts(
    row_positions: np.ndarray,
    col_positions: np.ndarray,
) -> np.ndarray:
    """Return (n_rows × n_cols, 2) float32 Cartesian product (row-major)."""
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
# Step 6: Annotation — rotated midpoint grid lines
# ---------------------------------------------------------------------------

def _draw_grid(
    image: np.ndarray,
    grid_pts: np.ndarray,   # (n_rows*n_cols, 2) float32 in image space
    is_detected: np.ndarray,
    n_rows: int,
    n_cols: int,
    avg_radius: float,
    config: AutoGridConfig,
) -> np.ndarray:
    """
    Draw the annotated grid on a copy of `image`.

    Grid lines are computed from the actual grid point positions (which may
    be rotated), so lines follow the real grid orientation rather than
    being forced axis-aligned.

    Boundary lines are drawn at midpoints between consecutive rows/columns;
    the outermost boundaries extend half-spacing beyond the edge circles.

    Green circles  = detected by HoughCircles
    Orange circles = reconstructed (inferred from grid fit)
    """
    canvas = image.copy()
    r  = max(5, int(avg_radius))
    lw = max(1, config.line_thickness)

    pts = grid_pts.reshape(n_rows, n_cols, 2)  # (n_rows, n_cols, 2)

    # ── Horizontal boundary lines (n_rows + 1) ───────────────────────────
    for bi in range(n_rows + 1):
        if bi == 0:
            # Outer top edge: reflect row 0 through row 1
            if n_rows > 1:
                row_pts = pts[0] + (pts[0] - pts[1]) * 0.5
            else:
                row_pts = pts[0] - np.array([[0, avg_radius]])
        elif bi == n_rows:
            # Outer bottom edge: reflect last row through second-to-last
            if n_rows > 1:
                row_pts = pts[-1] + (pts[-1] - pts[-2]) * 0.5
            else:
                row_pts = pts[-1] + np.array([[0, avg_radius]])
        else:
            row_pts = (pts[bi - 1] + pts[bi]) * 0.5

        # Draw line from left boundary point to right boundary point
        x0, y0 = int(round(float(row_pts[0, 0]))),  int(round(float(row_pts[0, 1])))
        x1, y1 = int(round(float(row_pts[-1, 0]))), int(round(float(row_pts[-1, 1])))
        cv2.line(canvas, (x0, y0), (x1, y1), config.color_h_lines, lw, cv2.LINE_AA)

    # ── Vertical boundary lines (n_cols + 1) ─────────────────────────────
    for bi in range(n_cols + 1):
        if bi == 0:
            if n_cols > 1:
                col_pts = pts[:, 0] + (pts[:, 0] - pts[:, 1]) * 0.5
            else:
                col_pts = pts[:, 0] - np.array([[avg_radius, 0]])
        elif bi == n_cols:
            if n_cols > 1:
                col_pts = pts[:, -1] + (pts[:, -1] - pts[:, -2]) * 0.5
            else:
                col_pts = pts[:, -1] + np.array([[avg_radius, 0]])
        else:
            col_pts = (pts[:, bi - 1] + pts[:, bi]) * 0.5

        x0, y0 = int(round(float(col_pts[0, 0]))),  int(round(float(col_pts[0, 1])))
        x1, y1 = int(round(float(col_pts[-1, 0]))), int(round(float(col_pts[-1, 1])))
        cv2.line(canvas, (x0, y0), (x1, y1), config.color_v_lines, lw, cv2.LINE_AA)

    # ── Circle markers ───────────────────────────────────────────────────
    for pt, det in zip(grid_pts, is_detected):
        cx_i = int(round(float(pt[0])))
        cy_i = int(round(float(pt[1])))
        color = config.color_detected if det else config.color_reconstructed
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

    Handles slightly rotated trays: estimates the grid rotation angle from
    row line-fits, corrects for it during grid fitting, then rotates the
    grid points back to the original image coordinate system.

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
          "col_positions":       (n_cols,) float32  (in rotation-corrected space)
          "row_positions":       (n_rows,) float32  (in rotation-corrected space)
          "avg_radius_px":       float
          "grid_angle_deg":      float  — estimated tray rotation angle
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

    # ── 2. Estimate tray rotation and correct circle coordinates ──────────
    angle = _estimate_grid_angle(centers, est_dy) if len(centers) >= 6 else 0.0
    min_angle = np.radians(config.min_correct_angle_deg)

    if abs(angle) > min_angle:
        logger.info("auto_grid: estimated rotation={:.3f}°, applying correction",
                    np.degrees(angle))
        img_origin = np.array([w / 2.0, h / 2.0])
        centers_fit = _rotate_points(centers, -angle, img_origin)
    else:
        angle = 0.0
        img_origin = np.array([w / 2.0, h / 2.0])
        centers_fit = centers

    # ── 3. Projection KDE → axis positions (on corrected coords) ──────────
    if len(centers_fit) >= 3:
        col_positions = _find_grid_positions(
            centers_fit[:, 0], n_cols, w, est_dx, config
        )
        row_positions = _find_grid_positions(
            centers_fit[:, 1], n_rows, h, est_dy, config
        )
    else:
        logger.warning(
            "auto_grid: only {} circles detected — falling back to uniform grid",
            len(centers),
        )
        col_positions = np.linspace(est_dx * 0.5, w - est_dx * 0.5, n_cols, dtype=np.float32)
        row_positions = np.linspace(est_dy * 0.5, h - est_dy * 0.5, n_rows, dtype=np.float32)

    # ── 4. Build grid in corrected space, rotate back to image space ───────
    grid_pts_corrected = _build_grid_pts(row_positions, col_positions)

    if abs(angle) > min_angle:
        grid_pts = _rotate_points(grid_pts_corrected, angle, img_origin)
    else:
        grid_pts = grid_pts_corrected

    # ── 5. Assign detected circles to cells (use original centers) ─────────
    snap_dist   = min(est_dx, est_dy) * config.snap_tol_frac
    is_detected = _assign_detections(centers, grid_pts, snap_dist)

    detected_count      = int(is_detected.sum())
    reconstructed_count = len(grid_pts) - detected_count

    logger.info(
        "auto_grid: detected={} reconstructed={} rotation={:.2f}° avg_r={:.1f}px",
        detected_count, reconstructed_count, np.degrees(angle), avg_radius,
    )

    # ── 6. Draw annotated image ───────────────────────────────────────────
    result_image = _draw_grid(
        image, grid_pts, is_detected, n_rows, n_cols, avg_radius, config,
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
        "grid_angle_deg":      float(np.degrees(angle)),
    }
