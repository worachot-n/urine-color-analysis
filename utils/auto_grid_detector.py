"""
Robust grid detector for bottle trays — classical CV only, no deep learning.

Designed to run on the FULL-RESOLUTION image without any letterbox/resize
step (letterboxing is a YOLO requirement and distorts circle geometry at
non-square aspect ratios).

Pipeline
--------
1. preprocess()               — grayscale + CLAHE normalisation
2. detect_circles()           — HoughCircles with rough image-adaptive params
3. estimate_actual_spacing()  — refine dx/dy from detected circle positions
                                (handles tray smaller than full image)
4. estimate_grid_angle()      — estimate tray rotation from row line-fits
5. find_grid_positions()      — Gaussian KDE projection + scipy peak detection
                                on rotation-corrected centres, with accurate
                                bandwidth and minimum-distance parameters
6. fit_uniform_grid()         — extend partial detections to full n_rows × n_cols
7. build_grid_pts()           — Cartesian product, then rotate back to image space
8. assign_detections()        — snap each detected circle to its nearest cell
9. draw_grid()                — rotated midpoint grid lines + circle markers

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

    # Fractions of the ROUGH spacing (image_dim / n_cells) — used only for
    # HoughCircles sizing.  The KDE uses the refined spacing from actual circles.
    min_dist_frac:   float = 0.55   # min_dist   = rough_spacing × frac
    radius_min_frac: float = 0.12   # min_radius = rough_spacing × frac
    radius_max_frac: float = 0.55   # max_radius = rough_spacing × frac

    # --- Spacing refinement ---
    # After circle detection, the actual spacing is estimated from nearest-
    # neighbour diffs in the sorted X/Y coordinate lists.
    # Diffs are accepted only in [rough * spacing_lo_frac, rough * spacing_hi_frac].
    spacing_lo_frac: float = 0.10   # ignore diffs smaller than this fraction of rough
    spacing_hi_frac: float = 0.99   # ignore diffs larger than this fraction of rough

    # --- Projection KDE (uses REFINED spacing) ---
    kde_bw_frac:        float = 0.30  # KDE bandwidth = refined_spacing × frac
    peak_min_h_frac:    float = 0.04  # min peak height = max_density × frac
    peak_min_dist_frac: float = 0.60  # min separation = refined_spacing × frac

    # --- Grid snap ---
    snap_tol_frac: float = 0.45   # max distance to snap = refined_spacing × frac

    # --- Rotation correction ---
    min_correct_angle_deg: float = 0.20

    # --- Visualisation (BGR) ---
    color_detected:      tuple = (0, 220,  0)
    color_reconstructed: tuple = (0, 140, 255)
    color_h_lines:       tuple = (255, 220, 50)
    color_v_lines:       tuple = (50,  200, 255)
    circle_thickness:    int   = 2
    line_thickness:      int   = 3


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
    rough_spacing: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    HoughCircles with parameters derived from the rough grid spacing.

    Returns:
        centers — (N, 2) float32 array of (x, y) positions
        radii   — (N,)   float32 array
    """
    min_r = max(5,  int(rough_spacing * config.radius_min_frac))
    max_r = max(15, int(rough_spacing * config.radius_max_frac))
    min_d = max(8,  int(rough_spacing * config.min_dist_frac))

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
# Step 3: Refined spacing estimation
# ---------------------------------------------------------------------------

def _estimate_actual_spacing(
    centers: np.ndarray,
    rough_dx: float,
    rough_dy: float,
    config: AutoGridConfig,
) -> Tuple[float, float]:
    """
    Estimate actual grid column/row spacing from detected circle positions.

    Sorts all X (resp. Y) coordinates, takes consecutive differences,
    and keeps only those in [rough * lo_frac, rough * hi_frac].
    The 25th-percentile of the kept values is the fundamental spacing —
    it rejects 2×/3× multiples that appear when circles are missing.

    This handles the common case where the tray occupies only a fraction
    of the full camera frame (so image_width/n_cols overestimates the true
    column spacing by 2–4×).
    """
    if len(centers) < 4:
        return rough_dx, rough_dy

    lo_x, hi_x = rough_dx * config.spacing_lo_frac, rough_dx * config.spacing_hi_frac
    lo_y, hi_y = rough_dy * config.spacing_lo_frac, rough_dy * config.spacing_hi_frac

    xs = np.sort(centers[:, 0])
    ys = np.sort(centers[:, 1])

    x_diffs = np.diff(xs)
    y_diffs = np.diff(ys)

    x_valid = x_diffs[(x_diffs >= lo_x) & (x_diffs <= hi_x)]
    y_valid = y_diffs[(y_diffs >= lo_y) & (y_diffs <= hi_y)]

    def _fundamental(diffs: np.ndarray, fallback: float) -> float:
        if len(diffs) < 3:
            return fallback
        # Keep the lower 35% — discards 2×/3× multiples
        threshold = float(np.percentile(diffs, 35))
        small = diffs[diffs <= threshold * 1.4]
        return float(np.median(small)) if len(small) > 0 else fallback

    est_dx = _fundamental(x_valid, rough_dx)
    est_dy = _fundamental(y_valid, rough_dy)

    logger.info(
        "auto_grid: spacing  dx {:.1f}→{:.1f}  dy {:.1f}→{:.1f}",
        rough_dx, est_dx, rough_dy, est_dy,
    )
    return est_dx, est_dy


# ---------------------------------------------------------------------------
# Step 4: Rotation estimation
# ---------------------------------------------------------------------------

def _rotate_points(pts: np.ndarray, angle_rad: float, origin: np.ndarray) -> np.ndarray:
    """Rotate 2-D points around `origin` by `angle_rad` (CCW positive)."""
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return ((pts.astype(np.float64) - origin) @ R.T + origin).astype(np.float32)


def _estimate_grid_angle(centers: np.ndarray, est_dy: float) -> float:
    """
    Estimate tray rotation angle (radians) from detected circle centres.

    Groups circles into approximate rows by Y proximity (±est_dy/2),
    fits a line per row, returns the median row slope angle.
    """
    if len(centers) < 6:
        return 0.0

    sorted_pts = centers[np.argsort(centers[:, 1])]

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
        m = float(np.linalg.lstsq(A, y, rcond=None)[0][0])
        angles.append(float(np.arctan(m)))

    if len(angles) < 2:
        return 0.0

    return float(np.median(angles))


# ---------------------------------------------------------------------------
# Step 5: 1-D axis-position estimation via projection KDE
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
    `peak_position mod spacing`, median vote is the grid origin.
    The grid is then centred on the detected peak cluster.
    """
    detected = np.sort(detected)

    if len(detected) >= 2:
        spacing = _estimate_fundamental_spacing(np.diff(detected), est_spacing)
    else:
        spacing = est_spacing

    votes = detected % spacing
    votes[votes > spacing * 0.5] -= spacing
    origin_frac = float(np.median(votes))
    if origin_frac < 0:
        origin_frac += spacing

    centre_detected = float(np.median(detected))
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
    est_spacing: float,     # REFINED spacing (not rough image-based)
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
# Step 6: Grid points and assignment
# ---------------------------------------------------------------------------

def _build_grid_pts(
    row_positions: np.ndarray,
    col_positions: np.ndarray,
) -> np.ndarray:
    """Return (n_rows × n_cols, 2) float32 Cartesian product (row-major)."""
    return np.array(
        [[cx, ry] for ry in row_positions for cx in col_positions],
        dtype=np.float32,
    )


def _assign_detections(
    centers: np.ndarray,
    grid_pts: np.ndarray,
    snap_dist: float,
) -> np.ndarray:
    """Mark each grid cell whose nearest detected circle is within snap_dist."""
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
# Step 7: Annotation
# ---------------------------------------------------------------------------

def draw_grid_lines(
    canvas: np.ndarray,
    grid_pts: np.ndarray,   # (n_rows*n_cols, 2) float32 in image space
    n_rows: int,
    n_cols: int,
    avg_radius: float,
    config: Optional[AutoGridConfig] = None,
) -> None:
    """
    Draw the full grid (n_rows+1 horizontal + n_cols+1 vertical lines)
    IN-PLACE on `canvas`. No circle markers — reusable as an overlay on top
    of other annotations.

    Boundary lines extend just outside the bottle radius so caps are never
    cut by the outer frame; interior lines pass through the midpoints
    between adjacent row/column centres.
    """
    if config is None:
        config = AutoGridConfig()
    lw = max(1, config.line_thickness)
    pts = grid_pts.reshape(n_rows, n_cols, 2)

    # Edge offset factor: push outer boundary outside the bottle radius so the
    # line never crosses cap edges. Interior boundaries (midpoints between
    # adjacent rows/cols) are unaffected.
    mid_r, mid_c = n_rows // 2, n_cols // 2
    next_r = mid_r + 1 if mid_r + 1 < n_rows else 1
    next_c = mid_c + 1 if mid_c + 1 < n_cols else 1
    dy_px = float(np.linalg.norm(pts[next_r, mid_c] - pts[mid_r, mid_c])) if n_rows >= 2 else 1.0
    dx_px = float(np.linalg.norm(pts[mid_r, next_c] - pts[mid_r, mid_c])) if n_cols >= 2 else 1.0
    edge_margin = 4.0
    fy = min(0.95, max(0.5, (avg_radius + edge_margin) / dy_px)) if dy_px > 0 else 0.5
    fx = min(0.95, max(0.5, (avg_radius + edge_margin) / dx_px)) if dx_px > 0 else 0.5

    # Precompute boundary midpoint arrays for all 4 sides
    top_pts   = pts[0]    + (pts[0]    - pts[1])    * fy   # (n_cols, 2) — above row 0
    bot_pts   = pts[-1]   + (pts[-1]   - pts[-2])   * fy   # (n_cols, 2) — below last row
    left_pts  = pts[:, 0] + (pts[:, 0] - pts[:, 1]) * fx  # (n_rows, 2) — left of col 0
    right_pts = pts[:,-1] + (pts[:,-1] - pts[:,-2]) * fx  # (n_rows, 2) — right of last col

    # 4 corner points: extrapolate outward in both axes from each corner cell
    TL = pts[0,  0 ] + (pts[0,  0 ] - pts[1,  0 ]) * fy + (pts[0,  0 ] - pts[0,  1 ]) * fx
    TR = pts[0, -1 ] + (pts[0, -1 ] - pts[1, -1 ]) * fy + (pts[0, -1 ] - pts[0, -2 ]) * fx
    BL = pts[-1,  0] + (pts[-1,  0] - pts[-2,  0]) * fy + (pts[-1,  0] - pts[-1,  1]) * fx
    BR = pts[-1, -1] + (pts[-1, -1] - pts[-2, -1]) * fy + (pts[-1, -1] - pts[-1, -2]) * fx

    # ── Horizontal lines (n_rows + 1): each spans from left boundary to right boundary ──
    for bi in range(n_rows + 1):
        if bi == 0:
            inner, le, re = top_pts, TL, TR
        elif bi == n_rows:
            inner, le, re = bot_pts, BL, BR
        else:
            inner = (pts[bi - 1] + pts[bi]) * 0.5
            le    = (left_pts[bi - 1]  + left_pts[bi])  * 0.5
            re    = (right_pts[bi - 1] + right_pts[bi]) * 0.5
        line = np.vstack([le, inner, re]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [line], False, config.color_h_lines, lw, cv2.LINE_AA)

    # ── Vertical lines (n_cols + 1): each spans from top boundary to bottom boundary ──
    for bi in range(n_cols + 1):
        if bi == 0:
            inner, te, be = left_pts, TL, BL
        elif bi == n_cols:
            inner, te, be = right_pts, TR, BR
        else:
            inner = (pts[:, bi - 1] + pts[:, bi]) * 0.5
            te    = (top_pts[bi - 1] + top_pts[bi]) * 0.5
            be    = (bot_pts[bi - 1] + bot_pts[bi]) * 0.5
        line = np.vstack([te, inner, be]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [line], False, config.color_v_lines, lw, cv2.LINE_AA)


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
    Draw the annotated grid on a copy of `image`: full grid lines plus a
    per-cell circle marker (green = detected by HoughCircles, orange =
    reconstructed/inferred).
    """
    canvas = image.copy()
    draw_grid_lines(canvas, grid_pts, n_rows, n_cols, avg_radius, config)

    r = max(5, int(avg_radius))
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

    Handles:
    - Tray smaller than the full camera frame (refines spacing from actual
      circle positions rather than image_dim / n_cells)
    - Slightly rotated trays (estimates rotation angle from row line-fits,
      corrects before KDE, rotates grid points back)

    Args:
        image:  Full-resolution (or ROI-cropped) BGR image.
        n_rows: Expected number of grid rows (from slot config).
        n_cols: Expected number of grid columns (from slot config).
        config: Optional tuning config; sensible defaults if None.

    Returns dict with keys:
        grid_pts            (n_rows*n_cols, 2) float32 — cell centres in image space
        is_detected         (n_rows*n_cols,) bool
        result_image        BGR annotated image (same resolution as input)
        detected_count      int
        reconstructed_count int
        col_positions       (n_cols,) float32  — in rotation-corrected space
        row_positions       (n_rows,) float32  — in rotation-corrected space
        avg_radius_px       float
        grid_angle_deg      float — estimated tray rotation
        est_dx              float — refined column spacing (px)
        est_dy              float — refined row spacing (px)
    """
    if config is None:
        config = AutoGridConfig()

    h, w = image.shape[:2]

    # Rough spacing — used only to size HoughCircles parameters
    rough_dx      = w / n_cols
    rough_dy      = h / n_rows
    rough_spacing = (rough_dx + rough_dy) / 2.0

    logger.info(
        "auto_grid: image={}×{}  grid={}×{}  rough_spacing={:.1f}px",
        w, h, n_cols, n_rows, rough_spacing,
    )

    # ── 1. Preprocess + detect circles ────────────────────────────────────
    gray           = _preprocess(image)
    centers, radii = _detect_circles(gray, config, rough_spacing)

    avg_radius = float(np.median(radii)) if len(radii) > 0 else rough_spacing * 0.38

    # ── 2. Refine spacing from actual circle positions ────────────────────
    if len(centers) >= 4:
        est_dx, est_dy = _estimate_actual_spacing(centers, rough_dx, rough_dy, config)
    else:
        est_dx, est_dy = rough_dx, rough_dy

    # ── 3. Estimate and correct for tray rotation ──────────────────────────
    angle = _estimate_grid_angle(centers, est_dy) if len(centers) >= 6 else 0.0
    min_angle = np.radians(config.min_correct_angle_deg)
    img_origin = np.array([w / 2.0, h / 2.0])

    if abs(angle) > min_angle:
        logger.info("auto_grid: rotation={:.3f}° — applying correction", np.degrees(angle))
        centers_fit = _rotate_points(centers, -angle, img_origin)
    else:
        angle = 0.0
        centers_fit = centers

    # ── 4. Projection KDE → axis positions (on rotation-corrected coords) ─
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

    # ── 5. Build grid in corrected space, rotate back to image space ───────
    grid_pts_corrected = _build_grid_pts(row_positions, col_positions)

    if abs(angle) > min_angle:
        grid_pts = _rotate_points(grid_pts_corrected, angle, img_origin)
    else:
        grid_pts = grid_pts_corrected

    # ── 6. Assign detected circles to cells (original, unrotated centres) ─
    snap_dist   = min(est_dx, est_dy) * config.snap_tol_frac
    is_detected = _assign_detections(centers, grid_pts, snap_dist)

    detected_count      = int(is_detected.sum())
    reconstructed_count = len(grid_pts) - detected_count

    logger.info(
        "auto_grid: detected={} reconstructed={} rotation={:.2f}° "
        "spacing=({:.1f}, {:.1f})px avg_r={:.1f}px",
        detected_count, reconstructed_count,
        np.degrees(angle), est_dx, est_dy, avg_radius,
    )

    # ── 7. Draw annotated image ───────────────────────────────────────────
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
        "est_dx":              est_dx,
        "est_dy":              est_dy,
    }
