"""
Standalone grid-circle detector for urine-analysis tray images.

Detects circular bottle caps on a dark background, reconstructs missing
positions using known grid dimensions (rows × cols from settings), and
draws midpoint-aligned grid lines between rows and columns.

No ML, no YOLO, no trained weights required.

Pipeline:
    detect_circles()         → raw (x,y) centres from HoughCircles
    estimate_grid_spacing()  → (dx, dy, origin_x, origin_y)
    reconstruct_grid()       → all (n_rows × n_cols) positions, detected + filled
    draw_grid_lines()        → annotated BGR image

Standalone usage:
    python utils/grid_circle_detector.py --image tray.jpg --rows 13 --cols 15
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GridCircleConfig:
    # HoughCircles parameters
    dp: float = 1.2
    min_dist: int = 55          # min distance between centres ≈ circle diameter
    param1: int = 60            # Canny high threshold
    param2: int = 25            # accumulator threshold — permissive to catch colored caps
    min_radius: int = 22        # radius ≈ 30px − 8px margin
    max_radius: int = 38        # radius ≈ 30px + 8px margin

    # Preprocessing
    blur_kernel: int = 5        # GaussianBlur kernel (must be odd)
    clahe_clip: float = 2.0
    clahe_tile: int = 8

    # Grid reconstruction
    snap_tolerance: float = 0.4  # fraction of spacing to accept as same grid cell
    min_spacing_px: int = 30     # guard against degenerate estimate

    # Visualization (BGR)
    color_detected: tuple = (0, 220, 0)        # green  — found by HoughCircles
    color_reconstructed: tuple = (0, 140, 255) # orange — filled-in missing
    color_lines_h: tuple = (255, 220, 0)       # horizontal grid lines
    color_lines_v: tuple = (0, 200, 255)       # vertical grid lines
    circle_thickness: int = 2
    line_thickness: int = 1


# ---------------------------------------------------------------------------
# Step 1: Circle detection
# ---------------------------------------------------------------------------

def detect_circles(gray: np.ndarray, config: GridCircleConfig) -> np.ndarray:
    """
    Detect circle centres using CLAHE preprocessing + cv2.HoughCircles.

    CLAHE on the L channel normalises white and colored (orange/red/yellow)
    caps to similar luminance contrast before edge detection.

    Returns:
        (N, 2) float32 array of (x, y) centres; shape (0, 2) if none found.
    """
    # CLAHE on L channel: equalises perceived lightness across hue variation
    bgr_proxy = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(bgr_proxy, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip,
        tileGridSize=(config.clahe_tile, config.clahe_tile),
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        enhanced_gray,
        (config.blur_kernel, config.blur_kernel),
        0,
    )

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=config.dp,
        minDist=config.min_dist,
        param1=config.param1,
        param2=config.param2,
        minRadius=config.min_radius,
        maxRadius=config.max_radius,
    )

    if circles is None:
        logger.warning("detect_circles: HoughCircles found no circles")
        return np.empty((0, 2), dtype=np.float32)

    centers = circles[0, :, :2].astype(np.float32)
    logger.info("detect_circles: found {} raw circles", len(centers))
    return centers


# ---------------------------------------------------------------------------
# Step 2: Grid spacing estimation
# ---------------------------------------------------------------------------

def _estimate_axis_spacing(sorted_coords: np.ndarray, min_spacing: int) -> float:
    """
    Estimate the fundamental grid period from sorted 1D projected coordinates.

    The sorted-diff distribution is multi-modal (gaps of 1×, 2×, 3× spacing
    when circles are missing).  We isolate the 1× peak by filtering to
    [min_spacing, 3.5 × min_spacing], then take the median.

    A de-aliasing step prevents returning 2× the true spacing when many
    circles are missing (50% loss would make 2× gaps dominate).
    """
    diffs = np.diff(sorted_coords)
    if len(diffs) == 0:
        return float(min_spacing)

    mask = (diffs >= min_spacing) & (diffs <= 3.5 * min_spacing)
    filtered = diffs[mask]

    if len(filtered) < 3:
        filtered = diffs[diffs >= min_spacing]

    if len(filtered) == 0:
        positive = diffs[diffs > 0]
        return float(np.median(positive)) if len(positive) > 0 else float(min_spacing)

    spacing = float(np.median(filtered))

    # De-aliasing: check if spacing/2 or spacing/3 is better supported
    for divisor in (2, 3):
        candidate = spacing / divisor
        if candidate >= min_spacing:
            near_candidate = int(np.sum(np.abs(diffs - candidate) < 0.3 * candidate))
            near_spacing = int(np.sum(np.abs(diffs - spacing) < 0.3 * spacing))
            if near_candidate >= near_spacing * 0.5:
                spacing = candidate
                break

    return spacing


def estimate_grid_spacing(
    centers: np.ndarray,
    n_rows: int,
    n_cols: int,
    image_shape: tuple[int, int],
    config: GridCircleConfig,
) -> tuple[float, float, float, float]:
    """
    Estimate grid spacing (dx, dy) and initial origin (origin_x, origin_y).

    Known grid dimensions (n_rows, n_cols) provide a last-resort fallback only
    when fewer than 4 circles are detected.  When circles are available the
    measured spacing is trusted directly — the image-dimension fallback is wrong
    whenever the tray does not fill the full letterboxed image (which is the
    common case after aspect-ratio padding).

    The initial origin is derived from the detected circles themselves (modular
    median) rather than a fixed dx/2 offset, so that RANSAC voting in
    reconstruct_grid snaps circles correctly regardless of where the tray sits
    within the frame.

    Returns:
        (dx, dy, origin_x, origin_y)
    """
    h, w = image_shape[:2]
    dx_fallback = w / n_cols
    dy_fallback = h / n_rows

    if len(centers) >= 4:
        xs = np.sort(centers[:, 0])
        ys = np.sort(centers[:, 1])
        dx_est = _estimate_axis_spacing(xs, config.min_spacing_px)
        dy_est = _estimate_axis_spacing(ys, config.min_spacing_px)

        # Trust the measured value as long as it is physically plausible.
        # The old ±30% cross-validation against w/n_cols was wrong whenever the
        # tray occupies less than the full letterboxed width/height (which is
        # always the case after aspect-ratio padding adds white bars).
        dx = dx_est if config.min_spacing_px < dx_est < w else dx_fallback
        dy = dy_est if config.min_spacing_px < dy_est < h else dy_fallback
    else:
        logger.warning(
            "estimate_grid_spacing: only {} circles detected — using image-dimension fallback",
            len(centers),
        )
        dx, dy = dx_fallback, dy_fallback

    # Better initial origin: derive from the fractional grid position of every
    # detected circle.  Each circle at cx satisfies cx ≡ origin_x (mod dx), so
    # median(cx mod dx) is a robust, data-driven estimate.  This is far better
    # than the old dx/2 heuristic, which failed whenever the tray was not
    # anchored at the image top-left corner.
    if len(centers) >= 4:
        origin_x = float(np.median(centers[:, 0] % dx))
        origin_y = float(np.median(centers[:, 1] % dy))
    else:
        origin_x = dx / 2.0
        origin_y = dy / 2.0

    logger.info(
        "estimate_grid_spacing: dx={:.1f}  dy={:.1f}  origin=({:.1f},{:.1f})",
        dx, dy, origin_x, origin_y,
    )
    return dx, dy, origin_x, origin_y


# ---------------------------------------------------------------------------
# Step 3: Grid reconstruction
# ---------------------------------------------------------------------------

def reconstruct_grid(
    centers: np.ndarray,
    n_rows: int,
    n_cols: int,
    dx: float,
    dy: float,
    origin_x: float,
    origin_y: float,
    config: GridCircleConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign detected circles to grid cells (RANSAC-style origin voting) and
    fill all n_rows × n_cols positions.

    Phase 1 — origin voting:
        Each detected circle that snaps cleanly to a grid cell votes for the
        sub-pixel origin offset that places it exactly on-grid.  The median
        vote is the refined origin — robust to missing circles and outliers.

    Phase 2 — fixed grid extent:
        n_rows and n_cols come from settings, so no extent estimation needed.
        Only the top-left anchor is derived from the detected circles.

    Phase 3 — fill all cells:
        Every (row, col) is given an ideal position; is_detected marks which
        were actually found by HoughCircles.

    Returns:
        grid_pts:    (n_rows * n_cols, 2) float32 — ideal grid positions
        is_detected: (n_rows * n_cols,)   bool    — True = HoughCircles found it
    """
    def _snap_and_vote(snap_tol_x: float, snap_tol_y: float) -> tuple[dict, list, list]:
        snapped: dict[tuple[int, int], tuple[float, float]] = {}
        votes_x: list[float] = []
        votes_y: list[float] = []
        for cx, cy in centers:
            col_f = (cx - origin_x) / dx
            row_f = (cy - origin_y) / dy
            col_i = int(round(col_f))
            row_i = int(round(row_f))
            if abs(col_f - col_i) * dx <= snap_tol_x and abs(row_f - row_i) * dy <= snap_tol_y:
                votes_x.append(cx - col_i * dx)
                votes_y.append(cy - row_i * dy)
                if (col_i, row_i) not in snapped:
                    snapped[(col_i, row_i)] = (cx, cy)
        return snapped, votes_x, votes_y

    tol_x = config.snap_tolerance * dx
    tol_y = config.snap_tolerance * dy

    snapped, votes_x, votes_y = _snap_and_vote(tol_x, tol_y)

    # Retry with doubled tolerance if too few snapped
    if len(votes_x) < 3:
        logger.warning(
            "reconstruct_grid: only {} votes with default tolerance — retrying with 2× tolerance",
            len(votes_x),
        )
        snapped, votes_x, votes_y = _snap_and_vote(tol_x * 2, tol_y * 2)

    if votes_x:
        refined_ox = float(np.median(votes_x))
        refined_oy = float(np.median(votes_y))
    else:
        logger.warning("reconstruct_grid: no circles snapped — using initial origin estimate")
        refined_ox, refined_oy = origin_x, origin_y

    # Determine top-left anchor from snapped circles
    if snapped:
        min_col = min(c for c, _ in snapped)
        min_row = min(r for _, r in snapped)
    else:
        min_col = 0
        min_row = 0

    final_ox = refined_ox + min_col * dx
    final_oy = refined_oy + min_row * dy

    # Normalise snapped indices relative to min_col/min_row
    detected_cells: set[tuple[int, int]] = set()
    for (ci, ri) in snapped:
        nc = ci - min_col
        nr = ri - min_row
        if 0 <= nr < n_rows and 0 <= nc < n_cols:
            detected_cells.add((nr, nc))

    grid_pts = np.zeros((n_rows * n_cols, 2), dtype=np.float32)
    is_detected = np.zeros(n_rows * n_cols, dtype=bool)

    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            grid_pts[idx, 0] = final_ox + c * dx
            grid_pts[idx, 1] = final_oy + r * dy
            is_detected[idx] = (r, c) in detected_cells

    detected_count = int(is_detected.sum())
    reconstructed_count = n_rows * n_cols - detected_count
    logger.info(
        "reconstruct_grid: {}×{} grid — {} detected, {} reconstructed",
        n_rows, n_cols, detected_count, reconstructed_count,
    )
    return grid_pts, is_detected


# ---------------------------------------------------------------------------
# Step 4: Draw grid lines
# ---------------------------------------------------------------------------

def draw_grid_lines(
    image: np.ndarray,
    grid_pts: np.ndarray,
    is_detected: np.ndarray,
    n_rows: int,
    n_cols: int,
    dx: float,
    dy: float,
    config: GridCircleConfig,
) -> np.ndarray:
    """
    Draw grid lines at midpoints BETWEEN circles and circle markers on a copy
    of the image.

    Lines follow the actual grid_pts positions per row/column, so they remain
    correct under slight perspective distortion (piecewise-linear polylines,
    same as _draw_calibrated_grid in app/shared/processor.py).

    Green circles  = detected by HoughCircles
    Orange circles = reconstructed (missing in detection)
    """
    canvas = image.copy()
    pts = grid_pts.reshape(n_rows, n_cols, 2)
    r_circle = max(2, int(dx * 0.42))

    # --- Horizontal lines ---
    # Top boundary
    top_line = np.array(
        [[int(pts[0, c, 0]), int(pts[0, c, 1] - dy / 2)] for c in range(n_cols)],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [top_line.reshape(-1, 1, 2)], False,
                  config.color_lines_h, config.line_thickness)

    # Inner horizontal lines (between row r and row r+1)
    for r in range(n_rows - 1):
        line = np.array(
            [
                [int(pts[r, c, 0]), int((pts[r, c, 1] + pts[r + 1, c, 1]) / 2)]
                for c in range(n_cols)
            ],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [line.reshape(-1, 1, 2)], False,
                      config.color_lines_h, config.line_thickness)

    # Bottom boundary
    bot_line = np.array(
        [[int(pts[-1, c, 0]), int(pts[-1, c, 1] + dy / 2)] for c in range(n_cols)],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [bot_line.reshape(-1, 1, 2)], False,
                  config.color_lines_h, config.line_thickness)

    # --- Vertical lines ---
    # Left boundary
    left_line = np.array(
        [[int(pts[r, 0, 0] - dx / 2), int(pts[r, 0, 1])] for r in range(n_rows)],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [left_line.reshape(-1, 1, 2)], False,
                  config.color_lines_v, config.line_thickness)

    # Inner vertical lines (between col c and col c+1)
    for c in range(n_cols - 1):
        line = np.array(
            [
                [int((pts[r, c, 0] + pts[r, c + 1, 0]) / 2), int(pts[r, c, 1])]
                for r in range(n_rows)
            ],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [line.reshape(-1, 1, 2)], False,
                      config.color_lines_v, config.line_thickness)

    # Right boundary
    right_line = np.array(
        [[int(pts[r, -1, 0] + dx / 2), int(pts[r, -1, 1])] for r in range(n_rows)],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [right_line.reshape(-1, 1, 2)], False,
                  config.color_lines_v, config.line_thickness)

    # --- Circle markers ---
    for idx in range(len(grid_pts)):
        cx = int(grid_pts[idx, 0])
        cy = int(grid_pts[idx, 1])
        color = config.color_detected if is_detected[idx] else config.color_reconstructed
        cv2.circle(canvas, (cx, cy), r_circle, color, config.circle_thickness)

    return canvas


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def detect_grid(
    image: np.ndarray,
    n_rows: int,
    n_cols: int,
    config: Optional[GridCircleConfig] = None,
) -> dict:
    """
    Full pipeline: raw BGR image → grid-annotated BGR image.

    Args:
        image:  BGR image (uint8, H×W×3).
        n_rows: Expected number of grid rows (from settings).
        n_cols: Expected number of grid columns (from settings).
        config: Optional GridCircleConfig; defaults used if None.
                When None, HoughCircles parameters are derived from the image
                dimensions and expected grid spacing so they stay valid for any
                input resolution.

    Returns:
        {
            "result_image":        BGR image with grid lines and circle markers,
            "grid_pts":            (n_rows*n_cols, 2) float32 array,
            "detected_count":      int,
            "reconstructed_count": int,
        }
    """
    if config is None:
        h, w = image.shape[:2]
        # Expected spacing in the shortest axis — the binding constraint for
        # min_dist and circle radius.  Using min() keeps us conservative: the
        # y-spacing is often smaller than x-spacing after letterbox padding
        # (white bars on top/bottom shorten the effective tray height in pixels).
        expected_spacing = min(w / n_cols, h / n_rows)
        config = GridCircleConfig(
            min_dist   = max(8,  int(expected_spacing * 0.55)),
            min_radius = max(5,  int(expected_spacing * 0.20)),
            max_radius = max(15, int(expected_spacing * 0.65)),
        )
        logger.debug(
            "detect_grid: adaptive params — spacing≈{:.1f}  min_dist={}  r=[{},{}]",
            expected_spacing, config.min_dist, config.min_radius, config.max_radius,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    centers = detect_circles(gray, config)

    dx, dy, origin_x, origin_y = estimate_grid_spacing(
        centers, n_rows, n_cols, image.shape, config
    )

    grid_pts, is_detected = reconstruct_grid(
        centers, n_rows, n_cols, dx, dy, origin_x, origin_y, config
    )

    result_image = draw_grid_lines(
        image, grid_pts, is_detected, n_rows, n_cols, dx, dy, config
    )

    detected_count = int(is_detected.sum())
    reconstructed_count = n_rows * n_cols - detected_count

    return {
        "result_image": result_image,
        "grid_pts": grid_pts,
        "detected_count": detected_count,
        "reconstructed_count": reconstructed_count,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Detect circle grid and draw grid lines")
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--rows", type=int, required=True, help="Expected number of grid rows")
    ap.add_argument("--cols", type=int, required=True, help="Expected number of grid columns")
    ap.add_argument("--output", default=None, help="Output annotated image path (optional)")
    ap.add_argument("--param2", type=int, default=None,
                    help="Override HoughCircles param2 (accumulator threshold)")
    ap.add_argument("--min-radius", type=int, default=None, dest="min_radius")
    ap.add_argument("--max-radius", type=int, default=None, dest="max_radius")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        logger.error("Cannot read image: {}", args.image)
        sys.exit(1)

    cfg = GridCircleConfig()
    if args.param2 is not None:
        cfg.param2 = args.param2
    if args.min_radius is not None:
        cfg.min_radius = args.min_radius
    if args.max_radius is not None:
        cfg.max_radius = args.max_radius

    result = detect_grid(img, args.rows, args.cols, cfg)

    logger.info(
        "Grid: {} rows × {} cols — {} detected + {} reconstructed",
        args.rows, args.cols,
        result["detected_count"],
        result["reconstructed_count"],
    )

    out_path = args.output or (Path(args.image).stem + "_grid.jpg")
    cv2.imwrite(out_path, result["result_image"])
    logger.success("Saved annotated image: {}", out_path)
