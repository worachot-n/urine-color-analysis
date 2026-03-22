"""
Image processing pipeline: red cap detection via HSV masking + grid-anchored contours.

Two detection strategies:
  detect_red_caps()       — global contour search (kept as utility)
  detect_bottles_by_slot() — grid-anchored per-slot search (primary, zero false negatives)

Pipeline (create_red_mask):
  1. CLAHE on HSV V-channel to normalise glare hotspots
  2. Dual-range HSV inRange (handles hue wrap-around at 0/180)
  3. GaussianBlur + morphological Open/Close to clean noise
"""

import cv2
import numpy as np

from configs.config import (
    HSV_RED_LOWER_1, HSV_RED_UPPER_1,
    HSV_RED_LOWER_2, HSV_RED_UPPER_2,
    MORPH_KERNEL_SIZE, MORPH_CLOSE_LARGE, GAUSSIAN_BLUR_KERNEL,
    MIN_CONTOUR_AREA, MAX_CONTOUR_AREA, CIRCULARITY_THRESHOLD,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
    SLOT_MIN_CONTOUR_AREA, SLOT_CIRCULARITY_THRESHOLD, SLOT_SEARCH_MARGIN,
)


def create_red_mask(frame):
    """
    Create a binary mask that isolates red regions in the frame.

    Uses two HSV ranges because red wraps around the hue axis (0° and 180°).
    Pipeline: HSV conversion → dual inRange → Gaussian blur → morphological
    Open (remove noise) → Close (fill gaps).

    Args:
        frame: BGR image (numpy array)

    Returns:
        uint8 binary mask, same H×W as frame
    """
    # Normalise brightness locally to recover red rings buried under glare hotspots
    hsv_pre = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    clahe   = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                               tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
    hsv_pre[:, :, 2] = clahe.apply(hsv_pre[:, :, 2])
    frame_eq = cv2.cvtColor(hsv_pre, cv2.COLOR_HSV2BGR)

    hsv = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv,
                        np.array(HSV_RED_LOWER_1, dtype=np.uint8),
                        np.array(HSV_RED_UPPER_1, dtype=np.uint8))
    mask2 = cv2.inRange(hsv,
                        np.array(HSV_RED_LOWER_2, dtype=np.uint8),
                        np.array(HSV_RED_UPPER_2, dtype=np.uint8))

    red_mask = cv2.bitwise_or(mask1, mask2)

    # Blur to reduce pixel noise
    red_mask = cv2.GaussianBlur(red_mask, GAUSSIAN_BLUR_KERNEL, 0)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    return red_mask


def detect_red_caps(frame):
    """
    Detect red circular rings using contour-based detection + circularity filter.

    Replaces HoughCircles to eliminate false negatives on partially-obscured rings.
    HoughCircles requires a nearly-complete circle to vote; contours catch broken
    or shadowed arcs that HoughCircles misses.

    Pipeline:
      1. Dual-range HSV red mask (widened saturation catches printed ring variants)
      2. GaussianBlur + morphological Open/Close (via create_red_mask)
      3. Extra large MORPH_CLOSE to bridge broken arc segments
      4. findContours → filter by area and circularity (4π·A/P² ≥ threshold)
      5. minEnclosingCircle to recover (cx, cy, r)

    Returns:
        list of (cx, cy, radius) tuples — integers. Empty list if none found.
    """
    red_mask = create_red_mask(frame)

    # Bridge broken ring arcs with a large closing pass
    big_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_CLOSE_LARGE, MORPH_CLOSE_LARGE)
    )
    closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, big_kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < CIRCULARITY_THRESHOLD:
            continue

        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        circles.append((int(cx), int(cy), int(r)))

    return circles


def detect_bottles_by_slot(frame, grid_cfg):
    """
    Grid-anchored bottle detection for zero false negatives.

    Instead of searching the entire frame globally, this function iterates
    every sample slot, crops the red mask to that slot's bounding box, and
    looks for ANY red contour within that area using relaxed local thresholds.

    Because we know exactly where to look, SLOT_MIN_CONTOUR_AREA and
    SLOT_CIRCULARITY_THRESHOLD can be far more permissive than the global
    equivalents without risking false positives from unrelated red objects.

    Args:
        frame:    BGR full-frame image (numpy array)
        grid_cfg: GridConfig instance (provides slot_data polygon coords)

    Returns:
        dict {slot_id: (cx, cy, radius)} — full-frame pixel coords,
        only for slots where a bottle contour was found.
    """
    red_mask = create_red_mask(frame)

    # One global large-close pass to bridge broken ring arcs
    big_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_CLOSE_LARGE, MORPH_CLOSE_LARGE)
    )
    closed_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, big_kernel)

    h_frame, w_frame = frame.shape[:2]
    results = {}

    for slot_id, info in grid_cfg.slot_data.items():
        coords = info['coords']

        x_min = int(np.min(coords[:, 0]))
        x_max = int(np.max(coords[:, 0]))
        y_min = int(np.min(coords[:, 1]))
        y_max = int(np.max(coords[:, 1]))

        x1 = max(0, x_min - SLOT_SEARCH_MARGIN)
        y1 = max(0, y_min - SLOT_SEARCH_MARGIN)
        x2 = min(w_frame, x_max + SLOT_SEARCH_MARGIN)
        y2 = min(h_frame, y_max + SLOT_SEARCH_MARGIN)

        local_mask = closed_mask[y1:y2, x1:x2]
        if local_mask.size == 0:
            continue

        contours, _ = cv2.findContours(
            local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_cnt  = None
        best_area = SLOT_MIN_CONTOUR_AREA - 1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < SLOT_MIN_CONTOUR_AREA:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < SLOT_CIRCULARITY_THRESHOLD:
                continue

            if area > best_area:
                best_area = area
                best_cnt  = cnt

        if best_cnt is not None:
            (lcx, lcy), r = cv2.minEnclosingCircle(best_cnt)
            # Convert from local-crop coords to full-frame coords
            cx = int(lcx) + x1
            cy = int(lcy) + y1
            results[slot_id] = (cx, cy, max(1, int(r)))

    return results
