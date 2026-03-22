"""
Image processing pipeline: red cap detection via HSV masking + Hough circles.

This module works on the FULL FRAME (not per-cell crops), matching CLAUDE.md:
  1. Convert frame to HSV
  2. Dual-range red mask (handles HSV wrap-around at 0/180)
  3. GaussianBlur + morphological Open/Close to clean noise
  4. HoughCircles on the cleaned mask

The caller (grid.py) then assigns each detected circle to a grid slot via
the majority-rule polygon overlap test.
"""

import cv2
import numpy as np

from configs.config import (
    HSV_RED_LOWER_1, HSV_RED_UPPER_1,
    HSV_RED_LOWER_2, HSV_RED_UPPER_2,
    MORPH_KERNEL_SIZE, MORPH_CLOSE_LARGE, GAUSSIAN_BLUR_KERNEL,
    MIN_CONTOUR_AREA, MAX_CONTOUR_AREA, CIRCULARITY_THRESHOLD,
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
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
