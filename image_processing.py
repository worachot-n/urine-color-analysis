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

from config import (
    HSV_RED_LOWER_1, HSV_RED_UPPER_1,
    HSV_RED_LOWER_2, HSV_RED_UPPER_2,
    MORPH_KERNEL_SIZE, GAUSSIAN_BLUR_KERNEL,
    HOUGH_DP, HOUGH_MIN_DIST,
    HOUGH_PARAM1, HOUGH_PARAM2,
    HOUGH_MIN_RADIUS, HOUGH_MAX_RADIUS,
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
    Detect red circular bottle caps in the full frame using Hough circles.

    Args:
        frame: BGR image (numpy array, full resolution)

    Returns:
        list of (cx, cy, radius) tuples — all detected circles, integers.
        Empty list if none found.
    """
    red_mask = create_red_mask(frame)

    circles = cv2.HoughCircles(
        red_mask,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS,
    )

    if circles is None:
        return []

    return [(int(c[0]), int(c[1]), int(c[2])) for c in circles[0]]
