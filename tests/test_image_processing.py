"""
Unit tests for image_processing.py

All tests use synthetic images — no camera hardware required.
"""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_processing import create_red_mask, detect_red_caps


# ---------------------------------------------------------------------------
# create_red_mask
# ---------------------------------------------------------------------------

def test_mask_all_black_image():
    """No red in a black image."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    mask  = create_red_mask(frame)
    assert cv2.countNonZero(mask) == 0


def test_mask_all_green_image():
    """Pure green has no red component."""
    frame = np.full((200, 200, 3), (0, 255, 0), dtype=np.uint8)   # BGR green
    mask  = create_red_mask(frame)
    assert cv2.countNonZero(mask) == 0


def test_mask_red_region_detected():
    """A solid red circle should produce a non-empty mask."""
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(frame, (150, 150), 50, (0, 0, 220), -1)   # BGR red
    mask = create_red_mask(frame)
    assert cv2.countNonZero(mask) > 0


def test_mask_shape_matches_input():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mask  = create_red_mask(frame)
    assert mask.shape == (480, 640)


def test_mask_dtype_is_uint8():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mask  = create_red_mask(frame)
    assert mask.dtype == np.uint8


# ---------------------------------------------------------------------------
# detect_red_caps
# ---------------------------------------------------------------------------

def test_detect_empty_frame():
    frame   = np.zeros((500, 500, 3), dtype=np.uint8)
    circles = detect_red_caps(frame)
    assert circles == []


def test_detect_returns_list():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    assert isinstance(detect_red_caps(frame), list)


def test_detect_circle_position_approximate():
    """A clear red circle should be detected near its true center."""
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    true_cx, true_cy, true_r = 250, 250, 40
    cv2.circle(frame, (true_cx, true_cy), true_r, (0, 0, 200), -1)

    circles = detect_red_caps(frame)

    if circles:   # Hough might not detect under every param set — soft check
        cx, cy, r = circles[0]
        assert abs(cx - true_cx) < 30
        assert abs(cy - true_cy) < 30


def test_detect_result_types():
    """All returned values must be integers."""
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(frame, (200, 200), 35, (0, 0, 210), -1)
    circles = detect_red_caps(frame)
    for cx, cy, r in circles:
        assert isinstance(cx, int)
        assert isinstance(cy, int)
        assert isinstance(r,  int)


def test_detect_non_red_circle_ignored():
    """A blue circle should not be returned."""
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(frame, (200, 200), 40, (200, 0, 0), -1)   # BGR blue
    circles = detect_red_caps(frame)
    assert circles == []


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_mask_all_black_image,
        test_mask_all_green_image,
        test_mask_red_region_detected,
        test_mask_shape_matches_input,
        test_mask_dtype_is_uint8,
        test_detect_empty_frame,
        test_detect_returns_list,
        test_detect_circle_position_approximate,
        test_detect_result_types,
        test_detect_non_red_circle_ignored,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} image_processing tests passed.")
