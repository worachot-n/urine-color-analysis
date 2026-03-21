"""
Unit tests for color_analysis.py

All tests are runnable without hardware — pure numpy/cv2 operations.
"""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.color_analysis import (
    delta_e_cie76,
    extract_bottle_color,
    build_reference_baseline,
    classify_sample,
)


# ---------------------------------------------------------------------------
# delta_e_cie76
# ---------------------------------------------------------------------------

def test_delta_e_identical():
    lab = (50.0, 0.0, 0.0)
    assert delta_e_cie76(lab, lab) == 0.0


def test_delta_e_white_vs_black():
    white = (100.0, 0.0, 0.0)
    black = (0.0,   0.0, 0.0)
    assert abs(delta_e_cie76(white, black) - 100.0) < 1e-9


def test_delta_e_symmetric():
    a = (60.0, 10.0, -5.0)
    b = (40.0, -3.0, 15.0)
    assert abs(delta_e_cie76(a, b) - delta_e_cie76(b, a)) < 1e-9


def test_delta_e_non_negative():
    assert delta_e_cie76((0.0, 0.0, 0.0), (100.0, 50.0, -50.0)) >= 0.0


# ---------------------------------------------------------------------------
# extract_bottle_color
# ---------------------------------------------------------------------------

def test_extract_returns_none_when_crop_degenerates():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # radius=5, inner_crop=15 → crop collapses
    assert extract_bottle_color(frame, 50, 50, 5, inner_crop_px=15) is None


def test_extract_returns_tuple_of_three():
    frame = np.full((300, 300, 3), (80, 160, 200), dtype=np.uint8)
    result = extract_bottle_color(frame, 150, 150, 50, inner_crop_px=5)
    assert result is not None
    assert len(result) == 3


def test_extract_uniform_color_is_consistent():
    """Same uniform color in different positions should return the same Lab."""
    frame = np.full((400, 400, 3), (30, 120, 200), dtype=np.uint8)
    lab1 = extract_bottle_color(frame, 100, 100, 40, inner_crop_px=5)
    lab2 = extract_bottle_color(frame, 300, 300, 40, inner_crop_px=5)
    assert lab1 is not None and lab2 is not None
    for c1, c2 in zip(lab1, lab2):
        assert abs(c1 - c2) < 1.0   # Identical uniform regions → same median


def test_extract_clamps_to_image_bounds():
    """Circle partially outside frame edge should still return a result."""
    frame = np.full((200, 200, 3), (100, 100, 100), dtype=np.uint8)
    result = extract_bottle_color(frame, 5, 5, 40, inner_crop_px=5)
    # After clamping there might still be a valid crop (or None if degenerate)
    # Either outcome is acceptable — just must not raise
    assert result is None or len(result) == 3


# ---------------------------------------------------------------------------
# build_reference_baseline
# ---------------------------------------------------------------------------

def test_build_baseline_returns_all_levels():
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    # Draw 3 circles per level with slightly different greys
    positions = {}
    for level in range(5):
        brightness = int(50 + level * 40)
        pts = []
        for k in range(3):
            cx = 50 + level * 80
            cy = 50 + k * 100
            cv2.circle(frame, (cx, cy), 30, (brightness, brightness, brightness), -1)
            pts.append((cx, cy, 30))
        positions[level] = pts

    baseline = build_reference_baseline(frame, positions)
    assert set(baseline.keys()) == {0, 1, 2, 3, 4}
    for lab in baseline.values():
        assert len(lab) == 3


def test_build_baseline_empty_positions():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    baseline = build_reference_baseline(frame, {})
    assert baseline == {}


# ---------------------------------------------------------------------------
# classify_sample
# ---------------------------------------------------------------------------

def test_classify_exact_match():
    baseline = {
        0: (80.0,  0.0,  0.0),
        1: (65.0,  5.0, 10.0),
        2: (50.0, 10.0, 20.0),
        3: (40.0, 15.0, 30.0),
        4: (30.0, 20.0, 40.0),
    }
    level, de, confident = classify_sample((50.0, 10.0, 20.0), baseline, threshold=15.0)
    assert level == 2
    assert de == 0.0
    assert confident is True


def test_classify_closest_wins():
    baseline = {0: (80.0, 0.0, 0.0), 1: (50.0, 0.0, 0.0), 2: (20.0, 0.0, 0.0)}
    # Sample is 4 units away from level 1
    level, de, _ = classify_sample((54.0, 0.0, 0.0), baseline)
    assert level == 1


def test_classify_uncertain_when_far():
    baseline = {0: (80.0, 0.0, 0.0)}
    # Delta E = 100 → well above any threshold
    level, de, confident = classify_sample((0.0, 0.0, 80.0), baseline, threshold=15.0)
    assert confident is False


def test_classify_none_sample():
    level, de, confident = classify_sample(None, {0: (50.0, 0.0, 0.0)})
    assert level is None
    assert confident is False


def test_classify_empty_baseline():
    level, de, confident = classify_sample((50.0, 0.0, 0.0), {})
    assert level is None
    assert confident is False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_delta_e_identical,
        test_delta_e_white_vs_black,
        test_delta_e_symmetric,
        test_delta_e_non_negative,
        test_extract_returns_none_when_crop_degenerates,
        test_extract_returns_tuple_of_three,
        test_extract_uniform_color_is_consistent,
        test_extract_clamps_to_image_bounds,
        test_build_baseline_returns_all_levels,
        test_build_baseline_empty_positions,
        test_classify_exact_match,
        test_classify_closest_wins,
        test_classify_uncertain_when_far,
        test_classify_none_sample,
        test_classify_empty_baseline,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} color_analysis tests passed.")
