"""
Unit tests for grid.py

Tests parse_slot_id, _circle_polygon_overlap, and compute_slot_polygons
(from calibration.py) without requiring grid_config.json on disk.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.grid import parse_slot_id, _circle_polygon_overlap
from utils.calibration import compute_slot_polygons


# ---------------------------------------------------------------------------
# parse_slot_id
# ---------------------------------------------------------------------------

def test_parse_A25_3():
    r = parse_slot_id("A25_3")
    assert r == {"group": "A2", "slot": 5, "expected_level": 3}


def test_parse_A11_0():
    r = parse_slot_id("A11_0")
    assert r["group"] == "A1"
    assert r["slot"] == 1
    assert r["expected_level"] == 0


def test_parse_A49_4():
    r = parse_slot_id("A49_4")
    assert r["group"] == "A4"
    assert r["slot"] == 9
    assert r["expected_level"] == 4


def test_parse_A31_2():
    r = parse_slot_id("A31_2")
    assert r["group"] == "A3"
    assert r["slot"] == 1
    assert r["expected_level"] == 2


# ---------------------------------------------------------------------------
# _circle_polygon_overlap
# ---------------------------------------------------------------------------

def test_overlap_circle_fully_inside():
    """Circle fully inside a large polygon → overlap ≈ circle area."""
    poly = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32)
    overlap = _circle_polygon_overlap(100, 100, 20, poly)
    expected = np.pi * 20 * 20
    assert overlap > expected * 0.90   # >90% of circle area


def test_overlap_circle_outside():
    """Circle completely outside polygon → 0."""
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    overlap = _circle_polygon_overlap(500, 500, 5, poly)
    assert overlap == 0.0


def test_overlap_circle_half_inside():
    """Circle center on polygon edge → roughly 50% overlap."""
    poly = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    overlap = _circle_polygon_overlap(0, 50, 20, poly)
    circle_area = np.pi * 20 * 20
    ratio = overlap / circle_area
    assert 0.3 < ratio < 0.7   # Roughly half


def test_overlap_non_negative():
    poly = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32)
    assert _circle_polygon_overlap(100, 100, 10, poly) >= 0.0


# ---------------------------------------------------------------------------
# compute_slot_polygons (calibration.py)
# ---------------------------------------------------------------------------

def _make_corners(w=1000, h=800):
    return [(0, 0), (w, 0), (w, h), (0, h)]


def test_compute_slot_polygons_counts():
    ref_slots, slot_data = compute_slot_polygons(_make_corners())
    assert len(ref_slots) == 5       # REF_L0…REF_L4
    assert len(slot_data) == 180     # 4 groups × 9 slots × 5 levels


def test_compute_slot_polygons_no_zz():
    """ZZ column must never appear in slot_data."""
    _, slot_data = compute_slot_polygons(_make_corners())
    for slot_id in slot_data:
        assert "ZZ" not in slot_id


def test_compute_slot_expected_levels():
    """Slot suffix must match expected_level."""
    _, slot_data = compute_slot_polygons(_make_corners())
    for slot_id, info in slot_data.items():
        suffix_level = int(slot_id.split("_")[1])
        assert info["expected_level"] == suffix_level


def test_compute_slot_groups():
    """All slots belong to groups A1-A4."""
    _, slot_data = compute_slot_polygons(_make_corners())
    for slot_id, info in slot_data.items():
        assert info["group"] in {"A1", "A2", "A3", "A4"}


def test_compute_ref_slots_have_four_corners():
    ref_slots, _ = compute_slot_polygons(_make_corners())
    for slot_id, info in ref_slots.items():
        assert len(info["coords"]) == 4


def test_compute_slot_data_have_four_corners():
    _, slot_data = compute_slot_polygons(_make_corners())
    for slot_id, info in slot_data.items():
        assert len(info["coords"]) == 4


def test_compute_ref_levels_0_to_4():
    ref_slots, _ = compute_slot_polygons(_make_corners())
    levels = {info["level"] for info in ref_slots.values()}
    assert levels == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_parse_A25_3,
        test_parse_A11_0,
        test_parse_A49_4,
        test_parse_A31_2,
        test_overlap_circle_fully_inside,
        test_overlap_circle_outside,
        test_overlap_circle_half_inside,
        test_overlap_non_negative,
        test_compute_slot_polygons_counts,
        test_compute_slot_polygons_no_zz,
        test_compute_slot_expected_levels,
        test_compute_slot_groups,
        test_compute_ref_slots_have_four_corners,
        test_compute_slot_data_have_four_corners,
        test_compute_ref_levels_0_to_4,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} grid tests passed.")
