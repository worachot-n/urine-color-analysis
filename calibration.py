"""
Calibration module: white balance lock and grid corner mapping.

Two responsibilities:
  1. capture_white_balance_frame() — picamera2 AWB/AE lock on white card
  2. run_calibration(image)        — interactive 4-corner grid mapping
                                     → computes all 195 slot polygons
                                     → saves grid_config.json

Grid layout (15×13 cells from 16×14 lines):
  Row 0  : Reference row — 5 groups of 3 cells each (REF_L0…REF_L4)
  Rows 1-12: Sample area — col 0 is ZZ dead zone, cols 1-15 are sample slots
  Row 13 : Unused (bottom border)

Column mapping per sample row:
  Col 0        → ZZ (excluded, never emitted to slot_data)
  Cols 1-3     → Level 0 block
  Cols 4-6     → Level 1 block
  Cols 7-9     → Level 2 block
  Cols 10-12   → Level 3 block
  Cols 13-15   → Level 4 block

Slot ID: A{group_num}{slot_num}_{level_idx}
  group_num  1-4   (A1=rows 1-3, A2=rows 4-6, A3=rows 7-9, A4=rows 10-12)
  slot_num   1-9   (within group; 3 rows × 3 cols-per-level-block)
  level_idx  0-4
"""

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from config import (
    GRID_CONFIG_FILE,
    CAPTURE_RESOLUTION,
    AWB_LOCK,
    AE_LOCK,
)


# ===========================================================================
# Camera capture
# ===========================================================================

def capture_white_balance_frame():
    """
    Capture a frame with picamera2 and lock AWB and AE gains.

    Place a white card in the scene before calling this. The function:
      1. Opens the camera and lets AWB/AE settle for 2 s
      2. Reads the current gains from metadata
      3. Locks those gains permanently on the camera object
      4. Captures a final locked frame

    Returns:
        (frame, controls)
          frame:    BGR numpy array, or None on failure
          controls: dict of locked camera controls (pass to capture_frame())
    """
    try:
        from picamera2 import Picamera2
        import time

        cam = Picamera2()
        cfg = cam.create_still_configuration(
            main={"size": CAPTURE_RESOLUTION, "format": "BGR888"}
        )
        cam.configure(cfg)
        cam.start()
        time.sleep(2)  # Let AWB/AE settle

        metadata = cam.capture_metadata()
        controls = {}

        if AWB_LOCK:
            controls["AwbEnable"]   = False
            controls["ColourGains"] = metadata.get("ColourGains", (1.0, 1.0))

        if AE_LOCK:
            controls["AeEnable"]     = False
            controls["ExposureTime"] = metadata.get("ExposureTime", 10000)
            controls["AnalogueGain"] = metadata.get("AnalogueGain", 1.0)

        cam.set_controls(controls)
        time.sleep(0.5)

        frame = cam.capture_array()
        cam.stop()
        cam.close()

        return frame, controls

    except Exception as e:
        print(f"capture_white_balance_frame error: {e}")
        return None, {}


def capture_frame(locked_controls=None):
    """
    Capture a single frame, optionally applying pre-locked AWB/AE controls.

    Args:
        locked_controls: dict from capture_white_balance_frame() (may be empty)

    Returns:
        BGR numpy array, or None on failure.
    """
    try:
        from picamera2 import Picamera2
        import time

        cam = Picamera2()
        cfg = cam.create_still_configuration(
            main={"size": CAPTURE_RESOLUTION, "format": "BGR888"}
        )
        cam.configure(cfg)
        if locked_controls:
            cam.set_controls(locked_controls)
        cam.start()
        time.sleep(1)
        frame = cam.capture_array()
        cam.stop()
        cam.close()
        return frame

    except Exception as e:
        print(f"capture_frame error: {e}")
        return None


# ===========================================================================
# Slot polygon computation
# ===========================================================================

def compute_slot_polygons(corners):
    """
    Compute all 195 slot polygons from the 4 outer grid corners using
    bilinear interpolation (perspective-correct for flat boards).

    Args:
        corners: [(x,y), (x,y), (x,y), (x,y)] in order:
                 top-left, top-right, bottom-right, bottom-left

    Returns:
        (reference_slots, slot_data)
          reference_slots: {slot_id: {'coords':[[x,y]×4], 'level':int, 'samples':3}}
          slot_data:       {slot_id: {'coords':[[x,y]×4], 'expected_level':int, 'group':str}}
    """
    tl, tr, br, bl = [np.array(c, dtype=np.float64) for c in corners]

    n_cols = 15   # 16 lines → 15 columns of cells
    n_rows = 13   # 14 lines → 13 rows of cells (row 13 unused in protocol)

    def lerp(p1, p2, t):
        return p1 + t * (p2 - p1)

    def cell_corners(col, row):
        """4-corner polygon [TL, TR, BR, BL] for cell (col, row)."""
        t0 = col       / n_cols
        t1 = (col + 1) / n_cols
        r0 = row       / n_rows
        r1 = (row + 1) / n_rows

        top_l = lerp(lerp(tl, tr, t0), lerp(bl, br, t0), r0)
        top_r = lerp(lerp(tl, tr, t1), lerp(bl, br, t1), r0)
        bot_r = lerp(lerp(tl, tr, t1), lerp(bl, br, t1), r1)
        bot_l = lerp(lerp(tl, tr, t0), lerp(bl, br, t0), r1)

        return [top_l.tolist(), top_r.tolist(), bot_r.tolist(), bot_l.tolist()]

    # ---- Reference row (row 0) — 5 groups of 3 cells ----
    reference_slots = {}
    for level_idx in range(5):
        slot_id   = f"REF_L{level_idx}"
        col_start = 1 + level_idx * 3

        left_c  = cell_corners(col_start,     0)
        right_c = cell_corners(col_start + 2, 0)

        coords = [
            left_c[0],   # TL of leftmost cell
            right_c[1],  # TR of rightmost cell
            right_c[2],  # BR of rightmost cell
            left_c[3],   # BL of leftmost cell
        ]
        reference_slots[slot_id] = {
            "coords":  coords,
            "level":   level_idx,
            "samples": 3,
        }

    # ---- Main grid (rows 1-12) ----
    # col 0 = ZZ → skipped (never added to slot_data)
    group_names = ["A1", "A2", "A3", "A4"]
    slot_data   = {}

    for group_idx, group in enumerate(group_names):
        group_num       = group_idx + 1
        group_row_start = 1 + group_idx * 3   # rows: 1, 4, 7, 10

        for within_row in range(3):
            grid_row         = group_row_start + within_row
            slot_row_offset  = within_row * 3    # 0, 3, 6

            for level_idx in range(5):
                col_start = 1 + level_idx * 3

                for col_offset in range(3):
                    grid_col = col_start + col_offset
                    slot_num = slot_row_offset + col_offset + 1   # 1-9
                    slot_id  = f"A{group_num}{slot_num}_{level_idx}"

                    slot_data[slot_id] = {
                        "coords":         cell_corners(grid_col, grid_row),
                        "expected_level": level_idx,
                        "group":          group,
                    }

    return reference_slots, slot_data


# ===========================================================================
# Interactive calibration
# ===========================================================================

def run_calibration(image=None):
    """
    Interactive grid calibration.

    Displays the image and asks the user to click the 4 outer corners of the
    grid (top-left, top-right, bottom-right, bottom-left). Then computes all
    slot polygons and saves grid_config.json.

    Args:
        image: Pre-loaded BGR frame. If None, captures from camera.

    Returns:
        Path to saved grid_config.json, or None if cancelled.
    """
    if image is None:
        print("Capturing calibration image from camera...")
        frame, _ = capture_white_balance_frame()
        if frame is None:
            print("Camera not available. Pass an image file with --image.")
            return None
    else:
        frame = image

    print("\n=== CALIBRATION MODE ===")
    print("Click 4 grid corners in order:")
    print("  1) Top-Left   2) Top-Right")
    print("  3) Bottom-Right   4) Bottom-Left")
    print("Keys: 'r' = reset  'q' = quit  Enter = confirm (after 4 clicks)")

    corners = []
    h, w    = frame.shape[:2]
    scale   = min(1400 / w, 900 / h, 1.0)
    disp_w  = int(w * scale)
    disp_h  = int(h * scale)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            img_x = int(x / scale)
            img_y = int(y / scale)
            corners.append((img_x, img_y))
            print(f"  Corner {len(corners)}: ({img_x}, {img_y})")

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", disp_w, disp_h)
    cv2.setMouseCallback("Calibration", on_mouse)

    while True:
        vis = frame.copy()

        for i, (cx, cy) in enumerate(corners):
            cv2.circle(vis, (cx, cy), 12, (0, 255, 0), -1)
            cv2.putText(vis, str(i + 1), (cx + 14, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)

        if len(corners) == 4:
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(vis, [pts.reshape(-1, 1, 2)],
                          isClosed=True, color=(0, 255, 255), thickness=4)
            cv2.putText(vis, "Press ENTER to save, 'r' to reset",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        disp = cv2.resize(vis, (disp_w, disp_h))
        cv2.imshow("Calibration", disp)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            corners.clear()
            print("Corners reset.")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("Calibration cancelled.")
            return None
        elif key in (13, 10) and len(corners) == 4:   # Enter
            break

    cv2.destroyAllWindows()

    print("\nComputing slot polygons...")
    reference_slots, slot_data = compute_slot_polygons(corners)

    config_data = {
        "system_metadata": {
            "project_name":    "Urine Color Analysis",
            "grid_dimensions": "16x14 lines",
            "calibration_date": datetime.now().strftime("%Y-%m-%d"),
            "corners": corners,
        },
        "reference_row": {
            "description": "Top row for dynamic color calibration (3 bottles per level)",
            "slots": reference_slots,
        },
        "main_grid": {
            "description": "Processing slots for Groups A1-A4 across Color Levels 0-4",
            "slot_data": slot_data,
        },
    }

    out_path = Path(GRID_CONFIG_FILE)
    with open(out_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"\nCalibration saved: {out_path}")
    print(f"  Reference slots : {len(reference_slots)}")
    print(f"  Sample slots    : {len(slot_data)}")

    return out_path
