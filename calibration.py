"""
Calibration module: white balance lock and grid corner mapping.

Two responsibilities:
  1. capture_white_balance_frame() — picamera2 AWB/AE lock on white card
  2. run_calibration(image)        — interactive 2-phase grid mapping
                                     Phase 1: click 4 outer corners
                                     Phase 2: drag individual lines to fine-tune
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

# Grid dimensions (lines, not cells).
# The grid has 16 cell columns (0-15) and 14 cell rows (0-13),
# so it needs 17 vertical lines and 15 horizontal lines.
_N_VLINES = 17   # 17 vertical lines → 16 cell columns (col 0 = ZZ, cols 1-15 = data)
_N_HLINES = 15   # 15 horizontal lines → 14 cell rows (row 0 = ref, rows 1-12 = samples, row 13 = unused)


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
# Grid intersection point helpers
# ===========================================================================

def _corners_to_grid_pts(corners):
    """
    Compute the (_N_HLINES, _N_VLINES, 2) array of grid intersection points
    from the 4 outer corners using bilinear interpolation.

    Args:
        corners: [(x,y)×4] in order: top-left, top-right, bottom-right, bottom-left

    Returns:
        numpy array of shape (14, 16, 2), dtype float64
        grid_pts[row, col] = (x, y) of the intersection of horizontal line `row`
                             and vertical line `col` in image coordinates.
    """
    tl, tr, br, bl = [np.array(c, dtype=np.float64) for c in corners]
    grid_pts = np.zeros((_N_HLINES, _N_VLINES, 2), dtype=np.float64)
    for r in range(_N_HLINES):
        t_row = r / (_N_HLINES - 1)   # 0 / 14 .. 14 / 14
        for c in range(_N_VLINES):
            t_col = c / (_N_VLINES - 1)   # 0 / 16 .. 16 / 16
            top = tl + t_col * (tr - tl)
            bot = bl + t_col * (br - bl)
            grid_pts[r, c] = top + t_row * (bot - top)
    return grid_pts


def compute_slot_polygons_from_grid(grid_pts):
    """
    Compute all 195 slot polygons from a (_N_HLINES, _N_VLINES, 2) grid_pts
    array (e.g. after interactive line editing).

    This is the primary polygon-generation path used by run_calibration().
    compute_slot_polygons(corners) is kept for tests and calls this internally.

    Args:
        grid_pts: numpy array shape (14, 16, 2)

    Returns:
        (reference_slots, slot_data)  — same schema as compute_slot_polygons()
    """
    def cell_quad(col, row):
        """4-corner polygon [TL, TR, BR, BL] for cell (col=0-14, row=0-12)."""
        return [
            grid_pts[row,     col    ].tolist(),
            grid_pts[row,     col + 1].tolist(),
            grid_pts[row + 1, col + 1].tolist(),
            grid_pts[row + 1, col    ].tolist(),
        ]

    # ---- Reference row (cell row 0) — 5 groups of 3 cells each ----
    reference_slots = {}
    for level_idx in range(5):
        col_start = 1 + level_idx * 3   # cols 1,4,7,10,13
        reference_slots[f"REF_L{level_idx}"] = {
            "coords": [
                grid_pts[0, col_start    ].tolist(),   # TL
                grid_pts[0, col_start + 3].tolist(),   # TR
                grid_pts[1, col_start + 3].tolist(),   # BR
                grid_pts[1, col_start    ].tolist(),   # BL
            ],
            "level":   level_idx,
            "samples": 3,
        }

    # ---- Main grid (cell rows 1-12) — col 0 = ZZ, skip ----
    group_names = ["A1", "A2", "A3", "A4"]
    slot_data   = {}

    for group_idx, group in enumerate(group_names):
        group_num       = group_idx + 1
        group_row_start = 1 + group_idx * 3

        for within_row in range(3):
            grid_row        = group_row_start + within_row
            slot_row_offset = within_row * 3

            for level_idx in range(5):
                col_start = 1 + level_idx * 3

                for col_offset in range(3):
                    grid_col = col_start + col_offset
                    slot_num = slot_row_offset + col_offset + 1
                    slot_id  = f"A{group_num}{slot_num}_{level_idx}"

                    slot_data[slot_id] = {
                        "coords":         cell_quad(grid_col, grid_row),
                        "expected_level": level_idx,
                        "group":          group,
                    }

    return reference_slots, slot_data


def compute_slot_polygons(corners):
    """
    Compute all 195 slot polygons from the 4 outer grid corners.

    Kept for backwards compatibility and unit tests.
    Internally delegates to compute_slot_polygons_from_grid().

    Args:
        corners: [(x,y)×4] in order: top-left, top-right, bottom-right, bottom-left

    Returns:
        (reference_slots, slot_data)
          reference_slots: {slot_id: {'coords':[[x,y]×4], 'level':int, 'samples':3}}
          slot_data:       {slot_id: {'coords':[[x,y]×4], 'expected_level':int, 'group':str}}
    """
    return compute_slot_polygons_from_grid(_corners_to_grid_pts(corners))


# ===========================================================================
# Phase 2 — Interactive line editor
# ===========================================================================

def _run_line_editor(frame, initial_grid_pts):
    """
    Interactive grid line editor (Phase 2 of calibration).

    Shows the auto-computed grid overlay on the image. The user can hover near
    any horizontal or vertical line to highlight it, then click-and-drag to
    move the entire line. This corrects any misalignment that the 4-corner
    bilinear computation could not handle.

    Horizontal lines move up/down only.
    Vertical lines move left/right only.

    Keys:
        drag    — move highlighted line
        r       — reset to initial 4-corner computation
        q       — cancel calibration
        Enter   — confirm and proceed to save

    Args:
        frame:            BGR image (full resolution)
        initial_grid_pts: (14, 16, 2) float64 array from _corners_to_grid_pts()

    Returns:
        Adjusted (14, 16, 2) grid_pts, or None if user cancelled.
    """
    h, w = frame.shape[:2]
    scale  = min(1400 / w, 900 / h, 1.0)
    disp_w = int(w * scale)
    disp_h = int(h * scale)

    # Work in image coordinates; scale only for display
    grid_pts = initial_grid_pts.copy()

    # Hover / drag state
    state = {
        'hover_type': None,   # 'h' | 'v' | None
        'hover_idx':  -1,
        'sel_type':   None,
        'sel_idx':    -1,
        'dragging':   False,
        'drag_x':     0.0,
        'drag_y':     0.0,
    }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _seg_dist(px, py, x1, y1, x2, y2):
        """Minimum distance from point (px,py) to segment (x1,y1)-(x2,y2)."""
        dx, dy = x2 - x1, y2 - y1
        len_sq = dx * dx + dy * dy
        if len_sq == 0:
            return np.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
        return np.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

    def _nearest_line(ix, iy):
        """
        Find the nearest grid line to image point (ix, iy).
        Returns (line_type, line_idx, distance_in_image_px).
        """
        best = ('h', 0, float('inf'))

        # Horizontal lines: row r connects grid_pts[r, 0..15]
        for r in range(_N_HLINES):
            for c in range(_N_VLINES - 1):
                x1, y1 = grid_pts[r, c]
                x2, y2 = grid_pts[r, c + 1]
                d = _seg_dist(ix, iy, x1, y1, x2, y2)
                if d < best[2]:
                    best = ('h', r, d)

        # Vertical lines: col c connects grid_pts[0..13, c]
        for c in range(_N_VLINES):
            for r in range(_N_HLINES - 1):
                x1, y1 = grid_pts[r, c]
                x2, y2 = grid_pts[r + 1, c]
                d = _seg_dist(ix, iy, x1, y1, x2, y2)
                if d < best[2]:
                    best = ('v', c, d)

        return best

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------

    HOVER_THRESH_DISP = 15   # display pixels

    def on_mouse(event, dx, dy, flags, _param):
        ix = dx / scale
        iy = dy / scale
        thresh = HOVER_THRESH_DISP / scale   # convert to image pixels

        if event == cv2.EVENT_MOUSEMOVE:
            if state['dragging']:
                ddx = ix - state['drag_x']
                ddy = iy - state['drag_y']
                state['drag_x'] = ix
                state['drag_y'] = iy
                if state['sel_type'] == 'h':
                    grid_pts[state['sel_idx'], :, 1] += ddy
                else:
                    grid_pts[:, state['sel_idx'], 0] += ddx
            else:
                ltype, lidx, dist = _nearest_line(ix, iy)
                if dist < thresh:
                    state['hover_type'] = ltype
                    state['hover_idx']  = lidx
                else:
                    state['hover_type'] = None
                    state['hover_idx']  = -1

        elif event == cv2.EVENT_LBUTTONDOWN:
            ltype, lidx, dist = _nearest_line(ix, iy)
            if dist < thresh:
                state['sel_type'] = ltype
                state['sel_idx']  = lidx
                state['dragging'] = True
                state['drag_x']   = ix
                state['drag_y']   = iy

        elif event == cv2.EVENT_LBUTTONUP:
            state['dragging'] = False

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    # Line colours: normal / hover / selected
    _C_H_NORM = (0,  210,  0)
    _C_H_HOV  = (0,  220, 255)
    _C_H_SEL  = (0,  255, 255)
    _C_V_NORM = (0,  210,  0)
    _C_V_HOV  = (255, 200, 0)
    _C_V_SEL  = (255, 255, 0)

    def _line_style(ltype, idx):
        is_sel = (state['sel_type'] == ltype and state['sel_idx'] == idx)
        is_hov = (state['hover_type'] == ltype and state['hover_idx'] == idx)
        if ltype == 'h':
            color = _C_H_SEL if is_sel else (_C_H_HOV if is_hov else _C_H_NORM)
        else:
            color = _C_V_SEL if is_sel else (_C_V_HOV if is_hov else _C_V_NORM)
        thick = 3 if (is_sel or is_hov) else 1
        return color, thick

    def draw():
        vis = frame.copy()

        for r in range(_N_HLINES):
            color, thick = _line_style('h', r)
            for c in range(_N_VLINES - 1):
                p1 = tuple(grid_pts[r, c    ].astype(int))
                p2 = tuple(grid_pts[r, c + 1].astype(int))
                cv2.line(vis, p1, p2, color, thick)

        for c in range(_N_VLINES):
            color, thick = _line_style('v', c)
            for r in range(_N_HLINES - 1):
                p1 = tuple(grid_pts[r,     c].astype(int))
                p2 = tuple(grid_pts[r + 1, c].astype(int))
                cv2.line(vis, p1, p2, color, thick)

        # Row / col index labels on the left and top edges
        for r in range(_N_HLINES):
            px, py = grid_pts[r, 0].astype(int)
            cv2.putText(vis, str(r), (max(0, px - 40), py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        for c in range(_N_VLINES):
            px, py = grid_pts[0, c].astype(int)
            cv2.putText(vis, str(c), (px, max(0, py - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Status bar (drawn twice for outline effect)
        hint = "Drag lines to adjust  |  r = reset  |  q = cancel  |  Enter = save"
        for fg, th in [((0, 0, 0), 4), ((255, 255, 255), 1)]:
            cv2.putText(vis, hint, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, fg, th)

        if state['sel_type']:
            axis = "Row" if state['sel_type'] == 'h' else "Col"
            label = f"Selected: {axis} {state['sel_idx']}"
            for fg, th in [((0, 0, 0), 4), ((0, 255, 255), 1)]:
                cv2.putText(vis, label, (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, fg, th)
        elif state['hover_type']:
            axis = "Row" if state['hover_type'] == 'h' else "Col"
            label = f"Hover: {axis} {state['hover_idx']}  (click to select)"
            for fg, th in [((0, 0, 0), 3), ((180, 220, 255), 1)]:
                cv2.putText(vis, label, (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, fg, th)

        disp = cv2.resize(vis, (disp_w, disp_h))
        cv2.imshow("Grid Editor — Phase 2", disp)

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    cv2.namedWindow("Grid Editor — Phase 2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Grid Editor — Phase 2", disp_w, disp_h)
    cv2.setMouseCallback("Grid Editor — Phase 2", on_mouse)

    print("\n=== PHASE 2 — GRID LINE EDITOR ===")
    print("  Hover near a line → highlights it")
    print("  Click + drag      → moves the entire line")
    print("  Horizontal lines (cyan when selected) move up/down only")
    print("  Vertical lines   (yellow when selected) move left/right only")
    print("  r = reset   q = cancel   Enter = confirm")

    while True:
        draw()
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            grid_pts[:] = initial_grid_pts
            state['sel_type']   = None
            state['hover_type'] = None
            print("  Grid reset to 4-corner computed values.")

        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("  Calibration cancelled.")
            return None

        elif key in (13, 10):   # Enter
            break

    cv2.destroyAllWindows()
    return grid_pts


# ===========================================================================
# Interactive calibration — entry point
# ===========================================================================

def run_calibration(image=None):
    """
    Two-phase interactive grid calibration.

    Phase 1 — 4-corner selection:
        Click the 4 outer corners of the grid (TL → TR → BR → BL).
        The system auto-computes the full 16×14 grid from those corners.

    Phase 2 — Line fine-tuning:
        The computed grid is shown as an overlay.
        Hover near any line and drag it to the correct position.
        Useful when the camera angle or lens distortion causes the bilinear
        auto-computation to misplace individual rows or columns.

    Press Enter in Phase 2 to save grid_config.json.

    Args:
        image: Pre-loaded BGR frame. If None, captures from camera.

    Returns:
        Path to saved grid_config.json, or None if cancelled.
    """
    if image is None:
        print("Capturing calibration image from camera...")
        frame, _ = capture_white_balance_frame()
        if frame is None:
            print("Camera not available. Pass an image file with --calib-image.")
            return None
    else:
        frame = image

    # ------------------------------------------------------------------
    # Phase 1 — 4-corner click
    # ------------------------------------------------------------------
    print("\n=== PHASE 1 — CORNER SELECTION ===")
    print("Click the 4 outer corners of the grid in order:")
    print("  1) Top-Left   2) Top-Right   3) Bottom-Right   4) Bottom-Left")
    print("Keys: r = reset   q = cancel   Enter = confirm (after 4 clicks)")

    h, w  = frame.shape[:2]
    scale = min(1400 / w, 900 / h, 1.0)
    disp_w, disp_h = int(w * scale), int(h * scale)

    corners = []

    def on_corner_click(event, dx, dy, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((int(dx / scale), int(dy / scale)))
            print(f"  Corner {len(corners)}: ({corners[-1][0]}, {corners[-1][1]})")

    cv2.namedWindow("Calibration — Phase 1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration — Phase 1", disp_w, disp_h)
    cv2.setMouseCallback("Calibration — Phase 1", on_corner_click)

    while True:
        vis = frame.copy()
        labels = ["1 TL", "2 TR", "3 BR", "4 BL"]
        for i, (cx, cy) in enumerate(corners):
            cv2.circle(vis, (cx, cy), 14, (0, 255, 0), -1)
            cv2.putText(vis, labels[i], (cx + 16, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)

        if len(corners) == 4:
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(vis, [pts.reshape(-1, 1, 2)],
                          isClosed=True, color=(0, 255, 255), thickness=4)
            hint = "Press Enter to continue to Phase 2"
        else:
            hint = f"Click corner {len(corners) + 1}  |  r = reset  |  q = cancel"

        for fg, th in [((0, 0, 0), 4), ((255, 255, 255), 1)]:
            cv2.putText(vis, hint, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, fg, th)

        cv2.imshow("Calibration — Phase 1", cv2.resize(vis, (disp_w, disp_h)))

        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            corners.clear()
            print("  Corners reset.")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("  Calibration cancelled.")
            return None
        elif key in (13, 10) and len(corners) == 4:
            break

    cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Phase 2 — Line fine-tuning
    # ------------------------------------------------------------------
    initial_grid_pts = _corners_to_grid_pts(corners)
    grid_pts = _run_line_editor(frame, initial_grid_pts)

    if grid_pts is None:
        return None

    # ------------------------------------------------------------------
    # Compute polygons and save
    # ------------------------------------------------------------------
    print("\nComputing slot polygons...")
    reference_slots, slot_data = compute_slot_polygons_from_grid(grid_pts)

    config_data = {
        "system_metadata": {
            "project_name":     "Urine Color Analysis",
            "grid_dimensions":  "16x14 lines",
            "calibration_date": datetime.now().strftime("%Y-%m-%d"),
            "corners":          corners,
            "grid_pts":         grid_pts.tolist(),
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
