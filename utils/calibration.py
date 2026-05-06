"""
Calibration module: white balance lock and grid corner mapping.

Two responsibilities:
  1. capture_white_balance_frame() — picamera2 AWB/AE lock on white card
  2. run_calibration(image)        — interactive 2-phase grid mapping
                                     Phase 1: click 4 outer corners
                                     Phase 2: drag individual lines to fine-tune
                                     → returns grid_pts dict for DB saving
                                     → caller POSTs to /settings/grid to persist

Grid layout (15 cols × 13 rows = 195 sample cells, from 16 v-lines × 14 h-lines):
  Row 0     : Reference row — 5 level groups of 3 cells each (L0…L4)
  Rows 1-13 : Sample area — 13 rows × 15 cols = 195 sample slots

Column mapping (same for all sample rows):
  Cols 1-3   → Level 0 block
  Cols 4-6   → Level 1 block
  Cols 7-9   → Level 2 block
  Cols 10-12 → Level 3 block
  Cols 13-15 → Level 4 block

Slot ID format: A{group}{slot}_{level}
  group  1-4  (A1=rows 1-3, A2=rows 4-6, A3=rows 7-9, A4=rows 10-12)
  slot   1-9  (within group: 3 rows × 3 cols per level block, row-major)
  level  0-4  (expected hydration level for that column zone)

grid_pts shape: (14, 16, 2) — 14 h-lines × 16 v-lines × (x, y)
  h-line 0     : top edge of reference row
  h-lines 1-13 : 13 sample row boundaries (h-line i = top of sample row i)
  h-line 13    : bottom edge of last sample row
  v-lines 0-15 : 15 column left/right boundaries (v-line i = left edge of col i+1,
                  v-line 15 = right edge of col 15)
"""

import cv2
import json
import logging
import os
import numpy as np
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

import tomllib
_cfg = tomllib.load(open(Path(__file__).parent.parent / "configs" / "config.toml", "rb"))
_cam = _cfg["camera"]
CAPTURE_RESOLUTION: tuple = tuple(_cam["capture_resolution"])
AWB_LOCK: bool            = bool(_cam["awb_lock"])
AE_LOCK: bool             = bool(_cam["ae_lock"])
CAMERA_ROTATE_180: bool   = bool(_cam["rotate_180"])

# Grid dimensions in lines (not cells).
# 16 v-lines → 15 column cells   (no ZZ dead zone)
# 14 h-lines → 13 sample rows + 1 reference row
_N_VLINES = 16
_N_HLINES = 14


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
        cfg = cam.create_still_configuration(main={"size": CAPTURE_RESOLUTION})
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

        tmp = "/tmp/awb_frame.jpg"
        cam.capture_file(tmp)
        cam.stop()
        cam.close()

        frame = cv2.imread(tmp)   # imread gives BGR — correct colors
        if frame is not None and CAMERA_ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

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
        cfg = cam.create_still_configuration(main={"size": CAPTURE_RESOLUTION})
        cam.configure(cfg)
        if locked_controls:
            cam.set_controls(locked_controls)
        cam.start()
        time.sleep(1)

        tmp = "/tmp/capture_frame.jpg"
        cam.capture_file(tmp)
        cam.stop()
        cam.close()

        frame = cv2.imread(tmp)   # imread gives BGR — correct colors
        if frame is not None and CAMERA_ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    except Exception as e:
        print(f"capture_frame error: {e}")
        return None


def capture_multi_snapshot(locked_controls=None, n=3, delay_ms=200,
                           exposure_variation=0.15):
    """
    Capture n consecutive frames for multi-snapshot consensus detection.

    The base ExposureTime is varied by ±exposure_variation across snapshots
    (snapshot 0 = normal, 1 = darker, 2 = brighter) to make real bottles
    robust across lighting conditions while reflections vary and disappear.

    Args:
        locked_controls:    dict from capture_white_balance_frame()
        n:                  number of snapshots (default 3)
        delay_ms:           milliseconds between shots
        exposure_variation: fractional ±variation of ExposureTime

    Returns:
        list of BGR numpy arrays (length n). May be shorter on error.
    """
    import time
    try:
        from picamera2 import Picamera2

        cam = Picamera2()
        cfg = cam.create_still_configuration(main={"size": CAPTURE_RESOLUTION})
        cam.configure(cfg)
        if locked_controls:
            cam.set_controls(locked_controls)
        cam.start()
        time.sleep(1.0)   # stabilise

        base_exposure = (locked_controls or {}).get("ExposureTime", None)
        frames = []
        variations = [0.0, -exposure_variation, +exposure_variation][:n]

        for i, var in enumerate(variations):
            if base_exposure and var != 0.0:
                cam.set_controls({"ExposureTime": int(base_exposure * (1 + var))})
                time.sleep(0.05)   # let the sensor settle briefly

            tmp = f"/tmp/snapshot_{i}.jpg"
            cam.capture_file(tmp)
            if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
                logger.error("[WORKER] snapshot_%d failed: file missing or 0 bytes at %s", i, tmp)
                continue
            logger.debug("[WORKER] snapshot_%d saved: %.1f KB", i, os.path.getsize(tmp) / 1024)
            frame = cv2.imread(tmp)
            if frame is None:
                logger.error("[WORKER] cv2.imread returned None for %s", tmp)
                continue
            if CAMERA_ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            frames.append(frame)

            if i < n - 1:
                time.sleep(delay_ms / 1000.0)

        cam.stop()
        cam.close()
        return frames

    except Exception as e:
        logger.error("capture_multi_snapshot error: %s", e)
        return []


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
        numpy array of shape (14, 17, 2), dtype float64
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
        grid_pts: numpy array shape (14, 17, 2)

    Returns:
        (reference_slots, slot_data)  — same schema as compute_slot_polygons()
    """
    def cell_quad(col, row):
        """4-corner polygon [TL, TR, BR, BL] for cell (col=0-15, row=0-11)."""
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

    # ---- Main grid (cell rows 1-12) — cell names from table/a.txt ----
    from utils.grid import load_grid_layout, parse_slot_id
    layout    = load_grid_layout()   # 12 rows × 16 cols (col 0 = ZZ)
    slot_data = {}

    for row_idx, row_cells in enumerate(layout):
        grid_row = row_idx + 1          # table row 0 → grid row 1
        for col_idx, slot_id in enumerate(row_cells):
            if not slot_id or slot_id == "ZZ":
                continue
            parsed = parse_slot_id(slot_id)
            slot_data[slot_id] = {
                "coords":         cell_quad(col_idx, grid_row),
                "expected_level": parsed["expected_level"],
                "group":          parsed["group"],
            }

    return reference_slots, slot_data


def compute_sample_roi(grid_pts):
    """
    Compute the bounding box of the sample area (rows 1-12 only, excluding reference row 0).

    Args:
        grid_pts: numpy array shape (14, 17, 2) — h-lines × v-lines × (x, y)

    Returns:
        [x1, y1, x2, y2] as ints — tight bounding box of the sample area.
        Use as the YOLO inference ROI so reference bottles are never seen.
    """
    grid_np = np.asarray(grid_pts)
    # Rows 1-12 span h-lines index 1 through 13 (14 h-lines total, indices 0-13)
    sample_pts = grid_np[1:14, :, :]   # shape (13, 17, 2)
    x1 = int(np.min(sample_pts[:, :, 0]))
    y1 = int(np.min(sample_pts[:, :, 1]))
    x2 = int(np.max(sample_pts[:, :, 0]))
    y2 = int(np.max(sample_pts[:, :, 1]))
    return [x1, y1, x2, y2]


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

    Press Enter in Phase 2 to finish. The caller is responsible for saving
    the returned grid_pts to the database via POST /settings/grid.

    Args:
        image: Pre-loaded BGR frame. If None, captures from camera.

    Returns:
        dict with keys {corners, grid_pts, calibration_date, ...}, or None if cancelled.
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
    # ------------------------------------------------------------------
    # Return grid data — caller saves to DB via POST /settings/grid
    # ------------------------------------------------------------------
    grid_data = {
        "calibration_date": datetime.now().strftime("%Y-%m-%d"),
        "corners":          corners,
        "grid_pts":         grid_pts.tolist(),
    }

    print(f"\nCalibration complete — POST grid_pts to /settings/grid to save to database.")
    return grid_data
