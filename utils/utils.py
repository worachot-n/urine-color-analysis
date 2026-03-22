"""
Utilities: structured logging and annotated image saving.

Every analysis cycle calls save_annotated_image() to write a timestamped
JPEG to logs/ containing:
  - Grid slot polygon outlines (green = sample, yellow = reference)
  - Detected red circles (color = OK / uncertain / error)
  - Text label per bottle: Slot ID + detected color level

Log files:
  logs/YYYY-MM-DD_HH-MM-SS.jpg   — annotated frame per analysis
  logs/system_YYYY-MM-DD.log     — text log for the day
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from configs.config import LOG_DIR, IMG_DIR, DEBUG_MODE


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def setup_logger(name="urine_analyzer"):
    """
    Configure the root logger with a console handler and a daily file handler.

    Attaching handlers to the root logger ensures that ALL module loggers
    (network, web_server, calibration, etc.) automatically inherit them and
    their logger.info() calls reach the terminal.

    Returns:
        logging.Logger instance named `name`
    """
    log_dir  = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"system_{datetime.now().strftime('%Y-%m-%d')}.log"

    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))

        root.addHandler(ch)
        root.addHandler(fh)

    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Annotated image saving
# ---------------------------------------------------------------------------

def save_annotated_image(frame, slot_assignments, grid_cfg,
                         unassigned_circles=None, rejected_slots=None, timestamp=None):
    """
    Save an annotated copy of the frame to logs/.

    Overlays drawn:
      - All slot polygon outlines (sample = green, reference = yellow)
      - Detected bottle circle (green = OK, orange = uncertain, red = error)
      - Label above circle: "{slot_id} L{level}"

    Args:
        frame:            Original BGR image (numpy array)
        slot_assignments: dict {slot_id: {
                              'cx': int, 'cy': int, 'radius': int,
                              'level': int | None,
                              'confident': bool,
                              'error': bool
                          }}
        grid_cfg:         GridConfig instance
        timestamp:        datetime (defaults to now)

    Returns:
        pathlib.Path to saved JPEG
    """
    if timestamp is None:
        timestamp = datetime.now()

    log_dir = Path(IMG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    vis = frame.copy()

    # -- Draw all slot polygons --
    for slot_id, info in grid_cfg.slot_data.items():
        pts = info['coords'].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 180, 0), thickness=1)

    for slot_id, info in grid_cfg.reference_slots.items():
        pts = info['coords'].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 200, 200), thickness=1)

    # -- Draw reference bottle sampling positions (gold circles + label) --
    ref_positions = grid_cfg.get_reference_positions()
    for level, positions in ref_positions.items():
        for cx, cy, r in positions:
            gold = (0, 200, 255)
            cv2.circle(vis, (cx, cy), r, gold, 2)
            cv2.circle(vis, (cx, cy), 4, gold, -1)
            label = f"REF_L{level}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tx = cx - tw // 2
            ty = cy - r - 5
            cv2.rectangle(vis, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
            cv2.putText(vis, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, gold, 1, cv2.LINE_AA)

    # -- Draw detected bottles --
    for slot_id, data in slot_assignments.items():
        cx     = data['cx']
        cy     = data['cy']
        radius = data['radius']
        level  = data.get('level')
        confident = data.get('confident', True)
        has_error = data.get('error', False)
        error_type = data.get("error_type")   # 'mismatch' | 'duplicate' | None
        expected   = data.get("expected_level")
        delta_e    = data.get("delta_e")

        if has_error or error_type == "duplicate":
            color = (0, 0, 255)       # red
        elif not confident:
            color = (0, 140, 255)     # orange
        else:
            color = (0, 220, 0)       # green

        # Draw YOLO bounding box if dimensions available, else circle fallback
        box_w = data.get('w')
        box_h = data.get('h')
        if box_w and box_h:
            bx1 = cx - box_w // 2
            by1 = cy - box_h // 2
            bx2 = cx + box_w // 2
            by2 = cy + box_h // 2
            thickness = 4 if error_type == "duplicate" else 2
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, thickness)
            label_top = by1
        else:
            cv2.circle(vis, (cx, cy), radius, color, 3)
            label_top = cy - radius
        cv2.circle(vis, (cx, cy), 4, color, -1)   # center dot always drawn

        # Label: [Slot ID] | [L#] | [de: value]
        if error_type == "duplicate":
            label = f"{slot_id} | DUP!"
        elif error_type == "mismatch" and level is not None:
            de_val = f"{delta_e:.1f}" if delta_e is not None else "?"
            label = f"{slot_id} | L{level} | de:{de_val} | ERR:MISMATCH(ExpL{expected})"
        elif level is not None:
            de_val = f"{delta_e:.1f}" if delta_e is not None else "?"
            label = f"{slot_id} | L{level} | de:{de_val}"
        else:
            label = slot_id

        # Dark background behind text for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tx = cx - tw // 2
        ty = label_top - 8
        cv2.rectangle(vis,
                      (tx - 3,      ty - th - 3),
                      (tx + tw + 3, ty + 3),
                      (0, 0, 0), -1)
        cv2.putText(vis, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # -- Draw unassigned HoughCircle detections (gray — detected outside any slot) --
    for cx, cy, r in (unassigned_circles or []):
        gray = (160, 160, 160)
        cv2.circle(vis, (cx, cy), r, gray, 1)
        cv2.circle(vis, (cx, cy), 3, gray, -1)
        cv2.putText(vis, "?", (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, gray, 1, cv2.LINE_AA)

    # -- Draw ghost-rejected slots (dark-red ✕ — red ring found but center is white/glare) --
    for cx, cy, r, slot_id in (rejected_slots or []):
        ghost = (0, 0, 180)   # dark red
        arm = max(8, r // 2)
        cv2.line(vis, (cx - arm, cy - arm), (cx + arm, cy + arm), ghost, 2)
        cv2.line(vis, (cx + arm, cy - arm), (cx - arm, cy + arm), ghost, 2)
        (tw, _th), _ = cv2.getTextSize(slot_id, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(vis, slot_id, (cx - tw // 2, cy - arm - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, ghost, 1, cv2.LINE_AA)

    filename = timestamp.strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
    out_path = log_dir / filename
    cv2.imwrite(str(out_path), vis)

    return out_path
