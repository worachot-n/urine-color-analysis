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

from config import LOG_DIR, DEBUG_MODE


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def setup_logger(name="urine_analyzer"):
    """
    Configure a logger that writes to both a daily log file and the console.

    Returns:
        logging.Logger instance
    """
    log_dir  = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"system_{datetime.now().strftime('%Y-%m-%d')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Annotated image saving
# ---------------------------------------------------------------------------

def save_annotated_image(frame, slot_assignments, grid_cfg, timestamp=None):
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

    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    vis = frame.copy()

    # -- Draw all slot polygons --
    for slot_id, info in grid_cfg.slot_data.items():
        pts = info['coords'].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 180, 0), thickness=1)

    for slot_id, info in grid_cfg.reference_slots.items():
        pts = info['coords'].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 200, 200), thickness=1)

    # -- Draw detected bottles --
    for slot_id, data in slot_assignments.items():
        cx     = data['cx']
        cy     = data['cy']
        radius = data['radius']
        level  = data.get('level')
        confident = data.get('confident', True)
        has_error = data.get('error', False)

        if has_error:
            color = (0, 0, 255)       # red
        elif not confident:
            color = (0, 140, 255)     # orange
        else:
            color = (0, 220, 0)       # green

        cv2.circle(vis, (cx, cy), radius, color, 3)
        cv2.circle(vis, (cx, cy), 4,      color, -1)

        label = slot_id
        if level is not None:
            label += f" L{level}"

        # Dark background behind text for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tx = cx - tw // 2
        ty = cy - radius - 8
        cv2.rectangle(vis,
                      (tx - 3,      ty - th - 3),
                      (tx + tw + 3, ty + 3),
                      (0, 0, 0), -1)
        cv2.putText(vis, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    filename = timestamp.strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
    out_path = log_dir / filename
    cv2.imwrite(str(out_path), vis)

    return out_path
