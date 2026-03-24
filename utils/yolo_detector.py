"""
YOLO26s + OpenVINO bottle detector with multi-snapshot consensus.

Single-class model:
    Class 0: bottle — all bottles (reference and sample alike)

Reference vs sample classification is done by grid position at inference time:
    - Bottles whose center falls within a reference slot (row 0) → ref_bottle
    - Bottles whose center falls within a main-grid slot (rows 1-12) → sample_bottle

Public API:
    YoloBottleDetector(model_path)
        .detect_once(frame)                       → list of [cx, cy, w, h, conf, cls]
        .consensus_filter(detections_list)        → list of confirmed [cx, cy, w, h, conf, cls]
        .geometric_validate_ref(boxes, grid)      → {level: [(cx, cy, r), ...]}
        .geometric_validate(boxes, grid)          → ({slot_id: {...}}, duplicate_slots)
        .detect_multi(frames, grid)               → (ref_positions, sample_hits, duplicate_slots)
        .write_result_json(assignments, counts, ts) → Path
"""

import json
from collections import Counter
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from configs.config import (
    YOLO_IMGSZ, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, YOLO_AUGMENT,
    YOLO_CONSENSUS_MIN, YOLO_CONSENSUS_IOU, YOLO_SLOT_MAX_DIST,
    YOLO_CLAHE_CLIP, YOLO_CLAHE_TILE, YOLO_ROI_PADDING,
    SAMPLE_ROI_TOP, SAMPLE_ROI_BOTTOM, SAMPLE_ROI_LEFT, SAMPLE_ROI_RIGHT,
    IMG_DIR,
)

class YoloBottleDetector:
    """
    Wraps a YOLOv8 OpenVINO model with CLAHE preprocessing,
    multi-snapshot consensus, and position-based ref/sample slot validation.
    """

    def __init__(self, model_path: str):
        from ultralytics import YOLO
        self.model = YOLO(model_path, task="detect")
        self._clahe = cv2.createCLAHE(
            clipLimit=YOLO_CLAHE_CLIP,
            tileGridSize=(YOLO_CLAHE_TILE, YOLO_CLAHE_TILE),
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE on L channel to normalise lighting for inference."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # Single-frame inference
    # ------------------------------------------------------------------

    def detect_once(self, frame: np.ndarray, roi: tuple = None) -> list:
        """
        Run YOLO26s on one frame.

        Args:
            frame: full-resolution BGR frame
            roi:   optional (x1, y1, x2, y2) crop region in pixel coords.
                   If provided, the frame is cropped before inference and all
                   returned box coordinates are offset back to full-frame space.

        Returns:
            list of [cx, cy, w, h, conf, cls] in original (full-frame) pixel coords
            cls: always 0 (single-class model — "bottle")
        """
        x_off, y_off = 0, 0
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            frame = frame[ry1:ry2, rx1:rx2]
            x_off, y_off = rx1, ry1

        enhanced = self._enhance(frame)

        results = self.model(
            enhanced,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            augment=YOLO_AUGMENT,
            verbose=False,
        )

        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx   = (x1 + x2) / 2 + x_off
                cy   = (y1 + y2) / 2 + y_off
                w    = x2 - x1
                h    = y2 - y1
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                boxes.append([cx, cy, w, h, conf, cls])
        return boxes

    # ------------------------------------------------------------------
    # Consensus filter
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(b1, b2) -> float:
        """IoU between two [cx, cy, w, h, ...] boxes."""
        ax1 = b1[0] - b1[2] / 2;  ax2 = b1[0] + b1[2] / 2
        ay1 = b1[1] - b1[3] / 2;  ay2 = b1[1] + b1[3] / 2
        bx1 = b2[0] - b2[2] / 2;  bx2 = b2[0] + b2[2] / 2
        by1 = b2[1] - b2[3] / 2;  by2 = b2[1] + b2[3] / 2

        ix1 = max(ax1, bx1);  ix2 = min(ax2, bx2)
        iy1 = max(ay1, by1);  iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (b1[2] * b1[3]) + (b2[2] * b2[3]) - inter
        return inter / union if union > 0 else 0.0

    def consensus_filter(self, detections_list: list) -> list:
        """
        Keep only detections that appear in at least YOLO_CONSENSUS_MIN snapshots.

        Args:
            detections_list: list of per-frame box lists
                             [ [[cx,cy,w,h,conf,cls], ...], [...], [...] ]

        Returns:
            list of confirmed [cx, cy, w, h, conf, cls] (averaged coords, modal cls)
        """
        if not detections_list:
            return []

        n_frames = len(detections_list)
        used = [set() for _ in range(n_frames)]
        confirmed = []

        for fi, frame_boxes in enumerate(detections_list):
            for bi, anchor in enumerate(frame_boxes):
                if bi in used[fi]:
                    continue                          # already claimed by an earlier anchor

                group = [anchor]
                used[fi].add(bi)

                for other_fi, other_boxes in enumerate(detections_list):
                    if other_fi == fi:
                        continue
                    best_j, best_iou = None, YOLO_CONSENSUS_IOU - 1e-9
                    for j, box in enumerate(other_boxes):
                        if j in used[other_fi]:
                            continue
                        iou = self._iou(anchor, box)
                        if iou > best_iou:
                            best_iou = iou
                            best_j = j
                    if best_j is not None:
                        group.append(other_boxes[best_j])
                        used[other_fi].add(best_j)

                if len(group) >= YOLO_CONSENSUS_MIN:
                    cx   = float(np.mean([b[0] for b in group]))
                    cy   = float(np.mean([b[1] for b in group]))
                    w    = float(np.mean([b[2] for b in group]))
                    h    = float(np.mean([b[3] for b in group]))
                    conf = float(np.mean([b[4] for b in group]))
                    cls  = Counter(int(b[5]) for b in group).most_common(1)[0][0]
                    confirmed.append([cx, cy, w, h, conf, cls])

        return confirmed

    # ------------------------------------------------------------------
    # Reference bottle geometric validation
    # ------------------------------------------------------------------

    def geometric_validate_ref(self, confirmed_boxes: list, grid_cfg) -> dict:
        """
        Map detected bottles to reference slots (row 0) by proximity, grouped by level.

        Each REF_Lx slot covers 3 physical bottles. The function assigns each
        detected bottle whose center falls near a reference slot to that level.
        Reference vs sample distinction is purely positional — no class filter.

        Args:
            confirmed_boxes: output of consensus_filter()
            grid_cfg:        GridConfig instance

        Returns:
            dict {level (int): [(cx, cy, radius), ...]}
            Compatible with build_reference_baseline() input format.
        """
        if not confirmed_boxes:
            return {}

        # Build reference slot centers {slot_id: (scx, scy, r, level)}
        ref_centers = {}
        for slot_id, info in grid_cfg.reference_slots.items():
            coords = info['coords']
            scx = float(np.mean(coords[:, 0]))
            scy = float(np.mean(coords[:, 1]))
            r = min(
                (float(np.max(coords[:, 0])) - float(np.min(coords[:, 0]))) / 2,
                (float(np.max(coords[:, 1])) - float(np.min(coords[:, 1]))) / 2,
            )
            ref_centers[slot_id] = (scx, scy, r, info['level'])

        # Build slot → list of candidate boxes
        slot_candidates: dict = {}
        for box in confirmed_boxes:
            bcx, bcy = box[0], box[1]
            best_slot = None
            best_dist = float('inf')
            for slot_id, (scx, scy, r, _level) in ref_centers.items():
                dist = float(np.hypot(bcx - scx, bcy - scy))
                if dist < YOLO_SLOT_MAX_DIST * r * 2 and dist < best_dist:
                    best_dist = dist
                    best_slot = slot_id
            if best_slot is not None:
                slot_candidates.setdefault(best_slot, []).append(box)

        # Keep one box per ref slot (highest confidence); group by level
        level_positions: dict = {}
        for slot_id, candidates in slot_candidates.items():
            best_box = max(candidates, key=lambda b: b[4])
            bcx, bcy = best_box[0], best_box[1]
            br = max(1, min(int(best_box[2]), int(best_box[3])) // 2)
            level = ref_centers[slot_id][3]
            level_positions.setdefault(level, []).append((int(bcx), int(bcy), br))

        return level_positions

    # ------------------------------------------------------------------
    # Sample bottle geometric validation
    # ------------------------------------------------------------------

    def geometric_validate(self, confirmed_boxes: list, grid_cfg) -> tuple:
        """
        Map detected bottles to main-grid slots (rows 1-12) by proximity.

        When multiple boxes compete for one slot, keep the highest-confidence
        box and record the slot in duplicate_slots.
        Reference vs sample distinction is purely positional — no class filter.

        Args:
            confirmed_boxes: output of consensus_filter()
            grid_cfg:        GridConfig instance

        Returns:
            (assigned, duplicate_slots)
              assigned:        dict {slot_id: {'cx','cy','w','h','conf'}}
              duplicate_slots: set of slot_ids that had more than one box competing
        """
        slot_centers = {}
        for slot_id, info in grid_cfg.slot_data.items():
            coords = info['coords']
            scx = float(np.mean(coords[:, 0]))
            scy = float(np.mean(coords[:, 1]))
            r = min(
                (float(np.max(coords[:, 0])) - float(np.min(coords[:, 0]))) / 2,
                (float(np.max(coords[:, 1])) - float(np.min(coords[:, 1]))) / 2,
            )
            slot_centers[slot_id] = (scx, scy, r)

        # Build slot → list of (dist, box) candidates
        slot_candidates: dict = {}
        for box in confirmed_boxes:
            bcx, bcy = box[0], box[1]
            best_slot = None
            best_dist = float('inf')
            for slot_id, (scx, scy, r) in slot_centers.items():
                dist = float(np.hypot(bcx - scx, bcy - scy))
                if dist < YOLO_SLOT_MAX_DIST * r * 2 and dist < best_dist:
                    best_dist = dist
                    best_slot = slot_id
            if best_slot is not None:
                slot_candidates.setdefault(best_slot, []).append((best_dist, box))

        assigned: dict = {}
        duplicate_slots: set = set()

        for slot_id, candidates in slot_candidates.items():
            if len(candidates) > 1:
                duplicate_slots.add(slot_id)
                # Keep box with highest confidence score
                best_box = max(candidates, key=lambda t: t[1][4])[1]
            else:
                best_box = candidates[0][1]
            assigned[slot_id] = {
                'cx': int(best_box[0]), 'cy': int(best_box[1]),
                'w':  int(best_box[2]), 'h':  int(best_box[3]),
                'conf': best_box[4],
            }

        return assigned, duplicate_slots

    # ------------------------------------------------------------------
    # ROI helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fixed_sample_roi(frame_shape: tuple) -> tuple:
        """
        Compute (x1, y1, x2, y2) from the fixed per-edge margins in config.toml.

        Each SAMPLE_ROI_* value is pixels trimmed FROM that edge:
            x1 = left,              y1 = top
            x2 = frame_w − right,   y2 = frame_h − bottom

        If a margin would make x2 ≤ x1 or y2 ≤ y1 (i.e. the margin is larger
        than the frame), that side falls back to the frame boundary so no crop
        is applied on that side.  This handles right=8000 on a 4608-px-wide
        frame gracefully (x2 → frame_w, no right-side crop).
        """
        fh, fw = frame_shape[:2]
        x1 = max(0, SAMPLE_ROI_LEFT)
        y1 = max(0, SAMPLE_ROI_TOP)
        x2 = fw - SAMPLE_ROI_RIGHT
        y2 = fh - SAMPLE_ROI_BOTTOM
        # Clamp: if margin overshoots, keep full frame on that side
        if x2 <= x1:
            x2 = fw
        if y2 <= y1:
            y2 = fh
        return x1, y1, x2, y2

    @staticmethod
    def _roi_from_corners(corners, frame_shape, padding: int = YOLO_ROI_PADDING) -> tuple:
        """
        Compute (x1, y1, x2, y2) ROI bounding box from 4 calibration corners + padding.
        Clamps to frame boundaries.
        """
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        fh, fw = frame_shape[:2]
        x1 = max(0,      int(min(xs)) - padding)
        y1 = max(0,      int(min(ys)) - padding)
        x2 = min(fw - 1, int(max(xs)) + padding)
        y2 = min(fh - 1, int(max(ys)) + padding)
        return x1, y1, x2, y2

    # ------------------------------------------------------------------
    # Full pipeline: frames → (ref_positions, sample_hits, duplicate_slots)
    # ------------------------------------------------------------------

    def detect_multi(self, frames: list, grid_cfg) -> tuple:
        """
        Run the full multi-snapshot → consensus → position-based geometric pipeline.

        Automatically crops each frame to the grid ROI (from grid_cfg.corners)
        before YOLO inference to eliminate false positives outside the grid area.

        Returns:
            (ref_positions, sample_hits, duplicate_slots)
              ref_positions:   dict {level: [(cx, cy, r), ...]} — bottles near row-0 slots
              sample_hits:     dict {slot_id: {cx, cy, w, h, conf}} — bottles near main-grid slots
              duplicate_slots: set of slot_ids with competing detections
        """
        roi = None
        if frames:
            # Fixed config-based crop takes priority — excludes reference row and
            # applies consistent margins defined in config.toml [sample_roi].
            roi = self._fixed_sample_roi(frames[0].shape)

        detections_list = [self.detect_once(f, roi=roi) for f in frames]
        confirmed = self.consensus_filter(detections_list)
        ref_positions = self.geometric_validate_ref(confirmed, grid_cfg)
        sample_hits, duplicate_slots = self.geometric_validate(confirmed, grid_cfg)
        return ref_positions, sample_hits, duplicate_slots

    # ------------------------------------------------------------------
    # JSON result writer
    # ------------------------------------------------------------------

    @staticmethod
    def write_result_json(slot_assignments: dict, counts: dict, timestamp: datetime) -> Path:
        """Write JSON summary of confirmed detections to logs/img/."""
        out_dir = Path(IMG_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = timestamp.strftime("%Y-%m-%d_%H-%M-%S") + "_result.json"
        payload = {
            "timestamp":        timestamp.isoformat(),
            "counts_per_level": counts,
            "confirmed_slots": {
                slot_id: {
                    "level":     data.get("level"),
                    "delta_e":   data.get("delta_e"),
                    "confident": data.get("confident"),
                    "error":     data.get("error"),
                }
                for slot_id, data in slot_assignments.items()
            },
        }
        out_path = out_dir / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return out_path
