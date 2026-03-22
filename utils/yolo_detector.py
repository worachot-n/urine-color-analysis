"""
YOLOv11 + OpenVINO bottle detector with multi-snapshot consensus.

Public API:
    YoloBottleDetector(model_path)
        .detect_once(frame)              → list of [cx, cy, w, h, conf]
        .consensus_filter(detections)    → list of confirmed [cx, cy, w, h, conf]
        .geometric_validate(boxes, grid) → {slot_id: {'cx','cy','w','h','conf'}}
        .detect_multi(frames, grid)      → {slot_id: {'cx','cy','w','h','conf'}}
        .write_result_json(assignments, counts, ts) → Path
"""

import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from configs.config import (
    YOLO_IMGSZ, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, YOLO_AUGMENT,
    YOLO_CONSENSUS_MIN, YOLO_CONSENSUS_IOU, YOLO_SLOT_MAX_DIST,
    YOLO_CLAHE_CLIP, YOLO_CLAHE_TILE,
    IMG_DIR,
)


class YoloBottleDetector:
    """
    Wraps a YOLOv11 OpenVINO model with CLAHE preprocessing,
    multi-snapshot consensus, and geometric slot validation.
    """

    def __init__(self, model_path: str):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
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

    def detect_once(self, frame: np.ndarray) -> list:
        """
        Run YOLOv11 on one frame.

        Returns:
            list of [cx, cy, w, h, conf] in original pixel coords
        """
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
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w  = x2 - x1
                h  = y2 - y1
                conf = float(box.conf[0])
                boxes.append([cx, cy, w, h, conf])
        return boxes

    # ------------------------------------------------------------------
    # Consensus filter
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(b1, b2) -> float:
        """IoU between two [cx, cy, w, h] boxes."""
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
                             [ [[cx,cy,w,h,conf], ...], [...], [...] ]

        Returns:
            list of confirmed [cx, cy, w, h, conf] boxes (averaged coords)
        """
        if not detections_list:
            return []

        anchors = list(detections_list[0])
        votes   = [[box] for box in anchors]

        for frame_boxes in detections_list[1:]:
            matched = set()
            for anchor_idx, anchor in enumerate(anchors):
                for j, box in enumerate(frame_boxes):
                    if j in matched:
                        continue
                    if self._iou(anchor, box) >= YOLO_CONSENSUS_IOU:
                        votes[anchor_idx].append(box)
                        matched.add(j)

        confirmed = []
        for group in votes:
            if len(group) >= YOLO_CONSENSUS_MIN:
                cx   = float(np.mean([b[0] for b in group]))
                cy   = float(np.mean([b[1] for b in group]))
                w    = float(np.mean([b[2] for b in group]))
                h    = float(np.mean([b[3] for b in group]))
                conf = float(np.mean([b[4] for b in group]))
                confirmed.append([cx, cy, w, h, conf])

        return confirmed

    # ------------------------------------------------------------------
    # Geometric validation
    # ------------------------------------------------------------------

    def geometric_validate(self, confirmed_boxes: list, grid_cfg) -> tuple:
        """
        Map confirmed boxes to grid slots using slot→candidates grouping.

        When multiple boxes compete for one slot, keep the highest-confidence
        box and record the slot in duplicate_slots.

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
    # Full pipeline: frames → slot assignments
    # ------------------------------------------------------------------

    def detect_multi(self, frames: list, grid_cfg) -> tuple:
        """
        Run the full multi-snapshot → consensus → geometric pipeline.

        Returns:
            (assigned, duplicate_slots)
              assigned:        dict {slot_id: {'cx','cy','w','h','conf'}}
              duplicate_slots: set of slot_ids that had competing detections
        """
        detections_list = [self.detect_once(f) for f in frames]
        confirmed = self.consensus_filter(detections_list)
        return self.geometric_validate(confirmed, grid_cfg)

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
