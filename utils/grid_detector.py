"""
Grid line detection — Phases B & C of the V2 analysis pipeline.

Phase B: U-Net segmentation — isolates white grid lines from the black background.
Phase C: Skeletonization + mathematical line fitting — finds 195 precise slot centres.

GridDetector raises FileNotFoundError at construction time if the U-Net weights file
(models/unet_grid.pt) is missing.  Train the network and place the weights file there
before starting the server.

Slot centre ordering:
  position_index = row_idx * COLS + col_idx + 1   (1-based, row-major)
  row_idx : 0–12  (13 sample rows, row 0 = reference excluded from output)
  col_idx : 0–14  (15 sample columns, col 0 = ZZ dead-zone excluded from output)

Grid line structure from a calibrated tray:
  14 horizontal lines → 13 row spans  (index 0 = reference, 1–12 = samples)
  17 vertical lines   → 16 col spans  (index 0 = ZZ dead zone, 1–15 = data)
  Intersections kept: rows 1–13 × cols 1–15 = 195 sample slots
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger

_IntersectionList = List[Tuple[int, int]]


@dataclasses.dataclass
class GridDetectionResult:
    sample_centres: list[tuple[int, int]]  # 195, rows 1-13 cols 1-15
    ref_centres:    list[tuple[int, int]]  # 15,  row 0 cols 1-15, left→right
    grid_spacing:   float                  # avg px gap between adjacent centres

# Grid constants matching the physical tray layout
_H_LINES = 14    # horizontal grid lines
_V_LINES = 16    # vertical grid lines (15 col spans; ZZ excluded from calibration area)
_ROWS    = 13    # sample rows (lines 1–13 → spans 1–13, span 0 = reference)
_COLS    = 15    # sample columns (lines 1–15 → spans 1–15, span 0 = ZZ dead zone)


class GridDetector:
    """
    Detects the 195 slot-centre coordinates from a raw tray image.

    Instantiate once at server startup; call detect_intersections() per scan.
    """

    def __init__(self, model_path: str = "models/unet_grid.pt") -> None:
        weights = Path(model_path)
        if not weights.exists():
            raise FileNotFoundError(
                f"U-Net weights not found: {weights.resolve()}\n"
                "Train the grid-line segmentation model and place the weights at "
                f"'{weights}' before starting the server."
            )

        import torch
        from models.unet import UNet

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net    = UNet().to(self._device)
        self._net.load_state_dict(
            torch.load(str(weights), map_location=self._device)
        )
        self._net.eval()
        logger.success("GridDetector: U-Net loaded from {} on {}", weights, self._device)

    # ------------------------------------------------------------------ public

    def detect_intersections(
        self,
        img_bgr: np.ndarray,
        input_size: int = 640,
        mask_threshold: float = 0.5,
    ) -> GridDetectionResult:
        """
        Run the full Phase B + C pipeline on a raw BGR image.

        Returns:
            GridDetectionResult with 195 sample_centres (rows 1-13, cols 1-15),
            15 ref_centres (row 0, cols 1-15), and average grid_spacing in pixels.
            sample_centres is empty if fitting fails.
        """
        h_orig, w_orig = img_bgr.shape[:2]

        # ── Phase B: U-Net segmentation ──────────────────────────────────────
        mask = self._segment(img_bgr, input_size, mask_threshold)   # 640×640 bool

        # ── Phase C: skeleton → line fitting → intersections ─────────────────
        sample_512, ref_512, spacing_512 = self._fit_intersections(mask)

        if len(sample_512) != 195:
            logger.warning(
                "GridDetector: expected 195 sample intersections, got {} — "
                "check U-Net weights and image quality",
                len(sample_512),
            )
            if not sample_512:
                return GridDetectionResult([], [], 50.0)

        if len(ref_512) < 15:
            logger.warning("GridDetector: only {} reference row centres found (expected 15)", len(ref_512))

        # Scale back to original image coordinates
        sx = w_orig / input_size
        sy = h_orig / input_size
        logger.info("GridDetector: {} sample + {} ref centres found", len(sample_512), len(ref_512))
        return GridDetectionResult(
            sample_centres=[(int(x * sx), int(y * sy)) for x, y in sample_512],
            ref_centres=   [(int(x * sx), int(y * sy)) for x, y in ref_512],
            grid_spacing=  spacing_512 * (sx + sy) / 2,
        )

    # ----------------------------------------------------------------- private

    def _segment(
        self,
        img_bgr: np.ndarray,
        input_size: int,
        threshold: float,
    ) -> np.ndarray:
        """Run U-Net; return boolean mask (True = grid line pixel)."""
        import torch

        gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        tensor  = torch.from_numpy(resized).float().div(255.0)   # [0,1]
        tensor  = tensor.unsqueeze(0).unsqueeze(0).to(self._device)  # (1,1,H,W)

        with torch.no_grad():
            logits = self._net(tensor)                           # (1,1,H,W)
            prob   = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H,W)

        mask = prob > threshold
        logger.debug(
            "U-Net segmentation: {:.1f}% pixels classified as grid lines",
            100.0 * mask.mean(),
        )
        return mask

    def _fit_intersections(self, mask: np.ndarray) -> tuple[_IntersectionList, _IntersectionList, float]:
        """
        Phase C:
          1. Skeletonize the binary mask.
          2. Detect lines with probabilistic Hough transform.
          3. Cluster into horizontal / vertical groups, fit one line per cluster.
          4. Compute all H×V intersection candidates.
          5. Keep only sample-area intersections (exclude reference row + ZZ col).
        """
        from skimage.morphology import skeletonize

        skeleton = skeletonize(mask).astype(np.uint8) * 255

        # Hough lines on skeleton (probabilistic for speed)
        lines = cv2.HoughLinesP(
            skeleton,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=20,
            maxLineGap=10,
        )
        if lines is None:
            logger.warning("GridDetector: Hough found no lines in skeleton")
            return [], [], 50.0

        h_segments: list[tuple[float, float, float, float]] = []
        v_segments: list[tuple[float, float, float, float]] = []

        for seg in lines[:, 0]:
            x1, y1, x2, y2 = map(float, seg)
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 30 or angle > 150:
                h_segments.append((x1, y1, x2, y2))
            else:
                v_segments.append((x1, y1, x2, y2))

        h_lines = self._cluster_and_fit(h_segments, axis="h", n_expected=_H_LINES)
        v_lines = self._cluster_and_fit(v_segments, axis="v", n_expected=_V_LINES)

        if len(h_lines) < 2 or len(v_lines) < 2:
            logger.warning(
                "GridDetector: insufficient lines after fitting (H={}, V={})",
                len(h_lines), len(v_lines),
            )
            return [], [], 50.0

        # Sort: horizontal lines by y-intercept (top→bottom), vertical by x-intercept
        h_lines.sort(key=lambda ab: ab[1])   # (slope, intercept) → sort by intercept
        v_lines.sort(key=lambda ab: ab[1])

        # Compute all H×V intersections
        all_pts: list[tuple[int, int]] = []
        for ha, hb in h_lines:
            for va, vb in v_lines:
                pt = _line_intersection(ha, hb, va, vb)
                if pt is not None:
                    all_pts.append(pt)

        # Select sample intersections: rows 1–13, cols 1–15 (0-indexed from sorted order)
        # h_lines[0] = top-most (reference row top edge); first SAMPLE row starts at index 1
        sample_pts: list[tuple[int, int]] = []
        n_h = len(h_lines)
        n_v = len(v_lines)

        for hi in range(1, min(n_h, _ROWS + 1)):        # rows 1 to 13
            for vi in range(1, min(n_v, _COLS + 1)):    # cols 1 to 15
                idx = hi * n_v + vi
                if idx < len(all_pts):
                    sample_pts.append(all_pts[idx])

        # Reference row: centre of each cell (average of 4 corners), cols 1-15
        # Cell corners: top-left=h[0]×v[vi], top-right=h[0]×v[vi+1],
        #               bot-left=h[1]×v[vi], bot-right=h[1]×v[vi+1]
        ref_pts: list[tuple[int, int]] = []
        for vi in range(1, min(n_v, _COLS + 1)):
            corner_indices = [
                0 * n_v + vi,       # top-left
                0 * n_v + (vi + 1), # top-right
                1 * n_v + vi,       # bottom-left
                1 * n_v + (vi + 1), # bottom-right
            ]
            corners = [all_pts[idx] for idx in corner_indices if idx < len(all_pts)]
            if corners:
                x = sum(c[0] for c in corners) // len(corners)
                y = sum(c[1] for c in corners) // len(corners)
                ref_pts.append((x, y))

        spacing = self._estimate_spacing(sample_pts)
        return sample_pts, ref_pts, spacing

    @staticmethod
    def _estimate_spacing(pts: list[tuple[int, int]]) -> float:
        """Median distance between adjacent centres in the first row of sample_pts."""
        if len(pts) < 2:
            return 50.0
        row = np.array(pts[:15], dtype=np.float32)
        diffs = np.linalg.norm(np.diff(row, axis=0), axis=1)
        return float(np.median(diffs)) if len(diffs) else 50.0

    @staticmethod
    def _cluster_and_fit(
        segments: list[tuple[float, float, float, float]],
        axis: str,
        n_expected: int,
    ) -> list[tuple[float, float]]:
        """
        Group line segments into n_expected clusters by their perpendicular position
        (y-midpoint for horizontal, x-midpoint for vertical), then fit one line per
        cluster with np.polyfit(deg=1).

        Returns list of (slope, intercept) pairs.
        """
        if not segments:
            return []

        # Compute midpoint along the clustering axis
        mids: np.ndarray
        if axis == "h":
            mids = np.array([(y1 + y2) / 2 for _, y1, _, y2 in segments])
        else:
            mids = np.array([(x1 + x2) / 2 for x1, _, x2, _ in segments])

        # Simple 1-D k-means-style clustering: sort + gap detection
        order = np.argsort(mids)
        sorted_mids = mids[order]
        gaps = np.diff(sorted_mids)

        if len(gaps) == 0:
            n_clusters = 1
        else:
            gap_threshold = max(np.percentile(gaps, 70), 2.0)
            split_points  = np.where(gaps > gap_threshold)[0] + 1
            # Aim for n_expected clusters; allow ±2
            n_clusters = len(split_points) + 1

        clusters: list[list[int]] = []
        prev = 0
        for sp in ([] if n_clusters == 1 else np.where(gaps > max(np.percentile(gaps, 70), 2.0))[0] + 1):
            clusters.append(order[prev:sp].tolist())
            prev = sp
        clusters.append(order[prev:].tolist())

        fitted: list[tuple[float, float]] = []
        for cluster_idx in clusters:
            segs_in = [segments[i] for i in cluster_idx]
            if axis == "h":
                xs = [x for x1, y1, x2, y2 in segs_in for x in (x1, x2)]
                ys = [y for x1, y1, x2, y2 in segs_in for y in (y1, y2)]
            else:
                xs = [y for x1, y1, x2, y2 in segs_in for y in (y1, y2)]
                ys = [x for x1, y1, x2, y2 in segs_in for x in (x1, x2)]

            if len(xs) < 2:
                continue
            try:
                slope, intercept = np.polyfit(xs, ys, 1)
                fitted.append((float(slope), float(intercept)))
            except np.linalg.LinAlgError:
                continue

        return fitted


def _line_intersection(
    ha: float, hb: float,
    va: float, vb: float,
) -> tuple[int, int] | None:
    """
    Find intersection of two lines expressed as y = ha*x + hb (horizontal)
    and x = va*y + vb (vertical, expressed as x = f(y)).

    Returns (x, y) integer pixel coordinates, or None if parallel/degenerate.
    """
    # h: y = ha*x + hb  →  ha*x - y + hb = 0
    # v: x = va*y + vb  →  x - va*y - vb = 0  →  (1/va)*x - y = vb/va  if va!=0
    # Substitute v into h:
    # y = ha*(va*y + vb) + hb  →  y - ha*va*y = ha*vb + hb
    # y*(1 - ha*va) = ha*vb + hb
    denom = 1.0 - ha * va
    if abs(denom) < 1e-9:
        return None
    y = (ha * vb + hb) / denom
    x = va * y + vb
    return (int(round(x)), int(round(y)))
