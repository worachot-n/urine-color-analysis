"""
Standalone YOLO detection visual log test.

Runs detect_once() on every image in data/ (or a single specified image),
draws annotated boxes + ROI boundary, and saves results to logs/img/detect_test/.

Usage:
    python tests/test_yolo_detect.py                      # all images in data/
    python tests/test_yolo_detect.py data/capture_xyz.jpg # single image
    python tests/test_yolo_detect.py --conf 0.35          # override threshold
"""

import sys
import os
import argparse
from pathlib import Path

# ── Make project root importable ──────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
_DATA_DIR   = _ROOT / "data"
_OUT_DIR    = _ROOT / "logs" / "img" / "detect_test"

# ── Colors (BGR) ───────────────────────────────────────────────────────────────
COLOR_BOX     = (80, 200, 120)    # green — detected bottle
COLOR_SMALL   = (0, 165, 255)    # orange — suspiciously small box (w or h < 30px)
COLOR_ROI     = (255, 220, 0)    # cyan-yellow — ROI boundary
COLOR_LABEL_BG = (30, 30, 30)


def _draw_box(img, box, min_size=30):
    cx, cy, w, h, conf, cls = box
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    color = COLOR_SMALL if (w < min_size or h < min_size) else COLOR_BOX
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    label = f"bottle {conf:.2f}  {int(w)}x{int(h)}"
    lw = len(label) * 8
    cv2.rectangle(img, (x1, y1 - 18), (x1 + lw, y1), COLOR_LABEL_BG, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Mark center
    cv2.circle(img, (int(cx), int(cy)), 4, color, -1)


def _draw_roi(img, roi):
    x1, y1, x2, y2 = roi
    cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_ROI, 3)
    cv2.putText(img, "ROI", (x1 + 6, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ROI, 2, cv2.LINE_AA)


def _draw_summary(img, boxes, roi, conf_threshold, mode="yolo"):
    total   = len(boxes)
    suspect = sum(1 for b in boxes if b[2] < 30 or b[3] < 30)
    lines = [
        f"Mode: {mode.upper()}   conf>={conf_threshold}",
        f"Detected: {total} bottle(s)",
        f"Suspect (tiny): {suspect}",
        f"ROI: {roi}",
    ]
    y = 30
    for line in lines:
        lw = len(line) * 9 + 10
        cv2.rectangle(img, (8, y - 18), (8 + lw, y + 4), (0, 0, 0), -1)
        cv2.putText(img, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26


def process_image(img_path: Path, model, conf_threshold: float, roi, out_dir: Path):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  [SKIP] Cannot read {img_path.name}")
        return

    boxes = model.detect_once(frame, roi=roi)

    # Scale down for display (max 1600px wide)
    scale = min(1.0, 1600 / frame.shape[1])
    vis_w = int(frame.shape[1] * scale)
    vis_h = int(frame.shape[0] * scale)
    vis = cv2.resize(frame, (vis_w, vis_h), interpolation=cv2.INTER_AREA)

    def s(v):   # scale a coordinate
        return int(v * scale)

    # Draw ROI boundary
    if roi is not None:
        rx1, ry1, rx2, ry2 = roi
        cv2.rectangle(vis, (s(rx1), s(ry1)), (s(rx2), s(ry2)), COLOR_ROI, 2)
        cv2.putText(vis, "ROI", (s(rx1) + 5, s(ry1) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ROI, 2, cv2.LINE_AA)

    # Draw detected boxes (scale coords to vis size)
    scaled_boxes = []
    for b in boxes:
        cx, cy, w, h, conf, cls = b
        sb = [cx*scale, cy*scale, w*scale, h*scale, conf, cls]
        scaled_boxes.append(sb)
        _draw_box(vis, sb)

    # Draw summary overlay
    roi_str = f"({roi[0]},{roi[1]})-({roi[2]},{roi[3]})" if roi else "none"
    _draw_summary(vis, boxes, roi_str, conf_threshold)

    # Save
    out_path = out_dir / f"det_{img_path.stem}.jpg"
    cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"  {img_path.name:45s}  {len(boxes):3d} bottle(s)"
          f"  tiny={sum(1 for b in boxes if b[2]<30 or b[3]<30)}"
          f"  → {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="YOLO detection visual log test")
    parser.add_argument("images", nargs="*", help="Image path(s) — default: all data/*.jpg")
    parser.add_argument("--conf", type=float, default=None,
                        help="Override conf_threshold (default: from config.toml)")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    import configs.config as config
    conf_threshold = args.conf if args.conf is not None else config.YOLO_CONF_THRESHOLD

    model_path = str(_ROOT / config.YOLO_MODEL_PATH)
    if not Path(model_path).exists():
        print(f"[ERROR] YOLO model not found: {model_path}")
        print("  Place the exported OpenVINO model in models/bottle_yolo26s_openvino/")
        sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────────────
    import configs.config as _c
    # Temporarily override conf threshold if --conf was passed
    from utils.yolo_detector import YoloBottleDetector
    model = YoloBottleDetector(model_path)
    if args.conf is not None:
        # Monkey-patch the threshold so detect_once uses our value
        import configs.config as _cfg_mod
        _cfg_mod.YOLO_CONF_THRESHOLD = args.conf
        import utils.yolo_detector as _yd_mod
        _yd_mod.YOLO_CONF_THRESHOLD = args.conf

    # ── Load ROI from grid calibration ───────────────────────────────────────
    roi = None
    try:
        from utils.grid import GridConfig
        gcfg = GridConfig()
        if gcfg.corners:
            # Use a sample image shape for ROI bounds (will be re-clamped per image)
            sample = cv2.imread(str(next(_DATA_DIR.glob("*.jpg"), None) or ""))
            if sample is not None:
                roi = YoloBottleDetector._roi_from_corners(gcfg.corners, sample.shape)
        print(f"[INFO] ROI: {roi}")
    except Exception as exc:
        print(f"[WARN] Grid config not loaded ({exc}) — running on full frame")

    # ── Collect images ───────────────────────────────────────────────────────
    if args.images:
        images = [Path(p) for p in args.images]
    else:
        images = sorted(_DATA_DIR.glob("*.jpg"))

    if not images:
        print(f"[ERROR] No images found in {_DATA_DIR}")
        sys.exit(1)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Run detection ────────────────────────────────────────────────────────
    print(f"\nYOLO detection test — conf>={conf_threshold}  model={Path(model_path).name}")
    print(f"Images: {len(images)}    Output: {_OUT_DIR}\n")

    total_detections = 0
    for img_path in images:
        # Recompute ROI clamped to this image's exact shape
        img_roi = None
        if roi is not None:
            frame_tmp = cv2.imread(str(img_path))
            if frame_tmp is not None:
                img_roi = YoloBottleDetector._roi_from_corners(
                    gcfg.corners, frame_tmp.shape
                ) if (hasattr(gcfg, 'corners') and gcfg.corners) else roi

        process_image(img_path, model, conf_threshold, img_roi, _OUT_DIR)

    print(f"\nDone. Results saved to: {_OUT_DIR}")


if __name__ == "__main__":
    main()
