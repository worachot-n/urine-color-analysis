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


def _draw_summary(img, boxes, roi, conf_threshold, mode="yolo", imgsz=640):
    total   = len(boxes)
    suspect = sum(1 for b in boxes if b[2] < 30 or b[3] < 30)
    lines = [
        f"Mode: {mode.upper()}   conf>={conf_threshold}   imgsz={imgsz}",
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


def process_image(img_path: Path, model, conf_threshold: float, roi, out_dir: Path, imgsz=640):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  [SKIP] Cannot read {img_path.name}")
        return

    # ── Step 1: explicit crop to sample ROI ──────────────────────────────────
    if roi is not None:
        rx1, ry1, rx2, ry2 = roi
        cropped = frame[ry1:ry2, rx1:rx2]
        x_off, y_off = rx1, ry1
    else:
        cropped = frame
        rx1, ry1, rx2, ry2 = 0, 0, frame.shape[1], frame.shape[0]
        x_off, y_off = 0, 0

    # Save the cropped image so YOLO input is visible for inspection
    crop_path = out_dir / f"crop_{img_path.stem}.jpg"
    cv2.imwrite(str(crop_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # ── Step 2: run YOLO on the cropped image (no roi arg) ───────────────────
    boxes_crop = model.detect_once(cropped)

    # ── Step 3: translate box coordinates back to full-frame space ───────────
    boxes = []
    for cx, cy, w, h, conf, cls in boxes_crop:
        boxes.append([cx + x_off, cy + y_off, w, h, conf, cls])

    # ── Step 4a: annotate cropped image with boxes in crop-space ─────────────
    vis_crop = cropped.copy()
    for cx, cy, w, h, conf, cls in boxes_crop:
        _draw_box(vis_crop, [cx, cy, w, h, conf, cls])
    roi_str = f"({rx1},{ry1})-({rx2},{ry2})"
    _draw_summary(vis_crop, boxes_crop, roi_str, conf_threshold, imgsz=imgsz)
    cv2.imwrite(str(crop_path), vis_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # ── Step 4b: annotate full frame for context ──────────────────────────────
    scale = min(1.0, 1600 / frame.shape[1])
    vis_w = int(frame.shape[1] * scale)
    vis_h = int(frame.shape[0] * scale)
    vis = cv2.resize(frame, (vis_w, vis_h), interpolation=cv2.INTER_AREA)

    def s(v):   # scale a coordinate
        return int(v * scale)

    # Dim the excluded area (outside ROI) with a dark overlay
    overlay = vis.copy()
    # Top strip
    if s(ry1) > 0:
        cv2.rectangle(overlay, (0, 0), (vis_w, s(ry1)), (0, 0, 0), -1)
    # Bottom strip
    if s(ry2) < vis_h:
        cv2.rectangle(overlay, (0, s(ry2)), (vis_w, vis_h), (0, 0, 0), -1)
    # Left strip
    if s(rx1) > 0:
        cv2.rectangle(overlay, (0, s(ry1)), (s(rx1), s(ry2)), (0, 0, 0), -1)
    # Right strip
    if s(rx2) < vis_w:
        cv2.rectangle(overlay, (s(rx2), s(ry1)), (vis_w, s(ry2)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

    # Draw bright ROI boundary
    cv2.rectangle(vis, (s(rx1), s(ry1)), (s(rx2), s(ry2)), COLOR_ROI, 3)
    cv2.putText(vis, f"DETECT AREA  {rx2-rx1}x{ry2-ry1}px",
                (s(rx1) + 6, s(ry1) + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ROI, 2, cv2.LINE_AA)

    # Draw detected boxes (translated to full-frame, scaled for vis)
    for b in boxes:
        cx, cy, w, h, conf, cls = b
        _draw_box(vis, [cx*scale, cy*scale, w*scale, h*scale, conf, cls])

    # Summary overlay
    _draw_summary(vis, boxes, roi_str, conf_threshold, imgsz=imgsz)

    # Save annotated full-frame result
    out_path = out_dir / f"det_{img_path.stem}.jpg"
    cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"  {img_path.name:45s}  {len(boxes):3d} bottle(s)"
          f"  tiny={sum(1 for b in boxes if b[2]<30 or b[3]<30)}"
          f"  → {out_path.name}  crop→{crop_path.name}")


def main():
    parser = argparse.ArgumentParser(description="YOLO detection visual log test")
    parser.add_argument("images", nargs="*", help="Image path(s) — default: all data/*.jpg")
    parser.add_argument("--conf", type=float, default=None,
                        help="Override conf_threshold (default: from config.toml)")
    args = parser.parse_args()

    # ── Load pyproject.toml (test tunables) ──────────────────────────────────
    import tomllib
    _pyproject = _ROOT / "pyproject.toml"
    with open(_pyproject, "rb") as _f:
        _pp = tomllib.load(_f)
    _ua = _pp.get("tool", {}).get("urine-analyzer", {})

    # ── Load config ──────────────────────────────────────────────────────────
    import configs.config as config
    _default_conf = float(_ua.get("yolo", {}).get("conf_threshold", 0.25))
    conf_threshold = args.conf if args.conf is not None else _default_conf

    _imgsz_raw = _ua.get("yolo", {}).get("imgsz", 640)
    _test_imgsz = _imgsz_raw if isinstance(_imgsz_raw, int) else list(_imgsz_raw)

    model_path = str(_ROOT / config.YOLO_MODEL_PATH)
    if not Path(model_path).exists():
        print(f"[ERROR] YOLO model not found: {model_path}")
        print("  Place the exported OpenVINO model in models/bottle_yolo26s_openvino/")
        sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────────────
    from utils.yolo_detector import YoloBottleDetector
    model = YoloBottleDetector(model_path)
    if args.conf is not None:
        # Monkey-patch the threshold so detect_once uses our value
        import configs.config as _cfg_mod
        _cfg_mod.YOLO_CONF_THRESHOLD = args.conf
        import utils.yolo_detector as _yd_mod
        _yd_mod.YOLO_CONF_THRESHOLD = args.conf

    # Always apply imgsz from pyproject.toml (rectangular to eliminate letterbox padding)
    import configs.config as _cfg_mod2
    _cfg_mod2.YOLO_IMGSZ = _test_imgsz
    import utils.yolo_detector as _yd_mod2
    _yd_mod2.YOLO_IMGSZ = _test_imgsz

    # ── Load crop margins from pyproject.toml ────────────────────────────────
    _sroi = _ua.get("sample_roi", {})
    _crop_top    = int(_sroi.get("top",    0))
    _crop_bottom = int(_sroi.get("bottom", 0))
    _crop_left   = int(_sroi.get("left",   0))
    _crop_right  = int(_sroi.get("right",  0))
    print(f"[INFO] Crop margins from pyproject.toml — "
          f"top={_crop_top} bottom={_crop_bottom} "
          f"left={_crop_left} right={_crop_right}")

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
        # Compute ROI from pyproject.toml margins, clamped to this image's resolution
        img_roi = None
        frame_tmp = cv2.imread(str(img_path))
        if frame_tmp is not None:
            fh, fw = frame_tmp.shape[:2]
            x1 = max(0, _crop_left)
            y1 = max(0, _crop_top)
            x2 = fw - _crop_right
            y2 = fh - _crop_bottom
            if x2 <= x1: x2 = fw   # margin too large — keep full width
            if y2 <= y1: y2 = fh   # margin too large — keep full height
            img_roi = (x1, y1, x2, y2)

        process_image(img_path, model, conf_threshold, img_roi, _OUT_DIR, imgsz=_test_imgsz)

    print(f"\nDone. Results saved to: {_OUT_DIR}")


if __name__ == "__main__":
    main()
