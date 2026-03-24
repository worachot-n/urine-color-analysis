"""
Local Windows test web server — validates Calibrate / Detect / Classify pipeline
using static images from the data/ folder.

Usage:
    cd urine-color-analysis
    python tests/test_web.py        # use system Python (recommended)

    # If the project venv doesn't have opencv/flask, run with global Python:
    # python313 tests/test_web.py

Then open: http://localhost:5001
"""

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# ── Make project root importable ──────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA_DIR    = _ROOT / "data"
_RESULT_DIR  = _ROOT / "logs" / "img"
_GRID_CFG    = _ROOT / "grid_config.json"
_TEMPLATES   = Path(__file__).parent / "templates"

# Ensure result directory exists
_RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=str(_TEMPLATES))
app.logger.setLevel("WARNING")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_frame(image_name: str):
    """Load a BGR frame from data/ folder. Returns (frame, error_response)."""
    path = _DATA_DIR / image_name
    if not path.exists():
        return None, (jsonify({"error": f"Image not found: {image_name}"}), 404)
    frame = cv2.imread(str(path))
    if frame is None:
        return None, (jsonify({"error": f"Cannot read image: {image_name}"}), 400)
    return frame, None


def _save_result(vis: np.ndarray, tag: str) -> str:
    """Save an annotated image to logs/img/ and return its URL path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:21]
    fname = f"test_{tag}_{ts}.jpg"
    out_path = _RESULT_DIR / fname
    cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return f"/result/{fname}"


def _draw_grid_overlay(frame: np.ndarray, ref_slots: dict, slot_data: dict) -> np.ndarray:
    """Draw all slot polygons on a copy of frame."""
    vis = frame.copy()
    for info in slot_data.values():
        pts = np.array(info["coords"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 180, 0), thickness=1)
    for info in ref_slots.values():
        pts = np.array(info["coords"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 200, 200), thickness=2)
    return vis


# ── Routes — static serving ───────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("test_ui.html")


@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory(str(_DATA_DIR), filename)


@app.route("/result/<path:filename>")
def serve_result(filename):
    return send_from_directory(str(_RESULT_DIR), filename)


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(str(_ROOT / "assets"), filename)


# ── API — image list ──────────────────────────────────────────────────────────

@app.route("/api/images")
def api_images():
    exts = {".jpg", ".jpeg", ".png"}
    images = sorted(
        f.name for f in _DATA_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    )
    return jsonify({"images": images})


# ── API — grid status ─────────────────────────────────────────────────────────

@app.route("/api/grid-status")
def api_grid_status():
    if _GRID_CFG.exists():
        try:
            with open(_GRID_CFG) as f:
                meta = json.load(f).get("system_metadata", {})
            date = meta.get("calibration_date", "")
        except Exception:
            date = ""
        return jsonify({"exists": True, "date": date})
    return jsonify({"exists": False, "date": None})


# ── API — calibration (3-step: compute → adjust → save) ──────────────────────

@app.route("/api/compute-grid", methods=["POST"])
def api_compute_grid():
    """Phase 1: 4 corners → return grid_pts JSON for canvas editor. Does NOT save."""
    body = request.get_json(force=True)
    corners_raw = body.get("corners", [])
    if len(corners_raw) != 4:
        return jsonify({"error": "Exactly 4 corners required"}), 400
    try:
        from utils.calibration import _corners_to_grid_pts
        grid_pts = _corners_to_grid_pts(corners_raw)
        return jsonify({"grid_pts": grid_pts.tolist()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/restore-grid")
def api_restore_grid():
    """Return saved corners + grid_pts from grid_config.json for the canvas editor."""
    if not _GRID_CFG.exists():
        return jsonify({"error": "grid_config.json not found — calibrate first"}), 404
    try:
        with open(_GRID_CFG) as f:
            cfg = json.load(f)
        meta = cfg.get("system_metadata", {})
        corners   = meta.get("corners")
        grid_pts  = meta.get("grid_pts")
        calib_date = meta.get("calibration_date", "")
        if not corners or len(corners) != 4:
            return jsonify({"error": "No corners saved in grid_config.json"}), 400
        if not grid_pts:
            from utils.calibration import _corners_to_grid_pts
            grid_pts = _corners_to_grid_pts(corners).tolist()
        return jsonify({"corners": corners, "grid_pts": grid_pts, "date": calib_date})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/save-grid", methods=["POST"])
def api_save_grid():
    """Phase 2: receive final corners + adjusted grid_pts, compute polygons, save grid_config.json."""
    body     = request.get_json(force=True)
    corners  = body.get("corners",  [])
    grid_pts = body.get("grid_pts", [])

    if len(corners) != 4:
        return jsonify({"error": "Need exactly 4 corners"}), 400
    if not grid_pts:
        return jsonify({"error": "grid_pts missing"}), 400

    try:
        from utils.calibration import compute_slot_polygons_from_grid, compute_sample_roi
        grid_np = np.array(grid_pts, dtype=np.float64)
        ref_slots, slot_data = compute_slot_polygons_from_grid(grid_np)
        sample_roi = compute_sample_roi(grid_np)

        def _ser(info: dict) -> dict:
            out = {}
            for k, v in info.items():
                out[k] = v.tolist() if isinstance(v, np.ndarray) else v
            return out

        payload = {
            "system_metadata": {
                "project_name":    "Urine Color Analysis",
                "grid_dimensions": "16x14 lines",
                "calibration_date": datetime.now().strftime("%Y-%m-%d"),
                "corners":    corners,
                "grid_pts":   grid_pts,
                "sample_roi": sample_roi,
            },
            "reference_row": {
                "slots": {sid: _ser(info) for sid, info in ref_slots.items()}
            },
            "main_grid": {
                "slot_data": {sid: _ser(info) for sid, info in slot_data.items()}
            },
        }
        with open(_GRID_CFG, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return jsonify({"ok": True, "slots": len(slot_data)})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/layout")
def api_layout():
    """Return slot ID layout table for canvas label overlay (ref row + 12 main rows)."""
    try:
        from utils.grid import load_grid_layout
        ref_row = (
            ["ZZ"]
            + ["REF_L0"] * 3
            + ["REF_L1"] * 3
            + ["REF_L2"] * 3
            + ["REF_L3"] * 3
            + ["REF_L4"] * 3
        )
        layout = [ref_row] + load_grid_layout()   # 13 rows: 1 ref + 12 main
        return jsonify({"layout": layout})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── API — color status ────────────────────────────────────────────────────────

_COLOR_CFG = _ROOT / "color.json"

@app.route("/api/color-status")
def api_color_status():
    if _COLOR_CFG.exists():
        try:
            data = json.loads(_COLOR_CFG.read_text())
            date = data.get("calibration_date", "")
        except Exception:
            date = ""
        return jsonify({"exists": True, "date": date})
    return jsonify({"exists": False, "date": None})


# ── API — save reference colors ───────────────────────────────────────────────

@app.route("/api/save-colors", methods=["POST"])
def api_save_colors():
    """
    Receive 15 reference bottle Lab+hex values, compute 5 averaged baselines,
    save to color.json.

    Body: {
      "bottles": {
        "0": [{"lab": [L,a,b], "hex": "#rrggbb"}, ...×3],
        "1": [...], "2": [...], "3": [...], "4": [...]
      }
    }
    """
    body = request.get_json(force=True)
    bottles = body.get("bottles", {})
    if len(bottles) != 5:
        return jsonify({"error": "Need exactly 5 levels (0-4)"}), 400

    baseline = {}
    for lvl_str, bottle_list in bottles.items():
        if not bottle_list:
            continue
        labs = [b["lab"] for b in bottle_list if "lab" in b]
        if not labs:
            continue
        avg_lab = [sum(v[i] for v in labs) / len(labs) for i in range(3)]
        # Hex from avg Lab (convert back to RGB for display)
        r = int(sum(int(b["hex"][1:3], 16) for b in bottle_list) / len(bottle_list))
        g = int(sum(int(b["hex"][3:5], 16) for b in bottle_list) / len(bottle_list))
        bv = int(sum(int(b["hex"][5:7], 16) for b in bottle_list) / len(bottle_list))
        avg_hex = f"#{r:02x}{g:02x}{bv:02x}"
        baseline[lvl_str] = {"lab": avg_lab, "hex": avg_hex}

    payload = {
        "calibration_date": datetime.now().strftime("%Y-%m-%d"),
        "bottles":  bottles,
        "baseline": baseline,
    }
    try:
        _COLOR_CFG.write_text(json.dumps(payload, indent=2))
        return jsonify({"ok": True, "levels": len(baseline)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


# ── API — detect ──────────────────────────────────────────────────────────────

@app.route("/api/detect", methods=["POST"])
def api_detect():
    body = request.get_json(force=True)
    image_name = body.get("image", "")

    frame, err = _load_frame(image_name)
    if err:
        return err

    boxes = []
    mode  = "opencv"
    warn  = None

    # ---- Compute ROI once — fixed config margins, shared by YOLO and OpenCV paths ----
    roi = None
    try:
        from utils.yolo_detector import YoloBottleDetector as _YD
        roi = _YD._fixed_sample_roi(frame.shape)
    except Exception:
        pass

    # ---- Try YOLO first ----
    try:
        import configs.config as config
        from utils.yolo_detector import YoloBottleDetector

        model_path = str(_ROOT / config.YOLO_MODEL_PATH)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        detector = YoloBottleDetector(model_path)
        raw_boxes = detector.detect_once(frame, roi=roi)   # [[cx,cy,w,h,conf,cls], ...]
        boxes = [
            {"cx": int(b[0]), "cy": int(b[1]), "w": int(b[2]), "h": int(b[3]),
             "conf": round(float(b[4]), 4), "cls": int(b[5])}
            for b in raw_boxes
        ]
        mode = "yolo"

    except FileNotFoundError as exc:
        warn = str(exc)
    except ImportError:
        warn = "ultralytics not installed — using OpenCV fallback"
    except Exception as exc:
        warn = f"YOLO error ({type(exc).__name__}): {exc}"

    # ---- OpenCV fallback ----
    if mode == "opencv":
        try:
            from utils.image_processing import detect_red_caps
            # Apply ROI crop to eliminate black-area false positives
            crop = frame
            x_off, y_off = 0, 0
            if roi is not None:
                rx1, ry1, rx2, ry2 = roi
                crop = frame[ry1:ry2, rx1:rx2]
                x_off, y_off = rx1, ry1
            circles = detect_red_caps(crop) or []
            boxes = [
                {"cx": int(cx) + x_off, "cy": int(cy) + y_off,
                 "w": int(r * 2), "h": int(r * 2), "conf": 1.0, "cls": 0}
                for cx, cy, r in circles
            ]
        except Exception as exc:
            return jsonify({"error": f"OpenCV detection failed: {exc}"}), 500

    # ---- Draw annotated image ----
    vis = frame.copy()
    cls_colors = {0: (80, 200, 120)}   # green for all bottles (single-class model)
    cls_names  = {0: "bottle"}

    for b in boxes:
        cx, cy, w, h = b["cx"], b["cy"], b["w"], b["h"]
        color = cls_colors.get(b["cls"], (200, 200, 200))
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_names.get(b['cls'], '?')} {b['conf']:.2f}"
        cv2.rectangle(vis, (x1, y1 - 16), (x1 + len(label) * 8, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    annotated_url = _save_result(vis, "detect")

    response = {
        "boxes": boxes,
        "count": len(boxes),
        "mode":  mode,
        "annotated_url": annotated_url,
    }
    if warn:
        response["warning"] = warn
    return jsonify(response)


# ── API — classify ────────────────────────────────────────────────────────────

@app.route("/api/classify", methods=["POST"])
def api_classify():
    body = request.get_json(force=True)
    image_name = body.get("image", "")

    # Grid required
    if not _GRID_CFG.exists():
        return jsonify({
            "error": "grid_config.json not found — run Calibrate tab first"
        }), 400

    frame, err = _load_frame(image_name)
    if err:
        return err

    try:
        import configs.config as config
        from utils.grid import GridConfig
        from utils.color_analysis import (
            build_reference_baseline, classify_sample,
            extract_bottle_color, delta_e_cie76, WHITE_LAB,
            load_static_baseline,
        )
        from utils.utils import save_annotated_image

        grid_cfg = GridConfig(str(_GRID_CFG))

        # ---- Detection: YOLO or OpenCV fallback ----
        yolo_hits: dict      = {}
        duplicate_slots: set = set()
        ref_positions_yolo: dict = {}
        detection_mode = "opencv"

        try:
            from utils.yolo_detector import YoloBottleDetector

            model_path = str(_ROOT / config.YOLO_MODEL_PATH)
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            detector = YoloBottleDetector(model_path)
            ref_positions_yolo, yolo_hits, duplicate_slots = detector.detect_multi(
                [frame] * config.YOLO_CONSENSUS_MIN, grid_cfg
            )
            detection_mode = "yolo"

        except (FileNotFoundError, ImportError):
            pass   # fall through to OpenCV
        except Exception as exc:
            app.logger.warning("YOLO detect_multi failed: %s", exc)

        # OpenCV fallback — map detected circles to slots via majority rule
        if detection_mode == "opencv":
            from utils.image_processing import detect_red_caps
            circles = detect_red_caps(frame) or []
            for cx, cy, r in circles:
                slot_id, _ = grid_cfg.find_slot_for_circle(cx, cy, r)
                if slot_id and slot_id not in yolo_hits:
                    yolo_hits[slot_id] = {"cx": cx, "cy": cy, "w": r * 2, "h": r * 2, "conf": 1.0}

        # ---- Reference baseline ----
        # Prefer pre-saved static baseline (color.json); fall back to dynamic sampling.
        static_baseline = load_static_baseline(str(_COLOR_CFG))
        if static_baseline:
            baseline = static_baseline
        else:
            calibrated_positions = grid_cfg.get_reference_positions()
            merged_ref = dict(calibrated_positions)
            merged_ref.update(ref_positions_yolo)
            baseline = build_reference_baseline(frame, merged_ref)

        if not baseline:
            return jsonify({"error": "Reference baseline empty — check reference row or save color.json"}), 500

        # ---- Classify each detected bottle ----
        slot_assignments: dict = {}
        rejected: list = []

        for slot_id, hit in yolo_hits.items():
            cx, cy = hit["cx"], hit["cy"]
            r = max(1, min(hit.get("w", 30), hit.get("h", 30)) // 2)

            sample_lab = extract_bottle_color(frame, cx, cy, r)
            if sample_lab is None:
                rejected.append(slot_id)
                continue

            # Ghost check
            de_white = delta_e_cie76(sample_lab, WHITE_LAB)
            if de_white < config.GHOST_DE_THRESHOLD:
                rejected.append(slot_id)
                slot_assignments[slot_id] = {
                    "cx": cx, "cy": cy, "radius": r,
                    "w": hit.get("w", r * 2), "h": hit.get("h", r * 2),
                    "level": None, "expected_level": None,
                    "delta_e": None, "confident": False,
                    "error": False, "ghost": True, "duplicate": False,
                }
                continue

            level, delta_e, confident = classify_sample(sample_lab, baseline)

            slot_info    = grid_cfg.slot_data.get(slot_id, {})
            expected     = slot_info.get("expected_level")
            is_duplicate = slot_id in duplicate_slots
            color_error  = (
                level is not None and expected is not None
                and level != expected and not is_duplicate
            )

            slot_assignments[slot_id] = {
                "cx": cx, "cy": cy, "radius": r,
                "w": hit.get("w", r * 2), "h": hit.get("h", r * 2),
                "level": level, "expected_level": expected,
                "delta_e": round(float(delta_e), 3) if delta_e is not None else None,
                "confident": confident,
                "error": color_error or is_duplicate,
                "error_type": "duplicate" if is_duplicate else ("mismatch" if color_error else None),
                "ghost": False,
                "duplicate": is_duplicate,
            }

        counts = {i: 0 for i in range(5)}
        for d in slot_assignments.values():
            lvl = d.get("level")
            if lvl is not None and 0 <= lvl <= 4:
                counts[lvl] += 1

        errors = [
            f"{sid}: expected L{d['expected_level']} got L{d['level']}"
            for sid, d in slot_assignments.items()
            if d.get("error") and not d.get("ghost")
        ]
        has_errors = bool(errors)

        # ---- Save annotated image ----
        ts = datetime.now()
        try:
            log_path = save_annotated_image(
                frame, slot_assignments, grid_cfg,
                rejected_slots=[
                    (slot_assignments[sid]["cx"], slot_assignments[sid]["cy"],
                     slot_assignments[sid]["radius"], sid)
                    for sid in rejected if sid in slot_assignments
                ],
                timestamp=ts,
            )
            annotated_url = f"/result/{log_path.name}"
        except Exception as exc:
            # Fallback: draw boxes manually
            vis = frame.copy()
            for sid, d in slot_assignments.items():
                cx, cy, r = d["cx"], d["cy"], d["radius"]
                color = (0, 80, 220) if d.get("ghost") else \
                        (0, 0, 220) if d.get("error") else (0, 200, 80)
                cv2.circle(vis, (cx, cy), r, color, 2)
                cv2.putText(vis, sid, (cx - 20, cy - r - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            annotated_url = _save_result(vis, "classify")

        # ---- JSON-safe slot assignments (remove numpy types) ----
        safe_slots = {}
        for sid, d in slot_assignments.items():
            safe_slots[sid] = {
                k: (int(v) if isinstance(v, (np.integer,)) else
                    float(v) if isinstance(v, (np.floating,)) else v)
                for k, v in d.items()
                if k not in ("cx", "cy", "radius", "w", "h")  # pixel coords not needed in table
            }
            safe_slots[sid]["cx"] = d["cx"]
            safe_slots[sid]["cy"] = d["cy"]

        return jsonify({
            "slot_assignments": safe_slots,
            "counts":           counts,
            "errors":           errors,
            "has_errors":       has_errors,
            "detection_mode":   detection_mode,
            "annotated_url":    annotated_url,
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Urine Analyzer — Local Test Lab")
    print(f"  Data folder : {_DATA_DIR}")
    print(f"  Grid config : {_GRID_CFG} ({'exists' if _GRID_CFG.exists() else 'NOT FOUND'})")
    print(f"  Results dir : {_RESULT_DIR}")
    print("=" * 60)
    print("  Open: http://localhost:5001")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
