"""
Flask web server — dashboard, web-based grid calibration, and captive-portal.

Routes:
  GET  /                          — Results dashboard
  GET  /api/status                — JSON scan state
  GET  /image/latest              — latest annotated scan JPG
  GET  /calibrate                 — calibration page (canvas UI)
  GET  /api/calibrate/image       — current calibration image
  POST /api/calibrate/capture     — capture frame from Pi camera
  POST /api/calibrate/upload      — upload image from browser
  POST /api/calibrate/compute-grid— 4 corners → grid_pts JSON
  POST /api/calibrate/save        — grid_pts → grid_config.json

  (Captive portal on port 80 during WiFi onboarding)
  GET  /wifi-setup                — HTML credential form
  POST /wifi-setup                — submit credentials

Public API (called from main.py):
    start_web_server(port)
    update_scan_result(counts, errors, image_path)
    notify_grid_saved()     — signal that grid_config.json was rewritten
    consume_grid_reload()   — True if user pressed Reload Grid on dashboard
    start_captive_portal(ip)
    stop_captive_portal()
"""

import json
import os
import sqlite3
import threading
import logging
from datetime import datetime
from pathlib import Path

from werkzeug.serving import make_server

import cv2
import numpy as np
from flask import (
    Flask, jsonify, render_template,
    request, send_file, redirect,
)

import configs.config as config
from utils import db as _db

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = str(Path(__file__).parent.parent / "templates")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_LOGS_DIR          = Path(config.LOG_DIR)
_CALIB_TEMP_IMAGE  = _LOGS_DIR / "calib_temp.jpg"

# ---------------------------------------------------------------------------
# Shared scan state
# ---------------------------------------------------------------------------
_scan_state: dict = {
    "counts":         {lvl: 0 for lvl in range(5)},
    "errors":         [],
    "last_scan_time": None,
    "image_path":     None,
}
_state_lock = threading.Lock()


def update_scan_result(
    counts: dict,
    errors: list[str],
    image_path: str | None,
    slot_assignments: dict | None = None,
) -> None:
    """Called by main.py after every scan to refresh dashboard data and persist to DB."""
    with _state_lock:
        _scan_state["counts"]         = dict(counts)
        _scan_state["errors"]         = list(errors)
        _scan_state["last_scan_time"] = datetime.now().isoformat(timespec="seconds")
        _scan_state["image_path"]     = str(Path(image_path).resolve()) if image_path else None
    scan_id = _db.save_scan_result(counts, errors, image_path)
    if slot_assignments:
        _db.save_slot_results(scan_id, slot_assignments)


# ---------------------------------------------------------------------------
# Grid-saved event (signals main.py to reload GridConfig)
# ---------------------------------------------------------------------------
_grid_saved_event = threading.Event()


def notify_grid_saved() -> None:
    """Called internally after /api/calibrate/save succeeds."""
    _grid_saved_event.set()


def consume_grid_saved() -> bool:
    """
    Called by main.py in its loop.
    Returns True (and clears the flag) if grid_config.json was just rewritten.
    """
    if _grid_saved_event.is_set():
        _grid_saved_event.clear()
        return True
    return False


# ---------------------------------------------------------------------------
# Grid-reload event (reload existing grid_config.json without recalibrating)
# ---------------------------------------------------------------------------
_grid_reload_event = threading.Event()


def consume_grid_reload() -> bool:
    """Called by main.py. Returns True if user requested a grid reload."""
    if _grid_reload_event.is_set():
        _grid_reload_event.clear()
        return True
    return False


# ---------------------------------------------------------------------------
# Calibration state
# ---------------------------------------------------------------------------
_calib_state: dict = {
    "image_path": None,
    "image_w":    0,
    "image_h":    0,
}
_calib_lock = threading.Lock()






# ---------------------------------------------------------------------------
# Flask app factory — dashboard + calibration
# ---------------------------------------------------------------------------

def _create_dashboard_app() -> Flask:
    app = Flask(__name__ + "_dashboard", template_folder=_TEMPLATES_DIR)
    app.logger.setLevel(logging.WARNING)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit

    # ---- Favicon ----
    @app.route("/favicon.ico")
    def favicon():
        return send_file(
            str(Path(__file__).parent.parent / "assets" / "logo.ico"),
            mimetype="image/x-icon",
        )

    # ---- Static assets (logo, icons, etc.) ----
    @app.route("/assets/<path:filename>")
    def assets(filename):
        assets_dir = str(Path(__file__).parent.parent / "assets")
        from flask import send_from_directory
        return send_from_directory(assets_dir, filename)

    # ---- Results page ----
    @app.route("/")
    def index():
        result = _db.get_latest_result() or {
            "counts":     {i: 0 for i in range(5)},
            "errors":     [],
            "timestamp":  None,
            "image_path": None,
            "id":         None,
        }
        return render_template(
            "dashboard.html",
            active         = "results",
            counts         = result["counts"],
            errors         = result["errors"],
            last_scan_time = result.get("timestamp"),
            scan_id        = result.get("id"),
            image_path     = result.get("image_path"),
        )

    @app.route("/api/status")
    def api_status():
        with _state_lock:
            return jsonify(dict(_scan_state))

    @app.route("/image/latest")
    def image_latest():
        with _state_lock:
            path = _scan_state["image_path"]
        serve_path = None
        if path and Path(path).is_file():
            serve_path = path
        else:
            # Fallback: serve newest .jpg in logs/img/ (handles stale/missing path)
            img_dir = Path(config.IMG_DIR)
            if img_dir.is_dir():
                jpegs = sorted(img_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
                if jpegs:
                    serve_path = str(jpegs[0].resolve())
        if serve_path is None:
            return "No image available", 404
        # conditional=False disables ETag/Last-Modified → prevents HTTP 304 responses
        resp = send_file(serve_path, mimetype="image/jpeg", conditional=False)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"]  = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    @app.route("/image/<int:scan_id>")
    def image_by_id(scan_id):
        """Serve the annotated image for a specific scan by DB row id."""
        with sqlite3.connect(Path(config.DB_PATH)) as con:
            row = con.execute(
                "SELECT image_path FROM scan_results WHERE id=?", (scan_id,)
            ).fetchone()
        if row and row[0] and Path(row[0]).is_file():
            return send_file(row[0], mimetype="image/jpeg")
        return "Image not found", 404

    @app.route("/api/history")
    def api_history():
        """Return recent scan results as JSON. ?limit=N (default 20)."""
        limit = min(int(request.args.get("limit", 20)), 200)
        return jsonify(_db.get_recent_results(limit))

    @app.route("/analysis")
    def analysis_page():
        return render_template("analysis.html", active="analysis")

    @app.route("/api/analysis")
    def api_analysis():
        """Return slot-level results filtered by ?start=YYYY-MM-DD&end=YYYY-MM-DD."""
        start = request.args.get("start", "")
        end   = request.args.get("end", "")
        return jsonify(_db.get_slot_results(start, end))

    # ---- Grid reload (no recalibration) ----
    @app.route("/api/grid/reload", methods=["POST"])
    def grid_reload():
        _grid_reload_event.set()
        logger.info("Grid reload requested via web")
        return jsonify({"status": "reload triggered"})

    # ---- Calibration page ----
    @app.route("/calibrate")
    def calibrate_page():
        last_calib = None
        try:
            cfg_path = Path(config.GRID_CONFIG_FILE)
            if cfg_path.is_file():
                last_calib = json.loads(cfg_path.read_text()) \
                    .get("system_metadata", {}).get("calibration_date")
        except Exception:
            pass
        return render_template("calibrate.html", active="calibrate", last_calib=last_calib)

    @app.route("/api/calibrate/image")
    def calib_image():
        with _calib_lock:
            path = _calib_state["image_path"]
        if path and Path(path).is_file():
            return send_file(path, mimetype="image/jpeg")
        return "No calibration image", 404

    @app.route("/api/calibrate/capture", methods=["POST"])
    def calib_capture():
        """Capture a frame from the Pi camera and save it as the calibration image."""
        try:
            from utils.calibration import capture_frame
            _LOGS_DIR.mkdir(parents=True, exist_ok=True)
            frame = capture_frame()
            if frame is None:
                return jsonify({"error": "Camera capture failed"}), 500
            cv2.imwrite(str(_CALIB_TEMP_IMAGE), frame)
            h, w = frame.shape[:2]
            with _calib_lock:
                _calib_state["image_path"] = str(_CALIB_TEMP_IMAGE)
                _calib_state["image_w"]    = w
                _calib_state["image_h"]    = h
            logger.info("Calibration image captured: %dx%d", w, h)
            return jsonify({"width": w, "height": h})
        except Exception as e:
            logger.error("calib_capture: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/calibrate/upload", methods=["POST"])
    def calib_upload():
        """Accept an uploaded image file as the calibration image."""
        if "file" not in request.files:
            return jsonify({"error": "No file in request"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400
        try:
            _LOGS_DIR.mkdir(parents=True, exist_ok=True)
            f.save(str(_CALIB_TEMP_IMAGE))
            frame = cv2.imread(str(_CALIB_TEMP_IMAGE))
            if frame is None:
                return jsonify({"error": "Cannot decode uploaded image"}), 400
            h, w = frame.shape[:2]
            with _calib_lock:
                _calib_state["image_path"] = str(_CALIB_TEMP_IMAGE)
                _calib_state["image_w"]    = w
                _calib_state["image_h"]    = h
            logger.info("Calibration image uploaded: %dx%d", w, h)
            return jsonify({"width": w, "height": h})
        except Exception as e:
            logger.error("calib_upload: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/calibrate/compute-grid", methods=["POST"])
    def calib_compute_grid():
        """
        Receive 4 corner coordinates (original image space),
        compute the full grid_pts array via bilinear interpolation,
        return it as a JSON list for the canvas to draw.
        """
        data = request.get_json(force=True)
        corners = data.get("corners", [])
        if len(corners) != 4:
            return jsonify({"error": "Need exactly 4 corners"}), 400
        try:
            from utils.calibration import _corners_to_grid_pts
            grid_pts = _corners_to_grid_pts(corners)  # shape (15, 17, 2)
            return jsonify({"grid_pts": grid_pts.tolist()})
        except Exception as e:
            logger.error("calib_compute_grid: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/calibrate/restore")
    def calib_restore():
        """Return saved corners + grid_pts from grid_config.json for canvas overlay."""
        try:
            cfg_path = Path(config.GRID_CONFIG_FILE)
            if not cfg_path.is_file():
                return jsonify({"error": "grid_config.json not found — calibrate first"}), 404
            cfg  = json.loads(cfg_path.read_text())
            meta = cfg.get("system_metadata", {})
            corners    = meta.get("corners")
            saved_gpts = meta.get("grid_pts")
            calib_date = meta.get("calibration_date", "")
            if not corners or len(corners) != 4:
                return jsonify({"error": "No corners saved in grid_config.json"}), 400
            if saved_gpts:
                # Use the exact grid_pts that were saved (preserves any fine-tuning)
                grid_pts_list = saved_gpts
            else:
                # Fallback for old files that only have corners
                from utils.calibration import _corners_to_grid_pts
                grid_pts_list = _corners_to_grid_pts(corners).tolist()
            return jsonify({
                "corners":          corners,
                "grid_pts":         grid_pts_list,
                "calibration_date": calib_date,
            })
        except Exception as e:
            logger.error("calib_restore: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/calibrate/layout")
    def calib_layout():
        """Return the 2D slot-name grid (14 rows × 16 cols) for canvas overlay."""
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
            layout = [ref_row] + load_grid_layout()   # 13 rows × 16 cols
            return jsonify({"layout": layout})
        except Exception as e:
            logger.error("calib_layout: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/calibrate/save", methods=["POST"])
    def calib_save():
        """
        Receive final corners + grid_pts (original image space),
        compute slot polygons, and save grid_config.json.
        """
        data     = request.get_json(force=True)
        corners  = data.get("corners",  [])
        grid_pts = data.get("grid_pts", [])

        if len(corners) != 4:
            return jsonify({"error": "Need exactly 4 corners"}), 400
        if not grid_pts:
            return jsonify({"error": "grid_pts missing"}), 400

        try:
            from utils.calibration import compute_slot_polygons_from_grid, compute_sample_roi
            grid_np = np.array(grid_pts, dtype=np.float64)  # (14, 17, 2)
            reference_slots, slot_data = compute_slot_polygons_from_grid(grid_np)
            sample_roi = compute_sample_roi(grid_np)

            output = {
                "system_metadata": {
                    "project_name":      "Urine Color Analysis",
                    "grid_dimensions":   "16x14 lines",
                    "calibration_date":  datetime.now().strftime("%Y-%m-%d"),
                    "corners":           corners,
                    "grid_pts":          grid_pts,
                    "sample_roi":        sample_roi,
                },
                "reference_row": {
                    "description": "Top row for dynamic color calibration (3 bottles per level)",
                    "slots":       reference_slots,
                },
                "main_grid": {
                    "description": "Processing slots for Groups A1-A4 across Color Levels 0-4",
                    "slot_data":   slot_data,
                },
            }

            out_path = Path(config.GRID_CONFIG_FILE)
            out_path.write_text(json.dumps(output, indent=2))
            logger.info("grid_config.json saved via web calibration")
            notify_grid_saved()
            return jsonify({"ok": True})
        except Exception as e:
            logger.error("calib_save: %s", e)
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/calibrate/color-status")
    def calib_color_status():
        path = Path(config.COLOR_JSON_FILE)
        if path.exists():
            try:
                d = json.loads(path.read_text())
                date = d.get("calibration_date", "")
            except Exception:
                date = ""
            return jsonify({"exists": True, "date": date})
        return jsonify({"exists": False, "date": None})

    @app.route("/api/calibrate/save-colors", methods=["POST"])
    def calib_save_colors():
        """
        Save 15 reference bottle Lab+hex values and 5 averaged baselines to color.json.
        Body: {"bottles": {"0": [{"lab":[L,a,b],"hex":"#rrggbb"},...×3], ...×5}}
        """
        data = request.get_json(force=True)
        bottles = data.get("bottles", {})
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
            r  = int(sum(int(b["hex"][1:3], 16) for b in bottle_list) / len(bottle_list))
            g  = int(sum(int(b["hex"][3:5], 16) for b in bottle_list) / len(bottle_list))
            bv = int(sum(int(b["hex"][5:7], 16) for b in bottle_list) / len(bottle_list))
            baseline[lvl_str] = {"lab": avg_lab, "hex": f"#{r:02x}{g:02x}{bv:02x}"}

        payload = {
            "calibration_date": datetime.now().strftime("%Y-%m-%d"),
            "bottles":  bottles,
            "baseline": baseline,
        }
        try:
            Path(config.COLOR_JSON_FILE).write_text(json.dumps(payload, indent=2))
            logger.info("color.json saved with %d levels", len(baseline))
            return jsonify({"ok": True, "levels": len(baseline)})
        except Exception as e:
            logger.error("calib_save_colors: %s", e)
            return jsonify({"ok": False, "error": str(e)}), 500

    return app


# ---------------------------------------------------------------------------
# Flask app factory — captive portal
# ---------------------------------------------------------------------------

def _create_captive_portal_app() -> Flask:
    app = Flask(__name__ + "_portal", template_folder=_TEMPLATES_DIR)
    app.logger.setLevel(logging.WARNING)

    @app.route("/api/wifi/scan")
    def wifi_scan():
        try:
            import utils.network as net
            networks = net.scan_wifi_networks()
            return jsonify({"networks": networks, "error": None})
        except Exception as e:
            logger.error("wifi_scan: %s", e)
            return jsonify({"networks": [], "error": str(e)})

    @app.route("/wifi-setup", methods=["GET"])
    def wifi_setup_get():
        return render_template("wifi_setup.html", message=None, msg_color="#333")

    @app.route("/wifi-setup", methods=["POST"])
    def wifi_setup_post():
        ssid     = (request.form.get("ssid")     or "").strip()
        password = (request.form.get("password") or "").strip()
        if not ssid:
            return render_template("wifi_setup.html", message="SSID is required.", msg_color="red")
        import utils.network as net
        net.notify_wifi_credentials(ssid, password)
        return render_template("wifi_connecting.html", ssid=ssid, password=password)

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def catch_all(path):
        return redirect("/wifi-setup")

    return app


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

_dashboard_server = None
_dashboard_thread = None
_portal_server    = None
_portal_thread    = None


def start_web_server(port: int = config.WEB_SERVER_PORT) -> None:
    """Start the dashboard + calibration Flask app in a background daemon thread."""
    global _dashboard_server, _dashboard_thread

    _db.init_db()
    app = _create_dashboard_app()
    _dashboard_server = make_server("0.0.0.0", port, app)
    _dashboard_thread = threading.Thread(
        target=_dashboard_server.serve_forever,
        daemon=True,
        name="dashboard-server",
    )
    _dashboard_thread.start()
    logger.info("Dashboard started on port %d", port)


def stop_web_server() -> None:
    """Gracefully shut down the dashboard server and free its port."""
    global _dashboard_server
    if _dashboard_server:
        _dashboard_server.shutdown()
        _dashboard_server = None
        logger.info("Dashboard server stopped")


def start_captive_portal(ap_ip: str = config.HOTSPOT_IP) -> None:
    """Start the captive-portal WiFi setup server on CAPTIVE_PORTAL_PORT in a daemon thread."""
    global _portal_server, _portal_thread

    port = config.CAPTIVE_PORTAL_PORT
    app  = _create_captive_portal_app()
    _portal_server = make_server("0.0.0.0", port, app)
    _portal_thread = threading.Thread(
        target=_portal_server.serve_forever,
        daemon=True,
        name="captive-portal",
    )
    _portal_thread.start()
    logger.info("Captive portal started at http://%s:%d/wifi-setup", ap_ip, port)


def stop_captive_portal() -> None:
    """Gracefully shut down the captive portal and free its port."""
    global _portal_server
    if _portal_server:
        _portal_server.shutdown()
        _portal_server = None
        logger.info("Captive portal stopped")
