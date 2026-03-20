"""
Flask web server — dashboard and captive-portal WiFi setup.

Two modes of operation:

1. Main dashboard (port WEB_SERVER_PORT, default 5000):
   GET /              — HTML dashboard with latest annotated image + counts
   GET /api/status    — JSON status payload
   GET /image/latest  — serves latest annotated JPG from logs/

2. Captive portal (port 80, during hotspot onboarding):
   GET  /wifi-setup   — HTML form: SSID + password
   POST /wifi-setup   — receives credentials, triggers network.connect_wifi

Both modes run in daemon threads so they never block main.py.

Public API:
    start_web_server(port)     — start dashboard in background thread
    stop_web_server()          — stop dashboard (best-effort)
    update_scan_result(...)    — push latest scan data to dashboard
    start_captive_portal(ip)   — start WiFi setup server on port 80
    stop_captive_portal()      — stop WiFi setup server
"""

import threading
import logging
import os
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_file, redirect

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state updated after each scan
# ---------------------------------------------------------------------------
_scan_state: dict = {
    "counts":         {lvl: 0 for lvl in range(5)},
    "errors":         [],
    "last_scan_time": None,
    "image_path":     None,
}
_state_lock = threading.Lock()


def update_scan_result(counts: dict, errors: list[str], image_path: str | None) -> None:
    """Called by main.py after every scan to refresh dashboard data."""
    with _state_lock:
        _scan_state["counts"]         = dict(counts)
        _scan_state["errors"]         = list(errors)
        _scan_state["last_scan_time"] = datetime.now().isoformat(timespec="seconds")
        _scan_state["image_path"]     = str(image_path) if image_path else None


# ---------------------------------------------------------------------------
# Dashboard HTML template
# ---------------------------------------------------------------------------
_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Urine Color Analyzer</title>
  <meta http-equiv="refresh" content="10">
  <style>
    body { font-family: monospace; background: #111; color: #eee; padding: 1rem; }
    h1   { color: #7cf; margin-bottom: 0.5rem; }
    .ts  { color: #888; font-size: 0.85rem; margin-bottom: 1rem; }
    table { border-collapse: collapse; margin-bottom: 1rem; }
    th, td { border: 1px solid #444; padding: 0.4rem 0.8rem; text-align: center; }
    th { background: #222; }
    .ok    { color: #4f4; }
    .error { color: #f44; }
    .errors-list { color: #fa0; }
    img { max-width: 100%; border: 1px solid #444; margin-top: 0.5rem; }
  </style>
</head>
<body>
  <h1>Urine Color Analyzer</h1>
  <div class="ts">Last scan: {{ last_scan_time or "No scan yet" }}</div>

  <table>
    <tr>
      {% for lvl in range(5) %}<th>L{{ lvl }}</th>{% endfor %}
    </tr>
    <tr>
      {% for lvl in range(5) %}<td>{{ counts[lvl] }}</td>{% endfor %}
    </tr>
  </table>

  {% if errors %}
    <p class="errors-list">Errors: {{ errors | join(", ") }}</p>
  {% else %}
    <p class="ok">&#10003; No errors</p>
  {% endif %}

  {% if image_path %}
    <img src="/image/latest" alt="Latest scan">
  {% endif %}
</body>
</html>"""


# ---------------------------------------------------------------------------
# WiFi setup (captive-portal) HTML template
# ---------------------------------------------------------------------------
_WIFI_SETUP_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>WiFi Setup</title>
  <style>
    body { font-family: sans-serif; background: #f5f5f5; display: flex;
           justify-content: center; padding-top: 3rem; }
    .card { background: #fff; padding: 2rem; border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,.2); min-width: 300px; }
    h2   { margin-top: 0; }
    input { width: 100%; padding: 0.5rem; margin: 0.4rem 0 1rem;
            box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
    button { width: 100%; padding: 0.6rem; background: #2196F3; color: #fff;
             border: none; border-radius: 4px; cursor: pointer; font-size: 1rem; }
    .msg { margin-top: 1rem; color: {{ msg_color }}; }
  </style>
</head>
<body>
  <div class="card">
    <h2>WiFi Setup</h2>
    <form method="post" action="/wifi-setup">
      <label>Network name (SSID)</label>
      <input name="ssid" type="text" placeholder="MyNetwork" required>
      <label>Password</label>
      <input name="password" type="password" placeholder="Password">
      <button type="submit">Connect</button>
    </form>
    {% if message %}
      <p class="msg">{{ message }}</p>
    {% endif %}
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------

def _create_dashboard_app() -> Flask:
    app = Flask(__name__ + "_dashboard")
    app.logger.setLevel(logging.WARNING)

    @app.route("/")
    def index():
        with _state_lock:
            state = dict(_scan_state)
        return render_template_string(
            _DASHBOARD_HTML,
            counts        = state["counts"],
            errors        = state["errors"],
            last_scan_time= state["last_scan_time"],
            image_path    = state["image_path"],
        )

    @app.route("/api/status")
    def api_status():
        with _state_lock:
            return jsonify(dict(_scan_state))

    @app.route("/image/latest")
    def image_latest():
        with _state_lock:
            path = _scan_state["image_path"]
        if path and Path(path).is_file():
            return send_file(path, mimetype="image/jpeg")
        return "No image available", 404

    return app


def _create_captive_portal_app() -> Flask:
    app = Flask(__name__ + "_portal")
    app.logger.setLevel(logging.WARNING)

    @app.route("/wifi-setup", methods=["GET"])
    def wifi_setup_get():
        return render_template_string(_WIFI_SETUP_HTML, message=None, msg_color="#333")

    @app.route("/wifi-setup", methods=["POST"])
    def wifi_setup_post():
        ssid     = (request.form.get("ssid")     or "").strip()
        password = (request.form.get("password") or "").strip()

        if not ssid:
            return render_template_string(
                _WIFI_SETUP_HTML,
                message="SSID is required.",
                msg_color="red",
            )

        # Notify network.py — it will attempt the connection
        import network
        network.notify_wifi_credentials(ssid, password)

        return render_template_string(
            _WIFI_SETUP_HTML,
            message="Connecting... The device will restart once connected.",
            msg_color="green",
        )

    # Redirect any other path to the setup form (captive portal behaviour)
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
    """Start the dashboard Flask app in a background daemon thread."""
    global _dashboard_thread

    app = _create_dashboard_app()

    _dashboard_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False),
        daemon=True,
        name="dashboard-server",
    )
    _dashboard_thread.start()
    logger.info("Dashboard started on port %d", port)


def stop_web_server() -> None:
    """Stop the dashboard server (best-effort; daemon threads auto-exit on main exit)."""
    logger.info("Dashboard server will stop when main process exits")


def start_captive_portal(ap_ip: str = config.HOTSPOT_IP) -> None:
    """Start the captive-portal WiFi setup server on port 80 in a daemon thread."""
    global _portal_thread

    app = _create_captive_portal_app()

    _portal_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=80, debug=False, use_reloader=False),
        daemon=True,
        name="captive-portal",
    )
    _portal_thread.start()
    logger.info("Captive portal started at http://%s/wifi-setup", ap_ip)


def stop_captive_portal() -> None:
    """Stop the captive-portal server (best-effort)."""
    logger.info("Captive portal will stop when main process exits")
