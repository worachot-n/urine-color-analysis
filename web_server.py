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
    start_captive_portal(ip)
    stop_captive_portal()
"""

import json
import os
import threading
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask, jsonify, render_template_string,
    request, send_file, redirect,
)

import config

logger = logging.getLogger(__name__)

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


def update_scan_result(counts: dict, errors: list[str], image_path: str | None) -> None:
    """Called by main.py after every scan to refresh dashboard data."""
    with _state_lock:
        _scan_state["counts"]         = dict(counts)
        _scan_state["errors"]         = list(errors)
        _scan_state["last_scan_time"] = datetime.now().isoformat(timespec="seconds")
        _scan_state["image_path"]     = str(image_path) if image_path else None


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
# Calibration state
# ---------------------------------------------------------------------------
_calib_state: dict = {
    "image_path": None,
    "image_w":    0,
    "image_h":    0,
}
_calib_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Shared navbar snippet (injected into all pages)
# ---------------------------------------------------------------------------
_NAVBAR = """
<nav>
  <a href="/" class="{% if active == 'results' %}active{% endif %}">&#128202; Results</a>
  <a href="/calibrate" class="{% if active == 'calibrate' %}active{% endif %}">&#127919; Calibrate</a>
</nav>
"""

_NAV_STYLE = """
nav { background:#1a1a2e; padding:0.6rem 1rem; margin-bottom:1.2rem;
      border-radius:6px; display:flex; gap:1.2rem; }
nav a { color:#7cf; text-decoration:none; font-size:1rem; padding:0.2rem 0.5rem;
        border-radius:4px; }
nav a.active, nav a:hover { background:#2a2a4e; color:#fff; }
"""

# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------
_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Urine Color Analyzer</title>
  <meta http-equiv="refresh" content="10">
  <style>
    body   { font-family: monospace; background:#111; color:#eee; padding:1rem; }
    h1     { color:#7cf; margin-bottom:0.4rem; }
    .ts    { color:#888; font-size:0.85rem; margin-bottom:1rem; }
    table  { border-collapse:collapse; margin-bottom:1rem; }
    th, td { border:1px solid #444; padding:0.4rem 0.8rem; text-align:center; }
    th     { background:#222; }
    .ok    { color:#4f4; }
    .errors-list { color:#fa0; }
    img    { max-width:100%; border:1px solid #444; margin-top:0.5rem; }
    """ + _NAV_STYLE + """
  </style>
</head>
<body>
  """ + _NAVBAR + """
  <h1>Urine Color Analyzer</h1>
  <div class="ts">Last scan: {{ last_scan_time or "No scan yet" }}</div>

  <table>
    <tr>{% for lvl in range(5) %}<th>L{{ lvl }}</th>{% endfor %}</tr>
    <tr>{% for lvl in range(5) %}<td>{{ counts[lvl] }}</td>{% endfor %}</tr>
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
# Calibration HTML
# ---------------------------------------------------------------------------
_CALIBRATE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Calibrate Grid</title>
  <style>
    body    { font-family: monospace; background:#111; color:#eee;
              padding:1rem; margin:0; }
    h1      { color:#7cf; margin-bottom:0.4rem; }
    """ + _NAV_STYLE + """
    .source-btns { display:flex; gap:1rem; margin-bottom:1rem; flex-wrap:wrap; }
    .btn    { padding:0.6rem 1.2rem; border:none; border-radius:5px;
              cursor:pointer; font-size:1rem; }
    .btn-blue  { background:#2196F3; color:#fff; }
    .btn-green { background:#4caf50; color:#fff; }
    .btn-red   { background:#f44336; color:#fff; }
    .btn:disabled { opacity:0.4; cursor:default; }
    #upload-input { display:none; }
    #canvas-wrap  { position:relative; display:inline-block;
                    border:2px solid #444; border-radius:4px; }
    #calib-canvas { display:block; cursor:crosshair;
                    max-width:90vw; max-height:72vh; }
    #status { margin-top:0.8rem; color:#fa0; min-height:1.2rem; }
    #instructions { color:#888; font-size:0.85rem; margin-bottom:0.6rem; }
    .step-label { color:#7cf; font-weight:bold; }
  </style>
</head>
<body>
  """ + _NAVBAR + """
  <h1>Grid Calibration</h1>

  <div class="source-btns">
    <button id="btn-capture" class="btn btn-blue" onclick="captureFromCamera()">
      &#128247; Capture from Camera
    </button>
    <label class="btn btn-blue" style="display:inline-block">
      &#128193; Upload Image
      <input id="upload-input" type="file" accept="image/*" onchange="uploadImage(this)">
    </label>
  </div>

  <p id="instructions">
    <span class="step-label">Step 1:</span> Choose an image source above.<br>
    <span class="step-label">Step 2:</span> Click the <b>4 outer corners</b> of the grid
      in order: Top-Left &#8594; Top-Right &#8594; Bottom-Right &#8594; Bottom-Left.<br>
    <span class="step-label">Step 3:</span> Drag grid lines to fine-tune if needed.<br>
    <span class="step-label">Step 4:</span> Click <b>Save Calibration</b>.
  </p>

  <div id="canvas-wrap" style="display:none">
    <canvas id="calib-canvas"></canvas>
  </div>

  <div style="margin-top:0.8rem; display:flex; gap:0.8rem; flex-wrap:wrap">
    <button id="btn-reset" class="btn btn-red" onclick="resetCorners()" disabled>
      &#8635; Reset Corners
    </button>
    <button id="btn-save" class="btn btn-green" onclick="saveCalibration()" disabled>
      &#10003; Save Calibration
    </button>
  </div>

  <div id="status"></div>

  <script>
  // -------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------
  let img        = null;
  let origW      = 0, origH = 0;
  let scaleX     = 1, scaleY = 1;
  let corners    = [];          // canvas-space [[x,y] x4]
  let gridPts    = null;        // canvas-space [15 rows][17 cols][2]
  let dragging   = null;        // {type:'h'|'v', index:n, startX, startY}

  const canvas   = document.getElementById('calib-canvas');
  const ctx      = canvas.getContext('2d');
  const wrap     = document.getElementById('canvas-wrap');
  const statusEl = document.getElementById('status');
  const btnSave  = document.getElementById('btn-save');
  const btnReset = document.getElementById('btn-reset');

  // -------------------------------------------------------------------------
  // Source selection
  // -------------------------------------------------------------------------
  async function captureFromCamera() {
    setStatus('Capturing from camera...');
    try {
      const res  = await fetch('/api/calibrate/capture', {method:'POST'});
      const data = await res.json();
      if (data.error) { setStatus('Error: ' + data.error); return; }
      await loadCalibImage(data.width, data.height);
    } catch(e) { setStatus('Capture failed: ' + e); }
  }

  async function uploadImage(input) {
    if (!input.files.length) return;
    setStatus('Uploading...');
    const fd = new FormData();
    fd.append('file', input.files[0]);
    try {
      const res  = await fetch('/api/calibrate/upload', {method:'POST', body:fd});
      const data = await res.json();
      if (data.error) { setStatus('Error: ' + data.error); return; }
      await loadCalibImage(data.width, data.height);
    } catch(e) { setStatus('Upload failed: ' + e); }
  }

  async function loadCalibImage(w, h) {
    origW = w; origH = h;
    img   = new Image();
    img.onload = () => {
      // Size canvas to fit viewport
      const maxW = Math.min(window.innerWidth * 0.92, 1400);
      const maxH = Math.min(window.innerHeight * 0.70, 900);
      scaleX = Math.min(maxW / origW, maxH / origH);
      scaleY = scaleX;
      canvas.width  = Math.round(origW * scaleX);
      canvas.height = Math.round(origH * scaleY);
      wrap.style.display = 'inline-block';
      resetCorners();
      setStatus('Click the 4 outer corners of the grid (TL \u2192 TR \u2192 BR \u2192 BL).');
    };
    img.src = '/api/calibrate/image?t=' + Date.now();
  }

  // -------------------------------------------------------------------------
  // Corner clicks
  // -------------------------------------------------------------------------
  function getEventPos(e) {
    const rect = canvas.getBoundingClientRect();
    if (e.touches) {
      return [e.touches[0].clientX - rect.left, e.touches[0].clientY - rect.top];
    }
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  canvas.addEventListener('click', async (e) => {
    if (corners.length >= 4 || gridPts) return;
    const [cx, cy] = getEventPos(e);
    corners.push([cx, cy]);
    draw();
    if (corners.length === 4) {
      setStatus('Computing grid overlay...');
      await computeGrid();
    } else {
      const labels = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left'];
      setStatus('Click corner ' + (corners.length + 1) + ': ' + labels[corners.length]);
    }
    btnReset.disabled = false;
  });

  // -------------------------------------------------------------------------
  // Grid computation
  // -------------------------------------------------------------------------
  async function computeGrid() {
    const origCorners = corners.map(([cx, cy]) => [cx / scaleX, cy / scaleY]);
    try {
      const res  = await fetch('/api/calibrate/compute-grid', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({corners: origCorners}),
      });
      const data = await res.json();
      if (data.error) { setStatus('Error: ' + data.error); return; }
      // Scale grid_pts to canvas coordinates
      gridPts = data.grid_pts.map(row =>
        row.map(([x, y]) => [x * scaleX, y * scaleY])
      );
      draw();
      btnSave.disabled  = false;
      setStatus('Grid computed. Drag lines to fine-tune, then click Save.');
    } catch(e) { setStatus('Grid computation failed: ' + e); }
  }

  // -------------------------------------------------------------------------
  // Draw
  // -------------------------------------------------------------------------
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (img) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // Corner markers
    const cornerLabels = ['1 TL', '2 TR', '3 BR', '4 BL'];
    corners.forEach(([cx, cy], i) => {
      ctx.beginPath();
      ctx.arc(cx, cy, 9, 0, 2*Math.PI);
      ctx.fillStyle = 'rgba(255,60,60,0.85)';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 11px monospace';
      ctx.fillText(cornerLabels[i] || (i+1), cx + 11, cy + 4);
    });

    // Grid lines
    if (!gridPts) return;
    ctx.lineWidth = 1;
    // Horizontal lines (15 rows)
    for (let r = 0; r < gridPts.length; r++) {
      ctx.beginPath();
      ctx.strokeStyle = r === 0 ? 'rgba(255,220,0,0.75)' : 'rgba(0,230,80,0.65)';
      gridPts[r].forEach(([x, y], c) => c === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
      ctx.stroke();
    }
    // Vertical lines (17 cols)
    for (let c = 0; c < gridPts[0].length; c++) {
      ctx.beginPath();
      ctx.strokeStyle = c === 0 ? 'rgba(255,80,80,0.6)' : 'rgba(0,230,80,0.65)';
      gridPts.forEach((row, r) => r === 0 ? ctx.moveTo(row[c][0], row[c][1])
                                          : ctx.lineTo(row[c][0], row[c][1]));
      ctx.stroke();
    }
  }

  // -------------------------------------------------------------------------
  // Line drag (Phase 2 fine-tuning)
  // -------------------------------------------------------------------------
  function lineHitTest(mx, my) {
    const THRESH = 10;
    if (!gridPts) return null;
    // Check horizontal lines
    for (let r = 0; r < gridPts.length; r++) {
      for (let c = 0; c < gridPts[r].length - 1; c++) {
        const [x1, y1] = gridPts[r][c];
        const [x2, y2] = gridPts[r][c + 1];
        if (mx < Math.min(x1,x2) - THRESH || mx > Math.max(x1,x2) + THRESH) continue;
        const dist = Math.abs((y2-y1)*mx - (x2-x1)*my + x2*y1 - y2*x1) /
                     Math.hypot(y2-y1, x2-x1);
        if (dist < THRESH) return {type:'h', index:r};
      }
    }
    // Check vertical lines
    for (let c = 0; c < gridPts[0].length; c++) {
      for (let r = 0; r < gridPts.length - 1; r++) {
        const [x1, y1] = gridPts[r][c];
        const [x2, y2] = gridPts[r+1][c];
        if (my < Math.min(y1,y2) - THRESH || my > Math.max(y1,y2) + THRESH) continue;
        const dist = Math.abs((y2-y1)*mx - (x2-x1)*my + x2*y1 - y2*x1) /
                     Math.hypot(y2-y1, x2-x1);
        if (dist < THRESH) return {type:'v', index:c};
      }
    }
    return null;
  }

  canvas.addEventListener('mousedown', e => {
    if (!gridPts || corners.length < 4) return;
    const [mx, my] = getEventPos(e);
    const hit = lineHitTest(mx, my);
    if (hit) { dragging = {...hit, startX:mx, startY:my}; e.preventDefault(); }
  });

  canvas.addEventListener('mousemove', e => {
    if (!dragging) return;
    const [mx, my] = getEventPos(e);
    const dx = mx - dragging.startX;
    const dy = my - dragging.startY;
    dragging.startX = mx; dragging.startY = my;
    if (dragging.type === 'h') {
      gridPts[dragging.index] = gridPts[dragging.index].map(([x, y]) => [x, y + dy]);
    } else {
      gridPts.forEach(row => { row[dragging.index][0] += dx; });
    }
    draw();
  });

  canvas.addEventListener('mouseup',    () => { dragging = null; });
  canvas.addEventListener('mouseleave', () => { dragging = null; });

  // Touch equivalents
  canvas.addEventListener('touchstart', e => {
    if (!gridPts) return;
    const [mx, my] = getEventPos(e);
    const hit = lineHitTest(mx, my);
    if (hit) { dragging = {...hit, startX:mx, startY:my}; e.preventDefault(); }
  }, {passive:false});

  canvas.addEventListener('touchmove', e => {
    if (!dragging) return;
    const [mx, my] = getEventPos(e);
    const dx = mx - dragging.startX, dy = my - dragging.startY;
    dragging.startX = mx; dragging.startY = my;
    if (dragging.type === 'h') {
      gridPts[dragging.index] = gridPts[dragging.index].map(([x, y]) => [x, y + dy]);
    } else {
      gridPts.forEach(row => { row[dragging.index][0] += dx; });
    }
    draw(); e.preventDefault();
  }, {passive:false});

  canvas.addEventListener('touchend', () => { dragging = null; });

  // -------------------------------------------------------------------------
  // Controls
  // -------------------------------------------------------------------------
  function resetCorners() {
    corners  = [];
    gridPts  = null;
    dragging = null;
    btnSave.disabled  = true;
    btnReset.disabled = true;
    draw();
    setStatus('Click the 4 outer corners of the grid (TL \u2192 TR \u2192 BR \u2192 BL).');
  }

  async function saveCalibration() {
    if (!gridPts) return;
    btnSave.disabled = true;
    setStatus('Saving calibration...');
    const origCorners = corners.map(([cx, cy]) => [cx / scaleX, cy / scaleY]);
    const origGridPts = gridPts.map(row => row.map(([x, y]) => [x / scaleX, y / scaleY]));
    try {
      const res  = await fetch('/api/calibrate/save', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({corners: origCorners, grid_pts: origGridPts}),
      });
      const data = await res.json();
      if (data.ok) {
        setStatus('\u2713 Calibration saved! Redirecting to Results...');
        setTimeout(() => window.location = '/', 1800);
      } else {
        setStatus('Save failed: ' + (data.error || 'Unknown error'));
        btnSave.disabled = false;
      }
    } catch(e) {
      setStatus('Save request failed: ' + e);
      btnSave.disabled = false;
    }
  }

  function setStatus(msg) { statusEl.textContent = msg; }
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# WiFi setup (captive-portal) HTML
# ---------------------------------------------------------------------------
_WIFI_SETUP_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>WiFi Setup</title>
  <style>
    body { font-family: sans-serif; background:#f5f5f5; display:flex;
           justify-content:center; padding-top:3rem; }
    .card { background:#fff; padding:2rem; border-radius:8px;
            box-shadow:0 2px 8px rgba(0,0,0,.2); min-width:300px; }
    h2   { margin-top:0; }
    input { width:100%; padding:0.5rem; margin:0.4rem 0 1rem;
            box-sizing:border-box; border:1px solid #ccc; border-radius:4px; }
    button { width:100%; padding:0.6rem; background:#2196F3; color:#fff;
             border:none; border-radius:4px; cursor:pointer; font-size:1rem; }
    .msg { margin-top:1rem; color:{{ msg_color }}; }
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
    {% if message %}<p class="msg">{{ message }}</p>{% endif %}
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Flask app factory — dashboard + calibration
# ---------------------------------------------------------------------------

def _create_dashboard_app() -> Flask:
    app = Flask(__name__ + "_dashboard")
    app.logger.setLevel(logging.WARNING)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit

    # ---- Results page ----
    @app.route("/")
    def index():
        with _state_lock:
            state = dict(_scan_state)
        return render_template_string(
            _DASHBOARD_HTML,
            active         = "results",
            counts         = state["counts"],
            errors         = state["errors"],
            last_scan_time = state["last_scan_time"],
            image_path     = state["image_path"],
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

    # ---- Calibration page ----
    @app.route("/calibrate")
    def calibrate_page():
        return render_template_string(_CALIBRATE_HTML, active="calibrate")

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
            from calibration import capture_frame
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
            from calibration import _corners_to_grid_pts
            grid_pts = _corners_to_grid_pts(corners)  # shape (15, 17, 2)
            return jsonify({"grid_pts": grid_pts.tolist()})
        except Exception as e:
            logger.error("calib_compute_grid: %s", e)
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
            from calibration import compute_slot_polygons_from_grid
            grid_np = np.array(grid_pts, dtype=np.float64)  # (15, 17, 2)
            reference_slots, slot_data = compute_slot_polygons_from_grid(grid_np)

            output = {
                "system_metadata": {
                    "project_name":      "Urine Color Analysis",
                    "grid_dimensions":   "16x14 lines",
                    "calibration_date":  datetime.now().strftime("%Y-%m-%d"),
                    "corners":           corners,
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

    return app


# ---------------------------------------------------------------------------
# Flask app factory — captive portal
# ---------------------------------------------------------------------------

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
                _WIFI_SETUP_HTML, message="SSID is required.", msg_color="red"
            )
        import network
        network.notify_wifi_credentials(ssid, password)
        return render_template_string(
            _WIFI_SETUP_HTML,
            message="Connecting... The device will restart once connected.",
            msg_color="green",
        )

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def catch_all(path):
        return redirect("/wifi-setup")

    return app


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

_dashboard_thread = None
_portal_thread    = None


def start_web_server(port: int = config.WEB_SERVER_PORT) -> None:
    """Start the dashboard + calibration Flask app in a background daemon thread."""
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
    logger.info("Captive portal will stop when main process exits")
