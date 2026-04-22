# Urine Color Analysis System v2

A distributed vision system for automated urine sample analysis using AI.
A **Raspberry Pi 4B** captures and undistorts images, controls status hardware, and POSTs to a server.
A **remote Ubuntu server** runs YOLO inference, performs CIE Lab color classification, stores results in PostgreSQL, and hosts the web dashboard. The grid is calibrated once manually and saved to the database.
Communication is secured via HTTPS (Cloudflare Tunnel) with a shared API key.

---

## Architecture

```
Raspberry Pi 4B                            Ubuntu Server
────────────────────────────────           ─────────────────────────────────────────
Button press (GPIO 24)                     POST /analyze
  → Relay Yellow ON (processing)             → Fetch active tray from DB
  → picamera2 capture (4608 × 2592)          → ROI crop (skips ref row + dead zone)
  → CameraUndistorter.undistort_frame()      → letterbox to 640 × 640 (white)
  → HTTPS POST + X-Auth-Token  ─────────►   → Load grid_json from active tray (DB)
  → TM1637 spinner animation               → YOLO: bottle detection → slot matching
  → LCD "Analyzing..."                       → Live CIE Lab baseline from ref row
  → parse JSON response        ◄─────────   → Zone-baseline refinement from sample area
  → update LCD + TM1637 count              → Wrong-zone + duplicate-label validation
  → scroll error coords on LCD             → Visual log saved to logs/img/
  → Relay Green (ok) / Red (error)         → Telegram report (Thai) sent
                                           → PostgreSQL (ScanSession + TestSlot rows)
                                           → return JSON {count, summary, slots, errors}
```

- Pi performs lens undistortion locally before uploading (focus locked via `camera_params.yaml`)
- Server requires an **active tray** to be selected via the `/trays` management page
- Grid is calibrated once via `/settings` (4-corner click → bilinear interpolation → saved to `grid_json` in DB)
- YOLO boxes are matched to 195 calibrated slot centres loaded from the active tray's `grid_json`
- Color reference is extracted live from the reference row (row 0) every capture, then **refined using actual sample-area bottle colors** grouped by expected column zone — no `color.json` needed
- Duplicate label detection flags bottles whose `layout_json` label appears in more than one occupied slot
- Dashboard renders an interactive CSS grid sized to the active tray's `rows × cols`, auto-refreshing every 10 s
- Tunnel via Cloudflare for public HTTPS access from the Pi

---

## Repository Layout

```
urine-color-analysis/
│
├── main.py                         # Entry point — --role client | server
│
├── app/
│   ├── server_app.py               # FastAPI — YOLO, PostgreSQL, dashboard, tray API, grid calibration
│   ├── client_app.py               # Pi loop — undistort, GPIO, relay, LCD, TM1637, HTTP POST
│   ├── shared/
│   │   ├── config.py               # pydantic-settings (config.toml + settings.yaml + .env)
│   │   └── processor.py            # crop_sample_roi(), letterbox_white_padding(),
│   │                               # scale_coordinates(), save_visual_log(),
│   │                               # generate_visual_report(), _render_annotated_canvas()
│   └── web/
│       ├── templates/
│       │   ├── layout.html         # Base template (nav + hamburger, 5 links)
│       │   ├── dashboard.html      # Interactive dynamic grid (rows×cols) — 10 s auto-refresh
│       │   ├── trays.html          # Tray management — create, activate, edit, delete
│       │   ├── trend.html          # Historical chart + date-range filter
│       │   ├── settings.html       # Grid calibration UI
│       │   └── test.html           # Manual upload + result preview (no DB / Telegram)
│       └── static/
│           ├── captures/           # Annotated JPEGs at /static/captures/
│           └── test/               # Test-upload results at /static/test/
│
├── utils/
│   ├── grid_detector.py            # GridDetectionResult dataclass
│   ├── camera_undistort.py         # CameraUndistorter — lens correction + focus lock
│   ├── yolo_detector.py            # YoloBottleDetector
│   ├── color_analysis.py           # CIE Lab, Delta E, annular-ring extraction, classifier
│   ├── grid.py                     # GridConfig — slot polygons, find_slot_for_circle()
│   └── calibration.py              # Grid calibration helpers
│
├── models/
│   ├── unet.py                     # U-Net model (unused — kept for reference)
│   ├── unet_grid.pt                # U-Net trained weights (git-ignored)
│   └── best.pt                     # YOLO PyTorch weights (git-ignored)
│
├── configs/
│   ├── config.toml                 # PRIMARY hardware config — GPIO, camera, relay, YOLO, color analysis
│   ├── camera_params.yaml          # Lens calibration — focus_value, K matrix, distortion
│   └── config.py                   # TOML loader (used by utils/ directly)
│
├── settings.yaml                   # Server / model / database fallback config
├── pyproject.toml                  # uv project — extras: common / pi / server
│
├── logs/
│   ├── img/                        # Annotated scan images (URINE_SCAN_*.jpg)
│   └── app_*.log                   # Rotating text logs
│
└── data/
    └── results.db                  # SQLite fallback database (git-ignored)
```

---

## Quick Start

### 1 — Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2 — Configure secrets

```bash
cp .env.template .env
# Edit .env:
#   API_KEY=your-long-random-secret
#   SERVER_URL=https://your-tunnel.trycloudflare.com
#   DATABASE_URL=postgresql://user:pass@localhost:5432/urine_db
#   TELEGRAM_TOKEN=your_bot_token
#   TELEGRAM_CHAT_ID=your_chat_id
```

Both machines must use the **same** `API_KEY`. `DATABASE_URL` defaults to SQLite if unset.

### 3 — Ubuntu server

```bash
uv sync --extra server --extra common
uv run main.py --role server
# Dashboard  → http://localhost:8000/dashboard
# Trays      → http://localhost:8000/trays    ← select active tray here first
# Test page  → http://localhost:8000/test
# Health     → http://localhost:8000/health
```

> **Before the first scan:** go to `/trays`, create a tray, and click **เลือกใช้งาน** to activate it.
> `/analyze` returns HTTP 400 if no tray is active.

### 4 — Raspberry Pi

```bash
uv sync --extra pi --extra common
uv run main.py --role client
```

---

## Dependencies

| Extra | Key packages |
|-------|-------------|
| `common` | `requests`, `pydantic-settings`, `loguru`, `numpy`, `pyyaml`, `python-dotenv`, `Pillow`, `psutil` |
| `server` | `fastapi`, `uvicorn`, `opencv-python`, `sqlalchemy`, `psycopg2-binary`, `ultralytics`, `torch`, `torchvision`, `segmentation-models-pytorch`, `scikit-image`, `scipy`, `jinja2`, `python-multipart` |
| `pi` | `picamera2`, `RPi.GPIO`, `raspberrypi-tm1637`, `rpi-lcd`, `smbus2` |

---

## Configuration

### `configs/config.toml` — hardware (primary source)

```toml
[gpio.button]
pin         = 24
debounce_ms = 50

[gpio.relay]
led_red    = 21
led_yellow = 20
led_green  = 12
active_low = true

[camera]
capture_resolution = [4608, 2592]
rotate_180         = true

[yolo]
model_path     = "models/best.pt"
conf_threshold = 0.40
iou_threshold  = 0.45

[color_analysis]
outer_crop_px = 15    # pixels to exclude from bottle edge (plastic ring)
inner_crop_px = 10    # pixels from bottle center to exclude (glare hot-spot)
                      # sampling region = annular ring between inner_crop_px and (radius - outer_crop_px)
confidence_margin = 3.0

[sample_roi]
top    = 220   # skip reference row
bottom = 20
left   = 200   # skip dead-zone column ZZ
right  = 800
```

### `configs/camera_params.yaml` — lens calibration

```yaml
metadata:
  focus_value: 1.0        # diopters ≈ 1/distance_metres; 100 cm tray → 1.0
  sensor_resolution:
    width:  4608
    height: 2592
calibration:
  camera_matrix:
    fx: 3254.7
    fy: 3254.7
    cx: 2304.0
    cy: 1296.0
  dist_coeffs: [-0.1234, 0.0567, -0.0001, 0.0002, -0.0123]
```

Replace the placeholder values with the output of your own `cv2.calibrateCamera()` run at full sensor resolution.

### `settings.yaml` — server fallbacks

```yaml
server:
  host: "0.0.0.0"
  port: 8000
model:
  path: "models/best.pt"
database:
  path: "data/results.db"
storage:
  captures_dir:   "app/web/static/captures"
  visual_log_dir: "logs/img"
```

### `.env` — secrets (never commit)

```env
API_KEY=your-long-random-secret
SERVER_URL=https://your-tunnel.trycloudflare.com
DATABASE_URL=postgresql://user:pass@localhost:5432/urine_db
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## API Reference

### `POST /analyze`

**Auth:** `X-Auth-Token: <API_KEY>` header required.
**Precondition:** An active tray must be selected via `/trays`.

```json
{
  "status":       "success",
  "session_id":   42,
  "tray_id":      3,
  "tray_name":    "ถาดหน่วยที่ 1",
  "count":        12,
  "total_physical_count": 12,
  "summary":      {"L0": 3, "L1": 4, "L2": 3, "L3": 2, "L4": 0},
  "is_clean":     false,
  "error_count":  2,
  "errors":       {"duplicate_slots": [31, 36], "wrong_color_slots": ["R01C05"]},
  "slots":        [{"position_index": 1, "color_result": 0, "is_error": false, "error_reason": null}, "..."],
  "timestamp":    "2026-04-21T10:30:00.000000",
  "image_id":     "uuid-string"
}
```

### `GET /api/latest`

Returns the most recent session with tray dimensions for the dashboard grid:
```json
{
  "tray":    {"id": 3, "tray_name": "...", "is_active": true, "rows": 13, "cols": 15},
  "session": {"id": 42, "scanned_at": "...", "color_0": 3, "...", "error_count": 2},
  "slots":   [{"position_index": 1, "color_result": 0, "is_error": false,
               "error_reason": null, "slot_label": "A01"}, "..."]
}
```

`error_reason` values: `null` (no error) · `"สีผิด"` (wrong colour zone) · `"วางเลขซ้ำ"` (duplicate label)

### Tray API

| Method | URL | Body / Notes |
|--------|-----|--------------|
| `GET` | `/api/trays` | List all trays |
| `POST` | `/api/trays` | `{tray_name, rows, cols}` |
| `PATCH` | `/api/trays/{id}` | `{tray_name?, rows?, cols?}` |
| `DELETE` | `/api/trays/{id}` | HTTP 409 if tray is active |
| `POST` | `/api/trays/{id}/activate` | Deactivates all others atomically |

### Other

| URL | Description |
|-----|-------------|
| `GET /health` | `{"status": "ok"}` |
| `GET /dashboard` | Interactive grid dashboard — auto-refreshes every 10 s |
| `GET /trays` | Tray management page |
| `GET /trend` | Historical chart |
| `GET /api/trend?from_date=&to_date=` | JSON trend data (Thai Buddhist dates) |
| `GET /api/sessions?tray_id=&from_date=` | Session history |
| `GET /settings` | Calibration UI |
| `POST /api/test-upload` | Full pipeline — no DB write, no Telegram |

---

## Database Schema (PostgreSQL)

```
trays            — one row per physical tray
  id, tray_name, is_active, rows, cols, total_slots, layout_json, grid_json, created_at
  grid_json: {calibration_date, corners, grid_pts[14][17][2], sample_centres×195, ref_centres×15, grid_spacing}

scan_sessions    — one row per scan event
  id, tray_id, scanned_at, color_0…color_4, error_count, is_clean,
  image_raw_path, image_annotated_path

test_slots       — rows×cols rows per session (195 for 13×15 tray)
  id, session_id, position_index, color_result (0-4 or null),
  is_error, error_reason ("สีผิด" | "วางเลขซ้ำ" | null), person_id

people           — personnel registry (for future identity linking)
  id, full_name, personnel_id, department
```

The server creates tables automatically on first startup (`Base.metadata.create_all`).
New columns on existing tables are added via idempotent `ALTER TABLE … ADD COLUMN` in `_run_migrations()`.

**SQLite fallback:** if `DATABASE_URL` is unset or starts with `sqlite://`, the server uses a local file at `logs/scan_results.db`. All features work identically.

---

## Image Processing Pipeline

```
Pi: 4608 × 2592 raw capture
        │
        ▼  CameraUndistorter.undistort_frame()
        Lens-corrected, focus-locked JPEG (no black borders)
        │
        ▼  POST /analyze
Server: cv2.imdecode() → full BGR
        │
        ▼  Fetch active tray (HTTP 400 if none)
        │
        ▼  crop_sample_roi(top=220, bottom=20, left=200, right=800)
        ROI — excludes reference row + dead-zone column ZZ
        │
        ▼  letterbox_white_padding(640)   [white fill, not black]
        640 × 640 padded image
        │
        ▼  Load grid from active tray grid_json (DB)
        _grid_pts_to_result() → 195 sample centres + 15 ref centres
        (HTTP 400 if tray is uncalibrated)
        │
        ▼  build_reference_baseline() on ref row
        Initial CIE Lab centroid per level (L0–L4) from reference row bottles
        │
        ▼  YOLO inference → scale_coordinates()
        Bottle boxes in full-image space
        │
        ▼  _match_detections_to_slots()
        Each box → nearest position_index (1–195)
        │
        ▼  _refine_baseline_from_slots()
        Median Lab per column zone (L0–L4) from actual sample-area bottles
        Corrects for lighting difference between reference row and sample area
        │
        ▼  CIE Lab classification via refined baseline
        level 0-4 per occupied slot (nearest centroid, CIE76 Delta E)
        │
        ▼  _build_slot_rows() with layout_map
        195 slot dicts: {position_index, color_result, is_error, error_reason}
        is_error + error_reason set for wrong colour zone OR duplicate label
        │
        ▼  _render_annotated_canvas() → save_visual_log()
        Annotated JPEG to logs/img/ + Telegram
        │
        ▼  INSERT ScanSession + 195 TestSlot rows → PostgreSQL
        return JSON
```

---

## Color Classification

Classification is **nearest-centroid in CIE Lab** (CIE76 Delta E), not iterative K-means.

### Two-phase baseline

1. **Reference row** (`build_reference_baseline`): 15 reference bottles at the top of the tray (3 per level) are sampled live each scan to build an initial {level → (L, a, b)} baseline.

2. **Sample-area refinement** (`_refine_baseline_from_slots`): All detected sample bottles are grouped by their expected column zone. The median Lab of each zone overrides the corresponding reference centroid. This corrects for systematic lighting differences between the reference row position and the sample area.

### Annular ring sampling

`extract_bottle_color` samples from a **circular ring** rather than a simple square crop:

```
outer_crop_px = 15  →  excludes plastic cap edge
inner_crop_px = 10  →  excludes center glare hot-spot
ring = pixels where  inner_crop_px ≤ dist_from_center ≤ (radius − outer_crop_px)
```

Median Lab is taken over non-glare ring pixels (L < 220 in OpenCV 8-bit encoding).
Both reference and sample bottles use the same ring parameters.

### Error detection

| Error | Condition | `error_reason` |
|-------|-----------|----------------|
| Wrong colour | `classified_level ≠ expected_level_for_column` | `"สีผิด"` |
| Duplicate label | Same `layout_json` label on 2+ occupied slots | `"วางเลขซ้ำ"` |

Duplicate takes precedence when both conditions apply.

---

## Tray Management Workflow

1. Navigate to `http://<server>:8000/trays`
2. Create a new tray — set **name**, **rows** (default 13), **cols** (default 15)
3. Click **เลือกใช้งาน** (Activate) on the desired tray
4. The active tray banner shows which tray is currently selected
5. All subsequent scans are linked to the active tray

The `/dashboard` grid automatically resizes to the active tray's `rows × cols` dimensions.

---

## Grid Calibration

1. Navigate to `http://<server>:8000/settings`
2. Load the latest capture or upload a tray photo
3. Click 4 outer corners: **TL → TR → BR → BL** (in order)
4. Optionally drag grid lines to fine-tune alignment
5. Click **บันทึกกริด** → server saves 195 slot centres + 15 ref centres to `trays.grid_json` in DB

> The calibration must be done once per tray before scanning. `/analyze` returns HTTP 400 if the active tray is uncalibrated.

---

## Camera Lens Calibration (Pi)

Calibrate once using a checkerboard at full sensor resolution (4608×2592):

```python
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, (4608, 2592), None, None
)
```

Save the values to `configs/camera_params.yaml`. The `CameraUndistorter` scales them automatically when a different capture resolution is used at runtime.

---

## Pi Client State Machine

### Relay (LED Tower)

| State | Trigger | Relay |
|-------|---------|-------|
| Idle / Ready | Boot + clean scan | **Green** ON |
| Processing | Button press → throughout upload + wait | **Yellow** ON |
| Error | Camera / network / timeout / wrong-zone / duplicate errors | **Red** ON (until next scan) |

### LCD + TM1637

| State | LCD Line 1 | LCD Line 2 | TM1637 |
|-------|-----------|-----------|--------|
| Startup | `Urine Analyzer` | `Ready` | `0000` |
| Capturing | `Capturing...` | — | `8888` |
| Uploading | `Uploading...` | `Please wait` | spinning |
| Analyzing | `Analyzing...` | `AI running` | spinning |
| Result OK | `Count: N` | colour summary | `000N` |
| Error slots | `Count: N` | `ERR @ R01C05  R03C12` (scrolling) | `000N` |
| Server offline | `ERR: SRV OFF` | `Check network` | `----` |
| Timeout | `ERR: TIMEOUT` | `Server slow` | `----` |

---

## Cloudflare Tunnel (HTTPS)

```bash
# On Ubuntu server
cloudflared tunnel --url http://localhost:8000
# Prints: https://xxxx.trycloudflare.com

# Set in .env on Pi:
SERVER_URL=https://xxxx.trycloudflare.com
```

Do **not** include a port number in `SERVER_URL` when using Cloudflare Tunnel.

---

## Local Development (no Pi hardware)

```bash
uv run main.py --role server

# Upload via the test page (no API key):
# http://localhost:8000/test

# Or via curl (API key required):
curl -X POST http://localhost:8000/analyze \
     -H "X-Auth-Token: changeme" \
     -F "file=@data/test.jpg"
```

---

## Running Tests

```bash
uv run pytest
uv run pytest -v
```

---

## .gitignore Recommendations

```gitignore
.env
data/
models/
logs/
app/web/static/captures/
app/web/static/test/
color.json
grid_config.json
*.pyc
__pycache__/
.venv/
```
