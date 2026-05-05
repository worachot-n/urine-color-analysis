# Urine Color Analysis System v3

A distributed vision system for automated urine sample analysis using AI.
A **Raspberry Pi 4B** captures and undistorts images, controls status hardware, and POSTs to a server.
A **Ubuntu server** runs YOLO inference, performs CIE Lab color classification, stores annotated images in Google Drive, appends results to Google Sheets, and hosts the web dashboard.

---

## Architecture

```
Raspberry Pi 4B                            Ubuntu Server
────────────────────────────────           ─────────────────────────────────────────
Button press (GPIO 24)                     POST /analyze
  → Relay Yellow ON (processing)             → crop_sample_roi (skip dead zones)
  → picamera2 capture (4608 × 2592)          → letterbox to 640 × 640 (white fill)
  → CameraUndistorter.undistort_frame()      → detect_grid (HoughCircles + RANSAC)
  → HTTPS POST + X-Auth-Token  ─────────►   → YOLO: bottle detection → slot matching
  → TM1637 spinner animation               → Live CIE Lab baseline from ref cells
  → LCD "Analyzing..."                       → classify_sample (nearest-centroid ΔE)
  → parse JSON response        ◄─────────   → upload annotated image → Google Drive
  → update LCD + TM1637 count              → append row → Google Sheets
  → scroll missing slots on LCD            → return JSON
  → Relay Green (ok) / Red (missing)
```

- Pi performs lens undistortion locally before uploading
- Grid is detected automatically each scan via classical CV — no calibration step
- Slot assignment is configured once via `/settings` and persisted to `server/slot_config.json`
- Color reference is extracted live from reference cells every scan — no static `color.json`
- Results (annotated JPEG + JSON) stored permanently in `server/static/results/`

---

## Repository Layout

```
urine-color-analysis/
│
├── client/
│   └── pi_client.py              # Pi loop — GPIO, camera, undistort, relay, LCD, TM1637, POST
│
├── server/
│   ├── api.py                    # FastAPI app — all 8 routes
│   ├── pipeline.py               # run_pipeline() + lab_to_hex() — shared by all callers
│   ├── slot_config.py            # SlotConfig dataclass + load/save/query helpers
│   ├── slot_config.json          # Persisted slot assignment (user-edited via /settings)
│   ├── integrations/
│   │   ├── drive.py              # upload_image() → Google Drive
│   │   └── sheets.py             # write_slot_config_to_sheet(), append_result_to_sheet()
│   ├── templates/
│   │   ├── base.html             # Shared layout (nav: Upload | Results | Settings)
│   │   ├── upload.html           # Drag-drop form → fetch /analyze → redirect to /result/<id>
│   │   ├── results.html          # History table (all scans, newest first)
│   │   ├── result.html           # Detail: annotated image + slot table + scatter plot + JSON
│   │   └── settings.html         # Interactive grid editor
│   └── static/
│       ├── results/              # Persistent annotated JPEGs + JSON (<scan_id>.{jpg,json})
│       └── js/
│           └── settings.js       # Grid rendering, popup state, API calls
│
├── utils/
│   ├── grid_circle_detector.py   # detect_grid() — HoughCircles + spacing + RANSAC
│   ├── color_analysis.py         # extract_bottle_color(), build_reference_baseline(), classify_sample()
│   └── camera_undistort.py       # CameraUndistorter — lens correction + focus lock
│
├── app/
│   └── shared/
│       └── processor.py          # crop_sample_roi(), letterbox_white_padding(), scale_coordinates()
│
├── configs/
│   ├── config.toml               # Hardware config — GPIO, camera, YOLO, color, google
│   ├── camera_params.yaml        # Lens calibration — focus_value, K, dist_coeffs
│   └── config.py                 # TOML loader
│
├── credentials/
│   ├── client_secrets.json       # OAuth2 client ID + secret (git-ignored)
│   └── token.json                # OAuth2 token, auto-written after first auth (git-ignored)
│
├── models/
│   └── best.pt                   # YOLO weights (git-ignored)
│
├── pyproject.toml                # uv project — extras: common / pi / server
└── logs/
    └── app_*.log                 # Rotating text logs
```

---

## Quick Start

### 1 — Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2 — Configure secrets

```bash
cp .env.template .env
# Edit .env:
#   API_KEY=your-long-random-secret
#   SERVER_URL=https://your-tunnel.trycloudflare.com
```

### 3 — Ubuntu server

```bash
uv sync --extra server --extra common
uv run uvicorn server.api:app --reload

# Upload page  → http://localhost:8000/upload
# History      → http://localhost:8000/results
# Settings     → http://localhost:8000/settings
# Health       → http://localhost:8000/health
```

> **Before the first scan:** go to `/settings`, assign slot IDs to cells, and click **Save**.
> `/analyze` returns HTTP 400 if no slots are configured.

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
| `server` | `fastapi`, `uvicorn`, `opencv-python`, `ultralytics`, `torch`, `torchvision`, `scikit-image`, `scipy`, `jinja2`, `python-multipart`, `google-api-python-client`, `google-auth-httplib2`, `google-auth-oauthlib` |
| `pi` | `picamera2`, `RPi.GPIO`, `rpi-lcd`, `raspberrypi-tm1637`, `smbus2` |

---

## Configuration

### `configs/config.toml`

```toml
[camera]
capture_resolution = [4608, 2592]
rotate_180         = true

[yolo]
model_path     = "models/best.pt"
conf_threshold = 0.20
iou_threshold  = 0.45

[color_analysis]
outer_crop_px     = 15    # pixels to exclude from bottle edge (plastic ring)
inner_crop_px     = 10    # pixels from bottle center to exclude (glare hot-spot)
confidence_margin = 3.0   # second_best_ΔE − best_ΔE required for "Confident"

[sample_roi]
top    = 220   # skip reference row
bottom = 20
left   = 200   # skip dead-zone column
right  = 800

[google]
credentials_file = "client_secrets.json"
token_file       = "token.json"
drive_folder_id  = ""          # Google Drive folder ID for annotated images
spreadsheet_id   = ""          # Single Google Spreadsheet ID (two tabs inside)
slots_tab        = "SlotAssignment"
results_tab      = "Results"
```

### `configs/camera_params.yaml` — lens calibration

```yaml
metadata:
  focus_value: 1.0
  sensor_resolution: {width: 4608, height: 2592}
calibration:
  camera_matrix: {fx: 3254.7, fy: 3254.7, cx: 2304.0, cy: 1296.0}
  dist_coeffs: [-0.1234, 0.0567, -0.0001, 0.0002, -0.0123]
```

Replace placeholder values with the output of `cv2.calibrateCamera()` at full sensor resolution.

### `.env` — secrets (never commit)

```env
API_KEY=your-long-random-secret
SERVER_URL=https://your-tunnel.trycloudflare.com
```

---

## Slot Configuration

Not all grid cells are used. The `/settings` page lets you assign a `slot_id` label to each cell you want to analyze. The pipeline ignores all unassigned cells.

```json
{
  "rows": 13,
  "cols": 15,
  "cells": {
    "1":  { "slot_id": "REF_L0", "is_reference": true,  "ref_level": 0 },
    "2":  { "slot_id": "REF_L1", "is_reference": true,  "ref_level": 1 },
    "3":  { "slot_id": "REF_L2", "is_reference": true,  "ref_level": 2 },
    "4":  { "slot_id": "REF_L3", "is_reference": true,  "ref_level": 3 },
    "5":  { "slot_id": "REF_L4", "is_reference": true,  "ref_level": 4 },
    "16": { "slot_id": "A01",    "is_reference": false, "ref_level": null }
  }
}
```

- Key = **1-based row-major** cell index: `cell_index = (row - 1) * cols + col`
- `is_reference: true` → used for dynamic color baseline; `ref_level` 0–4 required
- `is_reference: false` → sample bottle to classify

---

## API Reference

### `POST /analyze`

**Auth:** `X-Auth-Token: <API_KEY>` header required.
**Precondition:** At least one slot must be assigned via `/settings`.

```json
{
  "status": "success",
  "scan_id": "a1b2c3d4",
  "detected_count": 20,
  "total_assigned": 25,
  "missing_slots": ["A03", "B07"],
  "summary": {"L0": 5, "L1": 8, "L2": 4, "L3": 3, "L4": 0},
  "reference_labs": {
    "0": [{"lab": [210.0, 126.0, 134.0], "hex": "#f5e8c0"}],
    "1": [{"lab": [185.0, 131.0, 148.0], "hex": "#dfc07a"}]
  },
  "slots": {
    "A01": {"cell_index": 16, "detected": true, "color_level": 1,
            "delta_e": 3.2, "confident": true, "lab": [188.0, 130.0, 147.0], "hex": "#e2c47e"},
    "A03": {"cell_index": 18, "detected": false, "color_level": null,
            "delta_e": null, "confident": false, "lab": null, "hex": null}
  },
  "image_url": "/static/results/a1b2c3d4.jpg",
  "timestamp": "2026-05-05T10:30:00Z"
}
```

`/analyze` always returns JSON. The browser upload page uses `fetch()`, receives JSON, then does `window.location.href = '/result/' + data.scan_id`. The Pi receives the same JSON directly.

### Other Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/upload` | Upload form page |
| `GET` | `/results` | Scan history (newest first) |
| `GET` | `/result/{scan_id}` | Scan detail page |
| `GET` | `/settings` | Slot assignment page |
| `GET` | `/api/slots` | Return `slot_config.json` as JSON |
| `POST` | `/api/slots` | Save slot configuration |
| `GET` | `/health` | `{"status": "ok"}` |

---

## Image Processing Pipeline

```
Pi: 4608 × 2592 raw capture
        │
        ▼  CameraUndistorter.undistort_frame()
        Lens-corrected JPEG
        │
        ▼  POST /analyze
Server: cv2.imdecode() → full BGR
        │
        ▼  crop_sample_roi(top=220, bottom=20, left=200, right=800)
        ROI — excludes reference row + dead-zone column
        │
        ▼  letterbox_white_padding(640)   [white fill, NOT black]
        640 × 640 padded image
        │
        ▼  detect_grid(padded, rows, cols)   [HoughCircles + RANSAC]
        grid_pts in 640-space → convert to full-image coords
        │
        ▼  YOLO inference on padded image → scale to full-image coords
        │
        ▼  assign_bottles_to_slots(yolo_boxes, grid_pts_full, active_cells)
        │
        ▼  build_reference_baseline(frame_full, ref_positions)
        Dynamic per-frame CIE Lab centroid per level (L0–L4)
        │
        ▼  classify_all_samples()
        extract_bottle_color() (annular ring median) → classify_sample() (nearest ΔE)
        lab_to_hex() → "#rrggbb" for each bottle
        │
        ▼  render_annotated_image()   [green circles = detected, red = missing]
        │
        ┌─── CLOUD (fire-and-forget) ──────────────────┐
        │  upload_image() → Google Drive               │
        │  append_result_to_sheet() → Google Sheets    │
        └──────────────────────────────────────────────┘
        │
        ▼  save scan_result.json + annotated.jpg → server/static/results/
        ▼  return JSON
```

---

## Color Classification

Classification uses **nearest-centroid in CIE Lab** (CIE76 Delta E). Confidence is **margin-of-victory** (`second_best_ΔE − best_ΔE > margin`), not a fixed threshold.

### Annular ring sampling

`extract_bottle_color` samples the **median Lab** over an annular ring:

```
outer_crop_px = 15  →  excludes plastic cap edge
inner_crop_px = 10  →  excludes center glare hot-spot
ring = pixels where  inner_crop_px ≤ dist_from_center ≤ (radius − outer_crop_px)
```

Both reference and sample bottles use the same ring parameters. `cx`/`cy` must be in full-image coordinates.

### Dynamic reference baseline

`build_reference_baseline()` extracts Lab from every reference cell each scan and averages within each level group. No stored `color.json` — the baseline adapts to current lighting automatically.

---

## Google Sheets Integration

Single spreadsheet (`spreadsheet_id`) with two tabs:

**"SlotAssignment" tab** — cleared and rewritten whenever `POST /api/slots` saves the config.
Columns: `cell_index | slot_id | is_reference | ref_level | row | col`

**"Results" tab** — one batch appended per successful `/analyze` call. Never overwritten.

Summary row: `scan_id | timestamp | detected_count | total_assigned | missing_slots | L0 | L1 | L2 | L3 | L4`

Detail row (one per assigned cell): `scan_id | slot_id | cell_index | is_reference | ref_level | detected | color_level | delta_e | confident | L | a | b | hex`

The `hex` column cell background is set to the detected color via `userEnteredFormat.backgroundColor` — a visual color swatch directly in the spreadsheet.

### Google OAuth2 First-Time Setup

```toml
# configs/config.toml
[google]
credentials_file = "client_secrets.json"   # download from Google Cloud Console
token_file       = "token.json"
drive_folder_id  = "..."
spreadsheet_id   = "..."
```

On first use, the server opens a browser consent URL and writes `token.json` automatically. Subsequent runs auto-refresh. Never commit `token.json` or `client_secrets.json`.

Scopes: `https://www.googleapis.com/auth/drive.file` + `https://www.googleapis.com/auth/spreadsheets`

---

## Result Detail Page

Three panels at `/result/{scan_id}`:

1. **Annotated image** — green circles on detected bottles, red on missing assigned cells, slot_id labels
2. **Slot results table** — color swatch, level badge, ΔE, confidence per slot
3. **Color matching scatter plot** (Chart.js) — CIE Lab a*/b* space; reference cells as large markers per level, detected bottles as small labeled markers with tooltips (slot_id, Lab, ΔE); shows visually why each bottle was classified at its level
4. **Raw JSON** — collapsible `<details><pre>` block

---

## Pi Client

The Pi does exactly three things: capture, send, display.

### LCD 16×4 Display States

| State | Line 1 | Line 2 | Line 3 | Line 4 |
|---|---|---|---|---|
| Boot | `Urine Analyzer ` | `Ready          ` | ` ` | ` ` |
| Capturing | `Capturing...   ` | ` ` | ` ` | ` ` |
| Uploading | `Uploading...   ` | `Please wait    ` | ` ` | ` ` |
| Analyzing | `Analyzing...   ` | `AI running     ` | ` ` | ` ` |
| OK result | `Count: NN/NNN  ` | `L0:N L1:N L2:N ` | `L3:N L4:N      ` | `Status: OK     ` |
| Missing slots | `Count: NN/NNN  ` | `L0:N L1:N L2:N ` | `L3:N L4:N      ` | `Miss: A03 B07..` |
| Network error | `ERR: SRV OFF   ` | `Check network  ` | ` ` | ` ` |
| Timeout | `ERR: TIMEOUT   ` | `Server slow    ` | ` ` | ` ` |

Line 4 scrolls slot_ids for missing bottles at 300 ms/step in a daemon thread.

### TM1637 States

| State | Display |
|---|---|
| Boot | `0000` |
| Capturing | `8888` |
| Upload / analyze | Spinning `- - - -` at 250 ms |
| Result | `NNNN` (detected count) |
| Error | `----` |

### Relay LED Tower

```
Boot              → Green ON
Button pressed    → Yellow ON  (until result received)
All slots found   → Green ON
Any missing slot  → Red ON
Any error         → Red ON
```

Active-LOW relay: `GPIO.LOW` energises. Turn off previous channel before turning new one on (50 ms gap).

---

## Cloudflare Tunnel (HTTPS)

```bash
# On Ubuntu server
cloudflared tunnel --url http://localhost:8000
# Prints: https://xxxx.trycloudflare.com

# Set in .env on Pi:
SERVER_URL=https://xxxx.trycloudflare.com
```

---

## Local Development

```bash
uv run uvicorn server.api:app --reload

# Upload via the web page:
open http://localhost:8000/upload

# Or via curl:
curl -X POST http://localhost:8000/analyze \
     -H "X-Auth-Token: test" \
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
models/
logs/
credentials/
server/static/results/
*.pyc
__pycache__/
.venv/
```
