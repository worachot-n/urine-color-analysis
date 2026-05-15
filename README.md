# Urine Color Analysis System v3

A distributed vision system for automated urine sample analysis using AI.
A **Raspberry Pi 4B** captures and undistorts images, controls status hardware, and POSTs to a server.
A **Ubuntu server** runs YOLO inference, performs hybrid CIE Lab + histogram colour classification, stores results in SQLite, syncs to Google Sheets, and hosts the web dashboard.

---

## Architecture

```
Raspberry Pi 4B                            Ubuntu Server
────────────────────────────────           ─────────────────────────────────────────
Button press (GPIO 24)                     POST /analyze
  → Relay Yellow ON (processing)             → detect_grid_full (HoughCircles + KDE + RANSAC)
  → picamera2 capture (4608 × 2592)          → YOLO: bottle detection → slot matching
  → CameraUndistorter.undistort_frame()      → White-balance offset from WB reference cells
  → HTTPS POST + X-Auth-Token  ─────────►   → Live CIE Lab reference set from ref cells
  → TM1637 H0-H4 spinner animation           → classify_sample_path (hybrid chroma + histogram)
  → LCD "Analyzing..."                       → save to SQLite
  → parse JSON response        ◄─────────   → upload annotated image → Google Drive (async)
  → LCD: OK/Recheck + Total count            → sync rows → Google Sheets (async)
  → LCD line 3: scroll missing slot IDs      → Telegram notification (async)
  → LCD line 4: scroll result URL            → return JSON
  → TM1637 H0-H4: L0–L4 counts
  → Relay Green (ok) / Red (missing)
```

- Pi performs lens undistortion locally using calibrated `camera_params.yaml` before uploading
- Grid is detected automatically each scan via classical CV — no calibration step
- Slot assignment is configured once via `/settings` and persisted to `server/slot_config.json`
- White-reference cells enable per-scan chromatic adaptation (WB offset)
- Color reference extracted live from reference cells every scan — no static color table
- Results stored permanently in SQLite (`logs/urine_analysis.db`) and synced to Google Sheets
- Annotated images saved to `logs/img/` and served at `/img/<scan_id>.jpg`

---

## Repository Layout

```
urine-color-analysis/
│
├── main.py                           # Entry point: --role {server|client}
│
├── app/
│   ├── client_app.py                 # Pi loop — GPIO, camera, undistort, relay, LCD, TM1637, POST
│   ├── server_app.py                 # run_server() — launches FastAPI via uvicorn
│   └── shared/
│       ├── config.py                 # Unified config: TOML + settings.yaml + .env + env vars
│       └── processor.py              # letterbox_white_padding(), crop_sample_roi()
│
├── server/
│   ├── api.py                        # FastAPI app — all routes
│   ├── pipeline.py                   # run_pipeline() + lab_to_hex()
│   ├── slot_config.py                # SlotConfig + load/save/query helpers
│   ├── slot_config.json              # Persisted slot assignment
│   ├── integrations/
│   │   ├── drive.py                  # upload_image() → Google Drive
│   │   ├── sheets.py                 # append_detail/summary_to_sheet(), write_slot_config_to_sheet()
│   │   └── sqlite_backup.py          # SQLite offline store — primary result database
│   ├── templates/                    # Jinja2 templates (upload, results, result, settings, auto_grid)
│   └── static/                       # Tailwind CSS, Chart.js, NotoSansThai font (all local)
│
├── bot/
│   └── telegram_bot.py               # send_scan_report() — fire-and-forget; skipped if token blank
│
├── utils/
│   ├── auto_grid_detector.py         # detect_grid_full() — HoughCircles + KDE + RANSAC (used by pipeline)
│   ├── color_analysis.py             # extract_bottle_color(), build_reference_set(), classify_sample_path()
│   └── camera_undistort.py           # CameraUndistorter — lens correction + focus lock
│
├── scripts/
│   ├── setup-server.sh               # One-command Ubuntu install + systemd service
│   ├── setup-pi.sh                   # One-command Pi install + systemd service
│   ├── urine-server.service          # systemd unit template (server)
│   └── urine-pi.service              # systemd unit template (Pi client)
│
├── configs/
│   ├── config.toml                   # Hardware config — GPIO, camera, YOLO, color, google
│   ├── camera_params.yaml            # Lens calibration — focus_value, K, dist_coeffs
│   └── config.py                     # TOML loader
│
├── settings.yaml                     # Server/model/storage defaults
├── credentials/
│   └── credentials.json              # Service account key JSON (git-ignored)
├── models/
│   └── best.pt                       # YOLO weights (git-ignored)
├── tests/                            # pytest test suite
└── logs/
    ├── app_*.log                     # Rotating text logs
    ├── img/                          # Annotated JPEGs (served at /img/<scan_id>.jpg)
    └── urine_analysis.db             # SQLite scan store
```

---

## Quick Start

### Option A — Automated setup (recommended)

```bash
# Ubuntu server
bash scripts/setup-server.sh

# Raspberry Pi
bash scripts/setup-pi.sh
```

Each script installs dependencies, prompts for `.env` values, and installs a systemd service that starts on boot.

### Option B — Manual setup

#### 1 — Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2 — Configure secrets

```bash
cp .env.template .env
# Edit .env:
#   API_KEY=your-long-random-secret          # same value on server and Pi
#   SERVER_URL=https://your-tunnel.trycloudflare.com   # Pi only
```

#### 3 — Ubuntu server

```bash
uv sync --extra server --extra common
uv run main.py --role server

# Web UI
# Upload   → http://localhost:8000/upload
# History  → http://localhost:8000/results
# Settings → http://localhost:8000/settings
# Grid debug → http://localhost:8000/auto_grid
# Health   → http://localhost:8000/health
```

> **Before the first scan:** go to `/settings`, assign slot IDs to cells, mark reference cells, optionally mark white-reference cells, and click **Save**.
> `/analyze` returns HTTP 400 if no slots are configured.

#### 4 — Raspberry Pi

```bash
uv sync --extra pi --extra common
uv run main.py --role client
```

---

## Dependencies

| Extra | Key packages |
|-------|-------------|
| `common` | `requests`, `pydantic-settings`, `loguru`, `numpy`, `pyyaml`, `python-dotenv`, `Pillow`, `psutil` |
| `server` | `fastapi`, `uvicorn`, `opencv-python`, `ultralytics`, `torch`, `torchvision`, `scikit-image`, `scipy`, `jinja2`, `python-multipart`, `google-api-python-client`, `google-auth-httplib2` |
| `pi` | `picamera2`, `RPi.GPIO`, `rpi-lcd`, `raspberrypi-tm1637`, `smbus2` |

---

## Configuration

### `configs/config.toml`

```toml
[camera]
capture_resolution = [4608, 2592]
rotate_180         = true
awb_lock           = true
ae_lock            = true
jpeg_quality       = 95

[yolo]
model_path     = "models/best.pt"
conf_threshold = 0.20
iou_threshold  = 0.45

[color_analysis]
outer_crop_px             = 15      # exclude plastic ring at bottle edge
inner_crop_px             = 10      # exclude glare hot-spot at center
confidence_margin         = 3.0     # second_best_ΔE − best_ΔE for "Confident"
weight_chroma             = 0.5     # hybrid: chroma ΔE weight
weight_hist               = 0.5     # hybrid: histogram Bhattacharyya weight
max_path_distance         = 12.0    # beyond this → use nearest-level fallback
path_confident_distance   = 4.0     # within this → confident

[white_balance]
enabled = true

[google]
service_account_file = "credentials/credentials.json"
drive_folder_id      = ""          # Google Drive folder ID
spreadsheet_id       = ""          # Google Spreadsheet ID (three tabs inside)
slots_tab            = "SlotAssignment"
detail_tab           = "Detail"
summary_tab          = "Summary"

[sqlite]
db_path = "logs/urine_analysis.db"
```

### `configs/camera_params.yaml` — lens calibration

```yaml
metadata:
  focus_value: 1.0          # ~100 cm working distance (Pi Cam V3 diopters)
  sensor_resolution: {width: 4608, height: 2592}
calibration:
  camera_matrix: {fx: 3254.7, fy: 3254.7, cx: 2304.0, cy: 1296.0}
  dist_coeffs: [-0.1234, 0.0567, -0.0001, 0.0002, -0.0123]
```

Replace placeholder values with the output of `cv2.calibrateCamera()` at full sensor resolution.

### `.env` — secrets (never commit)

```env
# Both machines
API_KEY=your-long-random-secret

# Pi only
SERVER_URL=https://your-tunnel.trycloudflare.com
```

---

## Slot Configuration

Not all grid cells are used. The `/settings` page lets you assign a `slot_id` to each cell and declare its type:

```json
{
  "rows": 13,
  "cols": 15,
  "cells": {
    "1":  { "slot_id": "REF_L0", "is_reference": true,  "ref_level": 0, "is_white_reference": false },
    "2":  { "slot_id": "REF_L1", "is_reference": true,  "ref_level": 1, "is_white_reference": false },
    "6":  { "slot_id": "WB1",    "is_reference": false, "ref_level": null, "is_white_reference": true },
    "16": { "slot_id": "A01",    "is_reference": false, "ref_level": null, "is_white_reference": false }
  }
}
```

- Key = **1-based row-major** cell index: `cell_index = (row - 1) * cols + col`
- `is_reference: true` → used for dynamic CIE Lab baseline; `ref_level` 0–4 required
- `is_white_reference: true` → neutral white/grey patch for chromatic adaptation
- `is_reference` and `is_white_reference` are mutually exclusive
- `is_reference: false, is_white_reference: false` → sample bottle to classify

---

## API Reference

### `POST /analyze`

**Auth:** `X-Auth-Token: <API_KEY>` header required.
**Precondition:** At least one slot must be assigned via `/settings`.
**Field:** `file` (multipart/form-data, JPEG or PNG).

```json
{
  "status": "success",
  "scan_id": "a1b2c3d4",
  "detected_count": 20,
  "total_assigned": 25,
  "missing_slots": ["A03", "B07"],
  "summary": {"L0": 5, "L1": 8, "L2": 4, "L3": 3, "L4": 0},
  "reference_labs": {
    "0": [{"lab": [210.0, 126.0, 134.0], "hex": "#f5e8c0", "cell_index": 1}]
  },
  "slots": {
    "A01": {
      "cell_index": 16, "detected": true, "color_level": 1,
      "concentration_index": 0.25, "path_distance": 3.1,
      "delta_e": 3.2, "hist_bhatt": 0.12, "combined": 0.43,
      "confident": true, "best_fit": false, "force_l0": false, "soft_penalty": false,
      "lab": [188.0, 130.0, 147.0], "hex": "#e2c47e"
    }
  },
  "master_path": {"levels": [0,1,2,3,4], "points": [[...], ...]},
  "wb_offset": [2.1, -1.3],
  "image_url": "/img/a1b2c3d4.jpg",
  "timestamp": "2026-05-05T10:30:00Z"
}
```

The browser upload page redirects to `/result/<scan_id>` after receiving the JSON.
The Pi parses the same JSON to drive the LCD and relay.

### Other Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/upload` | Upload form page |
| `GET` | `/results` | Scan history (newest first, from SQLite) |
| `GET` | `/result/{scan_id}` | Scan detail page |
| `GET` | `/settings` | Slot assignment page |
| `GET` | `/auto_grid` | Grid CV debug page |
| `GET` | `/api/slots` | Return `slot_config.json` as JSON |
| `POST` | `/api/slots` | Save slot configuration |
| `POST` | `/api/sync` | Trigger Google Sheets sync of pending offline scans |
| `POST` | `/api/auto_grid` | Run grid reconstruction on uploaded image |
| `GET` | `/img/{scan_id}.jpg` | Annotated JPEG from `logs/img/` |
| `GET` | `/health` | `{"status": "ok"}` |

---

## Image Processing Pipeline

```
Pi: 4608 × 2592 raw capture
        │
        ▼  lock_focus_picamera2() — calibrated lens position
        ▼  CameraUndistorter.undistort_frame() — barrel distortion removed
        Lens-corrected JPEG
        │
        ▼  POST /analyze
Server: cv2.imdecode() → full BGR
        │
        ▼  detect_grid_full(rows, cols)   [HoughCircles + KDE + RANSAC]
        grid_pts (rows×cols, 2) in full-image coords + avg_radius_px
        │
        ▼  letterbox_white_padding(640)   [white fill, NOT black]
        640 × 640 padded image → YOLO inference → scale to full-image coords
        │
        ▼  assign_bottles_to_slots(yolo_boxes, grid_pts, active_cells)
        │
        ▼  compute_white_balance_offset(frame, wb_positions)
        [Δa*, Δb*] chromatic adaptation offset
        │
        ▼  build_reference_set + build_reference_histograms
        Per-level Lab lists + histogram templates (after outlier filter)
        │
        ▼  build_master_path(level_centroids)
        Continuous L0→L4 color path
        │
        ▼  classify_sample_path() per detected bottle
        achromatic check → force_l0 short-circuit
        OR project onto master path → hybrid score → level + confidence
        │
        ▼  render_annotated_image()  [≤1280px, coloured circles, Thai-aware labels]
        │
        ┌─── ASYNC (fire-and-forget) ──────────────────────────────────────────┐
        │  save_scan()              → SQLite logs/urine_analysis.db           │
        │  upload_image()           → Google Drive                            │
        │  append_detail/summary()  → Google Sheets (Detail + Summary tabs)   │
        │  send_scan_report()       → Telegram                                │
        └──────────────────────────────────────────────────────────────────────┘
        │
        ▼  save annotated.jpg → logs/img/<scan_id>.jpg
        ▼  return JSON
```

---

## Color Classification

Classification uses a **hybrid scorer**: chroma ΔE distance along a master color path + histogram Bhattacharyya distance to level templates. Confidence is **margin-of-victory**, not a fixed threshold.

### Annular ring sampling

`extract_bottle_color` samples the **median Lab** over an annular ring:

```
outer_crop_px = 15  →  excludes plastic cap edge
inner_crop_px = 10  →  excludes center glare hot-spot
ring = pixels where  inner_crop_px ≤ dist_from_center ≤ (radius − outer_crop_px)
```

Both reference and sample bottles use the same ring parameters.

### Dynamic reference baseline

`build_reference_set()` extracts Lab from every reference cell each scan. No stored colour table — the baseline adapts to current lighting automatically.

### White-balance correction

`compute_white_balance_offset()` measures the Lab of white-reference cells and computes (Δa*, Δb*) chromatic shift. Applied to histogram templates to normalise colour casts before classification.

### Achromatic short-circuit

Near-neutral samples (low chroma) are detected by `is_achromatic()` and force-classified L0 (clear) to prevent hue noise from misclassifying colourless samples as L1/L2.

---

## Google Sheets Integration

Single spreadsheet with three tabs:

**"SlotAssignment"** — cleared and rewritten whenever `POST /api/slots` saves.
Columns: `cell_index | slot_id | is_reference | ref_level | row | col`

**"Detail"** — one row per bottle per scan. Appended, never overwritten.
Columns: `scan_id | slot_id | cell_index | ... | detected | color_level | delta_e | confident | L | a | b | hex`
Cell backgrounds set to detected colour (visual swatch in spreadsheet).

**"Summary"** — one row per scan. Appended, never overwritten.
Columns: `scan_id | timestamp | detected_count | total_assigned | missing_slots | L0 | L1 | L2 | L3 | L4 | image_url`

### Offline resilience

If Google is unreachable, scans are saved to SQLite with `synced=0`. They are retried automatically on next server start, or on demand via `POST /api/sync`.

### Google Service Account Setup

1. Create a Service Account in Google Cloud Console.
2. Download the JSON key → save as `credentials/credentials.json`.
3. Share your Drive folder with the service account email (Contributor).
4. Share your Spreadsheet with the service account email (Editor).
5. Fill in `drive_folder_id` and `spreadsheet_id` in `configs/config.toml`.

No browser step, no `token.json`. Never commit `credentials/credentials.json`.

---

## Pi Client

The Pi does exactly three things: capture (with undistortion), send, display.

### LCD 16×4 Display States

| State | Line 1 | Line 2 | Line 3 | Line 4 |
|---|---|---|---|---|
| Boot | `Urine Analyzer ` | `Ready          ` | ` ` | ` ` |
| Capturing | `Capturing...   ` | ` ` | ` ` | ` ` |
| Uploading | `Uploading...   ` | `Please wait    ` | ` ` | ` ` |
| Analyzing | `Analyzing...   ` | `AI running     ` | ` ` | ` ` |
| OK result | `OK             ` | `Total:NN/NNN   ` | ` ` | `Full report:URL` ← scroll |
| Missing | `Recheck needed ` | `Total:NN/NNN   ` | `Recheck:A03 B07` ← scroll | `Full report:URL` ← scroll |
| Network error | `ERR: SRV OFF   ` | `Check network  ` | ` ` | ` ` |
| Timeout | `ERR: TIMEOUT   ` | `Server slow    ` | ` ` | ` ` |

Lines 3 and 4 scroll at 200 ms/step in daemon threads.

### TM1637 (5 displays: H0–H4)

| State | H0 | H1 | H2 | H3 | H4 |
|---|---|---|---|---|---|
| Boot | `0000` | `0000` | `0000` | `0000` | `0000` |
| Capturing | `8888` | `8888` | `8888` | `8888` | `8888` |
| Upload/analyze | spin | spin | spin | spin | spin |
| Result | L0 count | L1 count | L2 count | L3 count | L4 count |
| Error | `----` | `----` | `----` | `----` | `----` |

### Relay LED Tower

```
Boot              → Green ON
Button pressed    → Yellow ON  (until result received)
All slots found   → Green ON
Any missing slot  → Red ON
Any error         → Red ON
```

Active-LOW relay (`GPIO.LOW` energises). Turn off previous channel before turning new one on (50 ms gap).

---

## Service Management

```bash
systemctl status  urine-server   # or urine-pi
systemctl restart urine-server
journalctl -u urine-server -f    # live logs
```

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
uv run main.py --role server

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
*.pyc
__pycache__/
.venv/
```
