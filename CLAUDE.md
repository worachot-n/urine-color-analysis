# CLAUDE.md — Urine Analysis System v3

Codebase and developer context for AI Code Assistants (Claude, Cursor, Copilot).
Read this file before modifying any file in this repository.

---

## What Changed from v2 → v3 (and Why)

| Concern | v2 | v3 | Reason |
|---|---|---|---|
| Grid detection | Manual 4-corner click → bilinear → saved to DB | Auto classical CV per scan (`grid_circle_detector.py`) | No calibration step needed; works on any tray |
| Grid storage | PostgreSQL `trays.grid_json` | Not stored — reconstructed each scan | Grid is stateless; rows/cols come from slot_config |
| Sample storage | PostgreSQL (`ScanSession`, `TestSlot`, `Tray`) | Google Sheets (two tabs) | Simpler ops, accessible without DB admin |
| Image archive | Local `logs/img/` | Google Drive + local `server/static/results/` | Persistent, shareable, cloud-first |
| Color zoning | Fixed columns 1-3→L0, 4-6→L1… (hardcoded) | `slot_config.json` — any cell can be any ref level | Works for any tray layout; multiple refs per level |
| Web UI | Jinja2 `/dashboard` | `/upload` + `/results` + `/result/{id}` + `/settings` | History list, detail view, interactive grid editor |
| Test endpoint | `/test-upload` (separate) | `/analyze` (single endpoint) | No divergence between Pi and browser paths |
| Slot assignment | DB-driven fixed layout | `server/slot_config.json` — user-editable via `/settings` | Not all 195 cells used; user assigns slot_ids |
| Telegram bot | Inline in `server_app.py` | Removed | Out of scope for v3; add later if needed |
| Google auth | Service account JSON | OAuth2 (`google-auth-oauthlib`) → stored `token.json` | Works with personal Google accounts |
| Color result | Level only | Level + Lab values + hex color + ΔE scatter plot | Visualizes why each bottle was classified |

**Everything preserved from v2:**
- YOLO bottle detection (`ultralytics`) — still the only detection method
- CIE Lab color classification (`extract_bottle_color`, `classify_sample`, `delta_e_cie76`)
- Dynamic per-frame reference baseline (`build_reference_baseline`)
- White-padding letterbox for YOLO input
- `crop_sample_roi` ROI preprocessing
- Pi camera undistortion (`CameraUndistorter`)
- CLAHE preprocessing before YOLO/HoughCircles
- **LCD 16×4 display** — 4 lines
- **TM1637 7-segment displays** — count + spinner
- **3-channel relay LED tower** — idle / processing / error states
- `loguru` logging throughout
- `uv` as package manager

---

## Project Overview

A distributed urine sample analysis system. Bottles are placed on a physical grid tray. Per scan: classical CV (OpenCV HoughCircles + spacing estimation) detects and reconstructs the grid; YOLO detects bottles; CIE Lab color analysis classifies each bottle's hydration level (L0–L4) using a live reference-slot baseline. Results are stored in Google Sheets and annotated images uploaded to Google Drive.

| Role | Hardware | Responsibility |
|------|----------|----------------|
| **Client** | Raspberry Pi 4B | Capture image → POST to server → receive JSON result → display on LCD 16×4 / TM1637 / relay |
| **Server** | Ubuntu PC | Receive image → YOLO + OpenCV grid → color analysis → upload to Google Drive → append to Google Sheets → return JSON |

**Strict boundary:**

```
Raspberry Pi                          Server (Ubuntu)
─────────────────────────────         ──────────────────────────────────────────
capture image                         receive image
undistort                             run YOLO + OpenCV grid
POST /analyze ──────────────────────► detect + classify
                                      upload annotated image → Google Drive
                                      append row → Google Sheets
receive JSON result ◄────────────────return JSON
display on LCD / TM1637 / relay
```

The Pi has **no Google credentials and no Google SDK**. All cloud interaction is server-side only.

---

## Core Design Constraints (Non-negotiable)

| Task | Method | Reason |
|---|---|---|
| Bottle detection | **YOLO only** (PyTorch / ultralytics) | Handles varied sizes, occlusion, color |
| Grid detection | **OpenCV only** (HoughCircles + geometry) | No ML weights needed; deterministic |
| Color extraction | **Annular ring median** (not center point) | Avoids specular glare at cap center and plastic ring at edge |

Grid is **ground truth**. YOLO detections are assigned to pre-computed grid slots — never the reverse.

---

## Tech Stack

| Layer | Library / Tool |
|-------|----------------|
| Package manager | `uv` (never `pip install`) |
| Server framework | FastAPI + Uvicorn |
| Grid detection | `cv2.HoughCircles` + median-spacing estimation + RANSAC origin voting |
| Bottle detection | PyTorch / `ultralytics` YOLO |
| Image processing | OpenCV (`cv2`) |
| Color analysis | CIE Lab + Delta E (OpenCV + NumPy) |
| Config | `configs/config.toml` loaded via `tomllib` |
| Slot assignment | `server/slot_config.json` — JSON file, edited via `/settings` |
| Logging | `loguru` — use throughout, **never `print()`** |
| Cloud storage | Google Drive API v3 (`google-api-python-client`) |
| Cloud data | Google Sheets API v4 (`google-api-python-client`) |
| Google auth | OAuth2 flow (`google-auth-oauthlib`) → stored `token.json`; refresh via `google-auth-httplib2` |
| Multipart upload | `python-multipart` (FastAPI file ingestion) |
| Web UI | Jinja2 templates + Tailwind CSS (CDN) + Chart.js (CDN) |
| Pi camera | `picamera2` (libcamera) |
| Pi undistortion | `cv2` + `configs/camera_params.yaml` |
| Pi GPIO | `RPi.GPIO` (BCM mode) |
| Pi LCD | `rpi-lcd` — I2C 16×4 character display |
| Pi 7-segment | `raspberrypi-tm1637` — count + spinner |
| Pi relay | 3-channel active-LOW relay module |
| HTTP client | `requests` |
| Tunnel | Cloudflare Tunnel (`cloudflared`) |

---

## Repository Layout

```
urine-color-analysis/
├── client/
│   └── pi_client.py              # Pi loop — GPIO, camera, undistort, relay, LCD, TM1637, POST /analyze
├── server/
│   ├── api.py                    # FastAPI app — all routes
│   ├── pipeline.py               # run_pipeline() + lab_to_hex() — shared by all /analyze callers
│   ├── slot_config.py            # SlotConfig dataclass + load/save/query helpers
│   ├── slot_config.json          # Persisted slot assignment (user-edited via /settings)
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── drive.py              # upload_image() → Google Drive file_id
│   │   └── sheets.py             # write_slot_config_to_sheet(), append_result_to_sheet()
│   ├── templates/
│   │   ├── base.html             # Shared layout (nav: Upload | Results | Settings)
│   │   ├── upload.html           # Drag-drop form → fetch /analyze → redirect to /result/<id>
│   │   ├── results.html          # History table (all scans, newest first)
│   │   ├── result.html           # Detail: annotated image + slot table + Lab scatter plot + JSON
│   │   └── settings.html        # Interactive grid editor (click cell → popup → assign slot_id)
│   └── static/
│       ├── results/              # Persistent annotated JPEGs + JSON (<scan_id>.{jpg,json})
│       └── js/
│           └── settings.js       # Grid rendering, popup state management, API calls
├── utils/
│   ├── grid_circle_detector.py   # Standalone grid pipeline (detect_grid)
│   ├── color_analysis.py         # extract_bottle_color(), classify_sample(), build_reference_baseline()
│   └── camera_undistort.py       # CameraUndistorter — lens correction + focus lock
├── app/
│   └── shared/
│       └── processor.py          # crop_sample_roi(), letterbox_white_padding(), scale_coordinates()
├── configs/
│   ├── config.toml               # Hardware config — GPIO, camera, YOLO, color, google
│   ├── camera_params.yaml        # Lens calibration — focus_value, K, dist_coeffs
│   └── config.py                 # TOML loader — exposes flat constants
├── models/
│   └── best.pt                   # YOLO weights (git-ignored)
├── credentials/
│   ├── client_secrets.json       # OAuth2 client ID + secret (git-ignored)
│   └── token.json                # OAuth2 token, auto-written after first auth (git-ignored)
├── pyproject.toml                # uv project — extras: common / pi / server
└── logs/
    └── app_*.log                 # Rotating text logs
```

---

## Dependency Management

Always use `uv`. Never run `pip install`.

```toml
[project.optional-dependencies]
common = [
    "requests", "pydantic-settings", "loguru", "numpy", "pyyaml",
    "python-dotenv", "Pillow>=10.0", "psutil>=5.9"
]
server = [
    "fastapi", "uvicorn[standard]", "opencv-python", "python-multipart", "jinja2",
    "ultralytics>=8.4.24", "torch>=2.0", "torchvision>=0.15",
    # Google integration (SERVER ONLY — never install on Pi)
    "google-api-python-client>=2.0", "google-auth-httplib2>=0.2", "google-auth-oauthlib>=1.0"
]
pi = [
    "picamera2", "RPi.GPIO", "rpi-lcd", "raspberrypi-tm1637", "smbus2"
    # All Pi packages are gated on aarch64 Linux in pyproject.toml
]
```

```bash
uv sync --extra server --extra common   # server
uv sync --extra pi --extra common       # Pi
```

### Starting the server

```bash
uv run --extra server --extra common uvicorn server.api:app --reload
# Open http://localhost:8000/upload
```

### Google OAuth2 First-Time Setup

Fill in `configs/config.toml [google]`, then trigger any Google API call — the server will open a browser consent URL and write `token.json` automatically. Subsequent runs auto-refresh via `google-auth-httplib2`.

```toml
[google]
credentials_file = "client_secrets.json"
token_file       = "token.json"
drive_folder_id  = "..."          # Google Drive folder ID
spreadsheet_id   = "..."          # Single spreadsheet; two tabs inside
slots_tab        = "SlotAssignment"
results_tab      = "Results"
```

Scopes required:
```python
SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/spreadsheets",
]
```

Never commit `token.json` or `client_secrets.json`.

---

## Slot Config (`server/slot_config.json`)

Not all 195 grid cells are used. The user assigns a `slot_id` to each cell they want to analyze via the `/settings` page. The pipeline only processes cells that have an assigned `slot_id`.

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
    "16": { "slot_id": "A01",    "is_reference": false, "ref_level": null },
    "17": { "slot_id": "A02",    "is_reference": false, "ref_level": null }
  }
}
```

- Key = **1-based row-major** cell index: `cell_index = (row - 1) * cols + col`
- Cells absent from `"cells"` are completely ignored by the pipeline
- `is_reference: true` → used for color baseline; `ref_level` 0–4 required
- `is_reference: false` → sample bottle; `ref_level` is null
- Multiple cells can share the same `ref_level` — all are averaged into one baseline centroid

`server/slot_config.py` exposes:
```python
load_slot_config(path=None) -> SlotConfig
save_slot_config(cfg, path=None) -> None
active_cell_indices(cfg) -> set[int]   # all assigned cell indices
reference_cells(cfg) -> dict[int, int]  # cell_index → ref_level
sample_cells(cfg) -> dict[int, str]     # cell_index → slot_id
```

---

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/analyze` | `X-Auth-Token` | **Single endpoint** — Pi camera POST and browser file upload |
| `GET` | `/upload` | none | Upload form page |
| `GET` | `/results` | none | Scan history list (newest first) |
| `GET` | `/result/{scan_id}` | none | Scan detail page |
| `GET` | `/settings` | none | Slot assignment page |
| `GET` | `/api/slots` | none | Return `slot_config.json` as JSON |
| `POST` | `/api/slots` | none | Save slot configuration (also writes to Google Sheets SlotAssignment tab) |
| `GET` | `/health` | none | `{"status": "ok"}` |

`API_KEY` defaults to `"test"`, overridden via `API_KEY` environment variable.

### How `/analyze` serves both Pi and browser

`/analyze` always returns JSON. The browser upload page uses `fetch()`, receives JSON, then redirects:
```javascript
const data = await res.json();
window.location.href = '/result/' + data.scan_id;
```
The Pi receives the same JSON directly. No content-type negotiation needed.

### Result persistence

`/analyze` saves two files persistently to `server/static/results/`:
- `<scan_id>.jpg` — annotated image (50% scaled, quality 88)
- `<scan_id>.json` — full scan result dict

`GET /results` reads all `*.json` files sorted by `timestamp` descending.
`GET /result/{scan_id}` reads `<scan_id>.json`.

---

## `/analyze` Response Schema

```json
{
  "status": "success",
  "scan_id": "a1b2c3d4",
  "detected_count": 20,
  "total_assigned": 25,
  "missing_slots": ["A03", "B07"],
  "summary": { "L0": 5, "L1": 8, "L2": 4, "L3": 3, "L4": 0 },
  "reference_labs": {
    "0": [{ "lab": [210.0, 126.0, 134.0], "hex": "#f5e8c0" }],
    "1": [{ "lab": [185.0, 131.0, 148.0], "hex": "#dfc07a" },
          { "lab": [183.0, 130.0, 147.0], "hex": "#ddbf78" }]
  },
  "slots": {
    "A01": { "cell_index": 16, "detected": true,  "color_level": 1, "delta_e": 3.2, "confident": true,  "lab": [188.0, 130.0, 147.0], "hex": "#e2c47e" },
    "A03": { "cell_index": 18, "detected": false, "color_level": null, "delta_e": null, "confident": false, "lab": null, "hex": null }
  },
  "image_url": "/static/results/a1b2c3d4.jpg",
  "timestamp": "2026-05-05T10:30:00Z"
}
```

- `reference_labs` — per-level list of reference cells with their Lab + hex (for scatter plot and Sheets)
- `lab` / `hex` — extracted via annular ring median (not center pixel); `hex` = Lab → XYZ → sRGB → `#rrggbb`
- `scan_id` — browser redirects to `/result/<scan_id>`; Pi ignores it
- `missing_slots` — list of slot_ids where no YOLO detection was assigned; Pi uses for relay/LCD

---

## Full Server Pipeline (`server/pipeline.py`)

```
━━━ POST /analyze (JPEG bytes) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  │
  ▼ cv2.imdecode → BGR frame
  ▼ crop_sample_roi(top, bottom, left, right)     # remove dead zones
  ▼ letterbox_white_padding(640)                  # white fill, NOT black
  │   → padded (640×640), scale, pad_x, pad_y
  │
  ▼ detect_grid(padded, rows, cols)               # HoughCircles + CLAHE + RANSAC
  │   → grid_pts_640 (rows×cols, 2) in padded coords
  │   → convert to full-image coords via (pt - pad) / scale + roi_offset
  │
  ▼ YOLO inference on padded image                # ultralytics YOLO
  │   → boxes [cx,cy,w,h,conf] in padded coords → scale to full-image coords
  │
  ▼ assign_bottles_to_slots(yolo_boxes, grid_pts_full, active_cells, max_dist)
  │   → slot_hits: {cell_index: {cx, cy, w, h, conf}}
  │
  ▼ build_reference_baseline(frame_full, ref_positions)
  │   ref_positions built from reference_cells(slot_cfg) → {level: [(cx,cy,r),...]}
  │   extract_bottle_color() on each ref cell (annular ring median)
  │   → {level: (L, a, b)} dynamic per-frame baseline
  │
  ▼ classify_all_samples(slot_hits, sample_cells, baseline)
  │   extract_bottle_color() on each detected sample bottle
  │   classify_sample() → nearest centroid (minimum ΔE) + margin-of-victory confidence
  │   lab_to_hex() → "#rrggbb" for each bottle
  │   → result_slots: {slot_id: {cell_index, detected, color_level, delta_e, confident, lab, hex}}
  │
  ▼ build_scan_result(...)  → scan_result dict
  ▼ render_annotated_image(frame_full, ...) → JPEG bytes (50% scale)
  │
  ┌─── CLOUD (fire-and-forget, never blocks response) ──────────────────────────┐
  │ upload_image() → Google Drive                                               │
  │ append_result_to_sheet() → Google Sheets "Results" tab                     │
  └─────────────────────────────────────────────────────────────────────────────┘
  │
  ▼ save scan_result.json + annotated.jpg → server/static/results/
  ▼ return JSON  ──────────────────────────────────────────────────────────────►
```

### Why white padding?
White fill matches YOLO training preprocessing — black borders create false-contrast edges near bottle caps.

### Why ROI crop before letterbox?
Focusing the 640×640 budget on the sample area gives better YOLO coverage than the full frame.

### Why grid-first, then YOLO?
Grid defines slot positions. YOLO detections are *assigned to* grid slots — never used to infer structure. The grid is always complete (all `rows × cols` positions), even if 50% of bottles are missing.

### Why annular ring, not center point?
`extract_bottle_color()` samples the median Lab over the region between `inner_crop_px` (avoids specular glare at the center) and `radius - outer_crop_px` (avoids plastic cap ring at the edge). This applies equally to reference cells and sample bottles.

---

## Grid Detection Sub-Pipeline

Implemented in `utils/grid_circle_detector.py` (standalone combined version used by the server pipeline).

```
grayscale image
  │
  ▼ CLAHE on L channel (BGR→LAB, enhance L, back to gray)
  │   Normalises white AND colored (orange/yellow/red) caps to similar contrast
  │
  ▼ GaussianBlur (blur_kernel from GridCircleConfig)
  │
  ▼ cv2.HoughCircles (dp, min_dist, param1, param2, min_radius, max_radius)
  │   → raw centres (N, 2) — partial detection OK (30–50% missing tolerated)
  │
  ▼ estimate_grid_spacing(centres, rows, cols, image_shape)
  │   Sort projections on X and Y axes
  │   Median of consecutive diffs in [min_spacing, 3.5×min_spacing]
  │   De-aliasing: if spacing/2 better represented, halve it
  │   Cross-validate vs image_width/cols — prefer fallback if >30% deviation
  │   → (dx, dy, origin_x, origin_y)
  │
  ▼ reconstruct_grid(centres, rows, cols, dx, dy, origin_x, origin_y)
  │   RANSAC-style origin voting:
  │     Each detected centre votes for the sub-pixel origin that places it on-grid
  │     median(votes) → refined origin  (robust to up to 50% missing)
  │   Fill all rows × cols positions
  │   → grid_pts (rows×cols, 2), is_detected (rows×cols,) bool
```

Config (all thresholds from `GridCircleConfig` defaults, matching `[hough]` TOML section):
- `dp=1.2`, `min_dist=55`, `param1=60`, `param2=25`, `min_radius=22`, `max_radius=38`
- `blur_kernel=5`, `clahe_clip=2.0`, `clahe_tile=8`

**Config controls all thresholds — never hardcode in vision code.**

---

## Color Classification

Functions live in `utils/color_analysis.py`.

### `extract_bottle_color(frame, cx, cy, radius, outer_crop_px, inner_crop_px)`
Annular ring sampling:
- Excludes `outer_crop_px` pixels from bottle edge (plastic ring)
- Excludes `inner_crop_px` pixels from center (specular glare)
- Converts BGR → CIE Lab; applies glare filter (L > `glare_l_threshold`)
- Returns median (L, a, b) of valid ring pixels, or `None` if insufficient pixels
- **cx, cy must be in full-image coordinates** — not padded or ROI-relative

### `build_reference_baseline(frame, reference_positions)`
- `reference_positions`: `{ref_level: [(cx, cy, radius), ...]}` — multiple cells per level supported
- Calls `extract_bottle_color()` on each reference cell
- Averages Labs within each level group
- Returns `{level: (L, a, b)}` dynamic baseline calibrated to current frame's lighting
- **Never uses absolute Lab values — always relative to this frame's reference**

### `classify_sample(sample_lab, baseline, margin)`
- Nearest-centroid in CIE Lab (minimum Delta E CIE76)
- Confident when `second_best_ΔE − best_ΔE > margin` (margin-of-victory, not absolute threshold)
- Returns `(level, delta_e, confident)`

### `lab_to_hex(L_cv, a_cv, b_cv) -> str` (`server/pipeline.py`)
- Converts OpenCV 8-bit Lab encoding to `"#rrggbb"`
- Path: OpenCV Lab → standard Lab → XYZ D65 → linear sRGB → gamma sRGB → hex
- Used for both reference and sample bottles in scan results and Sheets

---

## Google Sheets Integration

Single spreadsheet (`spreadsheet_id`) with two tabs:

### Tab "SlotAssignment"
Cleared and rewritten whenever `POST /api/slots` saves the config.
Columns: `cell_index | slot_id | is_reference | ref_level | row | col`

### Tab "Results"
One batch appended per successful `/analyze` call. Never overwritten — full history accumulates.

**Summary row:** `scan_id | timestamp | detected_count | total_assigned | missing_slots | L0 | L1 | L2 | L3 | L4`

**Detail rows (one per assigned cell):** `scan_id | slot_id | cell_index | is_reference | ref_level | detected | color_level | delta_e | confident | L | a | b | hex`

The `hex` column cell background is set to the actual detected color via the Sheets API `userEnteredFormat.backgroundColor` — a visual color swatch directly in the spreadsheet.

Both calls are fire-and-forget (`try/except` wrapper) — a Google API failure never blocks the pipeline response.

---

## Settings Page (`/settings`)

Interactive visual grid. Each cell = one `(row, col)` position in the tray.

- Click any cell → popup with: slot_id input, reference checkbox, level dropdown (L0–L4)
- Popup **Save** → updates local JS state → re-renders cell → closes popup
- Popup **Clear** → removes cell from state → renders as empty
- **Load Saved** → `GET /api/slots` → reload and re-render (discards unsaved edits)
- **Reset All** → clear all cells in local state (not saved until **Save** is clicked)
- **Save** → `POST /api/slots` → persists to `slot_config.json` + writes to Google Sheets

Color coding: L0 = `#fff9c4`, L1 = `#ffe082`, L2 = `#ffb74d`, L3 = `#ef9a9a`, L4 = `#b71c1c` (white text), Sample = white, Empty = `#374151`

---

## Result Detail Page (`/result/{scan_id}`)

Three panels:
1. **Annotated image** — grid lines, green circles on detected bottles, red circles on missing assigned cells, slot_id labels
2. **Slot results table** — per slot: color swatch, level badge, ΔE, confidence; color-coded by level
3. **Color matching scatter plot** (Chart.js) — CIE Lab a*/b* space; reference cells as large markers, detected bottles as small labeled markers; tooltips show slot_id, Lab values, ΔE
4. **Raw JSON** — collapsible `<details><pre>` block

---

## Pi Client (`client/pi_client.py`)

The Pi does **exactly three things**: capture, send, display.

```
Boot
  │  GPIO.setmode(BCM), relay green, LCD "Urine Analyzer / Ready", TM1637 0000
  │
  └── wait GPIO.wait_for_edge(trigger_pin, FALLING)
        │
        ▼ _on_button_press()
          ├── relay yellow
          ├── LCD "Capturing..."  /  TM1637 8888
          ├── picam2.capture_array() → undistort_frame() → encode JPEG
          ├── LCD "Uploading..."  /  TM1637 spinner daemon
          ├── requests.post(SERVER_URL + "/analyze",
          │       files={"file": ("frame.jpg", jpeg_bytes, "image/jpeg")},
          │       headers={"X-Auth-Token": API_KEY},
          │       timeout=60)
          │         ├── HTTP 200 → _show_result(response.json())
          │         └── error   → _show_error(error_type)
          └── wait for next button press
```

### LCD 16×4 Display States

| State | Line 1 | Line 2 | Line 3 | Line 4 |
|---|---|---|---|---|
| Boot | `Urine Analyzer ` | `Ready          ` | ` ` | ` ` |
| Capturing | `Capturing...   ` | ` ` | ` ` | ` ` |
| Uploading | `Uploading...   ` | `Please wait    ` | ` ` | ` ` |
| Analyzing (>4 s) | `Analyzing...   ` | `AI running     ` | ` ` | ` ` |
| OK result | `Count: NN/NNN  ` | `L0:N L1:N L2:N ` | `L3:N L4:N      ` | `Status: OK     ` |
| Missing slots | `Count: NN/NNN  ` | `L0:N L1:N L2:N ` | `L3:N L4:N      ` | `Miss: A03 B07..` |
| Network error | `ERR: SRV OFF   ` | `Check network  ` | ` ` | ` ` |
| Timeout | `ERR: TIMEOUT   ` | `Server slow    ` | ` ` | ` ` |
| HTTP error | `ERR: HTTP NNN  ` | ` ` | ` ` | ` ` |
| Camera fail | `ERR: CAMERA    ` | `Capture failed ` | ` ` | ` ` |
| Server error | `ERR: AI FAIL   ` | ` ` | ` ` | ` ` |

Line 4 scrolls slot_ids for missing bottles: `_lcd_scroll(line=4, text)` — daemon thread at 300 ms/step.

### TM1637 States

| State | Display |
|---|---|
| Boot | `0000` |
| Capturing | `8888` (all segments on) |
| Upload / analyze | Spinning `- - - -` at 250 ms |
| Result | `NNNN` (detected count, zero-padded) |
| Error | `----` |

### Relay States

```
Boot              → Green ON
Button pressed    → Yellow ON  (until result received)
All slots found   → Green ON
Any missing slot  → Red ON     (until next button press)
Any error         → Red ON
```

Active-LOW relay: `GPIO.LOW` energises. Always turn previous channel OFF with 50 ms gap before turning new channel ON.

---

## Key Functions Reference

| Function | File | Notes |
|----------|------|-------|
| `detect_grid(image, rows, cols, config)` | `utils/grid_circle_detector.py` | Full standalone pipeline → `{grid_pts, detected_count, reconstructed_count}` |
| `extract_bottle_color(frame, cx, cy, r, ...)` | `utils/color_analysis.py` | Annular ring CIE Lab median — cx/cy in full-image coords |
| `build_reference_baseline(frame, ref_positions)` | `utils/color_analysis.py` | Dynamic per-frame baseline from multiple ref cells per level |
| `classify_sample(sample_lab, baseline, margin)` | `utils/color_analysis.py` | Nearest-centroid + margin-of-victory confidence |
| `delta_e_cie76(lab1, lab2)` | `utils/color_analysis.py` | CIE76 perceptual distance |
| `crop_sample_roi(img, top, bottom, left, right)` | `app/shared/processor.py` | Fixed-margin crop → (roi, x1, y1) |
| `letterbox_white_padding(img, 640)` | `app/shared/processor.py` | White fill letterbox → (padded, scale, pad_x, pad_y) |
| `run_pipeline(jpeg_bytes, slot_cfg)` | `server/pipeline.py` | Full pipeline → (scan_result dict, annotated_jpeg bytes) |
| `lab_to_hex(L_cv, a_cv, b_cv)` | `server/pipeline.py` | OpenCV 8-bit Lab → `"#rrggbb"` |
| `load_slot_config(path)` | `server/slot_config.py` | Load `slot_config.json` → SlotConfig |
| `save_slot_config(cfg, path)` | `server/slot_config.py` | Persist SlotConfig → JSON file |
| `active_cell_indices(cfg)` | `server/slot_config.py` | `set[int]` — all assigned cells |
| `reference_cells(cfg)` | `server/slot_config.py` | `{cell_index: ref_level}` |
| `sample_cells(cfg)` | `server/slot_config.py` | `{cell_index: slot_id}` |
| `upload_image(jpeg_bytes, filename, folder_id, ...)` | `server/integrations/drive.py` | Upload to Google Drive → file_id |
| `write_slot_config_to_sheet(cfg, spreadsheet_id, tab, ...)` | `server/integrations/sheets.py` | Clear + rewrite SlotAssignment tab |
| `append_result_to_sheet(scan_result, spreadsheet_id, tab, ...)` | `server/integrations/sheets.py` | Append summary + detail rows; set hex cell backgrounds |
| `CameraUndistorter(yaml_path)` | `utils/camera_undistort.py` | Pre-computes remap maps |
| `CameraUndistorter.undistort_frame(frame)` | `utils/camera_undistort.py` | Single `cv2.remap()` per frame |

---

## Error Handling Rules

1. `requests.post` (Pi client) must catch `ConnectionError`, `Timeout`, `HTTPError`, and bare `Exception` — never crash the main loop.
2. YOLO inference errors → HTTP 500 with `{"status": "error", "message": "..."}`.
3. HoughCircles finding zero circles → fall back to uniform grid from image dimensions + rows/cols (handled inside `estimate_grid_spacing`).
4. Google Drive upload failure → log warning, continue; `image_url` still points to local static file.
5. Google Sheets append failure → log warning, do not fail the HTTP response.
6. `GPIO.cleanup()` must run in a `finally` block on the Pi.
7. Camera (`picam2`) must be closed in `try/finally` inside `_capture_image()`.
8. Empty `slot_cfg.cells` → `/analyze` returns HTTP 400 — user must configure slots first.

---

## Logging Conventions

Use `loguru` exclusively. Never use `print()` or stdlib `logging`.

```python
from loguru import logger
logger.info("...")
logger.warning("...")
logger.error("...")
logger.success("...")
logger.exception("...")   # includes traceback
```

---

## What NOT To Do

- Do not use `pip install` — always `uv add` or edit `pyproject.toml` then `uv sync`
- Do not use deep learning for grid detection — only `cv2.HoughCircles` or blob/contour methods
- Do not use YOLO to infer grid structure — YOLO is detection only; grid comes from OpenCV
- Do not hardcode `rows`, `cols`, folder IDs, or thresholds in logic files
- Do not use black padding in `letterbox_white_padding` — must be white `(255, 255, 255)`
- Do not run inference on the Pi — capture + undistort + POST only
- Do not letterbox the full frame — `crop_sample_roi()` must be called first
- Do not commit `credentials/token.json`, `credentials/client_secrets.json`, `.env`, or `models/`
- Do not use a fixed absolute ΔE threshold for `classify_sample` — use margin-of-victory
- Do not use fixed column-zone color expectations (L0 = cols 1-3, etc.) — reference levels come from `slot_config.json`
- Do not import OpenVINO — project uses PyTorch/ultralytics
- Do not use `gspread` — use `google-api-python-client` directly for consistency with Drive
- Do not block the GPIO event loop with LCD/TM1637 writes — use daemon threads for scroll and spinner
- Do not call relay channel ON before previous channel is OFF — always turn off first with 50 ms gap
- Do not omit `missing_slots` from the API response — the Pi relay and LCD line 4 depend on it
- Do not put Google Drive or Sheets logic on the Pi — Pi has no Google credentials; server handles all cloud interaction
- Do not install `google-api-python-client`, `google-auth-*` on the Pi — they belong in the `server` extra only
- Do not sample color from the center point — always use `extract_bottle_color()` which samples the annular ring median
- Do not use a single endpoint for Pi vs browser test uploads — `/analyze` serves both identically
- Do not add a separate `/test-upload` endpoint — use `/analyze` for all image submissions
- Do not store scan results only in memory — persist `<scan_id>.json` and `<scan_id>.jpg` to `server/static/results/`

---

_Keep this file in sync when architecture, config keys, or API contracts change._
