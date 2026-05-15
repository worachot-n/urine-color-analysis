# CLAUDE.md — Urine Analysis System v3

Codebase and developer context for AI Code Assistants (Claude, Cursor, Copilot).
Read this file before modifying any file in this repository.

---

## What Changed from v2 → v3 (and Why)

| Concern | v2 | v3 | Reason |
|---|---|---|---|
| Grid detection | Manual 4-corner click → bilinear → saved to DB | Auto classical CV per scan (`auto_grid_detector.py`) | No calibration step needed; works on any tray |
| Grid storage | PostgreSQL `trays.grid_json` | Not stored — reconstructed each scan | Grid is stateless; rows/cols come from slot_config |
| Sample storage | PostgreSQL (`ScanSession`, `TestSlot`, `Tray`) | SQLite primary + Google Sheets sync | Offline-first; Sheets gets synced in background |
| Image archive | Local `logs/img/` | `logs/img/` (primary) + Google Drive (cloud backup) | Persistent local store; Drive for sharing |
| Color zoning | Fixed columns 1-3→L0, 4-6→L1… (hardcoded) | `slot_config.json` — any cell can be any ref level | Works for any tray layout; multiple refs per level |
| White balance | None | `is_white_reference` cells → chromatic adaptation offset | Compensates ambient light colour casts |
| Color classification | Nearest-centroid ΔE | Hybrid: chroma ΔE + histogram Bhattacharyya + master path | More robust across urine colour range |
| Web UI | Jinja2 `/dashboard` | `/upload` + `/results` + `/result/{id}` + `/settings` + `/auto_grid` | History list, detail view, interactive grid editor, CV debug |
| Test endpoint | `/test-upload` (separate) | `/analyze` (single endpoint) | No divergence between Pi and browser paths |
| Slot assignment | DB-driven fixed layout | `server/slot_config.json` — user-editable via `/settings` | Not all 195 cells used; user assigns slot_ids |
| Telegram bot | Inline in `server_app.py` | `bot/telegram_bot.py` — fire-and-forget on scan | Isolated; silently disabled if token missing |
| Google auth | Service account JSON | Service account JSON (`google-auth`) — headless, no browser | Fully automated; no token refresh needed |
| Sheets tabs | — | Three tabs: `SlotAssignment`, `Detail`, `Summary` | Separate per-bottle detail from per-scan summary |
| Offline resilience | None | SQLite backup + `/api/sync` endpoint | Scans never lost when Google is unreachable |
| Color result | Level only | Level + Lab values + hex color + ΔE scatter plot | Visualizes why each bottle was classified |

**Everything preserved from v2:**
- YOLO bottle detection (`ultralytics`) — still the only detection method
- CIE Lab color classification foundation (`extract_bottle_color`, `delta_e_cie76`)
- Dynamic per-frame reference baseline
- White-padding letterbox for YOLO input
- Pi camera undistortion (`CameraUndistorter`) — **now actively used in `app/client_app.py`**
- CLAHE preprocessing before YOLO/HoughCircles
- **LCD 16×4 display** — all 4 lines used
- **TM1637 7-segment displays** — 5 units (H0–H4), one per colour level
- **3-channel relay LED tower** — idle / processing / error states
- `loguru` logging throughout
- `uv` as package manager

---

## Project Overview

A distributed urine sample analysis system. Bottles are placed on a physical grid tray. Per scan: classical CV (OpenCV HoughCircles + KDE spacing) detects and reconstructs the grid; YOLO detects bottles; a hybrid CIE Lab + histogram classifier assigns each bottle a hydration level (L0–L4) using a live reference-slot baseline. Results are stored in SQLite locally and synced to Google Sheets; annotated images go to `logs/img/` and optionally Google Drive.

| Role | Hardware | Responsibility |
|------|----------|----------------|
| **Client** | Raspberry Pi 4B | Capture → undistort → POST to server → receive JSON → display on LCD 16×4 / TM1637 / relay |
| **Server** | Ubuntu PC | Receive image → YOLO + OpenCV grid → color analysis → SQLite → Google Sheets (async) → return JSON |

**Strict boundary:**

```
Raspberry Pi                          Server (Ubuntu)
─────────────────────────────         ──────────────────────────────────────────
capture image                         receive image
undistort (CameraUndistorter)         run YOLO + OpenCV grid
POST /analyze ──────────────────────► detect + classify
                                      save to SQLite
                                      upload annotated image → Google Drive (async)
                                      sync rows → Google Sheets (async)
receive JSON result ◄────────────────return JSON
display on LCD / TM1637 / relay
```

The Pi has **no Google credentials and no Google SDK**. All cloud interaction is server-side only.

---

## Core Design Constraints (Non-negotiable)

| Task | Method | Reason |
|---|---|---|
| Bottle detection | **YOLO only** (PyTorch / ultralytics) | Handles varied sizes, occlusion, color |
| Grid detection | **OpenCV only** (HoughCircles + KDE geometry) | No ML weights needed; deterministic |
| Color extraction | **Annular ring median** (not center point) | Avoids specular glare at cap center and plastic ring at edge |

Grid is **ground truth**. YOLO detections are assigned to pre-computed grid slots — never the reverse.

---

## Tech Stack

| Layer | Library / Tool |
|-------|----------------|
| Package manager | `uv` (never `pip install`) |
| Entry point | `main.py --role {server\|client}` |
| Server framework | FastAPI + Uvicorn |
| Grid detection | `cv2.HoughCircles` + KDE spacing estimation + RANSAC origin voting (`utils/auto_grid_detector.py`) |
| Bottle detection | PyTorch / `ultralytics` YOLO |
| Image processing | OpenCV (`cv2`) |
| Color analysis | Hybrid: CIE Lab chroma ΔE + histogram Bhattacharyya + master color path |
| Config | `configs/config.toml` loaded via `tomllib`; `settings.yaml` for server/model/storage |
| Slot assignment | `server/slot_config.json` — JSON file, edited via `/settings` |
| Logging | `loguru` — use throughout, **never `print()`** |
| Local storage | SQLite (`logs/urine_analysis.db`) — primary scan store |
| Cloud storage | Google Drive API v3 (`google-api-python-client`) |
| Cloud data | Google Sheets API v4 (`google-api-python-client`) — three tabs |
| Google auth | Service account (`google-auth`) — `credentials.json` only; no browser, no token file |
| Multipart upload | `python-multipart` (FastAPI file ingestion) |
| Web UI | Jinja2 templates + Tailwind CSS (local) + Chart.js (local) |
| Telegram | `bot/telegram_bot.py` — fire-and-forget; silently disabled if token absent |
| Pi camera | `picamera2` (libcamera) |
| Pi undistortion | `cv2` + `configs/camera_params.yaml` — `CameraUndistorter` initialized at boot |
| Pi GPIO | `RPi.GPIO` (BCM mode) |
| Pi LCD | `rpi-lcd` — I2C 16×4 character display |
| Pi 7-segment | `raspberrypi-tm1637` — **5 displays** (H0–H4), one per colour level |
| Pi relay | 3-channel active-LOW relay module |
| HTTP client | `requests` |
| Tunnel | Cloudflare Tunnel (`cloudflared`) |

---

## Repository Layout

```
urine-color-analysis/
├── main.py                           # Unified entry point — --role {server|client}
├── app/
│   ├── client_app.py                 # Pi loop — GPIO, camera, undistort, relay, LCD, TM1637, POST /analyze
│   ├── server_app.py                 # run_server() — launches uvicorn server.api:app
│   └── shared/
│       ├── config.py                 # Unified config: TOML + settings.yaml + .env + env vars
│       └── processor.py              # letterbox_white_padding(), crop_sample_roi(), scale_coordinates()
├── server/
│   ├── api.py                        # FastAPI app — all routes
│   ├── pipeline.py                   # run_pipeline() + lab_to_hex() — shared by all /analyze callers
│   ├── slot_config.py                # SlotConfig dataclass + load/save/query helpers
│   ├── slot_config.json              # Persisted slot assignment (user-edited via /settings)
│   ├── integrations/
│   │   ├── drive.py                  # upload_image() → Google Drive file_id
│   │   ├── sheets.py                 # write_slot_config_to_sheet(), append_detail/summary_to_sheet()
│   │   └── sqlite_backup.py          # SQLite offline store — init_db, save_scan, sync helpers
│   ├── templates/
│   │   ├── base.html                 # Shared layout (nav: Upload | Results | Settings | Grid)
│   │   ├── upload.html               # Drag-drop form → fetch /analyze → redirect to /result/<id>
│   │   ├── results.html              # History table (all scans, newest first)
│   │   ├── result.html               # Detail: annotated image + slot table + Lab scatter plot + JSON
│   │   ├── settings.html             # Interactive grid editor (click cell → popup → assign slot_id)
│   │   └── auto_grid.html            # Grid debug page (CV-only, no YOLO)
│   └── static/
│       ├── fonts/                    # NotoSansThai TTF + WOFF2 for Thai slot_id labels
│       └── js/
│           ├── settings.js           # Grid rendering, popup state management, API calls
│           ├── chart.umd.min.js      # Chart.js (local CDN)
│           └── tailwind.js           # Tailwind CSS (local CDN)
├── bot/
│   └── telegram_bot.py               # send_scan_report() — silently skipped if token blank
├── utils/
│   ├── auto_grid_detector.py         # detect_grid_full() — HoughCircles + KDE + RANSAC (used by pipeline)
│   ├── grid_circle_detector.py       # Legacy standalone grid pipeline (not used by pipeline)
│   ├── color_analysis.py             # extract_bottle_color(), build_reference_set(), classify_sample_path()
│   └── camera_undistort.py           # CameraUndistorter — lens correction + focus lock
├── scripts/
│   ├── setup-server.sh               # One-command Ubuntu server install + systemd service
│   ├── setup-pi.sh                   # One-command Pi client install + systemd service
│   ├── urine-server.service          # systemd unit template (server)
│   └── urine-pi.service              # systemd unit template (Pi client)
├── configs/
│   ├── config.toml                   # Hardware config — GPIO, camera, YOLO, color, google
│   ├── camera_params.yaml            # Lens calibration — focus_value, K, dist_coeffs
│   └── config.py                     # TOML loader — exposes flat constants
├── settings.yaml                     # Server/model/storage defaults (overridden by .env)
├── models/
│   └── best.pt                       # YOLO weights (git-ignored)
├── credentials/
│   └── credentials.json              # Service account key JSON (git-ignored)
├── tests/
│   ├── test_color_analysis.py
│   ├── test_grid.py
│   └── test_yolo_detect.py
└── logs/
    ├── app_*.log                     # Rotating text logs (10 MB, 7-day retention)
    ├── img/                          # Annotated JPEGs served at /img/<scan_id>.jpg
    └── urine_analysis.db             # SQLite scan store
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
    "scikit-image>=0.21", "scipy>=1.11",
    # Google integration (SERVER ONLY — never install on Pi)
    "google-api-python-client>=2.0", "google-auth-httplib2>=0.2"
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
uv run main.py --role server
# Open http://localhost:8000/upload
```

### One-command setup (installs dependencies + systemd service)

```bash
bash scripts/setup-server.sh   # Ubuntu
bash scripts/setup-pi.sh       # Raspberry Pi
```

### Google Service Account Setup

1. In Google Cloud Console, create a Service Account with no special roles.
2. Download its JSON key — save as `credentials/credentials.json`.
3. Share your target Google Drive folder with the service account email (Contributor role).
4. Share your Google Spreadsheet with the service account email (Editor role).
5. Fill in `configs/config.toml [google]`:

```toml
[google]
service_account_file = "credentials/credentials.json"
drive_folder_id      = "..."   # Google Drive folder ID
spreadsheet_id       = "..."   # Single spreadsheet; three tabs inside
slots_tab            = "SlotAssignment"
detail_tab           = "Detail"
summary_tab          = "Summary"
```

No browser step, no `token.json`. The server authenticates fully automatically on every start.
Google Sheets sync runs in the background — a failure never blocks the HTTP response.
Offline scans (those that failed to sync) are retried on the next server start via `_sync_pending()`.

Scopes used:
```python
# sheets.py
["https://www.googleapis.com/auth/spreadsheets"]
# drive.py
["https://www.googleapis.com/auth/drive.file"]
```

Never commit `credentials/credentials.json`.

---

## Slot Config (`server/slot_config.json`)

Not all 195 grid cells are used. The user assigns a `slot_id` to each cell via the `/settings` page.

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
- Cells absent from `"cells"` are completely ignored by the pipeline
- `is_reference: true` → colour reference; `ref_level` 0–4 required; `is_white_reference` must be false
- `is_white_reference: true` → neutral white patch for chromatic adaptation (WB offset); `is_reference` must be false
- `is_reference: false, is_white_reference: false` → sample bottle; `ref_level` is null
- Multiple cells can share the same `ref_level` — all are averaged into one baseline centroid
- `is_reference` and `is_white_reference` are mutually exclusive — `load_slot_config` raises `ValueError` if both are set

`server/slot_config.py` exposes:
```python
load_slot_config(path=None) -> SlotConfig
save_slot_config(cfg, path=None) -> None
active_cell_indices(cfg) -> set[int]      # all assigned cell indices
reference_cells(cfg) -> dict[int, int]    # cell_index → ref_level (colour refs only)
white_reference_cells(cfg) -> set[int]    # cell indices flagged as white-reference
sample_cells(cfg) -> dict[int, str]       # cell_index → slot_id (non-reference, non-WB)
```

---

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/analyze` | `X-Auth-Token` | **Single endpoint** — Pi camera POST and browser file upload |
| `GET` | `/` | none | Redirects to `/upload` |
| `GET` | `/upload` | none | Upload form page |
| `GET` | `/results` | none | Scan history list (newest first, from SQLite) |
| `GET` | `/result/{scan_id}` | none | Scan detail page |
| `GET` | `/settings` | none | Slot assignment page |
| `GET` | `/auto_grid` | none | Grid debug page (CV-only visualisation) |
| `GET` | `/api/slots` | none | Return `slot_config.json` as JSON |
| `POST` | `/api/slots` | none | Save slot configuration (also writes to Google Sheets SlotAssignment tab) |
| `POST` | `/api/sync` | none | Manually trigger Google Sheets sync of pending offline scans |
| `POST` | `/api/auto_grid` | none | Run full-res grid reconstruction on uploaded image |
| `GET` | `/health` | none | `{"status": "ok"}` |
| `GET` | `/img/{scan_id}.jpg` | none | Annotated JPEG served from `logs/img/` |

`API_KEY` defaults to `"test"`, overridden via `API_KEY` environment variable.

### How `/analyze` serves both Pi and browser

`/analyze` always returns JSON. The browser upload page uses `fetch()`, receives JSON, then redirects:
```javascript
const data = await res.json();
window.location.href = '/result/' + data.scan_id;
```
The Pi receives the same JSON directly. No content-type negotiation needed.

### Result persistence

`/analyze` saves results to two stores:
- **SQLite** (`logs/urine_analysis.db`) — primary; `GET /results` and `GET /result/{id}` read from here
- **Image file** (`logs/img/<scan_id>.jpg`) — annotated JPEG served at `/img/<scan_id>.jpg`

Google Sheets sync runs asynchronously after the response is returned. If it fails, the scan stays in SQLite as "pending" and is retried on next server start via `_sync_pending()` or `POST /api/sync`.

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
    "0": [{ "lab": [210.0, 126.0, 134.0], "hex": "#f5e8c0", "cell_index": 1 }]
  },
  "slots": {
    "A01": {
      "cell_index": 16, "detected": true, "color_level": 1,
      "concentration_index": 0.25, "path_distance": 3.1,
      "delta_e": 3.2, "hist_bhatt": 0.12, "combined": 0.43,
      "confident": true, "best_fit": false, "force_l0": false, "soft_penalty": false,
      "lab": [188.0, 130.0, 147.0], "hex": "#e2c47e"
    },
    "A03": {
      "cell_index": 18, "detected": false, "color_level": null,
      "concentration_index": null, "path_distance": null,
      "delta_e": null, "hist_bhatt": null, "combined": null,
      "confident": false, "best_fit": false, "force_l0": false, "soft_penalty": false,
      "lab": null, "hex": null
    }
  },
  "master_path": { "levels": [0,1,2,3,4], "points": [[...], ...] },
  "wb_offset": [2.1, -1.3],
  "image_url": "/img/a1b2c3d4.jpg",
  "timestamp": "2026-05-05T10:30:00Z"
}
```

- `concentration_index` — scalar 0–4 (continuous level position along the master color path)
- `path_distance` — ΔE distance from nearest master path segment
- `hist_bhatt` — histogram Bhattacharyya distance to nearest level template
- `combined` — weighted score (chroma + histogram)
- `force_l0` — true when achromatic detection short-circuits the path classifier
- `soft_penalty` — true when a soft penalty was applied (borderline chroma distance)
- `best_fit` — true when classified by fallback nearest-level rather than confident path projection
- `wb_offset` — [Δa*, Δb*] chromatic adaptation from white-reference cells (null if WB disabled)
- `missing_slots` — Pi uses for relay/LCD line 3 scroll

---

## Full Server Pipeline (`server/pipeline.py`)

```
━━━ POST /analyze (JPEG bytes) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  │
  ▼ cv2.imdecode → BGR frame (full resolution)
  │
  ▼ detect_grid_full(roi, rows, cols)       # HoughCircles + KDE + RANSAC
  │   → grid_pts_roi (rows×cols, 2) in full-image coords
  │   → avg_radius_px — median of detected circle radii
  │
  ▼ letterbox_white_padding(roi, 640)       # white fill, NOT black
  │   → padded (640×640), scale, pad_x, pad_y
  │
  ▼ YOLO inference on padded image
  │   → boxes [cx,cy,w,h,conf] → scale to full-image coords
  │
  ▼ assign_bottles_to_slots(yolo_boxes, grid_pts_full, active_cells, max_dist)
  │   → slot_hits: {cell_index: {cx, cy, w, h, conf}}
  │
  ▼ White-balance offset (from is_white_reference cells)
  │   compute_white_balance_offset(frame, wb_positions)
  │   → wb_offset [Δa*, Δb*] or None
  │
  ▼ build_reference_set(frame, ref_positions)
  │   extract_bottle_color() on each ref cell (annular ring median)
  │   filter_reference_outliers() — b*-based 2σ filter
  │   → {level: [(L, a, b), ...]} per-frame reference Labs
  │
  ▼ build_reference_histograms(frame, ref_positions, wb_offset)
  │   → {level: hist} Lab histogram templates per reference level
  │
  ▼ build_master_path(level_centroids)
  │   → master color path connecting L0→L1→L2→L3→L4 centroids
  │
  ▼ classify_sample_path(lab, hist, master_path, ref_hists, ...)
  │   Project sample onto master path → concentration_index
  │   Hybrid score = w_chroma×ΔE + w_hist×Bhattacharyya
  │   Achromatic short-circuit → force_l0 if near-neutral
  │   → {level, chroma_de, hist_bhatt, combined, confident, concentration_index, ...}
  │
  ▼ build_scan_result(...)  → scan_result dict
  ▼ render_annotated_image(frame_full, ...) → JPEG bytes (≤1280px wide)
  │
  ┌─── ASYNC (fire-and-forget, never blocks response) ──────────────────────────┐
  │ save_scan() → SQLite                                                        │
  │ upload_image() → Google Drive                                               │
  │ append_detail_to_sheet() + append_summary_to_sheet() → Google Sheets       │
  │ send_scan_report() → Telegram                                               │
  └─────────────────────────────────────────────────────────────────────────────┘
  │
  ▼ save annotated.jpg → logs/img/<scan_id>.jpg
  ▼ return JSON  ──────────────────────────────────────────────────────────────►
```

### Why white padding?
White fill matches YOLO training preprocessing — black borders create false-contrast edges near bottle caps.

### Why grid-first, then YOLO?
Grid defines slot positions. YOLO detections are *assigned to* grid slots — never used to infer structure. The grid is always complete (all `rows × cols` positions), even if 50% of bottles are missing.

### Why annular ring, not center point?
`extract_bottle_color()` samples the median Lab over the region between `inner_crop_px` (avoids specular glare) and `radius - outer_crop_px` (avoids plastic cap ring). Applies equally to reference cells and sample bottles.

---

## Grid Detection Sub-Pipeline

Implemented in `utils/auto_grid_detector.py` (`detect_grid_full`).

```
full-resolution image
  │
  ▼ CLAHE on L channel (BGR→LAB, enhance L, back to gray)
  │
  ▼ GaussianBlur
  │
  ▼ cv2.HoughCircles → raw centres (N, 2) — partial detection OK
  │
  ▼ KDE spacing estimation (replaces median-diff from v2)
  │   Kernel density estimation over pairwise X/Y gaps
  │   → dominant (dx, dy) spacing
  │
  ▼ RANSAC-style origin voting
  │   Each detected centre votes for sub-pixel origin placing it on-grid
  │   median(votes) → refined origin
  │   Fill all rows × cols positions
  │   → grid_pts (rows×cols, 2), avg_radius_px
```

`draw_grid_lines(canvas, grid_pts, rows, cols, radius)` — draws the reconstructed grid on an annotated image (used by both `/auto_grid` debug page and `_render_annotated`).

---

## Color Classification

Functions live in `utils/color_analysis.py`.

### `extract_bottle_color(frame, cx, cy, radius, outer_crop_px, inner_crop_px)`
Annular ring sampling (identical to v2):
- Excludes `outer_crop_px` pixels from bottle edge (plastic ring)
- Excludes `inner_crop_px` pixels from center (specular glare)
- Converts BGR → CIE Lab; applies glare filter (`L > glare_l_threshold`)
- Returns median (L, a, b) of valid ring pixels, or `None` if insufficient pixels
- **cx, cy must be in full-image coordinates**

### `build_reference_set(frame, ref_positions)`
- `ref_positions`: `{ref_level: [(cx, cy, radius), ...]}` — multiple cells per level supported
- Calls `extract_bottle_color()` on each reference cell
- Returns `{level: [(L, a, b), ...]}` — list of individual Labs per level (not averaged)

### `filter_reference_outliers(frame, ref_positions)`
- Removes per-level outliers using b*-based 2σ filter before reference is used
- Applied before both `build_reference_set` and `build_reference_histograms`

### `compute_white_balance_offset(frame, wb_positions)`
- Extracts Lab from white-reference cells; returns (Δa*, Δb*) chromatic adaptation offset
- Applied to histogram templates; not applied to the master path Lab distances

### `build_master_path(level_centroids)`
- Connects per-level Lab centroids into a continuous color path (L0→L1→L2→L3→L4)
- Returns `{levels, segments, points}` or `None` if fewer than 2 levels available

### `classify_sample_path(lab, hist, master_path, ref_hists, ...)`
- Projects sample Lab onto the master path → continuous `concentration_index`
- Computes hybrid score: `w_chroma × chroma_ΔE + w_hist × Bhattacharyya`
- Achromatic short-circuit: near-neutral samples are force-classified L0
- Soft penalty for borderline chroma distance
- Returns `{level, chroma_de, hist_bhatt, combined, confident, concentration_index, path_distance, ...}`

### `lab_to_hex(L_cv, a_cv, b_cv) -> str` (`server/pipeline.py`)
- Converts OpenCV 8-bit Lab encoding to `"#rrggbb"`
- Path: OpenCV Lab → standard Lab → XYZ D65 → linear sRGB → gamma sRGB → hex

---

## Google Sheets Integration

Single spreadsheet (`spreadsheet_id`) with three tabs:

### Tab "SlotAssignment"
Cleared and rewritten whenever `POST /api/slots` saves the config.
Columns: `cell_index | slot_id | is_reference | ref_level | row | col`

### Tab "Detail"
One row per assigned cell per scan. Never overwritten — full history accumulates.
Columns: `scan_id | slot_id | cell_index | is_reference | ref_level | detected | color_level | delta_e | confident | L | a | b | hex`
Cell background set to the detected hex colour via `userEnteredFormat.backgroundColor`.

### Tab "Summary"
One row per scan. Never overwritten.
Columns: `scan_id | timestamp | detected_count | total_assigned | missing_slots | L0 | L1 | L2 | L3 | L4 | image_url`

Both calls are fire-and-forget — a Google API failure logs a warning and the scan stays in SQLite as pending.

---

## SQLite Offline Backup (`server/integrations/sqlite_backup.py`)

Primary local scan store. Used by all `GET /results` and `GET /result/{id}` reads.

Tables:
- `scan_summary` — one row per scan (mirrors Summary tab + `synced` flag)
- `scan_detail` — one row per bottle per scan (mirrors Detail tab)
- `scan_images` — image file paths

Key functions:
```python
init_db(db_path)                          # create tables if not exist (called at startup)
save_scan(scan_id, scan_result, db_path)  # write summary + detail rows
mark_scan_synced(scan_id, db_path)        # set synced=1 after Sheets success
get_pending_scans(db_path) -> list[str]   # scan_ids where synced=0
count_pending(db_path) -> int
load_scan(scan_id, db_path) -> dict|None
load_all_scans(db_path) -> list[dict]     # newest-first, for /results page
```

---

## Settings Page (`/settings`)

Interactive visual grid. Each cell = one `(row, col)` position in the tray.

- Click any cell → popup with: slot_id input, reference checkbox, white-reference checkbox, level dropdown (L0–L4)
- `is_reference` and `is_white_reference` are mutually exclusive — only one can be set
- **Save** → `POST /api/slots` → persists to `slot_config.json` + writes to Google Sheets SlotAssignment tab

Color coding: L0 = `#fff9c4`, L1 = `#ffe082`, L2 = `#ffb74d`, L3 = `#ef9a9a`, L4 = `#b71c1c` (white text), White-ref = light blue, Sample = white, Empty = `#374151`

---

## Result Detail Page (`/result/{scan_id}`)

Four panels:
1. **Annotated image** — grid lines, coloured circles on detected bottles (colour = level), red circles on missing assigned cells, slot_id labels (Thai-aware via PIL/NotoSansThai)
2. **Slot results table** — per slot: color swatch, level badge, ΔE, confidence, concentration_index
3. **Color matching scatter plot** (Chart.js) — CIE Lab a*/b* space; reference cells as large markers, detected bottles as small labeled markers; tooltips show slot_id, Lab values, ΔE
4. **Raw JSON** — collapsible `<details><pre>` block

---

## Pi Client (`app/client_app.py`)

The Pi does **exactly three things**: capture (with undistortion), send, display.

```
Boot
  │  GPIO.setmode(BCM), relay green
  │  LCD "Urine Analyzer / Ready", TM1637 H0-H4: 0000
  │  CameraUndistorter initialized from configs/camera_params.yaml
  │
  └── poll GPIO.input(trigger_pin)
        │
        ▼ _on_button_press()
          ├── relay yellow
          ├── LCD line 1: "Capturing..."  /  TM1637 H0-H4: 8888
          ├── _capture_image(undistorter)
          │     picam2 capture → lock_focus_picamera2() → lock AWB/AE
          │     → undistorter.undistort_frame() → BGR→RGB flip → JPEG encode
          ├── LCD line 1: "Uploading..." / line 2: "Please wait"
          ├── TM1637 spinner + delayed LCD "Analyzing..." after 4s
          ├── requests.post(SERVER_URL + "/analyze",
          │       files={"file": ("capture.jpg", jpeg_bytes, "image/jpeg")},
          │       headers={"X-Auth-Token": API_KEY}, timeout=60)
          │
          ├── On success:
          │     LCD line 1: "OK" or "Recheck needed"
          │     LCD line 2: "Total:{detected}/{total}"
          │     LCD line 3: "Recheck:A03 B07..."  (scroll daemon if >16 chars)
          │     LCD line 4: "Full report:{server_url}/result/{scan_id}"  (scroll daemon)
          │     TM1637 H0–H4: L0, L1, L2, L3, L4 counts from summary
          │     Relay: green (no missing) / red (any missing)
          │
          └── On error: relay red, LCD line 1 error code, lines 2-4 blank
```

### LCD 16×4 Display States

| State | Line 1 | Line 2 | Line 3 | Line 4 |
|---|---|---|---|---|
| Boot | `Urine Analyzer ` | `Ready          ` | ` ` | ` ` |
| Capturing | `Capturing...   ` | ` ` | ` ` | ` ` |
| Uploading | `Uploading...   ` | `Please wait    ` | ` ` | ` ` |
| Analyzing (>4 s) | `Analyzing...   ` | `AI running     ` | ` ` | ` ` |
| OK result | `OK             ` | `Total:NN/NNN   ` | `                ` | `Full report:URL` (scroll) |
| Missing slots | `Recheck needed ` | `Total:NN/NNN   ` | `Recheck:A03 B07` (scroll) | `Full report:URL` (scroll) |
| Network error | `ERR: SRV OFF   ` | `Check network  ` | ` ` | ` ` |
| Timeout | `ERR: TIMEOUT   ` | `Server slow    ` | ` ` | ` ` |
| HTTP error | `ERR: HTTP NNN  ` | ` ` | ` ` | ` ` |
| Camera fail | `ERR: CAMERA    ` | `Capture failed ` | ` ` | ` ` |
| Server error | `ERR: AI FAIL   ` | ` ` | ` ` | ` ` |

Lines 3 and 4 use `_lcd_scroll_line(lcd, text, line, stop_event)` — daemon threads at 200 ms/step.

### TM1637 States (5 displays: H0–H4)

| State | H0 | H1 | H2 | H3 | H4 |
|---|---|---|---|---|---|
| Boot | `0000` | `0000` | `0000` | `0000` | `0000` |
| Capturing | `8888` | `8888` | `8888` | `8888` | `8888` |
| Upload / analyze | `- - -` spin | `- - -` spin | `- - -` spin | `- - -` spin | `- - -` spin |
| Result | L0 count | L1 count | L2 count | L3 count | L4 count |
| Error | `----` | `----` | `----` | `----` | `----` |

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
| `detect_grid_full(image, rows, cols)` | `utils/auto_grid_detector.py` | Full grid pipeline → `{grid_pts, avg_radius_px, ...}` |
| `draw_grid_lines(canvas, grid_pts, rows, cols, radius)` | `utils/auto_grid_detector.py` | Draw reconstructed grid on image |
| `extract_bottle_color(frame, cx, cy, r, ...)` | `utils/color_analysis.py` | Annular ring CIE Lab median — cx/cy in full-image coords |
| `build_reference_set(frame, ref_positions)` | `utils/color_analysis.py` | Per-frame reference Labs per level |
| `build_reference_histograms(frame, ref_positions, ...)` | `utils/color_analysis.py` | Lab histogram templates per level |
| `filter_reference_outliers(frame, ref_positions)` | `utils/color_analysis.py` | 2σ b*-based outlier removal |
| `compute_white_balance_offset(frame, wb_positions)` | `utils/color_analysis.py` | Chromatic adaptation (Δa*, Δb*) from white-ref cells |
| `build_master_path(level_centroids)` | `utils/color_analysis.py` | L0→L4 master color path |
| `classify_sample_path(lab, hist, master_path, ...)` | `utils/color_analysis.py` | Hybrid chroma+histogram classification |
| `is_achromatic(frame, cx, cy, radius)` | `utils/color_analysis.py` | Near-neutral detection → force_l0 short-circuit |
| `delta_e_cie76(lab1, lab2)` | `utils/color_analysis.py` | CIE76 perceptual distance |
| `letterbox_white_padding(img, 640)` | `app/shared/processor.py` | White fill letterbox → (padded, scale, pad_x, pad_y) |
| `run_pipeline(jpeg_bytes, slot_cfg)` | `server/pipeline.py` | Full pipeline → (scan_result dict, annotated_jpeg bytes) |
| `lab_to_hex(L_cv, a_cv, b_cv)` | `server/pipeline.py` | OpenCV 8-bit Lab → `"#rrggbb"` |
| `load_slot_config(path)` | `server/slot_config.py` | Load `slot_config.json` → SlotConfig |
| `save_slot_config(cfg, path)` | `server/slot_config.py` | Persist SlotConfig → JSON file |
| `reference_cells(cfg)` | `server/slot_config.py` | `{cell_index: ref_level}` colour refs |
| `white_reference_cells(cfg)` | `server/slot_config.py` | `set[int]` WB cell indices |
| `sample_cells(cfg)` | `server/slot_config.py` | `{cell_index: slot_id}` sample bottles |
| `upload_image(jpeg_bytes, filename, folder_id, ...)` | `server/integrations/drive.py` | Upload to Google Drive → file_id |
| `append_detail_to_sheet(scan_result, ...)` | `server/integrations/sheets.py` | Append per-bottle rows to Detail tab |
| `append_summary_to_sheet(scan_result, ...)` | `server/integrations/sheets.py` | Append per-scan summary row to Summary tab |
| `write_slot_config_to_sheet(cfg, ...)` | `server/integrations/sheets.py` | Clear + rewrite SlotAssignment tab |
| `init_db(db_path)` | `server/integrations/sqlite_backup.py` | Create tables if not exist |
| `save_scan(scan_id, scan_result, db_path)` | `server/integrations/sqlite_backup.py` | Write scan to SQLite |
| `get_pending_scans(db_path)` | `server/integrations/sqlite_backup.py` | scan_ids not yet synced to Sheets |
| `CameraUndistorter(yaml_path, capture_size)` | `utils/camera_undistort.py` | Pre-computes remap maps — init once at boot |
| `CameraUndistorter.undistort_frame(frame)` | `utils/camera_undistort.py` | Single `cv2.remap()` per frame |
| `lock_focus_picamera2(picam2, focus_value)` | `utils/camera_undistort.py` | Lock Pi Cam V3 lens to calibrated distance |

---

## Error Handling Rules

1. `requests.post` (Pi client) must catch `ConnectionError`, `Timeout`, `HTTPError`, and bare `Exception` — never crash the main loop.
2. YOLO inference errors → HTTP 500 with `{"status": "error", "message": "..."}`.
3. HoughCircles finding zero circles → fall back to uniform grid (handled inside `detect_grid_full`).
4. Google Drive upload failure → log warning, continue; `image_url` still points to local `/img/` file.
5. Google Sheets append failure → log warning, scan stays in SQLite as pending; retried by `_sync_pending()`.
6. `GPIO.cleanup()` must run in a `finally` block in `main.py` (not inside `run_client()`).
7. Camera (`picam2`) must be closed in `try/finally` inside `_capture_image()`.
8. Empty `slot_cfg.cells` → `/analyze` returns HTTP 400 — user must configure slots first.
9. `CameraUndistorter` init failure (yaml missing) → log warning, Pi continues without undistortion.
10. Telegram notification failure → log warning, never blocks response.

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

Logs rotate at 10 MB, retained for 7 days: `logs/app_<date>.log`.

---

## Setup Scripts (`scripts/`)

| Script | Machine | What it does |
|--------|---------|--------------|
| `setup-server.sh` | Ubuntu | Installs deps, creates `.env`, creates directories, installs + enables `urine-server.service` |
| `setup-pi.sh` | Pi | Installs deps, enables I2C + camera, adds user to hardware groups, creates `.env`, installs + enables `urine-pi.service` |

Service management:
```bash
systemctl status  urine-server     # or urine-pi
systemctl restart urine-server
journalctl -u urine-server -f      # live logs
```

---

## What NOT To Do

- Do not use `pip install` — always `uv add` or edit `pyproject.toml` then `uv sync`
- Do not use deep learning for grid detection — only `cv2.HoughCircles` or blob/contour methods
- Do not use YOLO to infer grid structure — YOLO is detection only; grid comes from OpenCV
- Do not hardcode `rows`, `cols`, folder IDs, or thresholds in logic files
- Do not use black padding in `letterbox_white_padding` — must be white `(255, 255, 255)`
- Do not run inference on the Pi — capture + undistort + POST only
- Do not commit `credentials/credentials.json`, `.env`, or `models/`
- Do not use a fixed absolute ΔE threshold for classification — use the hybrid path+histogram scorer
- Do not use fixed column-zone color expectations — reference levels come from `slot_config.json`
- Do not import OpenVINO — project uses PyTorch/ultralytics
- Do not use `gspread` — use `google-api-python-client` directly for consistency with Drive
- Do not block the GPIO event loop with LCD/TM1637 writes — use daemon threads for scroll and spinner
- Do not call relay channel ON before previous channel is OFF — always turn off first with 50 ms gap
- Do not omit `missing_slots` from the API response — Pi relay and LCD line 3 depend on it
- Do not put Google Drive or Sheets logic on the Pi — Pi has no Google credentials
- Do not install `google-api-python-client`, `google-auth-*` on the Pi — server extra only
- Do not sample color from the center point — always use `extract_bottle_color()` (annular ring)
- Do not add a separate `/test-upload` endpoint — `/analyze` serves Pi and browser identically
- Do not read scan results from `server/static/results/` — results are in SQLite; images in `logs/img/`
- Do not set `is_reference` and `is_white_reference` both true on the same cell — `ValueError`

---

_Keep this file in sync when architecture, config keys, or API contracts change._
