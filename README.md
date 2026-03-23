# Urine Color Analysis System

An automated vision system running on **Raspberry Pi 4B** that analyzes urine sample bottles placed on a physical 16×14 grid, classifies their color levels (L0–L4), and validates whether each bottle is in the correct zone.

---

## How It Works

A top-down camera captures the grid when a button is pressed. The system runs a hybrid pipeline:

1. **WiFi setup** on first boot — LCD + hotspot + captive-portal web form at `http://10.42.0.1`
2. **Web dashboard** at `http://<pi-ip>:5000` — results, annotated images, grid calibration
3. **Waits for button press** (GPIO 24) to start each scan
4. **Captures 3 snapshots** with 200 ms delay and ±15% exposure variation (multi-snapshot consensus)
5. **YOLO detection** (YOLOv8s / OpenVINO) — two-class model detects `ref_bottle` and `sample_bottle`
6. **Consensus filter** — keeps only detections that appear in ≥ 2 of 3 snapshots (eliminates glare ghosts)
7. **Geometric validation** — maps each confirmed box center to the nearest grid slot
8. **Ghost check** — rejects any detection whose CIE Lab color is too close to white paper (ΔE < 8)
9. **Auto-calibrated baseline** — `ref_bottle` YOLO detections from row 0 build a live color reference; calibrated slot positions are used as fallback for any missed level
10. **CIE Lab Delta E classification** — each sample bottle is assigned the level with the lowest ΔE distance to the live baseline; never raw RGB
11. **Placement validation** — classified level must match the slot's expected level encoded in the slot ID
12. **Hardware output** — relay LED tower (Red/Yellow/Green), 5× TM1637 displays, LCD 16×4
13. **Telegram report** — text summary + annotated JPEG after every scan
14. **SQLite log** — every scan saved to `logs/scan_results.db`

---

## Physical Grid Layout

```
Col:  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
     [ZZ] [------- REF_L0 -------] [--- REF_L1 ---] [--- REF_L2 ---] [--- REF_L3 ---] [--- REF_L4 ---]  ← Row 0 (Reference)
     ─────────────────────────────────────────────────────────────────────────────────────────────────
     [ZZ] [A11_0][A12_0][A13_0]   [A11_1]…[A13_1]   …   [A11_4]…[A13_4]   ← Row 1  ┐
     [ZZ] [A14_0][A15_0][A16_0]   …                                          ← Row 2  │ Group A1
     [ZZ] [A17_0][A18_0][A19_0]   …                                          ← Row 3  ┘
     …                                                                         Rows 4–6  Group A2
     …                                                                         Rows 7–9  Group A3
     [ZZ] [A47_0]…[A49_0]         …   [A47_4]…[A49_4]                        ← Row 12 ┘ Group A4
```

- **Col 0 (ZZ)** — exclusion zone, never processed
- **Row 0** — 15 fixed reference bottles (3 per level L0–L4), never moved
- **Rows 1–12** — 180 sample slots: 4 groups × 9 slots × 5 levels
- **Slot ID format:** `A{group}{slot}_{level}` — e.g. `A25_3` = Group A2, Slot 5, expected Level 3

---

## Hardware

| Component | GPIO / Interface |
|-----------|-----------------|
| Raspberry Pi 4B | — |
| Camera Module 3 | CSI (via `picamera2`) |
| Push Button | GPIO 24 / Pin 18 (active LOW, internal pull-up) |
| 3-Channel Relay (LED Tower) | Red=GPIO 21, Yellow=GPIO 20, Green=GPIO 12 (active LOW) |
| 5× TM1637 7-Segment Displays | CLK/DIO pairs — see `configs/config.toml` |
| LCD 16×4 I2C | SDA=GPIO 2, SCL=GPIO 3, addr `0x27`, bus 1 |

---

## Project Structure

```
urine-color-analysis/
│
├── main.py                     # Entry point — boot sequence, scan loop
│
├── configs/
│   ├── config.toml             # All constants: GPIO, camera, YOLO, thresholds, paths
│   └── config.py               # TOML loader — exposes flat constants
│
├── utils/
│   ├── calibration.py          # AWB lock, grid math, semi-auto corner calibration
│   ├── color_analysis.py       # CIE Lab extraction, Delta E, baseline builder, classifier
│   ├── db.py                   # SQLite scan result persistence
│   ├── grid.py                 # GridConfig loader, majority-rule slot assignment
│   ├── hardware.py             # Relay, TM1637, LCD, button drivers (graceful no-op on non-Pi)
│   ├── image_processing.py     # CLAHE, red mask, morphology, contour detection (legacy fallback)
│   ├── network.py              # WiFi detection, hotspot (nmcli), captive portal
│   ├── utils.py                # Logger, annotated image saving
│   ├── web_server.py           # Flask dashboard + calibration UI + WiFi setup portal
│   └── yolo_detector.py        # YOLOv8 OpenVINO wrapper — consensus, geometric validation
│
├── bot/
│   └── telegram_bot.py         # Telegram summary + image sender
│
├── scripts/
│   └── export_model.py         # YOLOv8s → OpenVINO export + two-class training spec
│
├── templates/
│   ├── base.html               # Shared layout (dark green / gold theme, Kanit font)
│   ├── dashboard.html          # Scan results page
│   ├── analysis.html           # Per-slot detail view
│   ├── calibrate.html          # Two-phase grid calibration (canvas + line editor)
│   ├── wifi_setup.html         # WiFi credential form
│   ├── wifi_connecting.html    # Connection progress indicator
│   └── global.css
│
├── assets/
│   ├── logo.svg
│   └── logo.ico
│
├── models/
│   └── bottle_yolov8s_openvino/  # Place exported OpenVINO model here (see scripts/export_model.py)
│       ├── yolov8s.xml
│       ├── yolov8s.bin
│       └── metadata.yaml
│
├── tests/
│   ├── test_color_analysis.py
│   ├── test_grid.py
│   ├── test_image_processing.py
│   ├── test_web.py             # Local Windows test server (port 5001) — see Test Lab section
│   └── templates/
│       └── test_ui.html        # Test Lab UI (Calibrate / Detect / Classify tabs)
│
├── grid_config.json            # Generated by calibration — slot polygons (195 slots)
├── logs/
│   ├── img/                    # Annotated scan JPEGs
│   └── scan_results.db         # SQLite history
└── data/                       # Local test images (for development on Windows)
```

---

## Installation

### Requirements

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager

### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Set up the project

```bash
cd urine-color-analysis
uv sync --group dev
```

Installs: `opencv-python`, `numpy`, `flask`, `python-dotenv`, `requests`, `pytest`.

### Raspberry Pi — additional packages

```bash
pip install -r requirements-pi.txt
```

Installs Pi-only packages: `picamera2`, `RPi.GPIO`, `smbus2`.

> All hardware calls degrade gracefully to no-ops when these libraries are absent — the full analysis pipeline runs on Windows/Mac for development.

---

## Model Setup

The YOLO detection engine requires a trained YOLOv8s model exported to OpenVINO format.

### Train (on a PC with GPU)

```bash
yolo train \
    model=yolov8s.pt \
    data=dataset/data.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    mosaic=1.0 blur_p=0.3 hsv_v=0.4 hsv_s=0.5 \
    flipud=0.0 fliplr=0.5 \
    project=runs/urine_detector name=v1
```

**Two-class dataset** (`data.yaml`):
```yaml
nc: 2
names: ['ref_bottle', 'sample_bottle']
```

- **Class 0 `ref_bottle`** — 15 reference bottles in the top row
- **Class 1 `sample_bottle`** — all test bottles in the main 16×14 grid
- Include 10–20% empty-grid images (no labels) as background negatives to suppress glare ghosts

### Export to OpenVINO

```bash
python scripts/export_model.py
```

Copy the output folder to `models/bottle_yolov8s_openvino/` on the Pi.

---

## Configuration

All settings live in **`configs/config.toml`** — edit that file only; `configs/config.py` is a read-only loader.

### Key parameters

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `[gpio.button]` | `pin` | `24` | Push button GPIO pin |
| `[camera]` | `capture_resolution` | `[4608, 2592]` | Camera Module 3 max |
| `[yolo]` | `model_path` | `models/bottle_yolov8s_openvino` | OpenVINO model directory |
| `[yolo]` | `conf_threshold` | `0.75` | YOLO confidence cutoff |
| `[yolo]` | `snapshots` | `3` | Captures per button press |
| `[yolo]` | `consensus_min_votes` | `2` | Snapshots a box must appear in |
| `[yolo]` | `slot_max_dist_ratio` | `0.60` | Max box→slot distance (× slot radius) |
| `[color_analysis]` | `ghost_de_threshold` | `8.0` | Min ΔE vs white paper to accept detection |
| `[color_analysis]` | `confidence_margin` | `3.0` | Best–second-best ΔE gap for "confident" |
| `[color_analysis]` | `ref_inner_crop_px` | `25` | Inner crop for reference bottles |
| `[color_analysis]` | `inner_crop_px` | `15` | Inner crop for sample bottles |
| `[system]` | `watchdog_timeout_sec` | `60` | Max processing time per scan |
| `[debug]` | `img_dir` | `logs/img/` | Annotated scan image output |

### Telegram

Add credentials to `.env`:

```env
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## Running

### On Raspberry Pi

```bash
python main.py
```

**Boot sequence:**
1. Hardware init — relay, LCD, button
2. Network — if no WiFi: launches hotspot `signal` (password: `angad@1234`); LCD shows IP
3. AWB lock — point camera at a white card; exposure + white balance locked
4. Web server starts at `http://<pi-ip>:5000`
5. Grid load — reads `grid_config.json`; LCD prompts calibration via web if missing
6. Press button → scan starts

### Scan cycle (each button press)

1. Yellow LED on
2. Capture 3 snapshots (200 ms apart, ±15% exposure variation)
3. YOLO inference × 3 → consensus filter → geometric slot assignment
4. Ghost rejection (ΔE < 8 vs white paper → skip)
5. Reference baseline built from detected `ref_bottle` positions
6. CIE Lab Delta E classification for each `sample_bottle`
7. Placement validation — classified level vs expected level in slot ID
8. Green LED = all OK / Red LED = mismatch or duplicate
9. TM1637 displays updated (count per level L0–L4)
10. LCD shows result or first error
11. Telegram: text summary + annotated image
12. Result saved to `logs/scan_results.db` + `logs/img/`

---

## Web Dashboard (`http://<pi-ip>:5000`)

### Results (`/`)
- Bottle counts per level (L0–L4)
- Error list (mismatches, duplicates)
- Latest annotated scan image
- Auto-refreshes every 10 seconds

### Analysis (`/analysis`)
- Per-slot breakdown: detected level, ΔE score, confidence, error flag

### Calibrate (`/calibrate`)

Used to map the physical grid on first run or after moving the camera.

1. **Capture** from Pi camera or **Upload** a photo
2. **Click 4 corners** of the grid: TL → TR → BR → BL
3. Grid polygons are computed automatically (bilinear interpolation across 16×14 lines)
4. **Fine-tune** — drag any grid line to correct misalignment
5. **Save** — writes `grid_config.json`; running scan loop reloads without restart

---

## Local Development — Test Lab

For developing and validating on Windows without Pi hardware:

```bash
cd urine-color-analysis
python tests/test_web.py
```

Open **http://localhost:5001**

Three tabs:

| Tab | What it tests |
|-----|---------------|
| **① Calibrate** | Click 4 corners on a `data/` image → computes `grid_config.json` → shows grid overlay |
| **② Detect** | Runs YOLO `detect_once()` on the image; falls back to OpenCV red-cap detection if model absent |
| **③ Classify** | Full pipeline — detection → consensus → baseline → ΔE classification; shows slot table + bar chart |

> **Python path note:** The project venv may not have `opencv` and `flask`. Run with the system Python if `uv sync` hasn't been run yet, or install manually: `pip install opencv-python-headless flask`.

---

## Running Tests

```bash
uv run pytest
# or verbose:
uv run pytest -v
```

---

## Output

### LED Tower

| State | LED |
|-------|-----|
| Processing | Yellow |
| All bottles correct | Green |
| Any error | Red |

### TM1637 Displays

Five displays (H0–H4) show the count of bottles classified at each color level.

### LCD Messages (16×4)

| Condition | Line 1 | Line 2 |
|-----------|--------|--------|
| No WiFi | `WiFi Not Found` | `Hotspot:signal` |
| WiFi connected | `WiFi Connected` | `IP:<ip>` |
| No grid config | `No grid config` | `Open <ip>:5000` |
| Scanning | `Scanning...` | — |
| All correct | `ALL OK` | level counts |
| Error detected | `Error:` | slot + level |
| Timeout | `ERROR:Timeout` | — |

### Telegram Report

```
Scan Result — 2026-03-22 14:32
L0:5 | L1:3 | L2:7 | L3:2 | L4:1
Errors: A11_0 Dup, A25_2 Mismatch (L3)
Status: ❌
```

Followed by the annotated scan image.

### Annotated Images (`logs/img/`)

Every scan saves `YYYY-MM-DD_HH-MM-SS.jpg` with:
- Grid slot outlines (green = sample, cyan = reference)
- Detected bottles — green box = OK, orange = uncertain, red = error/duplicate
- Label per bottle: `SlotID L{level} ΔE={value}`
- Red ✕ on rejected ghost detections

---

## Architecture Overview

```
Button press
    ↓
capture_multi_snapshot()          3 frames × (normal / -15% / +15% exposure)
    ↓
YoloBottleDetector.detect_once()  × 3 frames  →  [[cx,cy,w,h,conf,cls], ...]
    ↓
consensus_filter()                keep boxes in ≥ 2/3 snapshots  →  confirmed boxes
    ↓
geometric_validate_ref()          cls=0 → {level: [(cx,cy,r), ...]}  ref positions
geometric_validate()              cls=1 → {slot_id: {cx,cy,w,h,conf}}  sample hits
    ↓
build_reference_baseline()        ref positions → {level: (L,a,b)}  live standard
    ↓
for each sample hit:
    extract_bottle_color()        inner crop → median CIE Lab
    delta_e_cie76(lab, WHITE_LAB) ghost check  (ΔE < 8 → reject)
    classify_sample()             min ΔE to baseline → level + confidence
    validate vs expected_level    mismatch → error flag
    ↓
update_hardware()  +  save_annotated_image()  +  telegram_bot.send()
```
