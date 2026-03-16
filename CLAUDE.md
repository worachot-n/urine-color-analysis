# Claude Code Project Prompt: High-Precision Urine Color Analysis System

## Project Context

You are building an automated vision system that runs on a **Raspberry Pi 4B** with a **Camera Module 3**. The system analyzes urine sample bottles placed on a large physical grid (16 columns × 14 rows), classifies their color levels (0-4), and validates whether each bottle is in the correct zone. The grid handles **4 groups (A1-A4)**, each with **9 sample slots per group**, across **5 color levels (L0-L4)** — totaling up to **180 sample slots** plus **15 reference slots**. This is a real-world embedded system — not a simulation.

---

## Physical Setup

### The Grid
- A flat board with a printed grid: **16 columns × 14 rows** of lines (forming 15×13 cells).
- Each bottle has a **red circular sticker** on its cap for machine-vision detection.
- The grid is divided into two logical sections:

#### Reference Row (Row 0 — Top of Grid)
- **15 bottles** pre-filled with known color standards: **3 bottles per color level (L0-L4)**.
- These **never move**. They serve as the dynamic calibration reference every scan cycle.
- Slot IDs: `REF_L0`, `REF_L1`, `REF_L2`, `REF_L3`, `REF_L4` (each covers 3 bottle positions).

#### Main Grid (Rows 1-12 — Sample Area)
- **4 Groups:** A1, A2, A3, A4 (each group occupies 3 rows).
- **9 sample slots per group** (numbered 1-9, arranged in a 3×3 block).
- **5 color level columns** (L0-L4), each 3 cells wide.
- **Column 0** of every row is a dead zone marked `ZZ` (unused/structural).

#### Slot Naming Convention: `A{group}{slot}_{level}`
```
A11_0  =  Group A1, Slot 1, expected Color Level 0
A29_3  =  Group A2, Slot 9, expected Color Level 3
A45_4  =  Group A4, Slot 5, expected Color Level 4
```

#### Visual Layout (simplified):
```
Col:  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
     [ZZ] [--- REF_L0 ---] [--- REF_L1 ---] [--- REF_L2 ---] [--- REF_L3 ---] [--- REF_L4 ---]   ← Row 0 (Reference)
     ───────────────────────────────────────────────────────────────────────────────────────────
     [ZZ] [A11_0][A12_0][A13_0] [A11_1][A12_1][A13_1] [A11_2][A12_2][A13_2] [A11_3][A12_3][A13_3] [A11_4][A12_4][A13_4]  ← Row 1  ┐
     [ZZ] [A14_0][A15_0][A16_0] [A14_1][A15_1][A16_1] [A14_2][A15_2][A16_2] [A14_3][A15_3][A16_3] [A14_4][A15_4][A16_4]  ← Row 2  │ Group A1
     [ZZ] [A17_0][A18_0][A19_0] [A17_1][A18_1][A19_1] [A17_2][A18_2][A19_2] [A17_3][A18_3][A19_3] [A17_4][A18_4][A19_4]  ← Row 3  ┘
     [ZZ] [A21_0][A22_0][A23_0] [A21_1][A22_1][A23_1] [A21_2][A22_2][A23_2] [A21_3][A22_3][A23_3] [A21_4][A22_4][A23_4]  ← Row 4  ┐
     [ZZ] [A24_0][A25_0][A26_0] [A24_1][A25_1][A26_1] [A24_2][A25_2][A26_2] [A24_3][A25_3][A26_3] [A24_4][A25_4][A26_4]  ← Row 5  │ Group A2
     [ZZ] [A27_0][A28_0][A29_0] [A27_1][A28_1][A29_1] [A27_2][A28_2][A29_2] [A27_3][A28_3][A29_3] [A27_4][A28_4][A29_4]  ← Row 6  ┘
     [ZZ] [A31_0][A32_0][A33_0] [A31_1][A32_1][A33_1] [A31_2][A32_2][A33_2] [A31_3][A32_3][A33_3] [A31_4][A32_4][A33_4]  ← Row 7  ┐
     [ZZ] [A34_0][A35_0][A36_0] [A34_1][A35_1][A36_1] [A34_2][A35_2][A36_2] [A34_3][A35_3][A36_3] [A34_4][A35_4][A36_4]  ← Row 8  │ Group A3
     [ZZ] [A37_0][A38_0][A39_0] [A37_1][A38_1][A39_1] [A37_2][A38_2][A39_2] [A37_3][A38_3][A39_3] [A37_4][A38_4][A39_4]  ← Row 9  ┘
     [ZZ] [A41_0][A42_0][A43_0] [A41_1][A42_1][A43_1] [A41_2][A42_2][A43_2] [A41_3][A42_3][A43_3] [A41_4][A42_4][A43_4]  ← Row 10 ┐
     [ZZ] [A44_0][A45_0][A46_0] [A44_1][A45_1][A46_1] [A44_2][A45_2][A46_2] [A44_3][A45_3][A46_3] [A44_4][A45_4][A46_4]  ← Row 11 │ Group A4
     [ZZ] [A47_0][A48_0][A49_0] [A47_1][A48_1][A49_1] [A47_2][A48_2][A49_2] [A47_3][A48_3][A49_3] [A47_4][A48_4][A49_4]  ← Row 12 ┘
```

#### Key Structural Rules:
- Each group has **9 samples × 5 levels = 45 slots** per group.
- Total sample slots: **4 groups × 45 = 180 slots**.
- Total grid capacity: **180 sample + 15 reference = 195 bottle positions**.
- The `expected_level` for any slot is encoded in its name suffix (e.g., `_0` = Level 0, `_4` = Level 4).
- The `group` is encoded in the name (e.g., `A1x` = Group A1, `A3x` = Group A3).

### Hardware Components
| Component | Role |
|---|---|
| Raspberry Pi 4B | Main compute unit |
| Camera Module 3 | Image capture (fixed mount, top-down view) |
| HC-SR501 PIR Sensor (GPIO 23) | Motion trigger — starts scan after user walks away |
| 3-Channel Relay Module (GPIO 20, 21, 12) | Controls LED tower (Red/Yellow/Green) via relay switching |
| 5x TM1637 7-Segment Displays | Real-time count per color level (0-4) |
| LCD 20x4 (I2C on SDA=GPIO2, SCL=GPIO3) | Detailed error/status messages |
| Dual LED Light Bars (Left/Right) | Consistent top lighting |
| Polarizing Filter (on lens) | Eliminates specular glare on glass bottles |

---

## Software Architecture

### Language & Libraries
- **Python 3.9+**
- `opencv-python` — image capture and processing
- `numpy` — array operations and color math
- `RPi.GPIO` — GPIO control for LEDs, PIR sensor
- `tm1637` — 7-segment display driver
- `smbus2` — I2C communication for LCD
- `picamera2` — Raspberry Pi camera interface
- `scipy` (optional) — for Delta E calculation via `scipy.spatial.distance`

### Project File Structure
```
urine-color-analyzer/
├── main.py                  # Entry point: event loop, PIR trigger
├── config.py                # All constants, GPIO pins, HSV ranges, thresholds
├── grid_config.json         # Full grid layout: reference slots + 180 sample slots with coords
├── calibration.py           # Semi-auto corner mapping, white balance lock
├── image_processing.py      # Red mask, Hough circles, bottle localization
├── color_analysis.py        # Reference sampling, CIE Lab conversion, Delta E classification
├── hardware.py              # Relay-based LED tower, TM1637 displays, LCD control
├── grid.py                  # Load grid_config.json, point-in-polygon, slot lookup
├── utils.py                 # Logging, image saving for debug
└── tests/
    ├── test_color_analysis.py
    ├── test_grid.py
    └── test_image_processing.py
```

---

## Core Logic — Step by Step

### 1. Startup Sequence (`main.py`, `calibration.py`)

```
Boot → Lock White Balance & Auto-Exposure against white card
     → Load saved grid corner coordinates (or run semi-auto calibration if first run)
     → Initialize all hardware (LED tower, displays, LCD)
     → Enter PIR polling loop
```

- **White Balance Lock:** Capture a frame of a white reference card. Use `picamera2` controls to lock AWB gains and exposure. Store these values so the camera does not auto-adjust throughout the day.
- **Semi-Auto Calibration:** When the system is physically relocated, the user triggers a calibration mode. The system captures a frame, the user identifies 4 corner points of the grid (can be via terminal input or a simple GUI), and these coordinates are saved to `config.json`. All grid slot positions are derived from these 4 corners using perspective transform.

### 2. PIR Trigger & Capture (`main.py`)

```
PIR detects motion (GPIO 23) → Set YELLOW relay (GPIO 21) ON
                   → Wait PIR_COOLDOWN + settle delay for hand to leave
                   → Capture high-res frame
                   → Begin processing pipeline
```

- The delay values should be in `config.py` as `PIR_COOLDOWN_SEC = 5` and `PIR_SETTLE_DELAY_SEC = 2.5`.
- Capture at maximum resolution for accuracy, then resize only if needed for performance.

### 3. Image Processing Pipeline (`image_processing.py`)

#### 3a. Red Cap Detection
```python
# Detect red circular stickers on bottle caps
# Red exists in two HSV ranges (wraps around 0/180)

HSV_RED_LOWER_1 = (0, 120, 70)
HSV_RED_UPPER_1 = (10, 255, 255)
HSV_RED_LOWER_2 = (170, 120, 70)
HSV_RED_UPPER_2 = (180, 255, 255)

# Pipeline:
# 1. Convert frame to HSV
# 2. Create dual-range red mask (bitwise OR of both ranges)
# 3. Apply GaussianBlur (kernel 5x5) to reduce noise
# 4. Apply morphological Opening (remove small noise)
# 5. Apply morphological Closing (fill small gaps)
# 6. Run HoughCircles on the cleaned mask
```

#### 3b. Bottle Localization — Majority Rule (`grid.py`)
```
For each detected circle:
  1. Get circle center (cx, cy) and radius r
  2. For each grid slot polygon:
     - Use cv2.pointPolygonTest with the circle center
     - Calculate how much of the circle's contour area overlaps with the slot polygon
  3. Assign the bottle to the slot that contains the MAJORITY of its footprint
  4. Flag if a bottle straddles two slots (potential error)
  5. Flag if two bottles are assigned to the same slot (duplicate error)
```

This is critical — a simple center-point check fails when bottles sit on grid lines.

### 4. Dynamic Color Calibration (`color_analysis.py`)

This is the most important algorithm in the system.

#### 4a. Reference Row Sampling
```
Every scan cycle:
  1. Locate the 15 reference bottles (top section of grid)
  2. For each reference bottle:
     a. Crop a square region at the bottle center
     b. Apply an INNER CROP (shrink by 10-20px per side) to exclude cap edges
     c. Convert cropped region from BGR to CIE Lab color space
     d. Extract the median L, a, b values
  3. Group by known color level (3 bottles per level)
  4. Compute the median of each group → This is the LIVE STANDARD for that color level
```

#### 4b. Sample Classification
```
For each sample bottle detected in the main grid (up to 180 slots):
  1. Identify which slot ID the bottle occupies (e.g., "A25_2")
  2. Same inner-crop and median extraction as above
  3. Convert to CIE Lab
  4. Calculate Delta E (CIE76 or CIE2000) against each of the 5 live standards
  5. Classify as the color level with the LOWEST Delta E
  6. Apply a confidence threshold — if minimum Delta E > threshold, flag as "Uncertain"
  7. Validate: Compare classified level against the slot's expected_level (suffix of slot ID)
     - e.g., bottle in "A25_2" must classify as Level 2; if it classifies as Level 3 → ERROR
```

**Why this works:** By re-reading the reference bottles every cycle, the system self-calibrates against the current lighting. If the sun shifts, if a cloud passes, if someone turns on a room light — the reference bottles experience the exact same color shift as the sample bottles. The relative difference stays constant.

### 5. Validation & Output (`main.py`, `hardware.py`)

```
After classification:
  1. Check: Is the bottle's classified color correct for its zone?
     - The expected level is embedded in the slot ID suffix:
       "A25_2" → expected Level 2
       "A41_0" → expected Level 0
     - If classified_level != expected_level → COLOR MISMATCH ERROR
  2. Check: Are there any duplicate slot assignments?
     - Two bottles detected in the same slot → DUPLICATE ERROR
  3. Check: Are any slots in a group missing? (optional occupancy check)
  4. Count bottles per color level across all groups for TM1637 displays
  5. Update displays:
     - TM1637 H0-H4: Show total count of bottles at each color level
     - LCD: Show detailed status or error message

LED Tower Logic (controlled via 3-Channel Relay — ACTIVE LOW):
  - GREEN (Relay 3, GPIO 12): All detected bottles in correct zones, no errors
  - RED (Relay 1, GPIO 20): Any mismatch or duplicate detected
     - LCD shows specifics: "Error: A25_2 got Lv3"
     - LCD shows specifics: "Duplicate: A11_0"
  - YELLOW (Relay 2, GPIO 21): Only during processing (before result)
```

---

## Key Design Principles

1. **All color comparison must use CIE Lab Delta E** — never compare raw RGB or HSV values for classification. RGB/HSV is only for the red-cap detection step.

2. **Never trust absolute color values** — always compare samples RELATIVE to the reference row captured in the same frame, under the same lighting.

3. **Defensive coding for hardware:**
   - Wrap all GPIO and I2C calls in try/except
   - Implement a watchdog: if processing takes > 10 seconds, timeout and reset
   - Log every cycle with timestamp, detected bottles, classifications, and errors

4. **Debug mode:** Save annotated images (with detected circles, grid overlay, color values) to a `/debug/` folder for troubleshooting.

5. **Config-driven:** All thresholds, pin numbers, grid coordinates, HSV ranges, and timing values must live in `config.py` — never hardcoded in logic.

---

## Implementation Priority

Build and test in this order:

1. **`config.py`** — All constants and pin definitions
2. **`color_analysis.py`** — The Delta E calibration engine (testable with static images)
3. **`image_processing.py`** — Red detection + Hough circles (testable with static images)
4. **`grid.py`** — Grid mapping and majority-rule localization
5. **`hardware.py`** — LED, display, LCD drivers
6. **`calibration.py`** — White balance lock and corner mapping
7. **`main.py`** — Integration: PIR loop → capture → process → classify → display
8. **`tests/`** — Unit tests for color analysis, grid logic, and image processing

---

## Example Config Values (Starting Points)

```python
# config.py

# =============================================================================
# GPIO PIN ASSIGNMENTS (BCM Numbering)
# =============================================================================

# --- TM1637 7-Segment Displays (5 units, one per hydration level H0-H4) ---
TM1637_DISPLAYS = {
    "H0": {"CLK": 4,  "DIO": 17},   # Level 0 (Well Hydrated)
    "H1": {"CLK": 27, "DIO": 22},   # Level 1
    "H2": {"CLK": 5,  "DIO": 6},    # Level 2
    "H3": {"CLK": 13, "DIO": 19},   # Level 3
    "H4": {"CLK": 26, "DIO": 16},   # Level 4 (Dehydrated)
}

# --- LCD 20x4 via I2C ---
LCD_SDA = 2                 # I2C Data (physical pin 3)
LCD_SCL = 3                 # I2C Clock (physical pin 5)
LCD_I2C_ADDRESS = 0x27
LCD_COLS = 20
LCD_ROWS = 4

# --- 3-Channel Relay Module (controls LED Tower) ---
RELAY_LED_RED = 20          # Relay 1 → Red LED
RELAY_LED_YELLOW = 21       # Relay 2 → Yellow LED
RELAY_LED_GREEN = 12        # Relay 3 → Green LED

# --- PIR Sensor ---
PIN_PIR = 23
PIR_COOLDOWN_SEC = 5        # Minimum seconds between triggers
PIR_SETTLE_DELAY_SEC = 2.5  # Delay after trigger for hand to leave frame

# --- Power Reference (documentation only, not used in code) ---
# VCC_3V3 = Physical Pin 1
# VCC_5V  = Physical Pin 2
# GND     = Physical Pin 6

# =============================================================================
# CAMERA SETTINGS
# =============================================================================
CAPTURE_RESOLUTION = (4608, 2592)  # Camera Module 3 max
AWB_LOCK = True
AE_LOCK = True

# =============================================================================
# IMAGE PROCESSING
# =============================================================================

# --- Red Detection (HSV) ---
HSV_RED_LOWER_1 = (0, 120, 70)
HSV_RED_UPPER_1 = (10, 255, 255)
HSV_RED_LOWER_2 = (170, 120, 70)
HSV_RED_UPPER_2 = (180, 255, 255)

# --- Morphology ---
MORPH_KERNEL_SIZE = 5
GAUSSIAN_BLUR_KERNEL = (5, 5)

# --- Hough Circles ---
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 50
HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 30
HOUGH_MIN_RADIUS = 20
HOUGH_MAX_RADIUS = 60

# =============================================================================
# COLOR ANALYSIS
# =============================================================================
INNER_CROP_PX = 15          # Pixels to shrink from each side
DELTA_E_THRESHOLD = 15.0    # Max Delta E for confident classification
COLOR_LEVELS = 5            # Number of distinct urine color levels (H0-H4)
REFS_PER_LEVEL = 3          # Reference bottles per level

# =============================================================================
# GRID CONFIGURATION (loaded from grid_config.json)
# =============================================================================
GRID_CONFIG_FILE = "grid_config.json"

# Grid dimensions
GRID_COLS = 16              # 16 columns (col 0 = ZZ dead zone)
GRID_ROWS = 14              # 14 rows (row 0 = reference, rows 1-12 = samples, row 13 = unused)
GROUPS = ["A1", "A2", "A3", "A4"]
SLOTS_PER_GROUP = 9         # Slots 1-9 per group
LEVELS = [0, 1, 2, 3, 4]   # Color levels

# =============================================================================
# RELAY CONTROL LOGIC
# =============================================================================
# Note: Relay modules are typically ACTIVE LOW
# RELAY_ON  = GPIO.LOW  (relay energized, LED ON)
# RELAY_OFF = GPIO.HIGH (relay released, LED OFF)
RELAY_ACTIVE_LOW = True

# =============================================================================
# DEBUG
# =============================================================================
DEBUG_MODE = True
DEBUG_IMAGE_DIR = "/home/pi/urine-analyzer/debug/"
WATCHDOG_TIMEOUT_SEC = 10
```

---

## What to Build First

Start with **`color_analysis.py`** — this is the core intelligence of the system. Write these functions:

1. `extract_bottle_color(frame, cx, cy, radius, inner_crop_px)` → Returns median CIE Lab values
2. `build_reference_baseline(frame, reference_positions)` → Returns dict of {level: (L, a, b)}
3. `classify_sample(sample_lab, baseline, threshold)` → Returns (level, delta_e, confident: bool)
4. `delta_e_cie76(lab1, lab2)` → Float distance between two Lab colors

Make these functions testable with static images — no hardware dependency.

---

## Grid Configuration File Schema (`grid_config.json`)

The grid layout and all slot coordinates are stored in a separate JSON file. This file is generated during semi-auto calibration and loaded at startup. The code must parse this structure to know where every slot is.

```json
{
  "system_metadata": {
    "project_name": "Urine Color Analysis",
    "grid_dimensions": "16x14 lines",
    "calibration_date": "2026-03-16"
  },

  "reference_row": {
    "description": "Top row for dynamic color calibration (3 bottles per level)",
    "slots": {
      "REF_L0": {"coords": [[x,y], [x,y], [x,y], [x,y]], "level": 0, "samples": 3},
      "REF_L1": {"coords": [[x,y], [x,y], [x,y], [x,y]], "level": 1, "samples": 3},
      "REF_L2": {"coords": [[x,y], [x,y], [x,y], [x,y]], "level": 2, "samples": 3},
      "REF_L3": {"coords": [[x,y], [x,y], [x,y], [x,y]], "level": 3, "samples": 3},
      "REF_L4": {"coords": [[x,y], [x,y], [x,y], [x,y]], "level": 4, "samples": 3}
    }
  },

  "main_grid": {
    "description": "Processing slots for Groups A1-A4 across Color Levels 0-4",
    "layout": [
      ["ZZ", "A11_0", "A12_0", "A13_0", "A11_1", "A12_1", "A13_1", "A11_2", "A12_2", "A13_2", "A11_3", "A12_3", "A13_3", "A11_4", "A12_4", "A13_4"],
      ["ZZ", "A14_0", "A15_0", "A16_0", "A14_1", "A15_1", "A16_1", "A14_2", "A15_2", "A16_2", "A14_3", "A15_3", "A16_3", "A14_4", "A15_4", "A16_4"],
      ["ZZ", "A17_0", "A18_0", "A19_0", "A17_1", "A18_1", "A19_1", "A17_2", "A18_2", "A19_2", "A17_3", "A18_3", "A19_3", "A17_4", "A18_4", "A19_4"],
      ["ZZ", "A21_0", "A22_0", "A23_0", "A21_1", "A22_1", "A23_1", "A21_2", "A22_2", "A23_2", "A21_3", "A22_3", "A23_3", "A21_4", "A22_4", "A23_4"],
      ["ZZ", "A24_0", "A25_0", "A26_0", "A24_1", "A25_1", "A26_1", "A24_2", "A25_2", "A26_2", "A24_3", "A25_3", "A26_3", "A24_4", "A25_4", "A26_4"],
      ["ZZ", "A27_0", "A28_0", "A29_0", "A27_1", "A28_1", "A29_1", "A27_2", "A28_2", "A29_2", "A27_3", "A28_3", "A29_3", "A27_4", "A28_4", "A29_4"],
      ["ZZ", "A31_0", "A32_0", "A33_0", "A31_1", "A32_1", "A33_1", "A31_2", "A32_2", "A33_2", "A31_3", "A32_3", "A33_3", "A31_4", "A32_4", "A33_4"],
      ["ZZ", "A34_0", "A35_0", "A36_0", "A34_1", "A35_1", "A36_1", "A34_2", "A35_2", "A36_2", "A34_3", "A35_3", "A36_3", "A34_4", "A35_4", "A36_4"],
      ["ZZ", "A37_0", "A38_0", "A39_0", "A37_1", "A38_1", "A39_1", "A37_2", "A38_2", "A39_2", "A37_3", "A38_3", "A39_3", "A37_4", "A38_4", "A39_4"],
      ["ZZ", "A41_0", "A42_0", "A43_0", "A41_1", "A42_1", "A43_1", "A41_2", "A42_2", "A43_2", "A41_3", "A42_3", "A43_3", "A41_4", "A42_4", "A43_4"],
      ["ZZ", "A44_0", "A45_0", "A46_0", "A44_1", "A45_1", "A46_1", "A44_2", "A45_2", "A46_2", "A44_3", "A45_3", "A46_3", "A44_4", "A45_4", "A46_4"],
      ["ZZ", "A47_0", "A48_0", "A49_0", "A47_1", "A48_1", "A49_1", "A47_2", "A48_2", "A49_2", "A47_3", "A48_3", "A49_3", "A47_4", "A48_4", "A49_4"]
    ],
    "slot_data": {
      "A11_0": {"coords": [[x1,y1], [x2,y1], [x2,y2], [x1,y2]], "expected_level": 0, "group": "A1"},
      "A12_0": {"coords": [[x1,y1], [x2,y1], [x2,y2], [x1,y2]], "expected_level": 0, "group": "A1"},
      "...": "Repeat for all 180 slot IDs — each with 4-corner polygon coords, expected_level, and group"
    }
  }
}
```

### How the Code Must Use `grid_config.json`:

1. **`grid.py` loads this file at startup** and builds a lookup dict: `slot_id → polygon coords + expected_level + group`.
2. **`image_processing.py`** uses the polygon coords from `slot_data` for the Majority Rule bottle localization (pointPolygonTest against each slot's 4-corner polygon).
3. **`color_analysis.py`** uses `reference_row.slots` to know where to sample the 15 reference bottles.
4. **`main.py`** uses `expected_level` to validate each classified bottle, and `group` to aggregate results per group.
5. **`calibration.py`** generates this file — during semi-auto calibration, the user maps the 4 outer corners of the grid, and the system computes all 195 slot polygons via perspective-corrected interpolation.

### Slot ID Parsing Helper:
```python
def parse_slot_id(slot_id: str) -> dict:
    """Parse 'A25_3' → {'group': 'A2', 'slot': 5, 'expected_level': 3}"""
    # slot_id format: A{group_num}{slot_num}_{level}
    group = "A" + slot_id[1]          # "A2"
    slot = int(slot_id[2])            # 5
    level = int(slot_id.split("_")[1])  # 3
    return {"group": group, "slot": slot, "expected_level": level}
```
