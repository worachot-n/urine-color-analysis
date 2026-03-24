"""
Configuration loader — reads configs/config.toml and exposes flat constants.

To change any setting, edit configs/config.toml.
This file should not need to be edited directly.
"""

import tomllib
from pathlib import Path

_TOML_PATH = Path(__file__).parent / "config.toml"

with open(_TOML_PATH, "rb") as _f:
    _cfg = tomllib.load(_f)

# =============================================================================
# GPIO
# =============================================================================
TM1637_DISPLAYS: dict = {
    k: {"CLK": v["CLK"], "DIO": v["DIO"]}
    for k, v in _cfg["gpio"]["tm1637"].items()
}

LCD_SDA:         int  = _cfg["gpio"]["lcd"]["sda"]
LCD_SCL:         int  = _cfg["gpio"]["lcd"]["scl"]
LCD_I2C_ADDRESS: int  = _cfg["gpio"]["lcd"]["i2c_address"]
LCD_I2C_BUS:     int  = _cfg["gpio"]["lcd"]["i2c_bus"]
LCD_COLS:        int  = _cfg["gpio"]["lcd"]["cols"]
LCD_ROWS:        int  = _cfg["gpio"]["lcd"]["rows"]

RELAY_LED_RED:    int  = _cfg["gpio"]["relay"]["led_red"]
RELAY_LED_YELLOW: int  = _cfg["gpio"]["relay"]["led_yellow"]
RELAY_LED_GREEN:  int  = _cfg["gpio"]["relay"]["led_green"]
RELAY_ACTIVE_LOW: bool = _cfg["gpio"]["relay"]["active_low"]

PIN_BUTTON:         int = _cfg["gpio"]["button"]["pin"]
BUTTON_DEBOUNCE_MS: int = _cfg["gpio"]["button"]["debounce_ms"]

# Power supply pins (physical pin numbers — documentation only)
VCC_3V3: int = _cfg["gpio"]["power"]["vcc_3v3"]
VCC_5V:  int = _cfg["gpio"]["power"]["vcc_5v"]
GND:     int = _cfg["gpio"]["power"]["gnd"]

# =============================================================================
# Camera
# =============================================================================
CAPTURE_RESOLUTION: tuple = tuple(_cfg["camera"]["capture_resolution"])
AWB_LOCK:           bool  = _cfg["camera"]["awb_lock"]
AE_LOCK:            bool  = _cfg["camera"]["ae_lock"]
CAMERA_ROTATE_180:  bool  = _cfg["camera"]["rotate_180"]

# =============================================================================
# Image processing
# =============================================================================
HSV_RED_LOWER_1: tuple = tuple(_cfg["image_processing"]["hsv_red_lower_1"])
HSV_RED_UPPER_1: tuple = tuple(_cfg["image_processing"]["hsv_red_upper_1"])
HSV_RED_LOWER_2: tuple = tuple(_cfg["image_processing"]["hsv_red_lower_2"])
HSV_RED_UPPER_2: tuple = tuple(_cfg["image_processing"]["hsv_red_upper_2"])

MORPH_KERNEL_SIZE:    int   = _cfg["image_processing"]["morph_kernel_size"]
GAUSSIAN_BLUR_KERNEL: tuple = tuple(_cfg["image_processing"]["gaussian_blur_kernel"])
MORPH_CLOSE_LARGE:    int   = _cfg["image_processing"]["morph_close_large"]

MIN_CONTOUR_AREA:      int   = _cfg["image_processing"]["min_contour_area"]
MAX_CONTOUR_AREA:      int   = _cfg["image_processing"]["max_contour_area"]
CIRCULARITY_THRESHOLD: float = _cfg["image_processing"]["circularity_threshold"]

RED_RING_MIN_PIXELS: int  = _cfg["image_processing"]["red_ring_min_pixels"]

CLAHE_CLIP_LIMIT:   float = _cfg["image_processing"]["clahe_clip_limit"]
CLAHE_TILE_SIZE:    int   = _cfg["image_processing"]["clahe_tile_size"]
GLARE_L_THRESHOLD:  int   = _cfg["image_processing"]["glare_l_threshold"]
GLARE_MIN_VALID_PX: int   = _cfg["image_processing"]["glare_min_valid_px"]

# =============================================================================
# Color analysis
# =============================================================================
INNER_CROP_PX:     int   = _cfg["color_analysis"]["inner_crop_px"]
DELTA_E_THRESHOLD: float = _cfg["color_analysis"]["delta_e_threshold"]
CONFIDENCE_MARGIN:  float = _cfg["color_analysis"]["confidence_margin"]
PRESENCE_THRESHOLD: float = _cfg["color_analysis"]["presence_threshold"]
CONTRAST_THRESHOLD: float = _cfg["color_analysis"]["contrast_threshold"]
GHOST_DE_THRESHOLD: float = _cfg["color_analysis"]["ghost_de_threshold"]
REF_INNER_CROP_PX:  int   = _cfg["color_analysis"]["ref_inner_crop_px"]
COLOR_LEVELS:      int   = _cfg["color_analysis"]["color_levels"]
REFS_PER_LEVEL:    int   = _cfg["color_analysis"]["refs_per_level"]

# =============================================================================
# Grid
# =============================================================================
GRID_CONFIG_FILE:  str  = _cfg["grid"]["config_file"]
COLOR_JSON_FILE:   str  = _cfg["grid"]["color_json_file"]
GRID_COLS:         int  = _cfg["grid"]["cols"]
GRID_ROWS:         int  = _cfg["grid"]["rows"]
GROUPS:            list = _cfg["grid"]["groups"]
SLOTS_PER_GROUP:   int  = _cfg["grid"]["slots_per_group"]
LEVELS:            list = _cfg["grid"]["levels"]

# =============================================================================
# Network
# =============================================================================
HOTSPOT_SSID:         str = _cfg["network"]["hotspot_ssid"]
HOTSPOT_PASSWORD:     str = _cfg["network"]["hotspot_password"]
HOTSPOT_IP:           str = _cfg["network"]["hotspot_ip"]
WIFI_CONNECT_TIMEOUT: int = _cfg["network"]["wifi_connect_timeout"]

# =============================================================================
# Web server
# =============================================================================
WEB_SERVER_PORT:     int = _cfg["web_server"]["port"]
CAPTIVE_PORTAL_PORT: int = _cfg["web_server"]["captive_portal_port"]

# =============================================================================
# Telegram
# =============================================================================
TELEGRAM_TOKEN:   str = _cfg["telegram"]["token"]
TELEGRAM_CHAT_ID: str = _cfg["telegram"]["chat_id"]

# =============================================================================
# YOLO
# =============================================================================
YOLO_MODEL_PATH:         str   = _cfg["yolo"]["model_path"]
YOLO_IMGSZ:              int   = _cfg["yolo"]["imgsz"]
YOLO_CONF_THRESHOLD:     float = _cfg["yolo"]["conf_threshold"]
YOLO_IOU_THRESHOLD:      float = _cfg["yolo"]["iou_threshold"]
YOLO_AUGMENT:            bool  = _cfg["yolo"]["augment"]
YOLO_SNAPSHOTS:          int   = _cfg["yolo"]["snapshots"]
YOLO_SNAPSHOT_DELAY_MS:  int   = _cfg["yolo"]["snapshot_delay_ms"]
YOLO_EXPOSURE_VARIATION: float = _cfg["yolo"]["exposure_variation"]
YOLO_CONSENSUS_MIN:      int   = _cfg["yolo"]["consensus_min_votes"]
YOLO_CONSENSUS_IOU:      float = _cfg["yolo"]["consensus_iou"]
YOLO_BOX_MIN_PX:         int   = int(_cfg["yolo"].get("box_min_px", 140))
YOLO_BOX_MAX_PX:         int   = int(_cfg["yolo"].get("box_max_px", 200))
YOLO_SLOT_MAX_DIST:      float = _cfg["yolo"]["slot_max_dist_ratio"]
YOLO_CLAHE_CLIP:         float = _cfg["yolo"]["clahe_clip_limit"]
YOLO_CLAHE_TILE:         int   = _cfg["yolo"]["clahe_tile_size"]
YOLO_ROI_PADDING:        int   = int(_cfg["yolo"].get("roi_padding_px", 30))

# =============================================================================
# Sample ROI (fixed-margin crop for YOLO inference — excludes reference row)
# Each value is pixels to trim FROM that edge of the full frame:
#   x1 = left,              y1 = top
#   x2 = frame_w − right,   y2 = frame_h − bottom
# If a margin would make x2 ≤ x1 (margin too large for the frame),
# that side falls back to the frame boundary (i.e. no crop on that side).
# =============================================================================
_sroi = _cfg.get("sample_roi", {})
SAMPLE_ROI_TOP:    int = int(_sroi.get("top",    0))
SAMPLE_ROI_BOTTOM: int = int(_sroi.get("bottom", 0))
SAMPLE_ROI_LEFT:   int = int(_sroi.get("left",   0))
SAMPLE_ROI_RIGHT:  int = int(_sroi.get("right",  0))

# =============================================================================
# System / debug
# =============================================================================
WATCHDOG_TIMEOUT_SEC: int  = _cfg["system"]["watchdog_timeout_sec"]
DEBUG_MODE:           bool = _cfg["debug"]["enabled"]
LOG_DIR:              str  = _cfg["debug"]["log_dir"]
IMG_DIR:              str  = _cfg["debug"]["img_dir"]
DB_PATH:              str  = _cfg["debug"]["db_path"]
