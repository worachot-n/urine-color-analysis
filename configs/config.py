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

HOUGH_DP:         float = _cfg["image_processing"]["hough_dp"]
HOUGH_MIN_DIST:   int   = _cfg["image_processing"]["hough_min_dist"]
HOUGH_PARAM1:     int   = _cfg["image_processing"]["hough_param1"]
HOUGH_PARAM2:     int   = _cfg["image_processing"]["hough_param2"]
HOUGH_MIN_RADIUS: int   = _cfg["image_processing"]["hough_min_radius"]
HOUGH_MAX_RADIUS: int   = _cfg["image_processing"]["hough_max_radius"]

# =============================================================================
# Color analysis
# =============================================================================
INNER_CROP_PX:     int   = _cfg["color_analysis"]["inner_crop_px"]
DELTA_E_THRESHOLD: float = _cfg["color_analysis"]["delta_e_threshold"]
COLOR_LEVELS:      int   = _cfg["color_analysis"]["color_levels"]
REFS_PER_LEVEL:    int   = _cfg["color_analysis"]["refs_per_level"]

# =============================================================================
# Grid
# =============================================================================
GRID_CONFIG_FILE:  str  = _cfg["grid"]["config_file"]
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
# System / debug
# =============================================================================
WATCHDOG_TIMEOUT_SEC: int  = _cfg["system"]["watchdog_timeout_sec"]
DEBUG_MODE:           bool = _cfg["debug"]["enabled"]
LOG_DIR:              str  = _cfg["debug"]["log_dir"]
IMG_DIR:              str  = _cfg["debug"]["img_dir"]
DB_PATH:              str  = _cfg["debug"]["db_path"]
