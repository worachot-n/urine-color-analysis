"""
All system constants, GPIO pins, thresholds, and settings.
Nothing is hardcoded in logic modules — everything lives here.
"""

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

# --- LCD 16x4 via I2C ---
LCD_I2C_ADDRESS = 0x27
LCD_I2C_BUS     = 1          # Bus 1 on Raspberry Pi 3/4/5
LCD_COLS        = 16
LCD_ROWS        = 4

# --- 3-Channel Relay Module (controls LED Tower, ACTIVE LOW) ---
RELAY_LED_RED    = 20        # Relay 1 → Red LED
RELAY_LED_YELLOW = 21        # Relay 2 → Yellow LED
RELAY_LED_GREEN  = 12        # Relay 3 → Green LED
RELAY_ACTIVE_LOW = True      # GPIO.LOW = relay energized

# --- Push Button (replaces PIR) ---
PIN_BUTTON           = 24    # GPIO24 / Pin 18 — active LOW, internal pull-up
BUTTON_DEBOUNCE_MS   = 50    # Debounce time in milliseconds

# =============================================================================
# CAMERA SETTINGS
# =============================================================================
CAPTURE_RESOLUTION = (4608, 2592)   # Camera Module 3 maximum resolution
AWB_LOCK = True                      # Lock auto white-balance after warm-up
AE_LOCK  = True                      # Lock auto exposure after warm-up

# =============================================================================
# IMAGE PROCESSING — Red Cap Detection
# =============================================================================

# HSV ranges for red (wraps around 0/180 in OpenCV)
HSV_RED_LOWER_1 = (0,   120,  70)
HSV_RED_UPPER_1 = (10,  255, 255)
HSV_RED_LOWER_2 = (170, 120,  70)
HSV_RED_UPPER_2 = (180, 255, 255)

# Morphological cleanup
MORPH_KERNEL_SIZE    = 5
GAUSSIAN_BLUR_KERNEL = (5, 5)

# Hough circle parameters
HOUGH_DP         = 1.2
HOUGH_MIN_DIST   = 50
HOUGH_PARAM1     = 100
HOUGH_PARAM2     = 30
HOUGH_MIN_RADIUS = 20
HOUGH_MAX_RADIUS = 60

# =============================================================================
# COLOR ANALYSIS
# =============================================================================
INNER_CROP_PX      = 15     # Pixels to shrink from each side of bottle crop
DELTA_E_THRESHOLD  = 15.0   # Max Delta E for a confident classification
COLOR_LEVELS       = 5      # Number of distinct color levels (L0-L4)
REFS_PER_LEVEL     = 3      # Reference bottles per level in reference row

# =============================================================================
# GRID CONFIGURATION
# =============================================================================
GRID_CONFIG_FILE = "grid_config.json"

GRID_COLS      = 16          # 16 columns of lines (col 0 = ZZ dead zone)
GRID_ROWS      = 14          # 14 rows of lines (row 0 = reference, 1-12 = samples)
GROUPS         = ["A1", "A2", "A3", "A4"]
SLOTS_PER_GROUP = 9          # Slots 1-9 per group
LEVELS         = [0, 1, 2, 3, 4]

# =============================================================================
# NETWORK / HOTSPOT
# =============================================================================
HOTSPOT_SSID        = "signal"
HOTSPOT_PASSWORD    = "angad"
HOTSPOT_IP          = "192.168.4.1"    # Default AP gateway IP (dnsmasq)
WIFI_CONNECT_TIMEOUT = 30              # Seconds to wait for WiFi association

# =============================================================================
# WEB SERVER
# =============================================================================
WEB_SERVER_PORT      = 5000
CAPTIVE_PORTAL_PORT  = 8080   # Port 80 may be reserved; use 8080 for WiFi setup

# =============================================================================
# TELEGRAM (override with .env — these are fallback defaults)
# =============================================================================
TELEGRAM_TOKEN   = ""
TELEGRAM_CHAT_ID = ""

# =============================================================================
# SYSTEM SAFETY
# =============================================================================
WATCHDOG_TIMEOUT_SEC = 10    # Max seconds per processing cycle before timeout

# =============================================================================
# LOGGING / DEBUG
# =============================================================================
DEBUG_MODE = True
LOG_DIR    = "logs/"         # Every analysis saves an annotated image here
