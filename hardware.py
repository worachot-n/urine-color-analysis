"""
Hardware abstraction layer: relay (LED tower), TM1637 displays, LCD 16x4,
and push button.

All GPIO and I2C calls are wrapped in try/except so hardware failures
never crash the main pipeline. When running on a non-Pi host (no RPi.GPIO),
all functions silently no-op and return False/None.

Public API:

  Relay / LED tower:
    relay_init()          — set up GPIO pins
    led_yellow()          — processing state
    led_green()           — all OK
    led_red()             — error detected
    led_off()             — all LEDs off
    relay_cleanup()       — release GPIO

  TM1637 7-segment:
    tm1637_show(level, count)   — update one display
    tm1637_show_all(counts)     — update all 5 (dict {0:n … 4:n})

  LCD 16×4 I2C:
    lcd_init()                  — initialize display
    lcd_message(text, line)     — write text to line 1-4 (max 16 chars)
    lcd_clear()                 — blank all lines

  Push button (GPIO24):
    button_init()               — configure input with pull-up
    button_wait_press()         — block until button is pressed (active LOW)
    button_cleanup()            — release GPIO pin
"""

import time

from config import (
    RELAY_LED_RED, RELAY_LED_YELLOW, RELAY_LED_GREEN,
    TM1637_DISPLAYS,
    LCD_I2C_ADDRESS, LCD_I2C_BUS, LCD_COLS,
    PIN_BUTTON, BUTTON_DEBOUNCE_MS,
)

# ---------------------------------------------------------------------------
# Optional hardware imports
# ---------------------------------------------------------------------------
try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False

try:
    import smbus2 as smbus
    _SMBUS_AVAILABLE = True
except ImportError:
    try:
        import smbus
        _SMBUS_AVAILABLE = True
    except ImportError:
        _SMBUS_AVAILABLE = False

# Active-low relay states
_RELAY_ON  = 0   # GPIO.LOW  — relay energized
_RELAY_OFF = 1   # GPIO.HIGH — relay released

# LCD protocol constants
_LCD_CHR      = 1
_LCD_CMD      = 0
_LCD_BACKLIGHT = 0x08
_ENABLE        = 0b00000100
_E_PULSE       = 0.0005
_E_DELAY       = 0.0005
# HD44780 DDRAM addresses for 16×4 LCD (rows at 0x00, 0x40, 0x10, 0x50)
_LCD_LINE      = {1: 0x80, 2: 0xC0, 3: 0x90, 4: 0xD0}

# 7-segment digit codes for TM1637
_SEG = [0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F]

_lcd_bus = None   # smbus handle, set by lcd_init()


# ===========================================================================
# Relay / LED tower
# ===========================================================================

def relay_init():
    """
    Configure GPIO pins for the 3-channel relay module.
    All relays start OFF (RELAY_OFF = HIGH for active-low modules).
    """
    if not _GPIO_AVAILABLE:
        return False
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in [RELAY_LED_RED, RELAY_LED_YELLOW, RELAY_LED_GREEN]:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, _RELAY_OFF)
        return True
    except Exception as e:
        print(f"relay_init error: {e}")
        return False


def led_yellow():
    """Turn on YELLOW (processing), turn off RED and GREEN."""
    _set_led(RELAY_LED_YELLOW)


def led_green():
    """Turn on GREEN (all OK), turn off RED and YELLOW."""
    _set_led(RELAY_LED_GREEN)


def led_red():
    """Turn on RED (error), turn off GREEN and YELLOW."""
    _set_led(RELAY_LED_RED)


def led_off():
    """Turn off all LEDs."""
    _set_led(None)


def _set_led(active_pin):
    if not _GPIO_AVAILABLE:
        return
    try:
        for pin in [RELAY_LED_RED, RELAY_LED_YELLOW, RELAY_LED_GREEN]:
            GPIO.output(pin, _RELAY_ON if pin == active_pin else _RELAY_OFF)
    except Exception as e:
        print(f"_set_led error: {e}")


def relay_cleanup():
    """Turn off all LEDs and release GPIO pins."""
    if not _GPIO_AVAILABLE:
        return
    try:
        led_off()
        GPIO.cleanup([RELAY_LED_RED, RELAY_LED_YELLOW, RELAY_LED_GREEN])
    except Exception as e:
        print(f"relay_cleanup error: {e}")


# ===========================================================================
# TM1637 7-segment displays
# ===========================================================================

def tm1637_show(level, count):
    """
    Display an integer count on the TM1637 for hydration level H0-H4.

    Args:
        level: int 0-4 (selects display H0…H4)
        count: int to display (0-9999)
    """
    key = f"H{level}"
    if key not in TM1637_DISPLAYS:
        return False
    pins = TM1637_DISPLAYS[key]
    try:
        _tm1637_write(pins["CLK"], pins["DIO"], count)
        return True
    except Exception as e:
        print(f"tm1637_show H{level} error: {e}")
        return False


def tm1637_show_all(counts):
    """
    Update all 5 TM1637 displays.

    Args:
        counts: dict {0: n, 1: n, 2: n, 3: n, 4: n}
    """
    for level in range(5):
        tm1637_show(level, counts.get(level, 0))


def _tm1637_write(clk, dio, num):
    """Low-level TM1637 write via bit-banged GPIO."""
    if not _GPIO_AVAILABLE:
        return

    num = max(0, min(9999, int(num)))

    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(clk, GPIO.OUT)
        GPIO.setup(dio, GPIO.OUT)
        GPIO.output(clk, GPIO.HIGH)
        GPIO.output(dio, GPIO.HIGH)
        time.sleep(0.05)

        def write_byte(byte):
            for i in range(8):
                GPIO.output(clk, GPIO.LOW)
                time.sleep(0.0001)
                GPIO.output(dio, GPIO.HIGH if (byte & (1 << i)) else GPIO.LOW)
                time.sleep(0.0001)
                GPIO.output(clk, GPIO.HIGH)
                time.sleep(0.0001)
            # ACK clock pulse
            GPIO.output(clk, GPIO.LOW)
            time.sleep(0.0001)
            GPIO.output(clk, GPIO.HIGH)
            time.sleep(0.0001)

        def start():
            GPIO.output(dio, GPIO.HIGH); time.sleep(0.0001)
            GPIO.output(clk, GPIO.HIGH); time.sleep(0.0001)
            GPIO.output(dio, GPIO.LOW);  time.sleep(0.0001)

        def stop():
            GPIO.output(clk, GPIO.LOW);  time.sleep(0.0001)
            GPIO.output(dio, GPIO.LOW);  time.sleep(0.0001)
            GPIO.output(clk, GPIO.HIGH); time.sleep(0.0001)
            GPIO.output(dio, GPIO.HIGH); time.sleep(0.0001)

        # Set auto-increment address mode
        start()
        write_byte(0x40)
        stop()
        time.sleep(0.05)

        # Write 4 digits starting at address 0xC0
        start()
        write_byte(0xC0)
        for char in str(num).zfill(4):
            write_byte(_SEG[int(char)])
        stop()
        time.sleep(0.05)

        # Set brightness (full brightness = 0x8F)
        start()
        write_byte(0x8F)
        stop()

    except Exception as e:
        print(f"_tm1637_write error: {e}")


# ===========================================================================
# LCD 16×4 I2C
# ===========================================================================

def lcd_init():
    """
    Initialize the LCD 16×4 via I2C (HD44780 + PCF8574).

    Returns True on success, False if hardware unavailable or error.
    """
    global _lcd_bus
    if not _SMBUS_AVAILABLE:
        print("lcd_init: smbus not available")
        return False
    try:
        _lcd_bus = smbus.SMBus(LCD_I2C_BUS)
        time.sleep(0.05)                          # >40 ms power-on delay

        _lcd_write_byte(0x33, _LCD_CMD)           # Wake up (sends 0x30 twice)
        time.sleep(0.005)
        _lcd_write_byte(0x32, _LCD_CMD)           # Switch to 4-bit mode
        time.sleep(0.001)
        _lcd_write_byte(0x28, _LCD_CMD)           # Function set: 4-bit, 2-line, 5×8
        time.sleep(0.001)
        _lcd_write_byte(0x0C, _LCD_CMD)           # Display ON, cursor OFF, blink OFF
        time.sleep(0.001)
        _lcd_write_byte(0x01, _LCD_CMD)           # Clear display
        time.sleep(0.005)                          # Clear needs >1.52 ms
        _lcd_write_byte(0x06, _LCD_CMD)           # Entry mode: increment, no shift
        time.sleep(0.001)
        return True
    except Exception as e:
        print(f"lcd_init error: {e}")
        _lcd_bus = None
        return False


def lcd_message(text, line):
    """
    Write text to a LCD line (1-4). Truncated/padded to LCD_COLS characters.

    Args:
        text: str
        line: int 1-4
    """
    if _lcd_bus is None:
        return
    try:
        padded = str(text)[:LCD_COLS].ljust(LCD_COLS)
        _lcd_write_byte(_LCD_LINE.get(line, 0x80), _LCD_CMD)
        for char in padded:
            _lcd_write_byte(ord(char) if ord(char) < 128 else 0x3F, _LCD_CHR)
    except Exception as e:
        print(f"lcd_message error: {e}")


def lcd_clear():
    """Clear all LCD lines."""
    if _lcd_bus is None:
        return
    try:
        _lcd_write_byte(0x01, _LCD_CMD)
        time.sleep(0.005)
    except Exception as e:
        print(f"lcd_clear error: {e}")


def _lcd_write_byte(bits, mode):
    if _lcd_bus is None:
        return
    try:
        high = mode | (bits & 0xF0)        | _LCD_BACKLIGHT
        low  = mode | ((bits << 4) & 0xF0) | _LCD_BACKLIGHT
        _lcd_bus.write_byte(LCD_I2C_ADDRESS, high)
        _lcd_toggle(high)
        _lcd_bus.write_byte(LCD_I2C_ADDRESS, low)
        _lcd_toggle(low)
    except Exception as e:
        print(f"_lcd_write_byte error: {e}")


def _lcd_toggle(bits):
    try:
        time.sleep(_E_DELAY)
        _lcd_bus.write_byte(LCD_I2C_ADDRESS, bits | _ENABLE)
        time.sleep(_E_PULSE)
        _lcd_bus.write_byte(LCD_I2C_ADDRESS, bits & ~_ENABLE)
        time.sleep(_E_DELAY)
    except Exception:
        pass


# ===========================================================================
# Push Button — GPIO24, active LOW, internal pull-up
# ===========================================================================

def button_init():
    """
    Configure PIN_BUTTON as input with internal pull-up resistor.
    Button is active LOW (press connects pin to GND).

    Returns True on success, False if GPIO unavailable.
    """
    if not _GPIO_AVAILABLE:
        return False
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PIN_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        return True
    except Exception as e:
        print(f"button_init error: {e}")
        return False


def button_wait_press():
    """
    Block until the push button is pressed (active LOW with debounce).
    Falls back to keyboard Enter when GPIO is unavailable (for development).
    """
    if not _GPIO_AVAILABLE:
        input("[SIM] Press Enter to simulate button press...")
        return

    # Poll for LOW (button pressed) — compatible with I2C and all GPIO pins
    try:
        while GPIO.input(PIN_BUTTON) != GPIO.LOW:
            time.sleep(0.05)
        time.sleep(BUTTON_DEBOUNCE_MS / 1000.0)
        # Confirm still held after debounce
        if GPIO.input(PIN_BUTTON) == GPIO.LOW:
            return
        # Spurious press — keep waiting
        button_wait_press()
    except Exception as e:
        print(f"button_wait_press error: {e}")


def button_cleanup():
    """Release the button GPIO pin."""
    if not _GPIO_AVAILABLE:
        return
    try:
        GPIO.cleanup([PIN_BUTTON])
    except Exception as e:
        print(f"button_cleanup error: {e}")
