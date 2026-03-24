"""
Client-side application — Raspberry Pi 4B only.

Hardware:
  GPIO24 (BCM)  — push-button trigger (active LOW, internal pull-up)
  I2C LCD       — 16 × 4 character display at address 0x27
  TM1637        — 4-digit 7-segment display (shows integer bottle count)

Flow on each button press:
  1. LCD: "Capturing..."
  2. Capture 4608 × 2592 JPEG via picamera2
  3. LCD: "Uploading..."
  4. POST to SERVER_URL/analyze with X-Auth-Token header
  5. Parse JSON response → update LCD + TM1637
  6. Handle server-offline / inference-error gracefully

Install on Pi:
    uv sync --extra pi --extra common

Run:
    uv run main.py --role client --server-url https://your-tunnel.trycloudflare.com
"""

from __future__ import annotations

import sys
import time
from loguru import logger

from app.shared.config import cfg


# ─── Public entry point ───────────────────────────────────────────────────────

def run_client(server_url: str) -> None:
    """Initialise hardware and enter the main event loop."""
    _check_imports()

    import RPi.GPIO as GPIO  # noqa: N813

    lcd = _init_lcd()
    tm  = _init_tm1637()

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(cfg.gpio_trigger_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    _lcd(lcd, "Urine Analyzer", "Ready")
    _tm(tm, None)  # show "----"

    logger.info(
        "Client ready — waiting for button on GPIO{} | server: {}",
        cfg.gpio_trigger_pin,
        server_url,
    )

    try:
        while True:
            # wait_for_edge blocks until pin goes LOW (button press)
            GPIO.wait_for_edge(
                cfg.gpio_trigger_pin,
                GPIO.FALLING,
                bouncetime=200,
            )
            time.sleep(0.05)  # debounce settle
            _on_button_press(lcd, tm, server_url)
    except KeyboardInterrupt:
        logger.info("Shutdown requested — exiting client loop")
    finally:
        GPIO.cleanup()
        logger.info("GPIO cleanup complete")


# ─── Button handler ───────────────────────────────────────────────────────────

def _on_button_press(lcd, tm, server_url: str) -> None:
    import requests

    logger.info("Button pressed — starting capture")
    _lcd(lcd, "Capturing...", "")
    _tm(tm, None)

    img_bytes = _capture_image()
    if img_bytes is None:
        logger.error("Camera capture failed")
        _lcd(lcd, "ERR: CAMERA", "Capture failed")
        _tm(tm, "Err")
        return

    logger.info("Image captured ({:.1f} KB) — uploading to {}", len(img_bytes) / 1024, server_url)
    _lcd(lcd, "Uploading...", "Please wait")

    try:
        resp = requests.post(
            f"{server_url}/analyze",
            files={"file": ("capture.jpg", img_bytes, "image/jpeg")},
            headers={"X-Auth-Token": cfg.api_key},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

    except requests.exceptions.ConnectionError as exc:
        logger.error("Server unreachable: {}", exc)
        _lcd(lcd, "ERR: SRV OFF", "Check network")
        _tm(tm, "Err")
        return

    except requests.exceptions.Timeout:
        logger.error("Request timed out after 120 s")
        _lcd(lcd, "ERR: TIMEOUT", "Server slow")
        _tm(tm, "Err")
        return

    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        logger.error("HTTP {} from server: {}", status, exc)
        _lcd(lcd, f"ERR: HTTP {status}", "")
        _tm(tm, "Err")
        return

    except Exception as exc:
        logger.exception("Unexpected error during upload: {}", exc)
        _lcd(lcd, "ERR: AI FAIL", "")
        _tm(tm, "Err")
        return

    if data.get("status") != "success":
        msg = data.get("message", "unknown error")
        logger.error("Server returned error: {}", msg)
        _lcd(lcd, "ERR: AI FAIL", msg[:16])
        _tm(tm, "Err")
        return

    count        = data.get("count", 0)
    color_summary: dict = data.get("color_summary", {})
    errors:       dict = data.get("errors", {})
    dup_n  = len(errors.get("duplicate_slots",  []))
    wc_n   = len(errors.get("wrong_color_slots", []))

    logger.success(
        "Result received — count={}, colors={}, dups={}, wrong_color={}",
        count, color_summary, dup_n, wc_n,
    )

    _tm(tm, count)

    if dup_n > 0 or wc_n > 0:
        # Errors present — show count on line 1, error summary on line 2
        parts = []
        if dup_n > 0:
            parts.append(f"Dup:{dup_n}")
        if wc_n > 0:
            parts.append(f"WC:{wc_n}")
        logger.warning("Scan errors: {}", " ".join(parts))
        _lcd(lcd, f"Count: {count}", (" ".join(parts))[:16])
    else:
        color_str = "  ".join(f"{k}:{v}" for k, v in color_summary.items())
        _lcd(lcd, f"Count: {count}", color_str[:16] or "No color data")


# ─── Camera capture ───────────────────────────────────────────────────────────

def _capture_image() -> bytes | None:
    """Capture a still image and return it as JPEG bytes, or None on failure."""
    try:
        import io
        import cv2
        import numpy as np
        from picamera2 import Picamera2

        picam2 = Picamera2()
        still_cfg = picam2.create_still_configuration(
            main={"size": (cfg.camera_width, cfg.camera_height), "format": "RGB888"}
        )
        picam2.configure(still_cfg)
        picam2.start()
        time.sleep(2)  # allow auto-exposure to settle

        frame = picam2.capture_array()
        picam2.stop()
        picam2.close()

        # picamera2 returns RGB — convert to BGR for OpenCV / JPEG encode
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            logger.error("cv2.imencode failed")
            return None

        return buf.tobytes()

    except Exception as exc:
        logger.exception("Camera capture exception: {}", exc)
        return None


# ─── Display helpers ──────────────────────────────────────────────────────────

def _init_lcd():
    """Return an LCD object or None if the hardware is not available."""
    try:
        from rpi_lcd import LCD
        lcd = LCD()
        return lcd
    except Exception as exc:
        logger.warning("LCD init failed ({}), display disabled", exc)
        return None


def _init_tm1637():
    """Return a TM1637 object or None if the hardware is not available."""
    try:
        import tm1637
        tm = tm1637.TM1637(clk=cfg.tm1637_clk, dio=cfg.tm1637_dio)
        return tm
    except Exception as exc:
        logger.warning("TM1637 init failed ({}), display disabled", exc)
        return None


def _lcd(lcd, line1: str, line2: str) -> None:
    """Write two lines to the LCD (or log if not available)."""
    if lcd is None:
        logger.debug("LCD | {} | {}", line1, line2)
        return
    try:
        lcd.text(line1[:16], 1)
        lcd.text(line2[:16], 2)
    except Exception as exc:
        logger.warning("LCD write failed: {}", exc)


def _tm(tm, value: int | str | None) -> None:
    """
    Update the TM1637 display.

    value=None    → show "----"
    value="Err"   → show "Err "
    value=int     → show zero-padded integer (0000–9999)
    """
    if tm is None:
        logger.debug("TM1637 | {}", value)
        return
    try:
        if value is None:
            tm.show("----")
        elif value == "Err":
            tm.show("Err ")
        else:
            n = max(0, min(int(value), 9999))
            tm.number(n)
    except Exception as exc:
        logger.warning("TM1637 write failed: {}", exc)


# ─── Import guard ─────────────────────────────────────────────────────────────

def _check_imports() -> None:
    missing = []
    for pkg in ("RPi.GPIO", "picamera2"):
        try:
            __import__(pkg.replace(".", "_") if pkg == "RPi.GPIO" else pkg)
        except ImportError:
            missing.append(pkg)

    # RPi.GPIO ships as RPi package
    try:
        import RPi.GPIO  # noqa: F401
    except ImportError:
        if "RPi.GPIO" not in missing:
            missing.append("RPi.GPIO")

    if missing:
        logger.error(
            "Missing Pi packages: {}.  Install with: uv sync --extra pi --extra common",
            ", ".join(missing),
        )
        sys.exit(1)
