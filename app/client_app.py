"""
Client-side application — Raspberry Pi 4B only.

Hardware:
  GPIO24 (BCM)  — push-button trigger (active LOW, internal pull-up)
  I2C LCD       — 16 × 4 character display at address 0x27
  TM1637        — 4-digit 7-segment display (shows integer bottle count)
  Relay module  — 3-channel status lights (pins in configs/config.toml → [gpio.relay])
    Relay 1 (Red)    GPIO21 — Error state
    Relay 2 (Yellow) GPIO20 — Processing / waiting state
    Relay 3 (Green)  GPIO12 — Ready / OK state

Flow on each button press:
  1. Relay → Yellow  /  LCD: "Capturing..."  /  TM1637: 8888
  2. Capture 4608 × 2592 JPEG via picamera2
  3. LCD: "Uploading..."  /  TM1637: spinning animation
  4. POST to SERVER_URL/analyze with X-Auth-Token header (60 s timeout)
  5. LCD: "Analyzing..."  (switches after ~4 s upload phase)
  6. Parse JSON response:
       no errors  → Relay Green  /  LCD: "Count: N" + colour summary
       any errors → Relay Red    /  LCD: "Count: N" + error badges
  7. All system/network errors → Relay Red

Relay logic notes:
  • active_low = true  (default):  GPIO.LOW  = relay ON
  • active_low = false:            GPIO.HIGH = relay ON
  • A 50 ms gap is inserted before switching a relay ON to avoid inrush spikes.

Install on Pi:
    uv sync --extra pi --extra common

Run:
    uv run main.py --role client --server-url https://your-tunnel.trycloudflare.com
"""

from __future__ import annotations

import gc
import sys
import threading
import time
from loguru import logger

from app.shared.config import cfg


# ─── Spinner frames for TM1637 ────────────────────────────────────────────────

_SPIN_FRAMES = ["-   ", " -  ", "  - ", "   -"]


# ─── Public entry point ───────────────────────────────────────────────────────

def run_client(server_url: str) -> None:
    """
    Initialise hardware and enter the BLOCKING event loop.

    This function does NOT return until the process is killed (Ctrl+C).
    GPIO.cleanup() is intentionally NOT called here — it is the sole
    responsibility of main.py's finally block.

    GPIO Lifecycle
    --------------
    main.py                         run_client()
    ───────────────────────────────────────────────
    try:
      run_client(url)  ──────────►  GPIO.setmode(BCM)      ← step 1
                                    init LCD, TM1637        ← step 2
                                    GPIO.setup(pins)        ← step 3
                                    remove_event_detect()   ← step 4 (stale-state guard)
                                    while True:             ← step 5 (blocks here)
                                      wait_for_edge(...)
                                      _on_button_press(...)
                       ◄──────────  KeyboardInterrupt propagates up
    except KeyboardInterrupt: ...
    finally:
      GPIO.cleanup()                                        ← only here
    """
    _check_imports()

    import RPi.GPIO as GPIO  # noqa: N813

    # ── Step 1: GPIO mode — FIRST, before any GPIO or library call ───────────
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # ── Step 2: Patch TM1637.__del__ BEFORE creating the object ──────────────
    # ROOT CAUSE FIX: tm1637 library's __del__ calls GPIO.cleanup(), which
    # destroys the entire GPIO state as soon as the object is garbage-collected.
    # Replacing __del__ with a no-op stops the library hijacking our lifecycle.
    try:
        import tm1637 as _tm1637_mod
        _tm1637_mod.TM1637.__del__ = lambda self: None
        logger.debug("TM1637.__del__ patched — library GPIO.cleanup() disabled")
    except Exception:
        pass

    # ── Step 3: Peripheral init — each wrapped so a hardware fault can't abort ─
    lcd = None
    tm  = None
    try:
        lcd = _init_lcd()
    except Exception as exc:
        logger.warning("LCD init failed ({}), continuing without display", exc)
    try:
        tm = _init_tm1637()
    except Exception as exc:
        logger.warning("TM1637 init failed ({}), continuing without display", exc)
    try:
        _relay_setup(GPIO)
        _relay(GPIO, "idle")   # Green ON
    except Exception as exc:
        logger.warning("Relay setup failed ({}), continuing anyway", exc)

    _lcd(lcd, "Urine Analyzer", "Ready")
    _tm(tm, 0)   # 0000 on boot

    logger.info(
        "System Ready — GPIO{} trigger | relay R={} Y={} G={} | server: {}",
        cfg.gpio_trigger_pin,
        cfg.relay_red_pin, cfg.relay_yellow_pin, cfg.relay_green_pin,
        server_url,
    )

    # ── Step 4: Trigger pin LAST — after all other hardware has settled ───────
    GPIO.setup(cfg.gpio_trigger_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    time.sleep(0.5)   # allow pull-up to settle electrically

    try:
        GPIO.remove_event_detect(cfg.gpio_trigger_pin)
    except Exception:
        pass

    logger.info("WAITING for button press on GPIO{}…", cfg.gpio_trigger_pin)

    # ── Step 5: Fortress loop — NEVER exits unless KeyboardInterrupt ──────────
    #
    #   Outer except KeyboardInterrupt  — graceful Ctrl+C exit
    #   Inner except Exception          — any scan/hardware error → log & retry
    #   Neither ever returns to main.py prematurely.
    try:
        while True:
            try:
                # timeout=5000 ms: wakes every 5 s so the GC stays cooperative.
                # channel is None on timeout — no press, just loop again.
                channel = GPIO.wait_for_edge(
                    cfg.gpio_trigger_pin,
                    GPIO.FALLING,
                    bouncetime=500,
                    timeout=5000,
                )
                if channel is None:
                    continue

                logger.info("Button pressed! Starting analysis…")
                _on_button_press(lcd, tm, GPIO, server_url)

            except KeyboardInterrupt:
                raise   # let the outer handler catch it cleanly

            except Exception as exc:
                # Any OSError, RuntimeError, I2C [Errno 121], network failure, etc.
                # Log it and stay in the loop — never crash out to main.py.
                logger.error("Scan error (will retry on next press): {}", exc)
                try:
                    _relay(GPIO, "error")
                except Exception:
                    pass

    except KeyboardInterrupt:
        logger.info("Client stopping…")

    finally:
        _relay_all_off(GPIO)
        try:
            tm = None   # noqa: F841
            gc.collect()
        except Exception:
            pass
        # GPIO.cleanup() is NOT called here — main.py owns that single call.


# ─── Button handler ───────────────────────────────────────────────────────────

def _on_button_press(lcd, tm, GPIO, server_url: str) -> None:
    import requests

    _relay(GPIO, "processing")   # Yellow ON — stays on through entire scan
    _lcd(lcd, "Capturing...", "")
    _tm(tm, 8888)

    img_bytes = _capture_image()
    if img_bytes is None:
        logger.error("Camera capture failed")
        _relay(GPIO, "error")
        _lcd(lcd, "ERR: CAMERA", "Capture failed")
        _tm(tm, None)
        return

    size_kb = len(img_bytes) / 1024
    logger.info("Image captured ({:.1f} KB) — uploading to {}", size_kb, server_url)
    _lcd(lcd, "Uploading...", "Please wait")

    # TM1637 spinner + delayed LCD "Analyzing..." while POST is in flight
    stop_spin = threading.Event()
    spin_thread = threading.Thread(
        target=_tm_spin, args=(tm, stop_spin), daemon=True
    )
    lcd_thread = threading.Thread(
        target=_lcd_delayed, args=(lcd, stop_spin, "Analyzing...", "AI running", 4.0),
        daemon=True,
    )
    spin_thread.start()
    lcd_thread.start()

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{server_url}/analyze",
            files={"file": ("capture.jpg", img_bytes, "image/jpeg")},
            headers={"X-Auth-Token": cfg.api_key},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

    except requests.exceptions.ConnectionError as exc:
        stop_spin.set()
        logger.error("Server unreachable after {:.1f}s: {}", time.perf_counter() - t0, exc)
        _relay(GPIO, "error")
        _lcd(lcd, "ERR: SRV OFF", "Check network")
        _tm(tm, None)
        return

    except requests.exceptions.Timeout:
        stop_spin.set()
        logger.error("Request timed out after {:.1f}s (limit 60s)", time.perf_counter() - t0)
        _relay(GPIO, "error")
        _lcd(lcd, "ERR: TIMEOUT", "Server slow")
        _tm(tm, None)
        return

    except requests.exceptions.HTTPError as exc:
        stop_spin.set()
        status = exc.response.status_code if exc.response is not None else "?"
        logger.error("HTTP {} after {:.1f}s: {}", status, time.perf_counter() - t0, exc)
        _relay(GPIO, "error")
        _lcd(lcd, f"ERR: HTTP {status}", "")
        _tm(tm, None)
        return

    except Exception as exc:
        stop_spin.set()
        logger.exception("Unexpected error after {:.1f}s: {}", time.perf_counter() - t0, exc)
        _relay(GPIO, "error")
        _lcd(lcd, "ERR: AI FAIL", "")
        _tm(tm, None)
        return

    finally:
        stop_spin.set()

    elapsed = time.perf_counter() - t0
    logger.info("Server responded in {:.2f}s", elapsed)

    if data.get("status") != "success":
        msg = data.get("message", "unknown error")
        logger.error("Server returned error: {}", msg)
        _relay(GPIO, "error")
        _lcd(lcd, "ERR: AI FAIL", msg[:16])
        _tm(tm, None)
        return

    count         = data.get("total_physical_count", data.get("count", 0))
    color_summary: dict = data.get("summary", data.get("color_summary", {}))
    errors:       dict = data.get("errors", {})
    dup_n  = len(errors.get("duplicate_slots",  []))
    wc_n   = len(errors.get("wrong_color_slots", []))

    _tm(tm, count)

    if dup_n > 0 or wc_n > 0:
        parts = []
        if dup_n: parts.append(f"Dup:{dup_n}")
        if wc_n:  parts.append(f"WC:{wc_n}")
        logger.warning(
            "Scan errors — count={}, {} | latency={:.2f}s",
            count, " ".join(parts), elapsed,
        )
        _relay(GPIO, "error")
        _lcd(lcd, f"Count: {count}", (" ".join(parts))[:16])
    else:
        color_str = "  ".join(f"{k}:{v}" for k, v in color_summary.items())
        logger.success(
            "Result OK — count={}, colors={}, latency={:.2f}s",
            count, color_summary, elapsed,
        )
        _relay(GPIO, "idle")
        _lcd(lcd, f"Count: {count}", color_str[:16] or "No color data")


# ─── Camera capture ───────────────────────────────────────────────────────────

def _capture_image() -> bytes | None:
    """Capture a still image and return it as JPEG bytes, or None on failure."""
    try:
        import cv2
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

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            logger.error("cv2.imencode failed")
            return None

        return buf.tobytes()

    except Exception as exc:
        logger.exception("Camera capture exception: {}", exc)
        return None


# ─── Relay helpers ────────────────────────────────────────────────────────────

def _relay_setup(GPIO) -> None:
    """Set relay pins as outputs and drive them all to the OFF level."""
    off_level = GPIO.LOW if cfg.relay_active_low else GPIO.HIGH
    for pin in (cfg.relay_red_pin, cfg.relay_yellow_pin, cfg.relay_green_pin):
        GPIO.setup(pin, GPIO.OUT, initial=off_level)
    logger.debug(
        "Relay pins initialised — R={} Y={} G={} active_low={}",
        cfg.relay_red_pin, cfg.relay_yellow_pin, cfg.relay_green_pin,
        cfg.relay_active_low,
    )


def _relay(GPIO, state: str) -> None:
    """
    Switch exactly one relay ON and the other two OFF.

    state:
      "idle"       → Green ON   (ready / clean result)
      "processing" → Yellow ON  (capturing / uploading / waiting)
      "error"      → Red ON     (any error, validation or system)
    """
    on_level  = GPIO.LOW  if cfg.relay_active_low else GPIO.HIGH
    off_level = GPIO.HIGH if cfg.relay_active_low else GPIO.LOW

    state_map = {
        "idle":       (cfg.relay_green_pin,  [cfg.relay_red_pin,    cfg.relay_yellow_pin]),
        "processing": (cfg.relay_yellow_pin, [cfg.relay_red_pin,    cfg.relay_green_pin]),
        "error":      (cfg.relay_red_pin,    [cfg.relay_yellow_pin, cfg.relay_green_pin]),
    }

    if state not in state_map:
        logger.warning("Unknown relay state: {}", state)
        return

    on_pin, off_pins = state_map[state]
    for pin in off_pins:
        GPIO.output(pin, off_level)
    time.sleep(0.05)          # brief settle to avoid inrush
    GPIO.output(on_pin, on_level)
    logger.debug("Relay → {}", state)


def _relay_all_off(GPIO) -> None:
    """Drive all relay pins to OFF — called on shutdown."""
    off_level = GPIO.HIGH if cfg.relay_active_low else GPIO.LOW
    for pin in (cfg.relay_red_pin, cfg.relay_yellow_pin, cfg.relay_green_pin):
        try:
            GPIO.output(pin, off_level)
        except Exception:
            pass


# ─── Display helpers ──────────────────────────────────────────────────────────

def _init_lcd():
    """Return an LCD object or None if the hardware is not available."""
    try:
        from rpi_lcd import LCD
        return LCD()
    except Exception as exc:
        logger.warning("LCD init failed ({}), display disabled", exc)
        return None


def _init_tm1637():
    """Return a TM1637 object or None if the hardware is not available."""
    try:
        import tm1637
        return tm1637.TM1637(clk=cfg.tm1637_clk, dio=cfg.tm1637_dio)
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


def _lcd_delayed(lcd, stop_event: threading.Event, line1: str, line2: str, delay: float) -> None:
    """Switch LCD to line1/line2 after *delay* seconds unless stop_event fires first."""
    if not stop_event.wait(delay):
        _lcd(lcd, line1, line2)


def _tm(tm, value: int | str | None) -> None:
    """
    Update the TM1637 display.

    value=None → show "----"
    value=int  → show zero-padded integer (0000–9999)
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


def _tm_spin(tm, stop_event: threading.Event) -> None:
    """Cycle through spinner frames on TM1637 until stop_event is set."""
    i = 0
    while not stop_event.is_set():
        if tm is not None:
            try:
                tm.show(_SPIN_FRAMES[i % len(_SPIN_FRAMES)])
            except Exception:
                pass
        i += 1
        stop_event.wait(0.25)


# ─── Import guard ─────────────────────────────────────────────────────────────

def _check_imports() -> None:
    missing = []
    try:
        import RPi.GPIO  # noqa: F401
    except ImportError:
        missing.append("RPi.GPIO")
    try:
        import picamera2  # noqa: F401
    except ImportError:
        missing.append("picamera2")

    if missing:
        logger.error(
            "Missing Pi packages: {}.  Install with: uv sync --extra pi --extra common",
            ", ".join(missing),
        )
        sys.exit(1)
