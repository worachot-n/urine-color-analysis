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
                                    time.sleep(1.0)         ← step 4 (pull-up settle)
                                    while True:             ← step 5 (blocks here)
                                      poll GPIO.input()
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
    tms: list = [None] * 5
    try:
        lcd = _init_lcd()
    except Exception as exc:
        logger.warning("LCD init failed ({}), continuing without display", exc)
    try:
        tms = _init_tm1637_all()
    except Exception as exc:
        logger.warning("TM1637 init failed ({}), continuing without displays", exc)
    try:
        _relay_setup(GPIO)
        _relay(GPIO, "idle")   # Green ON
    except Exception as exc:
        logger.warning("Relay setup failed ({}), continuing anyway", exc)

    _lcd(lcd, "Urine Analyzer", "Ready")
    _tm_all(tms, 0)   # 0000 on all displays at boot

    logger.info(
        "System Ready — GPIO{} trigger | relay R={} Y={} G={} | server: {}",
        cfg.gpio_trigger_pin,
        cfg.relay_red_pin, cfg.relay_yellow_pin, cfg.relay_green_pin,
        server_url,
    )

    # ── Step 4: Trigger pin LAST — after all other hardware has settled ───────
    GPIO.setup(cfg.gpio_trigger_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    time.sleep(1.0)   # allow pull-up resistor to settle electrically

    logger.info("WAITING for button press on GPIO{}… (polling mode)", cfg.gpio_trigger_pin)

    # ── Step 5: Fortress loop — NEVER exits unless KeyboardInterrupt ──────────
    #
    #   Polling instead of wait_for_edge eliminates RuntimeError: Error waiting
    #   for edge, which can occur when the kernel GPIO event queue overflows or
    #   the pin is briefly reconfigured by another library.
    #
    #   Outer except KeyboardInterrupt  — graceful Ctrl+C exit
    #   Inner except Exception          — any scan/hardware error → log & retry
    #   Neither ever returns to main.py prematurely.
    try:
        while True:
            try:
                if GPIO.input(cfg.gpio_trigger_pin) == GPIO.LOW:
                    logger.info("Button pressed! Starting analysis…")
                    time.sleep(0.05)   # debounce: ignore contact bounce
                    _on_button_press(lcd, tms, GPIO, server_url)
                    time.sleep(1.0)    # lock-out: ignore spurious re-triggers

                time.sleep(0.1)        # polling interval — keeps CPU idle

            except KeyboardInterrupt:
                raise   # let the outer handler catch it cleanly

            except Exception as exc:
                # Any OSError, RuntimeError, I2C [Errno 121], network failure, etc.
                # Log it and stay in the loop — never crash out to main.py.
                logger.error("Loop error (will retry): {}", exc)
                time.sleep(1.0)
                try:
                    _relay(GPIO, "error")
                except Exception:
                    pass

    except KeyboardInterrupt:
        logger.info("Client stopping…")

    finally:
        _relay_all_off(GPIO)
        try:
            tms = [None] * 5   # noqa: F841 — release TM1637 objects before GPIO cleanup
            gc.collect()
        except Exception:
            pass
        # GPIO.cleanup() is NOT called here — main.py owns that single call.


# ─── Button handler ───────────────────────────────────────────────────────────

def _on_button_press(lcd, tms: list, GPIO, server_url: str) -> None:
    import requests

    # Stop any previous error-coord scroll before starting a new scan
    prev_scroll = getattr(_on_button_press, "_scroll_stop", None)
    if prev_scroll is not None:
        prev_scroll.set()
        _on_button_press._scroll_stop = None
    _lcd_line(lcd, "", 3)
    _lcd_line(lcd, "", 4)

    _relay(GPIO, "processing")   # Yellow ON — stays on through entire scan
    _lcd(lcd, "Capturing...", "")
    _tm_all(tms, 8888)

    img_bytes = _capture_image()
    if img_bytes is None:
        logger.error("Camera capture failed")
        _relay(GPIO, "error")
        _lcd(lcd, "ERR: CAMERA", "Capture failed")
        _tm_all(tms, None)
        return

    size_kb = len(img_bytes) / 1024
    logger.info("Image captured ({:.1f} KB) — uploading to {}", size_kb, server_url)
    _lcd(lcd, "Uploading...", "Please wait")

    # TM1637 spinner (all 5 displays) + delayed LCD "Analyzing..." while POST is in flight
    stop_spin = threading.Event()
    spin_thread = threading.Thread(
        target=_tm_spin, args=(tms, stop_spin), daemon=True
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
        _tm_all(tms, None)
        return

    except requests.exceptions.Timeout:
        stop_spin.set()
        logger.error("Request timed out after {:.1f}s (limit 60s)", time.perf_counter() - t0)
        _relay(GPIO, "error")
        _lcd(lcd, "ERR: TIMEOUT", "Server slow")
        _tm_all(tms, None)
        return

    except requests.exceptions.HTTPError as exc:
        stop_spin.set()
        status = exc.response.status_code if exc.response is not None else "?"
        logger.error("HTTP {} after {:.1f}s: {}", status, time.perf_counter() - t0, exc)
        _relay(GPIO, "error")
        _lcd(lcd, f"ERR: HTTP {status}", "")
        _tm_all(tms, None)
        return

    except Exception as exc:
        stop_spin.set()
        logger.exception("Unexpected error after {:.1f}s: {}", time.perf_counter() - t0, exc)
        _relay(GPIO, "error")
        _lcd(lcd, "ERR: AI FAIL", "")
        _tm_all(tms, None)
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
        _tm_all(tms, None)
        return

    count         = data.get("total_physical_count", data.get("count", 0))
    color_summary: dict = data.get("summary", data.get("color_summary", {}))
    errors:       dict = data.get("errors", {})
    aruco_id             = data.get("aruco_id")
    error_count          = data.get("error_count", 0)

    # V2: wrong_color_slots is a list of "R01C05" coordinate strings
    wc_coords: list[str] = errors.get("wrong_color_slots", [])
    dup_n  = len(errors.get("duplicate_slots", []))
    wc_n   = len(wc_coords)

    # LCD line 1: tray ID + count
    tray_str = str(aruco_id) if aruco_id is not None else "?"
    lcd_line1 = f"Tray:{tray_str} N:{count}"

    # Show per-level counts: H0→L0, H1→L1, H2→L2, H3→L3, H4→L4
    _tm_levels(tms, color_summary)

    # Clear lines 3-4 before result display
    _lcd_line(lcd, "", 3)
    _lcd_line(lcd, "", 4)

    if dup_n > 0 or wc_n > 0:
        parts = []
        if dup_n: parts.append(f"Dup:{dup_n}")
        if wc_n:  parts.append(f"WC:{wc_n}")
        logger.warning(
            "Scan errors — count={}, aruco={}, {} | latency={:.2f}s",
            count, aruco_id, " ".join(parts), elapsed,
        )
        _relay(GPIO, "error")
        _lcd(lcd, lcd_line1, (" ".join(parts))[:16])

        # Start scrolling error coordinates on LCD line 3 in a daemon thread.
        # The scroll stops when stop_spin (already set above) is replaced by a
        # new stop event that we fire at the start of the NEXT button press.
        # We store it on the thread itself so the button handler can find it.
        scroll_stop = threading.Event()
        scroll_thread = threading.Thread(
            target=_lcd_scroll, args=(lcd, wc_coords, scroll_stop), daemon=True
        )
        scroll_thread.stop_event = scroll_stop
        # Stop previous scroll if still running (previous scan)
        prev = getattr(_on_button_press, "_scroll_stop", None)
        if prev is not None:
            prev.set()
        _on_button_press._scroll_stop = scroll_stop
        scroll_thread.start()

    else:
        color_str = "  ".join(f"{k}:{v}" for k, v in color_summary.items())
        logger.success(
            "Result OK — count={}, aruco={}, colors={}, latency={:.2f}s",
            count, aruco_id, color_summary, elapsed,
        )
        # Stop any previous scroll
        prev = getattr(_on_button_press, "_scroll_stop", None)
        if prev is not None:
            prev.set()
        _on_button_press._scroll_stop = None
        _relay(GPIO, "idle")
        _lcd(lcd, lcd_line1, color_str[:16] or "No color data")


# Attribute slot for the active scroll-stop event (set by _on_button_press)
_on_button_press._scroll_stop = None


# ─── Camera capture ───────────────────────────────────────────────────────────

def _capture_image() -> bytes | None:
    """Capture a still image and return it as JPEG bytes, or None on failure.

    Strategy:
      1. Start with AE + AWB enabled and wait 2 s for the algorithms to converge.
      2. Read the converged ColourGains / ExposureTime / AnalogueGain from metadata.
      3. Lock whichever of AWB / AE are enabled in config so the final capture is
         taken with identical, stable settings — no colour shift mid-frame.
      4. Encode as RGB JPEG (picamera2 RGB888 arrays are already in RGB order).
    """
    try:
        import io
        from PIL import Image
        from picamera2 import Picamera2

        picam2 = Picamera2()
        still_cfg = picam2.create_still_configuration(
            main={"size": (cfg.camera_width, cfg.camera_height), "format": "RGB888"}
        )
        picam2.configure(still_cfg)
        picam2.start()
        time.sleep(2)  # allow AE / AWB to converge under real scene lighting

        # Lock gains/exposure so the actual capture frame is colour-stable.
        if cfg.camera_awb_lock or cfg.camera_ae_lock:
            metadata = picam2.capture_metadata()
            lock_controls: dict = {}
            if cfg.camera_awb_lock:
                lock_controls["AwbEnable"]   = False
                lock_controls["ColourGains"] = metadata["ColourGains"]
                logger.debug("AWB locked — ColourGains={}", metadata["ColourGains"])
            if cfg.camera_ae_lock:
                lock_controls["AeEnable"]      = False
                lock_controls["ExposureTime"]  = metadata["ExposureTime"]
                lock_controls["AnalogueGain"]  = metadata["AnalogueGain"]
                logger.debug(
                    "AE locked — ExposureTime={}µs AnalogueGain={:.2f}",
                    metadata["ExposureTime"], metadata["AnalogueGain"],
                )
            picam2.set_controls(lock_controls)
            time.sleep(0.5)  # let locked settings take effect before capture

        frame = picam2.capture_array()
        picam2.stop()
        picam2.close()

        # picamera2 "RGB888" is a V4L2 format name where bytes are stored in
        # BGR order (Blue first).  Reverse the channel axis before handing to
        # PIL so that Pillow encodes a true-RGB JPEG.  Without this swap the
        # red and blue channels are exchanged, producing a blue/cyan tint.
        img = Image.fromarray(frame[:, :, ::-1])  # BGR → RGB
        if cfg.camera_rotate_180:
            img = img.rotate(180)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=cfg.camera_jpeg_quality)
        logger.debug(
            "Image encoded — size={}×{} quality={} bytes={:.1f}KB",
            img.width, img.height, cfg.camera_jpeg_quality, buf.tell() / 1024,
        )
        return buf.getvalue()

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


def _init_tm1637_all() -> list:
    """Initialise all 5 TM1637 displays (H0–H4). Returns a list of length 5;
    failed/missing units are represented as None."""
    pins = [
        (cfg.tm1637_h0_clk, cfg.tm1637_h0_dio),
        (cfg.tm1637_h1_clk, cfg.tm1637_h1_dio),
        (cfg.tm1637_h2_clk, cfg.tm1637_h2_dio),
        (cfg.tm1637_h3_clk, cfg.tm1637_h3_dio),
        (cfg.tm1637_h4_clk, cfg.tm1637_h4_dio),
    ]
    result = []
    for i, (clk, dio) in enumerate(pins):
        try:
            import tm1637
            result.append(tm1637.TM1637(clk=clk, dio=dio))
            logger.debug("TM1637 H{} initialised (CLK={} DIO={})", i, clk, dio)
        except Exception as exc:
            logger.warning("TM1637 H{} init failed ({}), display disabled", i, exc)
            result.append(None)
    return result


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


def _lcd_line(lcd, text: str, line: int) -> None:
    """Write *text* to a single LCD line (1-4)."""
    if lcd is None:
        return
    try:
        lcd.text(text[:16], line)
    except Exception as exc:
        logger.warning("LCD line {} write failed: {}", line, exc)


def _lcd_scroll(lcd, error_coords: list[str], stop_event: threading.Event) -> None:
    """
    Scroll a list of error coordinates across LCD line 3 at 200 ms per shift.

    Runs until stop_event is set.  Displays static text if total string < 16 chars.
    Format: "ERR @ R01C05  R03C12  " (repeating scroll)
    """
    if not error_coords:
        return
    sep     = "  "
    payload = sep.join(error_coords) + sep
    msg     = "ERR @ " + payload

    if len(msg) <= 16:
        # Short enough to display statically
        _lcd_line(lcd, msg, 3)
        _lcd_line(lcd, "", 4)
        return

    # Scroll continuously
    pos = 0
    while not stop_event.is_set():
        window = (msg * 2)[pos:pos + 16]
        _lcd_line(lcd, window, 3)
        pos = (pos + 1) % len(msg)
        stop_event.wait(0.2)


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


def _tm_all(tms: list, value) -> None:
    """Broadcast the same value to all TM1637 displays in the list."""
    for tm in tms:
        _tm(tm, value)


def _tm_levels(tms: list, summary: dict) -> None:
    """Show per-level counts: tms[0]→L0, tms[1]→L1, …, tms[4]→L4."""
    for i, tm in enumerate(tms):
        _tm(tm, summary.get(f"L{i}", 0))


def _tm_spin(tms_or_tm, stop_event: threading.Event) -> None:
    """Cycle through spinner frames on all TM1637 displays until stop_event is set."""
    tm_list = tms_or_tm if isinstance(tms_or_tm, list) else [tms_or_tm]
    i = 0
    while not stop_event.is_set():
        for tm in tm_list:
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
