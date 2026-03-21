#!/usr/bin/env python3
"""
Urine Color Analysis System — Entry Point

Run with:
    python main.py

Boot sequence:
  1. Hardware init  (relay, LCD, button)
  2. Network setup  (WiFi check → hotspot + captive portal if needed)
  3. AWB lock       (camera white-balance lock)
  4. Web server     (dashboard + calibration UI at http://<ip>:5000)
  5. Grid load      (grid_config.json — calibrate via web if missing)
  6. Wait for button (GPIO24) → scan → Telegram → LCD URL → repeat

Grid calibration is done through the web dashboard at /calibrate —
no CLI flags or SSH required.
"""

import sys
import time
import threading
from pathlib import Path
from datetime import datetime

import cv2

import configs.config as config
import utils.web_server as web_server
import bot.telegram_bot as telegram_bot
from utils.grid import GridConfig
from utils.color_analysis import (
    extract_bottle_color,
    build_reference_baseline,
    classify_sample,
)
from utils.image_processing import detect_red_caps
from utils.hardware import (
    relay_init, led_yellow, led_green, led_red, led_off,
    tm1637_show_all,
    lcd_init, lcd_message, lcd_clear,
    button_init, button_wait_press,
    hardware_cleanup,
)
from utils.calibration import capture_white_balance_frame, capture_frame
from utils import setup_logger, save_annotated_image

logger = setup_logger()

# Locked camera controls populated once during AWB lock step
_camera_controls: dict = {}


# ===========================================================================
# Analysis pipeline
# ===========================================================================

def analyze_frame(frame, grid_cfg):
    """
    Full analysis pipeline on one captured frame.

    Returns dict with: counts, slot_assignments, duplicate_slots,
    has_errors, errors (human-readable list).
    Returns None if reference baseline could not be built.
    """
    ref_positions = grid_cfg.get_reference_positions()
    baseline = build_reference_baseline(frame, ref_positions)

    if not baseline:
        logger.warning("Reference baseline empty — cannot classify samples")
        return None

    circles = detect_red_caps(frame)
    logger.info("Hough detected %d circles", len(circles))

    slot_assignments: dict = {}
    seen_slots:       set  = set()
    duplicate_slots:  set  = set()

    for cx, cy, radius in circles:
        slot_id, overlap = grid_cfg.find_slot_for_circle(cx, cy, radius)

        if slot_id is None:
            continue

        if slot_id in seen_slots:
            duplicate_slots.add(slot_id)
            logger.warning("Duplicate: %s", slot_id)
        seen_slots.add(slot_id)

        sample_lab = extract_bottle_color(frame, cx, cy, radius)
        if sample_lab is not None:
            level, delta_e, confident = classify_sample(sample_lab, baseline)
        else:
            level, delta_e, confident = None, None, False

        slot_info      = grid_cfg.slot_data.get(slot_id, {})
        expected_level = slot_info.get("expected_level")
        color_error    = (
            level is not None
            and expected_level is not None
            and level != expected_level
        )
        if color_error:
            logger.warning(
                "Mismatch: %s expected L%s got L%s (ΔE=%.1f)",
                slot_id, expected_level, level, delta_e or 0
            )

        slot_assignments[slot_id] = {
            "cx":             cx,
            "cy":             cy,
            "radius":         radius,
            "level":          level,
            "expected_level": expected_level,
            "delta_e":        delta_e,
            "confident":      confident,
            "error":          color_error or (slot_id in duplicate_slots),
        }

    counts: dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for data in slot_assignments.values():
        lvl = data.get("level")
        if lvl is not None and 0 <= lvl <= 4:
            counts[lvl] += 1

    has_errors = bool(duplicate_slots) or any(
        d["error"] for d in slot_assignments.values()
    )

    errors: list[str] = []
    for slot_id in duplicate_slots:
        errors.append(f"{slot_id} Dup")
    for slot_id, data in slot_assignments.items():
        if data["error"] and slot_id not in duplicate_slots:
            got = data.get("level", "?")
            errors.append(f"{slot_id} Mismatch (L{got})")

    return {
        "counts":           counts,
        "slot_assignments": slot_assignments,
        "duplicate_slots":  duplicate_slots,
        "has_errors":       has_errors,
        "errors":           errors,
    }


# ===========================================================================
# Scan cycle
# ===========================================================================

def run_scan_cycle(grid_cfg, web_ip: str = ""):
    """
    One complete scan: capture → analyze → update hardware → Telegram → LCD URL.

    Returns True on success, False on timeout or error.
    """
    led_yellow()
    lcd_clear()
    lcd_message("Scanning...", 1)
    logger.info("Scan cycle started")

    result_box   = [None]
    error_box    = [None]
    log_path_box = [None]
    ts           = datetime.now()

    def _do_work():
        try:
            frame = capture_frame(_camera_controls)
            if frame is None:
                error_box[0] = "Camera capture failed"
                return
            result = analyze_frame(frame, grid_cfg)
            result_box[0] = result
            if result is not None:
                log_path = save_annotated_image(
                    frame, result["slot_assignments"], grid_cfg, timestamp=ts
                )
                log_path_box[0] = log_path
                logger.info("Log image saved: %s", log_path)
        except Exception as exc:
            error_box[0] = str(exc)
            logger.exception("Error in analysis worker")

    worker = threading.Thread(target=_do_work, daemon=True)
    worker.start()
    worker.join(timeout=config.WATCHDOG_TIMEOUT_SEC)

    if worker.is_alive():
        logger.error("Watchdog: scan timed out")
        lcd_clear()
        lcd_message("ERROR:Timeout", 1)
        led_red()
        return False

    if error_box[0]:
        logger.error("Scan error: %s", error_box[0])
        lcd_clear()
        lcd_message("ERROR:See log", 1)
        led_red()
        return False

    result = result_box[0]
    if result is None:
        led_red()
        return False

    counts   = result["counts"]
    errors   = result["errors"]
    log_path = log_path_box[0]

    tm1637_show_all(counts)
    web_server.update_scan_result(counts, errors, log_path)
    telegram_bot.send_scan_report(counts, errors, log_path, ts)

    lcd_clear()
    if result["has_errors"]:
        led_red()
        first_err = errors[0] if errors else "Unknown"
        lcd_message("Error:", 1)
        lcd_message(first_err[:16], 2)
        logger.warning("Scan complete: ERRORS — %s", errors)
    else:
        led_green()
        lcd_message("ALL OK", 1)
        lcd_message(f"L0:{counts[0]}L1:{counts[1]}L2:{counts[2]}", 2)
        lcd_message(f"L3:{counts[3]} L4:{counts[4]}", 3)
        logger.info("Scan complete: OK — counts=%s", counts)

    if web_ip:
        lcd_message(f"{web_ip}:5000", 4)

    return True


# ===========================================================================
# Live loop
# ===========================================================================

def run_live_mode(grid_cfg, web_ip: str = "", lcd_lines=None):
    """
    Continuous button-triggered scan loop.

    Checks for a grid reload signal (from web calibration save) and WiFi
    connectivity at the top of each iteration. If WiFi is lost, automatically
    falls back to hotspot + captive-portal mode and resumes once reconnected.
    """
    from utils.network import run_network_setup, is_wifi_connected

    def _show_ready():
        lcd_clear()
        lcd_message("Ready", 1)
        lcd_message("Press button", 2)
        if web_ip:
            lcd_message(f"{web_ip}:5000", 4)
        led_off()

    _show_ready()
    logger.info("Live mode: waiting for button press on GPIO%d", config.PIN_BUTTON)

    try:
        while True:
            # --- WiFi loss detection & automatic fallback ---
            if not is_wifi_connected():
                logger.warning("WiFi connection lost — falling back to network setup")
                lcd_clear()
                lcd_message("WiFi Lost!", 1)
                lcd_message("Restarting net", 2)
                led_off()
                try:
                    new_ssid, new_ip = run_network_setup(lcd_lines=lcd_lines)
                    web_ip = new_ip or ""
                    logger.info("Network restored: ssid=%s ip=%s", new_ssid, new_ip)
                except Exception as e:
                    logger.error("Network setup failed: %s", e)
                _show_ready()

            # --- Grid reload (web calibration save OR dashboard Reload button) ---
            if web_server.consume_grid_saved() or web_server.consume_grid_reload():
                try:
                    grid_cfg = GridConfig()
                    logger.info("GridConfig reloaded")
                    lcd_clear()
                    lcd_message("Grid reloaded!", 1)
                    lcd_message("Press button", 2)
                    if web_ip:
                        lcd_message(f"{web_ip}:5000", 4)
                except Exception as e:
                    logger.error("GridConfig reload failed: %s", e)

            button_wait_press()
            logger.info("Button pressed — starting scan")
            run_scan_cycle(grid_cfg, web_ip=web_ip)
            lcd_message("Press button", 2)
            lcd_message("to re-scan", 3)

    except KeyboardInterrupt:
        logger.info("Live mode stopped")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    # ---- Hardware init ----
    relay_init()
    lcd_init()
    button_init()

    # ---- Network setup ----
    from utils.network import run_network_setup

    def _lcd4(l1, l2, l3, l4):
        lcd_clear()
        if l1: lcd_message(l1, 1)
        if l2: lcd_message(l2, 2)
        if l3: lcd_message(l3, 3)
        if l4: lcd_message(l4, 4)

    logger.info("Running network setup...")
    ssid, web_ip = run_network_setup(lcd_lines=_lcd4)
    web_ip = web_ip or ""

    # ---- AWB lock ----
    logger.info("Locking white balance (point camera at white card)...")
    lcd_clear()
    lcd_message("Locking AWB...", 1)
    _, controls = capture_white_balance_frame()
    _camera_controls.update(controls)

    if controls:
        logger.info("White balance locked")
        lcd_message("AWB Locked", 1)
    else:
        logger.warning("AWB lock failed — continuing without lock")
        lcd_message("AWB: No lock", 1)
    time.sleep(1)

    # ---- Web server ----
    web_server.start_web_server(port=config.WEB_SERVER_PORT)
    logger.info("Web dashboard: http://%s:%d", web_ip, config.WEB_SERVER_PORT)
    print(f"\n  Dashboard:  http://{web_ip}:{config.WEB_SERVER_PORT}\n")

    # ---- Grid load (or prompt to calibrate via web) ----
    grid_cfg = None
    if Path(config.GRID_CONFIG_FILE).exists():
        try:
            grid_cfg = GridConfig()
            logger.info(
                "Grid loaded: %d sample slots, %d reference slots",
                len(grid_cfg.slot_data),
                len(grid_cfg.reference_slots),
            )
        except Exception as e:
            logger.error("Failed to load grid_config.json: %s", e)
            grid_cfg = None

    if grid_cfg is None:
        # No grid yet — guide user to calibrate via web
        lcd_clear()
        lcd_message("No grid config", 1)
        lcd_message("Open browser:", 2)
        lcd_message(f"{web_ip}:5000", 3)
        lcd_message("/calibrate", 4)
        logger.info(
            "grid_config.json not found. Open http://%s:%d/calibrate to calibrate.",
            web_ip, config.WEB_SERVER_PORT,
        )
        # Wait until calibration is saved via web
        while not web_server.consume_grid_saved():
            time.sleep(0.5)
        try:
            grid_cfg = GridConfig()
            logger.info("Grid loaded after web calibration")
        except Exception as e:
            logger.error("Cannot load grid after calibration: %s", e)
            sys.exit(1)

    # ---- Wait for first button press then enter main loop ----
    # Show WiFi status on LCD so user can confirm connectivity before scanning
    from utils.network import get_current_ssid, get_current_ip
    _cur_ssid = get_current_ssid() or ssid or "Unknown"
    _cur_ip   = get_current_ip()   or web_ip or "?.?.?.?"
    lcd_clear()
    lcd_message("WiFi Status: OK", 1)
    lcd_message(f"SSID:{_cur_ssid[:10]}", 2)
    lcd_message(f"IP:{_cur_ip}", 3)
    lcd_message("Press NEXT(GP24)", 4)
    button_wait_press()

    try:
        run_live_mode(grid_cfg, web_ip=web_ip, lcd_lines=_lcd4)
    finally:
        hardware_cleanup()


if __name__ == "__main__":
    main()
