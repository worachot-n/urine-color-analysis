#!/usr/bin/env python3
"""
Urine Color Analysis System — Entry Point

Boot sequence:
  1. Network setup  — WiFi check → hotspot + captive portal if needed
  2. Hardware init  — relay, LCD, button
  3. AWB lock       — camera white-balance lock (live mode only)
  4. Grid load      — grid_config.json
  5. Web dashboard  — background Flask server (port 5000)
  6. Wait for button (GPIO24) → scan → Telegram → LCD URL → repeat

Operation modes:

  Setup & Calibration:
    python main.py --calibrate [--calib-image path.jpg]
      Generates grid_config.json (required before inference modes).

  Live mode (button-triggered):
    python main.py --live
      Locks AWB, runs network setup, then loops on button press.

  Image mode (offline):
    python main.py --image path.jpg
      Analyze a single pre-stored image (no camera, no network setup).
"""

import argparse
import sys
import time
import threading
from pathlib import Path
from datetime import datetime

import cv2

import config
from grid import GridConfig
from color_analysis import (
    extract_bottle_color,
    build_reference_baseline,
    classify_sample,
)
from image_processing import detect_red_caps
from hardware import (
    relay_init, led_yellow, led_green, led_red, led_off, relay_cleanup,
    tm1637_show_all,
    lcd_init, lcd_message, lcd_clear,
    button_init, button_wait_press, button_cleanup,
)
from calibration import capture_white_balance_frame, capture_frame, run_calibration
from utils import setup_logger, save_annotated_image
import web_server
import telegram_bot

logger = setup_logger()

# Locked camera controls populated once during AWB lock step
_camera_controls: dict = {}


# ===========================================================================
# Analysis pipeline  (unchanged from original)
# ===========================================================================

def analyze_frame(frame, grid_cfg):
    """
    Full analysis pipeline on one captured frame.

    Returns dict with keys: counts, slot_assignments, duplicate_slots,
    has_errors, errors (list of human-readable strings).
    Returns None if baseline could not be built.
    """
    ref_positions = grid_cfg.get_reference_positions()
    baseline = build_reference_baseline(frame, ref_positions)

    if not baseline:
        logger.warning("Reference baseline empty — cannot classify samples")
        return None

    logger.debug("Baseline built for levels: %s", sorted(baseline.keys()))

    circles = detect_red_caps(frame)
    logger.info("Hough detected %d circles", len(circles))

    slot_assignments: dict = {}
    seen_slots:       set  = set()
    duplicate_slots:  set  = set()

    for cx, cy, radius in circles:
        slot_id, overlap = grid_cfg.find_slot_for_circle(cx, cy, radius)

        if slot_id is None:
            logger.debug(
                "Circle (%d,%d) r=%d — no slot match (overlap=%.2f)",
                cx, cy, radius, overlap
            )
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

    # Build human-readable error list for LCD/Telegram
    errors: list[str] = []
    for slot_id in duplicate_slots:
        errors.append(f"{slot_id} Dup")
    for slot_id, data in slot_assignments.items():
        if data["error"] and slot_id not in duplicate_slots:
            exp = data.get("expected_level", "?")
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

def run_scan_cycle(grid_cfg, source_image=None, web_ip: str = ""):
    """
    One complete scan: capture → analyze → update hardware → Telegram → LCD URL.

    Args:
        grid_cfg:     GridConfig instance
        source_image: Pre-loaded BGR frame (image mode). None = camera capture.
        web_ip:       Current device IP for LCD dashboard URL display.

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
            frame = source_image if source_image is not None else capture_frame(_camera_controls)
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

    counts    = result["counts"]
    errors    = result["errors"]
    log_path  = log_path_box[0]

    # --- Update TM1637 displays ---
    tm1637_show_all(counts)

    # --- Update web dashboard ---
    web_server.update_scan_result(counts, errors, log_path)

    # --- Send Telegram report ---
    telegram_bot.send_scan_report(counts, errors, log_path, ts)

    # --- Update LCD and LED ---
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

    # --- Show web URL on LCD line 4 ---
    if web_ip:
        lcd_message(f"{web_ip}:5000", 4)

    return True


# ===========================================================================
# Live mode (button-triggered)
# ===========================================================================

def run_live_mode(grid_cfg, web_ip: str = ""):
    """
    Continuous button-triggered scan loop.

    Waits for GPIO24 push button press, then runs a scan cycle.
    Falls back to keyboard Enter when GPIO is unavailable.
    """
    lcd_clear()
    lcd_message("Ready", 1)
    lcd_message("Press button", 2)
    if web_ip:
        lcd_message(f"{web_ip}:5000", 4)
    led_off()
    logger.info("Live mode: waiting for button press on GPIO%d", config.PIN_BUTTON)

    try:
        while True:
            button_wait_press()
            logger.info("Button pressed — starting scan")
            run_scan_cycle(grid_cfg, web_ip=web_ip)
            # After scan, prompt for next press
            lcd_message("Press button", 2)
            lcd_message("to re-scan", 3)

    except KeyboardInterrupt:
        logger.info("Live mode stopped")

    finally:
        button_cleanup()


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Urine Color Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --calibrate            Setup & Calibration: map grid corners → grid_config.json
  --live                 Inference: button-triggered continuous scan
  --image PATH           Inference: analyze a static image file

Examples:
  python main.py --calibrate
  python main.py --calibrate --calib-image my_photo.jpg
  python main.py --live
  python main.py --image sample.jpg
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--calibrate", action="store_true",
                       help="Run grid calibration to generate grid_config.json")
    group.add_argument("--live",  action="store_true",
                       help="Button-triggered live scan (requires Raspberry Pi)")
    group.add_argument("--image", metavar="PATH",
                       help="Analyze a single static image file")

    parser.add_argument("--calib-image", metavar="PATH",
                        help="Image to use for calibration (skips camera capture)")

    args = parser.parse_args()

    # ---- CALIBRATION MODE ------------------------------------------------
    if args.calibrate:
        logger.info("Calibration mode")
        calib_frame = None
        if args.calib_image:
            calib_frame = cv2.imread(args.calib_image)
            if calib_frame is None:
                logger.error("Cannot read: %s", args.calib_image)
                sys.exit(1)

        result = run_calibration(image=calib_frame)
        if result:
            print(f"\nCalibration saved: {result}")
        else:
            logger.error("Calibration cancelled or failed")
            sys.exit(1)
        return

    # ---- INFERENCE MODE --------------------------------------------------

    if not Path(config.GRID_CONFIG_FILE).exists():
        print(f"ERROR: {config.GRID_CONFIG_FILE} not found.")
        print("Run calibration first:  python main.py --calibrate")
        sys.exit(1)

    # Hardware init
    relay_init()
    lcd_init()
    button_init()

    logger.info("Loading grid config...")
    grid_cfg = GridConfig()
    logger.info(
        "Grid loaded: %d sample slots, %d reference slots",
        len(grid_cfg.slot_data),
        len(grid_cfg.reference_slots),
    )

    web_ip = ""

    # ---- Live mode -------------------------------------------------------
    if args.live:
        # 1. Network setup
        from network import run_network_setup
        from hardware import lcd_message as _lcd

        def _lcd4(l1, l2, l3, l4):
            lcd_clear()
            if l1: _lcd(l1, 1)
            if l2: _lcd(l2, 2)
            if l3: _lcd(l3, 3)
            if l4: _lcd(l4, 4)

        logger.info("Running network setup...")
        ssid, web_ip = run_network_setup(lcd_lines=_lcd4)
        web_ip = web_ip or ""

        # 2. White balance lock
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

        # 3. Start web dashboard
        web_server.start_web_server(port=config.WEB_SERVER_PORT)
        logger.info("Web dashboard: http://%s:%d", web_ip, config.WEB_SERVER_PORT)

        # 4. Wait for button, then enter main loop
        lcd_clear()
        lcd_message("Press button", 1)
        lcd_message("to start", 2)
        if web_ip:
            lcd_message(f"{web_ip}:5000", 4)
        button_wait_press()

        run_live_mode(grid_cfg, web_ip=web_ip)

    # ---- Image mode ------------------------------------------------------
    elif args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            logger.error("Image not found: %s", args.image)
            sys.exit(1)

        logger.info("Analyzing: %s", args.image)
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.error("Could not read image")
            sys.exit(1)

        # Start web dashboard (shows results after scan)
        web_server.start_web_server(port=config.WEB_SERVER_PORT)

        success = run_scan_cycle(grid_cfg, source_image=frame)
        if not success:
            relay_cleanup()
            lcd_clear()
            sys.exit(1)

        # Keep server alive briefly so the user can check the dashboard
        print(f"\nDashboard: http://localhost:{config.WEB_SERVER_PORT}")
        print("Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    # Cleanup
    relay_cleanup()
    button_cleanup()
    lcd_clear()


if __name__ == "__main__":
    main()
