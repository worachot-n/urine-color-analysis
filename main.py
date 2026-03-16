#!/usr/bin/env python3
"""
Urine Color Analysis System — Entry Point

Two operation modes:

  Setup & Calibration Mode
    python main.py --calibrate [--image path.jpg]
      - Interactive 4-corner grid alignment
      - Generates grid_config.json with all 195 slot polygons
      - Optionally loads an existing image instead of capturing

  Inference Mode
    python main.py --live
      - PIR sensor (GPIO 23) triggers scan cycles
      - Locks white balance on startup
      - Runs continuously until Ctrl+C

    python main.py --image path.jpg
      - Analyze a single pre-recorded image file
      - Useful for offline testing without camera hardware
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
)
from calibration import capture_white_balance_frame, capture_frame, run_calibration
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

    Steps:
      1. Sample reference row → build live Lab baseline
      2. Detect red caps (Hough circles on full frame)
      3. Assign circles to slots (majority rule)
      4. Classify each sample via Delta E against live baseline
      5. Validate: classified level must match slot's expected_level
      6. Count bottles per level

    Args:
        frame:    BGR image (numpy array)
        grid_cfg: GridConfig instance

    Returns:
        dict with keys: counts, slot_assignments, duplicate_slots, has_errors
        Returns None if baseline could not be built.
    """
    # Step 1: Reference baseline
    ref_positions = grid_cfg.get_reference_positions()
    baseline = build_reference_baseline(frame, ref_positions)

    if not baseline:
        logger.warning("Reference baseline empty — cannot classify samples")
        return None

    logger.debug(f"Baseline built for levels: {sorted(baseline.keys())}")

    # Step 2: Detect caps
    circles = detect_red_caps(frame)
    logger.info(f"Hough detected {len(circles)} circles")

    # Step 3: Assign to slots
    slot_assignments: dict = {}
    seen_slots: set = set()
    duplicate_slots: set = set()

    for cx, cy, radius in circles:
        slot_id, overlap = grid_cfg.find_slot_for_circle(cx, cy, radius)

        if slot_id is None:
            logger.debug(f"Circle ({cx},{cy}) r={radius} — no slot match (overlap={overlap:.2f})")
            continue

        if slot_id in seen_slots:
            duplicate_slots.add(slot_id)
            logger.warning(f"Duplicate: {slot_id}")
        seen_slots.add(slot_id)

        # Step 4: Classify
        sample_lab = extract_bottle_color(frame, cx, cy, radius)
        if sample_lab is not None:
            level, delta_e, confident = classify_sample(sample_lab, baseline)
        else:
            level, delta_e, confident = None, None, False

        # Step 5: Validate
        slot_info      = grid_cfg.slot_data.get(slot_id, {})
        expected_level = slot_info.get('expected_level')
        color_error    = (
            level is not None
            and expected_level is not None
            and level != expected_level
        )
        if color_error:
            logger.warning(
                f"Mismatch: {slot_id} expected L{expected_level} got L{level} "
                f"(ΔE={delta_e:.1f})"
            )

        slot_assignments[slot_id] = {
            'cx':             cx,
            'cy':             cy,
            'radius':         radius,
            'level':          level,
            'expected_level': expected_level,
            'delta_e':        delta_e,
            'confident':      confident,
            'error':          color_error or (slot_id in duplicate_slots),
        }

    # Step 6: Counts
    counts: dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for data in slot_assignments.values():
        lvl = data.get('level')
        if lvl is not None and 0 <= lvl <= 4:
            counts[lvl] += 1

    has_errors = bool(duplicate_slots) or any(
        d['error'] for d in slot_assignments.values()
    )

    return {
        'counts':           counts,
        'slot_assignments': slot_assignments,
        'duplicate_slots':  duplicate_slots,
        'has_errors':       has_errors,
    }


# ===========================================================================
# Scan cycle
# ===========================================================================

def run_scan_cycle(grid_cfg, source_image=None):
    """
    One complete scan: capture → analyze → update hardware → save log.

    A watchdog thread enforces config.WATCHDOG_TIMEOUT_SEC. If analysis
    exceeds the timeout, the cycle is aborted and the RED light is shown.

    Args:
        grid_cfg:     GridConfig instance
        source_image: Pre-loaded BGR frame (image mode). None = camera capture.

    Returns:
        True on success, False on timeout or error.
    """
    led_yellow()
    lcd_message("Scanning...", 1)
    logger.info("Scan cycle started")

    result_box = [None]
    error_box  = [None]
    ts         = datetime.now()

    def _do_work():
        try:
            if source_image is not None:
                frame = source_image
            else:
                frame = capture_frame(_camera_controls)
                if frame is None:
                    error_box[0] = "Camera capture failed"
                    return

            result = analyze_frame(frame, grid_cfg)
            result_box[0] = result

            if result is not None:
                log_path = save_annotated_image(
                    frame, result['slot_assignments'], grid_cfg, timestamp=ts
                )
                logger.info(f"Log image saved: {log_path}")

        except Exception as exc:
            error_box[0] = str(exc)
            logger.exception("Error in analysis worker")

    worker = threading.Thread(target=_do_work, daemon=True)
    worker.start()
    worker.join(timeout=config.WATCHDOG_TIMEOUT_SEC)

    if worker.is_alive():
        logger.error("Watchdog: scan timed out")
        lcd_clear()
        lcd_message("ERROR: Timeout", 1)
        led_red()
        return False

    if error_box[0]:
        logger.error(f"Scan error: {error_box[0]}")
        lcd_clear()
        lcd_message("ERROR: See log", 1)
        led_red()
        return False

    result = result_box[0]
    if result is None:
        led_red()
        return False

    # Update TM1637 displays
    tm1637_show_all(result['counts'])

    # Update LCD and relay LED
    lcd_clear()
    counts = result['counts']

    if result['has_errors']:
        led_red()
        # Show first error on LCD
        for slot_id, data in result['slot_assignments'].items():
            if not data.get('error'):
                continue
            if slot_id in result['duplicate_slots']:
                lcd_message(f"DUP: {slot_id}", 1)
            else:
                exp = data.get('expected_level', '?')
                got = data.get('level', '?')
                lcd_message(f"ERR: {slot_id}", 1)
                lcd_message(f"Exp L{exp} Got L{got}", 2)
            break
        logger.warning(f"Scan complete: ERRORS — {result['duplicate_slots']}")
    else:
        led_green()
        lcd_message("ALL OK", 1)
        lcd_message(f"L0:{counts[0]} L1:{counts[1]} L2:{counts[2]}", 2)
        lcd_message(f"L3:{counts[3]} L4:{counts[4]}", 3)
        logger.info(f"Scan complete: OK — counts={counts}")

    return True


# ===========================================================================
# Live mode (PIR-triggered)
# ===========================================================================

def run_live_mode(grid_cfg):
    """
    Continuous PIR-triggered scan loop.

    Polls GPIO PIN_PIR every 50 ms. When a RISING edge is detected and
    cooldown has elapsed:
      1. Wait PIR_SETTLE_DELAY_SEC for hand to leave frame
      2. Run scan cycle

    Falls back to keyboard simulation when GPIO is unavailable.
    """
    gpio_available = False
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(config.PIN_PIR, GPIO.IN)
        gpio_available = True
        logger.info(f"PIR sensor ready on GPIO{config.PIN_PIR}")
    except Exception as e:
        logger.warning(f"GPIO not available ({e}) — using keyboard simulation")

    lcd_clear()
    lcd_message("Ready", 1)
    led_off()
    logger.info("Live mode: waiting for motion...")

    last_trigger = 0.0

    try:
        while True:
            if gpio_available:
                import RPi.GPIO as GPIO
                if GPIO.input(config.PIN_PIR) == GPIO.HIGH:
                    now = time.time()
                    if now - last_trigger >= config.PIR_COOLDOWN_SEC:
                        last_trigger = now
                        logger.info("PIR triggered")
                        time.sleep(config.PIR_SETTLE_DELAY_SEC)
                        run_scan_cycle(grid_cfg)
                time.sleep(0.05)
            else:
                input("\n[Simulation] Press Enter to trigger scan...")
                time.sleep(config.PIR_SETTLE_DELAY_SEC)
                run_scan_cycle(grid_cfg)

    except KeyboardInterrupt:
        logger.info("Live mode stopped")

    finally:
        if gpio_available:
            import RPi.GPIO as GPIO
            GPIO.cleanup(config.PIN_PIR)


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
  --live                 Inference: PIR-triggered continuous scan
  --image PATH           Inference: analyze a static image file

Examples:
  python main.py --calibrate
  python main.py --calibrate --image my_photo.jpg
  python main.py --live
  python main.py --image sample.jpg
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--calibrate", action="store_true",
                       help="Run grid calibration to generate grid_config.json")
    group.add_argument("--live",  action="store_true",
                       help="PIR-triggered live scan (requires Raspberry Pi)")
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
                logger.error(f"Cannot read: {args.calib_image}")
                sys.exit(1)

        result = run_calibration(image=calib_frame)
        if result:
            print(f"\nCalibration saved: {result}")
        else:
            logger.error("Calibration cancelled or failed")
            sys.exit(1)
        return

    # ---- INFERENCE MODE --------------------------------------------------

    # Require grid_config.json
    if not Path(config.GRID_CONFIG_FILE).exists():
        print(f"ERROR: {config.GRID_CONFIG_FILE} not found.")
        print("Run calibration first:  python main.py --calibrate")
        sys.exit(1)

    # Initialize hardware
    relay_init()
    lcd_init()

    logger.info("Loading grid config...")
    grid_cfg = GridConfig()
    logger.info(
        f"Grid loaded: {len(grid_cfg.slot_data)} sample slots, "
        f"{len(grid_cfg.reference_slots)} reference slots"
    )

    # ---- Live mode -------------------------------------------------------
    if args.live:
        logger.info("Locking white balance (point camera at white card)...")
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

        run_live_mode(grid_cfg)

    # ---- Image mode ------------------------------------------------------
    elif args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            logger.error(f"Image not found: {args.image}")
            sys.exit(1)

        logger.info(f"Analyzing: {args.image}")
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.error("Could not read image")
            sys.exit(1)

        success = run_scan_cycle(grid_cfg, source_image=frame)
        if not success:
            sys.exit(1)

    # Cleanup
    relay_cleanup()
    lcd_clear()


if __name__ == "__main__":
    main()
