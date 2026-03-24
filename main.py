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
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

import bot.telegram_bot as telegram_bot
import configs.config as config
import utils.web_server as web_server
from utils.calibration import capture_frame, capture_white_balance_frame, capture_multi_snapshot
from utils.color_analysis import (
    build_reference_baseline,
    classify_sample,
    extract_bottle_color,
    delta_e_cie76,
    load_static_baseline,
    WHITE_LAB,
)
from utils.image_processing import build_slot_red_mask, slot_has_red_ring
from utils.grid import GridConfig
from utils.hardware import (
    button_init,
    button_wait_press,
    hardware_cleanup,
    lcd_clear,
    lcd_init,
    lcd_message,
    led_green,
    led_off,
    led_red,
    led_yellow,
    relay_init,
    tm1637_show_all,
)
import numpy as np
from utils.utils import save_annotated_image, setup_logger
from utils.yolo_detector import YoloBottleDetector

logger = setup_logger()

# Locked camera controls populated once during AWB lock step
_camera_controls: dict = {}

# Lazy-loaded YOLO detector (pre-loaded at startup — see main())
_yolo_detector: "YoloBottleDetector | None" = None

# Prevents overlapping scan cycles if a previous scan is still running
_scan_in_progress = threading.Event()


def _get_yolo() -> YoloBottleDetector:
    global _yolo_detector
    if _yolo_detector is None:
        _yolo_detector = YoloBottleDetector(config.YOLO_MODEL_PATH)
    return _yolo_detector


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

    # Two-step presence verification:
    #   Test 1 — Red Ring: slot bbox must have ≥ RED_RING_MIN_PIXELS in the healed red mask
    #   Test 2 — Contrast:  slot center must differ from white paper by ≥ CONTRAST_THRESHOLD
    # Slots failing Test 1 → empty (no red ring found).
    # Slots failing Test 2 → ghost/glare (ring found but center is blown-out white).
    closed_red_mask = build_slot_red_mask(frame)

    slot_assignments: dict = {}
    rejected_slots:   list = []   # (cx, cy, r, slot_id) — ghost detections for visual log
    unassigned_circles: list = []
    duplicate_slots: set = set()

    for slot_id, info in grid_cfg.slot_data.items():
        coords = info['coords']

        cx = int(np.mean(coords[:, 0]))
        cy = int(np.mean(coords[:, 1]))
        x_min = int(np.min(coords[:, 0]))
        x_max = int(np.max(coords[:, 0]))
        y_min = int(np.min(coords[:, 1]))
        y_max = int(np.max(coords[:, 1]))
        r = max(1, min((x_max - x_min) // 2, (y_max - y_min) // 2))

        # ── Test 1: Red Ring ────────────────────────────────────────────────
        if not slot_has_red_ring(closed_red_mask, coords):
            continue   # slot is empty — no red ring evidence

        # ── Test 2: Object Contrast (anti-ghost) ────────────────────────────
        sample_lab = extract_bottle_color(frame, cx, cy, r)
        if sample_lab is None:
            rejected_slots.append((cx, cy, r, slot_id))
            continue

        delta_to_white = delta_e_cie76(sample_lab, WHITE_LAB)
        if delta_to_white < config.CONTRAST_THRESHOLD:
            rejected_slots.append((cx, cy, r, slot_id))
            logger.debug("Ghost rejected: %s (ΔE-white=%.1f)", slot_id, delta_to_white)
            continue

        # ── Both tests passed → classify ────────────────────────────────────
        level, delta_e, confident = classify_sample(sample_lab, baseline)

        expected_level = info.get("expected_level")
        color_error = (
            level is not None and expected_level is not None and level != expected_level
        )
        if color_error:
            logger.warning(
                "Mismatch: %s expected L%s got L%s (ΔE=%.1f)",
                slot_id, expected_level, level, delta_e or 0,
            )

        slot_assignments[slot_id] = {
            "cx": cx, "cy": cy, "radius": r,
            "level": level, "expected_level": expected_level,
            "delta_e": delta_e, "confident": confident,
            "error": color_error,
            "error_type": "mismatch" if color_error else None,
        }

    logger.info(
        "Detection: %d bottles confirmed, %d ghosts rejected",
        len(slot_assignments), len(rejected_slots),
    )

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
        "counts": counts,
        "slot_assignments": slot_assignments,
        "duplicate_slots": duplicate_slots,
        "has_errors": has_errors,
        "errors": errors,
        "unassigned_circles": unassigned_circles,
        "rejected_slots": rejected_slots,
    }


# ===========================================================================
# YOLOv8 analysis pipeline
# ===========================================================================


def analyze_frame_yolo(frames: list, grid_cfg):
    """
    YOLOv8 + consensus + color-classification pipeline.

    Args:
        frames:   list of BGR frames from capture_multi_snapshot()
        grid_cfg: GridConfig instance

    Returns:
        Same dict shape as analyze_frame() for downstream compatibility.
        Returns None if reference baseline could not be built.
    """
    detector = _get_yolo()
    ref_positions_yolo, yolo_hits, duplicate_slots = detector.detect_multi(frames, grid_cfg)

    # Merge: YOLO ref detections override calibrated positions where available;
    # calibrated grid_config positions are the fallback for any missed levels.
    calibrated_positions = grid_cfg.get_reference_positions()
    merged_ref_positions = dict(calibrated_positions)
    merged_ref_positions.update(ref_positions_yolo)

    fallback_levels = set(calibrated_positions.keys()) - set(ref_positions_yolo.keys())
    if fallback_levels:
        logger.warning("YOLO missed ref levels %s — using calibrated positions as fallback",
                       sorted(fallback_levels))

    logger.info("YOLO: %d ref bottles, %d sample hits, %d duplicate slots",
                sum(len(v) for v in ref_positions_yolo.values()),
                len(yolo_hits), len(duplicate_slots))

    # Prefer pre-saved static baseline (color.json) over dynamic live sampling.
    # Static baseline is set once during calibration and is unaffected by YOLO
    # missing the reference row (which is now excluded from the sample-area ROI).
    static = load_static_baseline(config.COLOR_JSON_FILE)
    if static:
        baseline = static
        logger.info("Using static color baseline from %s", config.COLOR_JSON_FILE)
    else:
        baseline = build_reference_baseline(frames[0], merged_ref_positions)

    if not baseline:
        logger.warning("Reference baseline empty — cannot classify samples")
        return None

    slot_assignments: dict = {}
    unassigned_circles: list = []
    rejected_slots:     list = []

    for slot_id, hit in yolo_hits.items():
        cx, cy = hit['cx'], hit['cy']
        r = max(1, min(hit['w'], hit['h']) // 2)

        sample_lab = extract_bottle_color(frames[0], cx, cy, r)
        if sample_lab is None:
            rejected_slots.append((cx, cy, r, slot_id))
            continue

        # Physical presence check — reject glare / white paper reflections
        delta_to_white = delta_e_cie76(sample_lab, WHITE_LAB)
        if delta_to_white < config.GHOST_DE_THRESHOLD:
            rejected_slots.append((cx, cy, r, slot_id))
            logger.debug("YOLO ghost rejected: %s (ΔE-white=%.1f)", slot_id, delta_to_white)
            continue

        level, delta_e, confident = classify_sample(sample_lab, baseline)

        slot_info = grid_cfg.slot_data.get(slot_id, {})
        expected  = slot_info.get("expected_level")
        color_error = (
            level is not None and expected is not None and level != expected
        )
        if color_error:
            logger.warning(
                "Mismatch: %s expected L%s got L%s (ΔE=%.1f)",
                slot_id, expected, level, delta_e or 0,
            )

        slot_assignments[slot_id] = {
            "cx": cx, "cy": cy, "radius": r,
            "w": hit['w'], "h": hit['h'],
            "level": level, "expected_level": expected,
            "delta_e": delta_e, "confident": confident,
            "error": color_error,
            "error_type": "mismatch" if color_error else None,
        }

    counts: dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for data in slot_assignments.values():
        lvl = data.get("level")
        if lvl is not None and 0 <= lvl <= 4:
            counts[lvl] += 1

    has_errors = bool(duplicate_slots) or any(d["error"] for d in slot_assignments.values())
    errors: list[str] = [f"{sid} Dup" for sid in duplicate_slots]
    errors += [
        f"{sid} Mismatch (L{d.get('level', '?')})"
        for sid, d in slot_assignments.items() if d["error"]
    ]

    return {
        "counts": counts,
        "slot_assignments": slot_assignments,
        "duplicate_slots": duplicate_slots,
        "has_errors": has_errors,
        "errors": errors,
        "unassigned_circles": unassigned_circles,
        "rejected_slots": rejected_slots,
    }


# ===========================================================================
# Scan cycle
# ===========================================================================


def run_scan_cycle(grid_cfg, web_ip: str = ""):
    """
    One complete scan: capture → analyze → update hardware → Telegram → LCD URL.

    Returns True on success, False on timeout or error.
    """
    if _scan_in_progress.is_set():
        logger.warning("Scan already in progress — ignoring button press")
        return False
    _scan_in_progress.set()
    try:
        return _run_scan_cycle_inner(grid_cfg, web_ip=web_ip)
    finally:
        _scan_in_progress.clear()


def _run_scan_cycle_inner(grid_cfg, web_ip: str = ""):
    try:
        led_yellow()
        lcd_clear()
        lcd_message("Scanning...", 1)
    except Exception:
        pass
    logger.info("Scan cycle started")

    result_box = [None]
    error_box = [None]
    log_path_box = [None]
    ts = datetime.now()

    def _do_work():
        try:
            frames = capture_multi_snapshot(
                _camera_controls,
                n=config.YOLO_SNAPSHOTS,
                delay_ms=config.YOLO_SNAPSHOT_DELAY_MS,
                exposure_variation=config.YOLO_EXPOSURE_VARIATION,
            )
            if not frames:
                error_box[0] = "Camera capture failed"
                return
            result = analyze_frame_yolo(frames, grid_cfg)
            result_box[0] = result
            if result is not None:
                log_path = save_annotated_image(
                    frames[0], result["slot_assignments"], grid_cfg,
                    unassigned_circles=result["unassigned_circles"],
                    rejected_slots=result["rejected_slots"],
                    timestamp=ts,
                )
                log_path_box[0] = log_path
                YoloBottleDetector.write_result_json(
                    result["slot_assignments"], result["counts"], ts
                )
                logger.info("Log image + JSON saved: %s", log_path)
        except Exception as exc:
            error_box[0] = str(exc)
            logger.exception("Error in YOLO analysis worker")

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

    counts = result["counts"]
    errors = result["errors"]
    log_path = log_path_box[0]

    try:
        tm1637_show_all(counts)
    except Exception:
        pass
    try:
        web_server.update_scan_result(counts, errors, log_path, result["slot_assignments"])
    except Exception:
        logger.warning("web_server.update_scan_result failed — continuing")
    try:
        telegram_bot.send_scan_report(counts, errors, log_path, ts)
    except Exception:
        logger.warning("Telegram report failed — continuing")

    try:
        lcd_clear()
        if result["has_errors"]:
            led_red()
            first_err = errors[0] if errors else "Unknown"
            lcd_message("Error:", 1)
            lcd_message(first_err[:16], 2)
        else:
            led_green()
            lcd_message("ALL OK", 1)
            lcd_message(f"L0:{counts[0]}L1:{counts[1]}L2:{counts[2]}", 2)
            lcd_message(f"L3:{counts[3]} L4:{counts[4]}", 3)
        if web_ip:
            lcd_message(f"{web_ip}:5000", 4)
    except Exception:
        pass
    if result["has_errors"]:
        logger.warning("Scan complete: ERRORS — %s", errors)
    else:
        logger.info("Scan complete: OK — counts=%s", counts)

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
    from utils.network import is_wifi_connected, run_network_setup

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
            try:
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
                try:
                    run_scan_cycle(grid_cfg, web_ip=web_ip)
                except Exception as exc:
                    logger.exception("run_scan_cycle crashed: %s", exc)
                    try:
                        led_red()
                    except Exception:
                        pass
                try:
                    lcd_message("Press button", 2)
                    lcd_message("to re-scan", 3)
                except Exception:
                    pass

            except KeyboardInterrupt:
                raise   # let outer handler catch it
            except Exception as exc:
                logger.exception("Unexpected error in main loop — continuing: %s", exc)
                try:
                    led_red()
                except Exception:
                    pass

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
        if l1:
            lcd_message(l1, 1)
        if l2:
            lcd_message(l2, 2)
        if l3:
            lcd_message(l3, 3)
        if l4:
            lcd_message(l4, 4)

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
            web_ip,
            config.WEB_SERVER_PORT,
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

    # ---- Pre-load YOLO model so first scan doesn't hit the watchdog ----
    logger.info("Pre-loading YOLO model...")
    lcd_clear()
    lcd_message("Loading model...", 1)
    try:
        _get_yolo()
        logger.info("YOLO model ready")
        lcd_message("Model ready", 1)
    except Exception as e:
        logger.error("YOLO model load failed: %s", e)
        lcd_message("Model FAILED", 1)
    time.sleep(1)

    # ---- Wait for first button press then enter main loop ----
    # Show WiFi status on LCD so user can confirm connectivity before scanning
    from utils.network import get_current_ip, get_current_ssid

    _cur_ssid = get_current_ssid() or ssid or "Unknown"
    _cur_ip = get_current_ip() or web_ip or "?.?.?.?"
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
