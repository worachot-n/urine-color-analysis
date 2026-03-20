"""
Telegram notification module.

Sends a text summary and annotated JPG image to a Telegram group after
every scan cycle.

Credentials are loaded from the .env file:
    TELEGRAM_TOKEN   — Bot API token from @BotFather
    TELEGRAM_CHAT_ID — Target group or chat ID

If either value is missing or the request fails, the function logs a warning
and returns without raising — Telegram is non-critical to system operation.

Public API:
    send_scan_report(counts, errors, image_path, timestamp) -> bool
"""

import os
import logging
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

import config

logger = logging.getLogger(__name__)

load_dotenv()

# Prefer .env values, fall back to config.py defaults
_TOKEN   = os.getenv("TELEGRAM_TOKEN")   or config.TELEGRAM_TOKEN
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or config.TELEGRAM_CHAT_ID

_API_BASE = "https://api.telegram.org/bot{token}/{method}"
_TIMEOUT  = 15   # seconds per request


def _api_url(method: str) -> str:
    return _API_BASE.format(token=_TOKEN, method=method)


def _is_configured() -> bool:
    if not _TOKEN or not _CHAT_ID:
        logger.debug("Telegram not configured — skipping notification")
        return False
    return True


def send_scan_report(
    counts:     dict,
    errors:     list[str],
    image_path: str | Path | None = None,
    timestamp:  datetime | None   = None,
) -> bool:
    """
    Send a scan summary text and optional annotated image to Telegram.

    Args:
        counts:     dict {0: n, 1: n, 2: n, 3: n, 4: n} — bottles per level
        errors:     list of error strings, e.g. ["A11_0 Dup", "A25_2 Mismatch (L3)"]
        image_path: path to the annotated JPG (None = text-only)
        timestamp:  datetime of the scan (defaults to now)

    Returns:
        True if all Telegram requests succeeded, False otherwise.
    """
    if not _is_configured():
        return False

    if timestamp is None:
        timestamp = datetime.now()

    # --- Build text summary ---
    ts_str    = timestamp.strftime("%Y-%m-%d %H:%M")
    counts_str = " | ".join(f"L{lvl}:{counts.get(lvl, 0)}" for lvl in range(5))

    if errors:
        error_str  = ", ".join(errors[:10])   # cap at 10 to stay readable
        status_str = "\u274c Error"
    else:
        error_str  = "None"
        status_str = "\u2705 OK"

    text = (
        f"*Scan Result \u2014 {ts_str}*\n"
        f"{counts_str}\n"
        f"Errors: {error_str}\n"
        f"Status: {status_str}"
    )

    success = True

    # --- Send text message ---
    try:
        resp = requests.post(
            _api_url("sendMessage"),
            data={"chat_id": _CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=_TIMEOUT,
        )
        if not resp.ok:
            logger.warning("Telegram sendMessage failed: %s", resp.text)
            success = False
    except Exception as e:
        logger.warning("Telegram sendMessage error: %s", e)
        success = False

    # --- Send annotated image ---
    if image_path and Path(image_path).is_file():
        try:
            with open(image_path, "rb") as f:
                resp = requests.post(
                    _api_url("sendPhoto"),
                    data={"chat_id": _CHAT_ID, "caption": f"Scan {ts_str}"},
                    files={"photo": f},
                    timeout=_TIMEOUT,
                )
            if not resp.ok:
                logger.warning("Telegram sendPhoto failed: %s", resp.text)
                success = False
        except Exception as e:
            logger.warning("Telegram sendPhoto error: %s", e)
            success = False
    elif image_path:
        logger.debug("Telegram: image_path not found — %s", image_path)

    return success
