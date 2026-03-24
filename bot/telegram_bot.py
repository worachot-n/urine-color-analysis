"""
Telegram notification module — server-side.

Sends a text summary and the annotated JPEG to a Telegram chat after
every successful /analyze call.

Credentials come from .env (or environment variables):
    TELEGRAM_TOKEN   — Bot API token from @BotFather
    TELEGRAM_CHAT_ID — Target group or personal chat ID

Both values are optional.  If either is missing the module skips silently.
All errors are logged as warnings — Telegram is non-critical to server operation.

Public API:
    send_scan_report(count, color_summary, image_path, timestamp) -> bool
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import requests
from loguru import logger

from app.shared.config import cfg


_TIMEOUT  = 15   # seconds per Telegram API call
_API_BASE = "https://api.telegram.org/bot{token}/{method}"


def _url(method: str) -> str:
    return _API_BASE.format(token=cfg.telegram_token, method=method)


def _ready() -> bool:
    """Return True only when both token and chat_id are configured."""
    if not cfg.telegram_token or not cfg.telegram_chat_id:
        return False
    return True


def send_scan_report(
    count:         int,
    color_summary: dict[str, int],
    image_path:    str | Path | None = None,
    timestamp:     datetime | None   = None,
) -> bool:
    """
    Send a scan summary text + annotated image to Telegram.

    Args:
        count:         Total bottles detected.
        color_summary: {level_label: count} dict, e.g. {"L0": 3, "L1": 4}.
        image_path:    Absolute path to the annotated JPEG (None → text only).
        timestamp:     Scan datetime (defaults to now).

    Returns:
        True if every Telegram request succeeded, False otherwise.
    """
    if not _ready():
        logger.debug("Telegram not configured — skipping notification")
        return False

    if timestamp is None:
        timestamp = datetime.utcnow()

    # ── Build message text ──────────────────────────────────────────────
    ts_str = timestamp.strftime("%Y-%m-%d %H:%M UTC")

    if color_summary:
        colors_str = "  ".join(f"{k}:{v}" for k, v in sorted(color_summary.items()))
    else:
        colors_str = "—"

    text = (
        f"*Scan Result — {ts_str}*\n"
        f"Bottles: *{count}*\n"
        f"Colors: {colors_str}"
    )

    success = True

    # ── Send text message ───────────────────────────────────────────────
    try:
        resp = requests.post(
            _url("sendMessage"),
            data={
                "chat_id":    cfg.telegram_chat_id,
                "text":       text,
                "parse_mode": "Markdown",
            },
            timeout=_TIMEOUT,
        )
        if not resp.ok:
            logger.warning("Telegram sendMessage failed ({}): {}", resp.status_code, resp.text)
            success = False
        else:
            logger.debug("Telegram text sent OK")
    except Exception as exc:
        logger.warning("Telegram sendMessage error: {}", exc)
        success = False

    # ── Send annotated image ────────────────────────────────────────────
    img = Path(image_path) if image_path else None
    if img and img.is_file():
        try:
            with open(img, "rb") as fh:
                resp = requests.post(
                    _url("sendPhoto"),
                    data={
                        "chat_id": cfg.telegram_chat_id,
                        "caption": f"Scan {ts_str} — {count} bottles",
                    },
                    files={"photo": fh},
                    timeout=_TIMEOUT,
                )
            if not resp.ok:
                logger.warning("Telegram sendPhoto failed ({}): {}", resp.status_code, resp.text)
                success = False
            else:
                logger.debug("Telegram photo sent OK")
        except Exception as exc:
            logger.warning("Telegram sendPhoto error: {}", exc)
            success = False
    elif image_path:
        logger.debug("Telegram: image file not found — {}", image_path)

    return success
