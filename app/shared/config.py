"""
Centralised configuration for both roles (client + server).

Load order (highest priority last wins):
  1. settings.yaml   — default runtime values
  2. .env            — secrets and deployment overrides (API_KEY, SERVER_URL)
  3. Environment variables — same names as .env keys

Usage:
    from app.shared.config import cfg
    print(cfg.server_port, cfg.api_key)
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _yaml_defaults() -> dict:
    """Load settings.yaml relative to the project root (two levels above this file)."""
    root = Path(__file__).resolve().parent.parent.parent
    path = root / "settings.yaml"
    if path.exists():
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return data
    return {}


_y = _yaml_defaults()
_srv = _y.get("server", {})
_mdl = _y.get("model", {})
_db  = _y.get("database", {})
_st  = _y.get("storage", {})
_cam = _y.get("camera", {})
_gpio = _y.get("gpio", {})
_disp = _y.get("display", {})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    # ── Secrets (from .env / environment) ─────────────────────────────────
    api_key: str    = Field(default="changeme",              alias="API_KEY")
    server_url: str = Field(default="http://localhost:8000", alias="SERVER_URL")

    # ── Telegram (optional) ────────────────────────────────────────────
    telegram_token:   str = Field(default="", alias="TELEGRAM_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    # ── FastAPI server ─────────────────────────────────────────────────────
    server_host: str = _srv.get("host", "0.0.0.0")
    server_port: int = _srv.get("port", 8000)

    # ── YOLO model ─────────────────────────────────────────────────────────
    model_path:       str   = _mdl.get("path",                   "models/bottle_yolo26s_openvino_model")
    model_input_size: int   = _mdl.get("input_size",             640)
    model_conf:       float = _mdl.get("confidence_threshold",   0.25)
    model_iou:        float = _mdl.get("iou_threshold",          0.45)

    # ── Persistence ────────────────────────────────────────────────────────
    database_path: str = _db.get("path", "data/results.db")
    captures_dir:  str = _st.get("captures_dir", "app/web/static/captures")

    # ── Camera ─────────────────────────────────────────────────────────────
    camera_width:  int = _cam.get("width",  4608)
    camera_height: int = _cam.get("height", 2592)

    # ── GPIO ───────────────────────────────────────────────────────────────
    gpio_trigger_pin: int = _gpio.get("trigger_pin", 24)

    # ── Display ────────────────────────────────────────────────────────────
    lcd_address:  int = _disp.get("lcd_address",  0x27)
    tm1637_clk:   int = _disp.get("tm1637_clk",   5)
    tm1637_dio:   int = _disp.get("tm1637_dio",   6)


cfg = Settings()
