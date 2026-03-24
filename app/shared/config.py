"""
Centralised configuration for both roles (client + server).

Load order (highest priority last wins):
  1. settings.yaml          — server/model/database defaults
  2. configs/config.toml    — hardware (GPIO, camera, relay) — overrides YAML for those sections
  3. .env                   — secrets and deployment overrides (API_KEY, SERVER_URL)
  4. Environment variables  — same names as .env keys

Usage:
    from app.shared.config import cfg
    print(cfg.server_port, cfg.relay_red_pin)
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parent.parent.parent


# ─── YAML loader (server / model / database / storage) ───────────────────────

def _yaml_defaults() -> dict:
    """Load settings.yaml from the project root."""
    path = _ROOT / "settings.yaml"
    if path.exists():
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    return {}


# ─── TOML loader (hardware — GPIO, camera, relay, display) ───────────────────

def _toml_defaults() -> dict:
    """Load configs/config.toml from the project root."""
    path = _ROOT / "configs" / "config.toml"
    if path.exists():
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    return {}


_y = _yaml_defaults()
_t = _toml_defaults()

# YAML sections (server-side)
_srv = _y.get("server",   {})
_mdl = _y.get("model",    {})
_db  = _y.get("database", {})
_st  = _y.get("storage",  {})

# TOML sections (hardware — Pi-side); fall back to YAML if TOML absent
_t_gpio  = _t.get("gpio",   {})
_t_btn   = _t_gpio.get("button", {})
_t_relay = _t_gpio.get("relay",  {})
_t_lcd   = _t_gpio.get("lcd",    {})
_t_tm    = _t_gpio.get("tm1637", {})   # dict of H0-H4 entries
_t_cam   = _t.get("camera", {})
_t_tg    = _t.get("telegram", {})

# YAML fallbacks for GPIO (used only when config.toml is absent)
_y_gpio  = _y.get("gpio",    {})
_y_relay = _y.get("relay",   {})
_y_cam   = _y.get("camera",  {})
_y_disp  = _y.get("display", {})

# ── Helpers ───────────────────────────────────────────────────────────────────

def _cam_dim(index: int, fallback: int) -> int:
    """Extract camera width/height from TOML capture_resolution list."""
    res = _t_cam.get("capture_resolution")
    if isinstance(res, (list, tuple)) and len(res) > index:
        return int(res[index])
    return fallback


def _tm_pin(level: str, key: str, fallback: int) -> int:
    """Read a CLK/DIO pin for a given TM1637 level (e.g. 'H2')."""
    entry = _t_tm.get(level, {})
    return int(entry.get(key, fallback))


# ─── Settings class ───────────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    # ── Secrets (from .env / environment) ──────────────────────────────────
    api_key: str    = Field(default="changeme",              alias="API_KEY")
    server_url: str = Field(default="http://localhost:8000", alias="SERVER_URL")

    # ── Telegram — loaded from config.toml; overrideable via .env ──────────
    telegram_token:   str = Field(
        default=_t_tg.get("token",   ""), alias="TELEGRAM_TOKEN"
    )
    telegram_chat_id: str = Field(
        default=_t_tg.get("chat_id", ""), alias="TELEGRAM_CHAT_ID"
    )

    # ── FastAPI server ──────────────────────────────────────────────────────
    server_host: str = _srv.get("host", "0.0.0.0")
    server_port: int = _srv.get("port", 8000)

    # ── YOLO model ──────────────────────────────────────────────────────────
    model_path:       str   = _t.get("yolo", {}).get("model_path",
                              _mdl.get("path", "models/best.pt"))
    model_input_size: int   = _t.get("yolo", {}).get("imgsz",
                              _mdl.get("input_size", 640))
    model_conf:       float = _t.get("yolo", {}).get("conf_threshold",
                              _mdl.get("confidence_threshold", 0.25))
    model_iou:        float = _t.get("yolo", {}).get("iou_threshold",
                              _mdl.get("iou_threshold", 0.45))

    # ── Persistence ─────────────────────────────────────────────────────────
    database_path:  str = _t.get("debug", {}).get("db_path",
                          _db.get("path", "data/results.db"))
    captures_dir:   str = _st.get("captures_dir", "app/web/static/captures")
    visual_log_dir: str = _t.get("debug", {}).get("img_dir",
                          _st.get("visual_log_dir", "logs/img"))

    # ── Camera — config.toml [camera] preferred ─────────────────────────────
    camera_width:  int = _cam_dim(0, _y_cam.get("width",  4608))
    camera_height: int = _cam_dim(1, _y_cam.get("height", 2592))

    # ── GPIO button — config.toml [gpio.button] preferred ───────────────────
    gpio_trigger_pin:    int = _t_btn.get("pin",         _y_gpio.get("trigger_pin", 24))
    button_debounce_ms:  int = _t_btn.get("debounce_ms", 50)

    # ── Relay (status lights) — config.toml [gpio.relay] preferred ──────────
    relay_red_pin:    int  = _t_relay.get("led_red",    _y_relay.get("red_pin",    21))
    relay_yellow_pin: int  = _t_relay.get("led_yellow", _y_relay.get("yellow_pin", 20))
    relay_green_pin:  int  = _t_relay.get("led_green",  _y_relay.get("green_pin",  12))
    relay_active_low: bool = _t_relay.get("active_low", _y_relay.get("active_low", True))

    # ── LCD — config.toml [gpio.lcd] preferred ──────────────────────────────
    lcd_address: int = _t_lcd.get("i2c_address", _y_disp.get("lcd_address", 0x27))

    # ── TM1637 — config.toml [gpio.tm1637.H2] preferred (H2 = main display) ─
    tm1637_clk: int = _tm_pin("H2", "CLK", _y_disp.get("tm1637_clk", 5))
    tm1637_dio: int = _tm_pin("H2", "DIO", _y_disp.get("tm1637_dio", 6))


cfg = Settings()
