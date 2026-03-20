"""
Network management: WiFi detection, Access Point (hotspot) mode,
and a captive-portal web form to configure WiFi credentials.

Uses nmcli (NetworkManager CLI) for WiFi control on Raspberry Pi OS.
All subprocess calls fail gracefully so the rest of the system keeps running.

Public API:
    is_wifi_connected()            -> bool
    get_current_ssid()             -> str | None
    get_current_ip(iface)          -> str | None
    start_hotspot(ssid, password)  -> str  (AP IP address)
    stop_hotspot()                 -> None
    connect_wifi(ssid, password)   -> bool
    run_network_setup(lcd_lines)   -> (ssid, ip)
        lcd_lines is a callable(line1, line2, line3, line4) for LCD output
"""

import subprocess
import socket
import threading
import time
import logging

from config import (
    HOTSPOT_SSID,
    HOTSPOT_PASSWORD,
    HOTSPOT_IP,
    WIFI_CONNECT_TIMEOUT,
    CAPTIVE_PORTAL_PORT,
)

logger = logging.getLogger(__name__)

# Name used for the nmcli hotspot connection profile
_HOTSPOT_CON_NAME = "urine-hotspot"

# Shared state written by the captive-portal Flask thread
_wifi_config_event  = threading.Event()
_wifi_config_result = {}   # {"ssid": ..., "password": ...}


# ---------------------------------------------------------------------------
# WiFi status helpers
# ---------------------------------------------------------------------------

def is_wifi_connected() -> bool:
    """Return True if any wireless interface has an active IP address."""
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "TYPE,STATE", "con", "show", "--active"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if line.startswith("wifi:") and "activated" in line:
                return True
        return False
    except Exception as e:
        logger.warning("is_wifi_connected: %s", e)
        return False


def get_current_ssid() -> str | None:
    """Return the SSID of the currently active WiFi connection, or None."""
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if line.startswith("yes:"):
                return line.split(":", 1)[1]
        return None
    except Exception as e:
        logger.warning("get_current_ssid: %s", e)
        return None


def get_current_ip(iface: str = "wlan0") -> str | None:
    """Return the IPv4 address of *iface*, or None."""
    try:
        result = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", iface],
            capture_output=True, text=True, timeout=5
        )
        for token in result.stdout.split():
            if "/" in token and not token.startswith("127"):
                return token.split("/")[0]
        return None
    except Exception as e:
        logger.warning("get_current_ip: %s", e)
        return None


# ---------------------------------------------------------------------------
# Hotspot (Access Point) control
# ---------------------------------------------------------------------------

def start_hotspot(ssid: str = HOTSPOT_SSID, password: str = HOTSPOT_PASSWORD) -> str:
    """
    Create a WiFi hotspot using nmcli.

    Returns the AP IP address string (HOTSPOT_IP) on success,
    or the configured HOTSPOT_IP as a fallback if the command fails.
    """
    try:
        # Remove any stale hotspot connection profile first
        subprocess.run(
            ["nmcli", "con", "delete", _HOTSPOT_CON_NAME],
            capture_output=True, timeout=5
        )
    except Exception:
        pass

    try:
        subprocess.run(
            [
                "nmcli", "dev", "wifi", "hotspot",
                "ifname", "wlan0",
                "ssid", ssid,
                "password", password,
                "con-name", _HOTSPOT_CON_NAME,
            ],
            capture_output=True, timeout=15
        )
        logger.info("Hotspot '%s' started", ssid)
    except Exception as e:
        logger.error("start_hotspot: %s", e)

    return HOTSPOT_IP


def stop_hotspot() -> None:
    """Tear down the hotspot connection profile."""
    try:
        subprocess.run(
            ["nmcli", "con", "down", _HOTSPOT_CON_NAME],
            capture_output=True, timeout=10
        )
        subprocess.run(
            ["nmcli", "con", "delete", _HOTSPOT_CON_NAME],
            capture_output=True, timeout=5
        )
        logger.info("Hotspot stopped")
    except Exception as e:
        logger.warning("stop_hotspot: %s", e)


# ---------------------------------------------------------------------------
# WiFi connection
# ---------------------------------------------------------------------------

def connect_wifi(ssid: str, password: str) -> bool:
    """
    Connect to a WiFi network using nmcli.

    Returns True if connected within WIFI_CONNECT_TIMEOUT seconds.
    """
    try:
        subprocess.run(
            ["nmcli", "dev", "wifi", "connect", ssid, "password", password],
            capture_output=True, timeout=WIFI_CONNECT_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        logger.warning("connect_wifi: timeout connecting to '%s'", ssid)
        return False
    except Exception as e:
        logger.error("connect_wifi: %s", e)
        return False

    # Poll for connection
    deadline = time.time() + WIFI_CONNECT_TIMEOUT
    while time.time() < deadline:
        if is_wifi_connected():
            logger.info("Connected to '%s'", ssid)
            return True
        time.sleep(1)

    logger.warning("connect_wifi: failed to associate with '%s'", ssid)
    return False


# ---------------------------------------------------------------------------
# WiFi network scan
# ---------------------------------------------------------------------------

def scan_wifi_networks() -> list[dict]:
    """
    Scan for nearby WiFi networks using nmcli.

    Returns a list of dicts sorted by signal strength (strongest first):
        [{"ssid": str, "signal": int (0-100), "security": str}, ...]

    Duplicate SSIDs are collapsed — only the entry with the strongest signal
    is kept. Hidden networks (empty SSID) are labelled "<Hidden>".
    """
    try:
        result = subprocess.run(
            ["nmcli", "--terse", "--fields", "SSID,SIGNAL,SECURITY",
             "dev", "wifi", "list"],
            capture_output=True, text=True, timeout=10
        )
    except Exception as e:
        logger.warning("scan_wifi_networks: %s", e)
        return []

    seen: dict = {}
    for line in result.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 2:
            continue
        ssid     = parts[0].strip() or "<Hidden>"
        signal   = int(parts[1]) if parts[1].strip().isdigit() else 0
        security = parts[2].strip() if len(parts) > 2 else ""
        if ssid not in seen or signal > seen[ssid]["signal"]:
            seen[ssid] = {"ssid": ssid, "signal": signal, "security": security}

    networks = sorted(seen.values(), key=lambda x: x["signal"], reverse=True)
    logger.info("WiFi scan found %d network(s)", len(networks))
    return networks


# ---------------------------------------------------------------------------
# Captive-portal signal (called by web_server.py when form is submitted)
# ---------------------------------------------------------------------------

def notify_wifi_credentials(ssid: str, password: str) -> None:
    """
    Called by the web server when the user submits WiFi credentials.
    Stores the credentials and signals the waiting run_network_setup().
    """
    global _wifi_config_result
    _wifi_config_result = {"ssid": ssid, "password": password}
    _wifi_config_event.set()


def get_pending_wifi_config() -> dict | None:
    """
    Return pending WiFi credentials if submitted, else None.
    Clears the event so subsequent calls return None until next submission.
    """
    if _wifi_config_event.is_set():
        _wifi_config_event.clear()
        return dict(_wifi_config_result)
    return None


# ---------------------------------------------------------------------------
# High-level setup flow
# ---------------------------------------------------------------------------

def run_network_setup(lcd_lines=None) -> tuple[str | None, str | None]:
    """
    Orchestrate the full network setup sequence.

    If WiFi is already connected, returns (ssid, ip) immediately.
    Otherwise launches a hotspot + captive-portal and waits for the user
    to submit WiFi credentials via the web form.

    Args:
        lcd_lines: optional callable(line1, line2, line3, line4) for LCD output.
                   Pass None to skip LCD updates.

    Returns:
        (ssid, ip) — the connected SSID and local IP address.
    """
    def show(l1="", l2="", l3="", l4=""):
        if lcd_lines:
            try:
                lcd_lines(l1, l2, l3, l4)
            except Exception:
                pass
        logger.info("LCD | %s | %s | %s | %s", l1, l2, l3, l4)

    # --- Already connected? ---
    if is_wifi_connected():
        ssid = get_current_ssid() or "Unknown"
        ip   = get_current_ip()   or "?.?.?.?"
        show("WiFi Connected", f"SSID: {ssid[:14]}", f"IP:{ip}", "Press NEXT to")
        logger.info("Already connected: ssid=%s ip=%s", ssid, ip)
        return ssid, ip

    # --- No connection — start hotspot and captive portal ---
    ap_ip = start_hotspot(HOTSPOT_SSID, HOTSPOT_PASSWORD)
    portal_url = f"{ap_ip}:{CAPTIVE_PORTAL_PORT}"
    show("WiFi Not Found", f"Hotspot:{HOTSPOT_SSID}", f"Pass:{HOTSPOT_PASSWORD}", portal_url)

    # Import here to avoid circular dependency (web_server imports network)
    from web_server import start_captive_portal, stop_captive_portal
    start_captive_portal(ap_ip)

    logger.info("Waiting for WiFi credentials via captive portal...")

    # Poll until credentials arrive
    while True:
        config = get_pending_wifi_config()
        if config:
            ssid     = config["ssid"]
            password = config["password"]
            show("Connecting...", f"SSID:{ssid[:11]}", "", "")
            stop_captive_portal()
            stop_hotspot()

            if connect_wifi(ssid, password):
                ip = get_current_ip() or "?.?.?.?"
                show("WiFi Connected", f"SSID:{ssid[:11]}", f"IP:{ip}", "Press NEXT to")
                return ssid, ip
            else:
                # Retry — restart hotspot
                show("Connect Failed", "Restart hotspot", f"Hotspot:{HOTSPOT_SSID}", f"Pass:{HOTSPOT_PASSWORD}")
                start_hotspot(HOTSPOT_SSID, HOTSPOT_PASSWORD)
                start_captive_portal(ap_ip)
        time.sleep(0.5)
