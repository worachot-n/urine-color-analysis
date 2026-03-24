# CLAUDE.md — Urine Analysis System v2

This file provides context for AI Code Assistants (Cursor, Claude, Copilot) about the architecture, conventions, and constraints of this project. Read this before modifying any file.

---

## Project Overview

A distributed urine sample analysis system that detects and counts urine collection bottles using a YOLO object detection model.

| Role       | Hardware        | Responsibility                                                                             |
| ---------- | --------------- | ------------------------------------------------------------------------------------------ |
| **Client** | Raspberry Pi 4B | Capture image via PiCamera2, trigger via GPIO24, display results on LCD + TM1637           |
| **Server** | Ubuntu PC       | Run FastAPI, perform YOLO inference via OpenVINO, store results in SQLite, serve Dashboard |

Communication: Pi POSTs a raw JPEG to `https://www.<YOUR_DOMAIN>.com/analyze` → Server returns JSON.
The server URL is a **public HTTPS domain via Cloudflare Tunnel** — not a local IP. No port number in the URL.

---

## Tech Stack

| Layer            | Library / Tool                                                              |
| ---------------- | --------------------------------------------------------------------------- |
| Package Manager  | `uv` (not pip, not poetry)                                                  |
| Server Framework | FastAPI + Uvicorn                                                           |
| AI Inference     | OpenVINO Runtime + YOLO (yolo26s)                                           |
| Image Processing | OpenCV (`cv2`)                                                              |
| Database         | SQLAlchemy + SQLite (`data/results.db`)                                     |
| Web Templates    | Jinja2                                                                      |
| Config           | `pydantic-settings` reading `settings.yaml`                                 |
| Logging          | `loguru` — use throughout, never `print()`                                  |
| Pi Camera        | `picamera2`                                                                 |
| Pi GPIO          | `RPi.GPIO` (BCM mode, pin 24)                                               |
| Pi Display       | `rpi-lcd` (I2C LCD) + `raspberrypi-tm1637` (7-segment)                      |
| HTTP Client      | `requests`                                                                  |
| Tunnel           | Cloudflare Tunnel (`cloudflared`) — exposes local FastAPI over public HTTPS |

---

## Repository Layout

```
urine-analysis-v2/
├── .python-version          # Managed by uv (e.g. 3.11)
├── pyproject.toml           # Single source of truth for all deps
├── settings.yaml            # Runtime config (never hardcode IPs or pins here)
├── main.py                  # CLI entry: --role client|server
├── app/
│   ├── __init__.py
│   ├── client_app.py        # Pi-only: camera, GPIO, LCD, TM1637, requests
│   ├── server_app.py        # Ubuntu-only: FastAPI, OpenVINO, DB, Dashboard
│   ├── shared/
│   │   ├── __init__.py
│   │   ├── config.py        # Pydantic-settings Config class
│   │   └── processor.py     # letterbox_image() + scale_coords()
│   └── web/
│       ├── templates/       # Jinja2 HTML (dashboard.html, etc.)
│       └── static/          # CSS, JS, and saved analysis images
├── models/
│   └── yolo26s_openvino/    # model.xml + model.bin (gitignored, large files)
└── data/
    └── results.db           # SQLite — server-side only, gitignored
```

---

## Dependency Management

Always use `uv`. Never run bare `pip install`.

```toml
# pyproject.toml structure
[project.optional-dependencies]
common = ["requests", "pydantic-settings", "loguru", "numpy"]
pi     = ["picamera2", "RPi.GPIO", "raspberrypi-tm1637", "rpi-lcd"]
server = ["fastapi", "uvicorn[standard]", "openvino", "opencv-python",
          "sqlalchemy", "jinja2", "python-multipart"]
```

**Install commands:**

```bash
# On Ubuntu Server
uv sync --extra server --extra common

# On Raspberry Pi
uv sync --extra pi --extra common
```

---

## Running the System

### Ubuntu Server

```bash
# 1. Start FastAPI (listens on localhost:8000)
uv run main.py --role server

# 2. In a separate terminal — start Cloudflare Tunnel
cloudflared tunnel run <TUNNEL_NAME>
# This exposes https://www.<YOUR_DOMAIN>.com → localhost:8000
# Dashboard available at https://www.<YOUR_DOMAIN>.com/dashboard
```

> **Tunnel must be running before the Pi sends any requests.**
> FastAPI itself only binds to `127.0.0.1:8000` (or `0.0.0.0:8000`). Cloudflare handles TLS and the public domain.

### Raspberry Pi (Client)

```bash
uv run main.py --role client --server-url https://www.<YOUR_DOMAIN>.com
```

No port number needed — Cloudflare Tunnel handles routing on standard HTTPS port 443.

---

## Configuration (`settings.yaml`)

```yaml
server:
  host: "127.0.0.1" # Bind to localhost only — Cloudflare Tunnel handles external access
  port: 8000

tunnel:
  public_url: "https://www.<YOUR_DOMAIN>.com" # Cloudflare Tunnel public URL (no trailing slash)

model:
  path: "models/yolo26s_openvino"
  input_size: 640 # YOLO input: 640x640
  confidence_threshold: 0.5
  iou_threshold: 0.45

camera:
  width: 4608
  height: 2592

gpio:
  trigger_pin: 24 # BCM numbering

display:
  lcd_address: 0x27 # I2C address
  tm1637_clk: 5
  tm1637_dio: 6
```

All values must be loaded via `app/shared/config.py` (Pydantic Settings). Never hardcode these values in logic files.

---

## Image Processing Pipeline (Critical)

The model expects **640×640 white-padded** images, but the camera produces **4608×2592**.

### `processor.py` must implement:

**`letterbox_image(img, target_size=640, pad_color=(255,255,255))`**

- Compute scale = `target_size / max(original_w, original_h)`
- Resize proportionally
- Pad **with white** (not black) to reach 640×640
- Return: `(padded_img, scale, pad_x, pad_y)`

**`scale_coords(boxes, scale, pad_x, pad_y)`**

- Inverse transform: `x_orig = (x_padded - pad_x) / scale`
- Return bounding boxes in original 4608×2592 coordinate space

> **Why white padding?** Bottle labels and liquid colors are dark/colored on white backgrounds. Black padding creates false contrast edges that confuse the model. White padding keeps the visual context natural.

---

## API Contract

### `POST /analyze`

**Request:** `multipart/form-data` with field `file` (JPEG image)

**Response:**

```json
{
  "status": "success",
  "count": 12,
  "color_summary": {
    "yellow": 8,
    "amber": 4
  },
  "timestamp": "2025-01-15T10:30:00",
  "image_id": "uuid-string"
}
```

**Error response:**

```json
{
  "status": "error",
  "message": "Model inference failed: <reason>"
}
```

---

## Database Schema (SQLAlchemy)

```python
class AnalysisResult(Base):
    __tablename__ = "results"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    timestamp  = Column(DateTime, default=datetime.utcnow)
    count      = Column(Integer)
    color_json = Column(String)   # JSON string of color_summary dict
    image_path = Column(String)   # Relative path under app/web/static/
```

---

## LCD & TM1637 Display Logic

| State                  | LCD Line 1       | LCD Line 2     | TM1637 |
| ---------------------- | ---------------- | -------------- | ------ |
| Startup                | `Urine Analyzer` | `Ready`        | `----` |
| Button pressed         | `Capturing...`   | _(blank)_      | `----` |
| Waiting for server     | `Processing...`  | `Please wait`  | `----` |
| Result received        | `Count: 12`      | `Y:8 A:4`      | `12`   |
| Server unreachable     | `ERR: SRV OFF`   | `Check tunnel` | `Err`  |
| Cloudflare 524 timeout | `ERR: TIMEOUT`   | `Retry later`  | `Err`  |
| Inference failed       | `ERR: AI FAIL`   | _(blank)_      | `Err`  |
| SSL error              | `ERR: SSL`       | `Check domain` | `Err`  |

---

## Logging Conventions

Use `loguru` exclusively. Never use `print()` or the stdlib `logging` module.

```python
from loguru import logger

logger.info("Server started on {host}:{port}", host=cfg.server.host, port=cfg.server.port)
logger.debug("Inference input shape: {shape}", shape=input_tensor.shape)
logger.warning("Low confidence detection ignored: {score:.3f}", score=conf)
logger.error("Failed to connect to server: {err}", err=str(e))
logger.success("Analysis complete. Count={count}", count=result["count"])
```

Log format in `main.py`:

```python
logger.add("logs/app_{time}.log", rotation="10 MB", retention="7 days", level="DEBUG")
```

---

## Startup Lifecycle (Server)

```
main.py --role server
  └── server_app.py: create_app()
        ├── @app.on_event("startup")
        │     ├── Load OpenVINO model (once, store in app.state.model)
        │     ├── Create DB tables if not exist
        │     └── logger.success("Model loaded. Ready.")
        ├── POST /analyze  → infer → save DB → return JSON
        └── GET  /dashboard → render Jinja2 template with last 50 results
```

---

## Startup Lifecycle (Client)

```
main.py --role client
  └── client_app.py: run()
        ├── GPIO.setup(24, IN, pull_up_down=PUD_DOWN)
        ├── LCD: "Urine Analyzer / Ready"
        ├── TM1637: "----"
        └── loop:
              wait for GPIO24 rising edge
              └── capture() → POST to /analyze → parse JSON → update displays
```

---

## Error Handling Rules

1. **All `requests.post` calls** must catch these exceptions:
   ```python
   except requests.exceptions.ConnectionError:
       lcd.show("ERR: SRV OFF", "Check tunnel")
   except requests.exceptions.Timeout:
       lcd.show("ERR: TIMEOUT", "Retry later")
   except requests.exceptions.SSLError:
       lcd.show("ERR: SSL", "Check domain")
   ```
   Never let any of these crash the main loop.
2. **Model inference errors** must be caught and logged; return HTTP 500 with `{"status": "error", ...}`.
3. **GPIO cleanup** must always run in a `finally` block: `GPIO.cleanup()`.
4. **Camera resource** must be released in `finally`: `picam2.close()`.
5. **DB session** must use context manager (`with Session() as session:`).

---

## What NOT To Do

- ❌ Do not use `pip install` — always `uv add` or edit `pyproject.toml` then `uv sync`
- ❌ Do not import `picamera2` or `RPi.GPIO` in `server_app.py`
- ❌ Do not import `openvino` or `cv2` in `client_app.py`
- ❌ Do not hardcode IP addresses, pin numbers, or file paths — read from `settings.yaml` via config
- ❌ Do not use black padding in `letterbox_image` — must be white `(255, 255, 255)`
- ❌ Do not run inference on the Pi — the Pi only sends the image, never processes it
- ❌ Do not block the GPIO event loop with long `time.sleep()` calls
- ❌ Do not store images outside `app/web/static/` on the server
- ❌ Do not bind FastAPI to `0.0.0.0` in production — bind to `127.0.0.1` and let Cloudflare Tunnel handle external traffic
- ❌ Do not hardcode `http://` for the server URL on the Pi — the Cloudflare Tunnel URL is always `https://`
- ❌ Do not include a port number in `--server-url` when using Cloudflare Tunnel (Tunnel uses standard port 443)

---

## Testing Connection Before First Run

Run this script on the Pi to verify the Cloudflare Tunnel is reachable and measure latency:

```bash
uv run python -c "
import requests, time, sys
url = sys.argv[1].rstrip('/') + '/health'
print(f'Testing: {url}')
for i in range(5):
    t = time.time()
    try:
        r = requests.get(url, timeout=10)   # Tunnel may add ~50-200ms latency
        ms = (time.time() - t) * 1000
        print(f'[{i+1}] {ms:.1f}ms — HTTP {r.status_code}')
    except requests.exceptions.SSLError as e:
        print(f'[{i+1}] SSL ERROR: {e}')
    except requests.exceptions.ConnectionError as e:
        print(f'[{i+1}] CONNECTION FAIL: {e}')
    time.sleep(1)
" https://www.<YOUR_DOMAIN>.com
```

**Expected:** latency 50–250ms (Cloudflare adds overhead vs LAN). If SSL error occurs, verify the domain is correctly configured in Cloudflare dashboard.

Server must expose `GET /health` returning `{"status": "ok", "tunnel": "cloudflare"}`.

---

## Cloudflare Tunnel Setup (One-time)

Cloudflare Tunnel replaces the need for port forwarding or exposing a public IP. The Pi connects to the server via a secure public HTTPS domain.

### On Ubuntu Server — install and configure:

```bash
# Install cloudflared
curl -L https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg
# (follow Cloudflare docs for your distro)

# Authenticate
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create urine-analysis

# Create config at ~/.cloudflared/config.yml:
```

```yaml
# ~/.cloudflared/config.yml
tunnel: <TUNNEL_ID>
credentials-file: /home/<USER>/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: www.<YOUR_DOMAIN>.com
    service: http://127.0.0.1:8000
  - service: http_status:404
```

```bash
# Run tunnel (or set up as systemd service)
cloudflared tunnel run urine-analysis
```

### Important behaviors to handle in code:

| Scenario                                 | Behavior                                                               |
| ---------------------------------------- | ---------------------------------------------------------------------- |
| Tunnel is down                           | `requests` raises `ConnectionError` — show `ERR: SRV OFF` on LCD       |
| Large image upload (4608×2592 JPEG ~3MB) | Set `requests.post(..., timeout=30)` — Tunnel adds latency             |
| Cloudflare 524 timeout                   | Server took >100s — log warning, show `ERR: TIMEOUT` on LCD            |
| SSL certificate                          | Handled automatically by Cloudflare — no manual cert management needed |

### Timeout settings for `client_app.py`:

```python
# Cloudflare Tunnel adds ~100-300ms overhead vs LAN
# Large JPEG upload needs generous timeout
response = requests.post(
    f"{cfg.tunnel.public_url}/analyze",
    files={"file": image_bytes},
    timeout=30       # 30s for upload + inference
)
```

---

## Git Ignore Recommendations

```gitignore
data/
models/
logs/
app/web/static/images/
*.pyc
__pycache__/
.venv/
```

---

_Last updated: auto-generated by Claude from project spec. Keep this file in sync when architecture changes._
