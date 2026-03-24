#!/usr/bin/env python3
"""
Urine Color Analysis System — Unified Entry Point

Usage
-----
Ubuntu server:
    uv run main.py --role server

Raspberry Pi client:
    uv run main.py --role client --server-url https://your-tunnel.trycloudflare.com

Install dependencies
--------------------
Ubuntu:
    uv sync --extra server --extra common

Raspberry Pi:
    uv sync --extra pi --extra common
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

# File log — rotated at 10 MB, kept for 7 days
Path("logs").mkdir(exist_ok=True)
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} — {message}",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Urine Color Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--role",
        choices=["client", "server"],
        required=True,
        help="Run as 'server' (Ubuntu/FastAPI) or 'client' (Raspberry Pi)",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="Full HTTPS URL of the analysis server (client mode only). "
             "Overrides SERVER_URL in .env.",
    )
    args = parser.parse_args()

    if args.role == "server":
        logger.info("Starting in SERVER mode")
        from app.server_app import run_server
        run_server()

    elif args.role == "client":
        # Resolve server URL: CLI flag > .env/settings
        from app.shared.config import cfg
        server_url = args.server_url or cfg.server_url
        if not server_url or server_url == "http://localhost:8000":
            logger.error(
                "No server URL provided.  Pass --server-url or set SERVER_URL in .env"
            )
            sys.exit(1)
        logger.info("Starting in CLIENT mode → {}", server_url)
        from app.client_app import run_client
        run_client(server_url=server_url)


if __name__ == "__main__":
    main()
