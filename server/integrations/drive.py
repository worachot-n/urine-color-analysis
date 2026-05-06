"""
Google Drive integration — upload annotated scan images.

Uploads are fire-and-forget: any failure is logged but never propagates to
the pipeline caller, so a missing service account file or network error does
not break the /analyze response.

Config (configs/config.toml [google]):
    service_account_file = "credentials.json"
    drive_folder_id      = "..."
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

_SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
]


def _get_creds(service_account_file: str):
    """Load service account credentials; return None if unavailable."""
    try:
        from google.oauth2 import service_account as sa

        if not Path(service_account_file).exists():
            logger.debug("drive: {} not found — skipping upload", service_account_file)
            return None
        return sa.Credentials.from_service_account_file(service_account_file, scopes=_SCOPES)
    except Exception as e:
        logger.debug("drive: credential load failed: {}", e)
        return None


def upload_image(
    jpeg_bytes: bytes,
    filename: str,
    folder_id: str,
    service_account_file: str = "credentials.json",
) -> str | None:
    """
    Upload *jpeg_bytes* to Google Drive folder *folder_id*.

    Returns the Drive file ID, or None on failure.
    """
    try:
        import io
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseUpload

        creds = _get_creds(service_account_file)
        if creds is None:
            return None

        service = build("drive", "v3", credentials=creds)
        media = MediaIoBaseUpload(io.BytesIO(jpeg_bytes), mimetype="image/jpeg")
        meta = {"name": filename, "parents": [folder_id]}
        file = service.files().create(body=meta, media_body=media, fields="id").execute()
        file_id = file.get("id")
        logger.info("drive: uploaded {} → file_id={}", filename, file_id)
        return file_id
    except Exception as e:
        logger.warning("drive: upload failed for {}: {}", filename, e)
        return None
