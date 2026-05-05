"""
Google Drive integration — upload annotated scan images.

Uploads are fire-and-forget: any failure is logged but never propagates to
the pipeline caller, so a missing OAuth token or network error does not
break the /analyze response.

Config (configs/config.toml [google]):
    credentials_file = "client_secrets.json"
    token_file       = "token.json"
    drive_folder_id  = "..."
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

_SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
]


def _get_creds(credentials_file: str, token_file: str):
    """Load or refresh OAuth2 credentials; return None if unavailable."""
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request

        creds = None
        token_path = Path(token_file)
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not Path(credentials_file).exists():
                    logger.debug("drive: client_secrets.json not found — skipping upload")
                    return None
                flow = InstalledAppFlow.from_client_secrets_file(credentials_file, _SCOPES)
                creds = flow.run_local_server(port=0)
            token_path.write_text(creds.to_json())

        return creds
    except Exception as e:
        logger.debug("drive: credential load failed: {}", e)
        return None


def upload_image(
    jpeg_bytes: bytes,
    filename: str,
    folder_id: str,
    credentials_file: str = "client_secrets.json",
    token_file: str = "token.json",
) -> str | None:
    """
    Upload *jpeg_bytes* to Google Drive folder *folder_id*.

    Returns the Drive file ID, or None on failure.
    """
    try:
        import io
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseUpload

        creds = _get_creds(credentials_file, token_file)
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
