"""
Google Sheets integration — two tabs in a single spreadsheet.

Tab "SlotAssignment": cleared and rewritten whenever POST /api/slots saves.
Tab "Results":        one batch appended per successful /analyze call.

Both calls are fire-and-forget; any failure is logged but never surfaces to
the caller.

Config (configs/config.toml [google]):
    spreadsheet_id = "..."
    slots_tab      = "SlotAssignment"
    results_tab    = "Results"
"""

from __future__ import annotations

from pathlib import Path
from loguru import logger

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
]


def _get_creds(credentials_file: str, token_file: str):
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
                    logger.debug("sheets: client_secrets.json not found — skipping")
                    return None
                flow = InstalledAppFlow.from_client_secrets_file(credentials_file, _SCOPES)
                creds = flow.run_local_server(port=0)
            token_path.write_text(creds.to_json())

        return creds
    except Exception as e:
        logger.debug("sheets: credential load failed: {}", e)
        return None


def _build_service(credentials_file: str, token_file: str):
    try:
        from googleapiclient.discovery import build
        creds = _get_creds(credentials_file, token_file)
        if creds is None:
            return None
        return build("sheets", "v4", credentials=creds)
    except Exception as e:
        logger.debug("sheets: service build failed: {}", e)
        return None


def _hex_to_rgb_float(hex_color: str) -> dict:
    """Convert '#rrggbb' to {red, green, blue} floats in [0, 1]."""
    h = hex_color.lstrip("#")
    return {
        "red":   int(h[0:2], 16) / 255.0,
        "green": int(h[2:4], 16) / 255.0,
        "blue":  int(h[4:6], 16) / 255.0,
    }


def write_slot_config_to_sheet(
    cfg,  # SlotConfig
    spreadsheet_id: str,
    tab: str,
    credentials_file: str = "client_secrets.json",
    token_file: str = "token.json",
) -> None:
    """Clear and rewrite the SlotAssignment tab with current slot config."""
    try:
        service = _build_service(credentials_file, token_file)
        if service is None:
            return

        header = ["cell_index", "slot_id", "is_reference", "ref_level", "row", "col"]
        rows = [header]
        for cell_idx, cell in sorted(cfg.cells.items()):
            row_num = (cell_idx - 1) // cfg.cols + 1
            col_num = (cell_idx - 1) % cfg.cols + 1
            rows.append([
                cell_idx,
                cell.slot_id,
                str(cell.is_reference),
                str(cell.ref_level) if cell.ref_level is not None else "",
                row_num,
                col_num,
            ])

        sheet = service.spreadsheets()
        range_name = f"{tab}!A1"
        sheet.values().clear(spreadsheetId=spreadsheet_id, range=f"{tab}!A:Z").execute()
        sheet.values().update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption="RAW",
            body={"values": rows},
        ).execute()
        logger.info("sheets: SlotAssignment tab updated ({} cells)", len(cfg.cells))
    except Exception as e:
        logger.warning("sheets: write_slot_config failed: {}", e)


def append_result_to_sheet(
    scan_result: dict,
    spreadsheet_id: str,
    tab: str,
    credentials_file: str = "client_secrets.json",
    token_file: str = "token.json",
) -> None:
    """Append one summary row + N detail rows to the Results tab."""
    try:
        service = _build_service(credentials_file, token_file)
        if service is None:
            return

        sid = scan_result.get("scan_id", "")
        ts  = scan_result.get("timestamp", "")
        det = scan_result.get("detected_count", 0)
        tot = scan_result.get("total_assigned", 0)
        miss = ",".join(scan_result.get("missing_slots", []))
        summary = scan_result.get("summary", {})

        summary_row = [
            f"SCAN:{sid}", ts, det, tot, miss,
            summary.get("L0", 0), summary.get("L1", 0), summary.get("L2", 0),
            summary.get("L3", 0), summary.get("L4", 0),
            "", "", "", "", "", "", "", "",  # align with detail columns
        ]

        detail_header = [
            "scan_id", "slot_id", "cell_index", "is_reference", "ref_level",
            "detected", "color_level", "delta_e", "confident", "L", "a", "b", "hex",
        ]

        rows_to_append = [summary_row, detail_header]
        hex_indices: list[int] = []

        base_row = len(rows_to_append)  # 0-indexed offset relative to first appended row

        all_slots: dict = {}
        # sample slots
        for slot_id, slot_data in scan_result.get("slots", {}).items():
            all_slots[slot_id] = {**slot_data, "_is_reference": False, "_ref_level": None}
        # reference slots embedded in reference_labs
        for ref_level_str, refs in scan_result.get("reference_labs", {}).items():
            for i, ref in enumerate(refs):
                key = f"REF_L{ref_level_str}_{i}"
                all_slots[key] = {
                    "cell_index": None,
                    "detected": True,
                    "_is_reference": True,
                    "_ref_level": int(ref_level_str),
                    "color_level": int(ref_level_str),
                    "delta_e": None,
                    "confident": True,
                    "lab": ref.get("lab"),
                    "hex": ref.get("hex"),
                }

        for slot_id, d in all_slots.items():
            lab = d.get("lab") or [None, None, None]
            hex_color = d.get("hex") or ""
            row_data = [
                sid,
                slot_id,
                d.get("cell_index", ""),
                str(d.get("_is_reference", False)),
                str(d.get("_ref_level", "")),
                str(d.get("detected", False)),
                str(d.get("color_level", "")),
                d.get("delta_e", ""),
                str(d.get("confident", False)),
                lab[0] if lab[0] is not None else "",
                lab[1] if lab[1] is not None else "",
                lab[2] if lab[2] is not None else "",
                hex_color,
            ]
            if hex_color:
                hex_indices.append(len(rows_to_append))
            rows_to_append.append(row_data)

        sheet = service.spreadsheets()
        # Append all rows
        append_resp = sheet.values().append(
            spreadsheetId=spreadsheet_id,
            range=f"{tab}!A1",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": rows_to_append},
        ).execute()

        # Apply background color to hex cells in the "hex" column (col M = index 12)
        if hex_indices:
            updated_range = append_resp["updates"]["updatedRange"]
            start_row_str = updated_range.split("!")[1].split(":")[0]
            import re
            start_row_1based = int(re.search(r"\d+", start_row_str).group())
            hex_col_index = 12  # 0-based: columns A-M → 0-12

            requests = []
            for rel_idx in hex_indices:
                abs_row = start_row_1based + rel_idx - 1  # 1-based
                hex_color = rows_to_append[rel_idx][12]
                if hex_color and hex_color.startswith("#") and len(hex_color) == 7:
                    requests.append({
                        "repeatCell": {
                            "range": {
                                "sheetId": _get_sheet_id(service, spreadsheet_id, tab),
                                "startRowIndex": abs_row - 1,
                                "endRowIndex": abs_row,
                                "startColumnIndex": hex_col_index,
                                "endColumnIndex": hex_col_index + 1,
                            },
                            "cell": {
                                "userEnteredFormat": {
                                    "backgroundColor": _hex_to_rgb_float(hex_color)
                                }
                            },
                            "fields": "userEnteredFormat.backgroundColor",
                        }
                    })
            if requests:
                service.spreadsheets().batchUpdate(
                    spreadsheetId=spreadsheet_id,
                    body={"requests": requests},
                ).execute()

        logger.info("sheets: appended scan {} ({} rows)", sid, len(rows_to_append))
    except Exception as e:
        logger.warning("sheets: append_result failed: {}", e)


def _get_sheet_id(service, spreadsheet_id: str, tab_name: str) -> int:
    """Return the numeric sheetId for a tab by name; 0 if not found."""
    try:
        meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        for s in meta.get("sheets", []):
            if s["properties"]["title"] == tab_name:
                return s["properties"]["sheetId"]
    except Exception:
        pass
    return 0
