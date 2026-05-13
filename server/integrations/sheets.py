"""
Google Sheets integration — three tabs in a single spreadsheet.

Tab "Detail":         one row per bottle per scan (detailed metrics).
Tab "Summary":        one row per scan (level distribution counts).
Tab "SlotAssignment": cleared and rewritten whenever POST /api/slots saves.

All calls are fire-and-forget; any failure is logged but never surfaces to
the caller.

Config (configs/config.toml [google]):
    spreadsheet_id       = "..."
    detail_tab           = "Detail"
    summary_tab          = "Summary"
    slots_tab            = "SlotAssignment"
    service_account_file = "credentials.json"
"""

from __future__ import annotations

import re
from pathlib import Path
from loguru import logger

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

_DETAIL_HEADER = [
    "scan_id", "created_at",
    "is_reference", "ref_level",
    "slot_id", "cell_index", "row", "column",
    "L", "a", "b", "hex",
    "path_idx", "path_delta", "hist_delta", "delta_e",
    "color_level", "is_forced", "is_achromatic",
    "detected", "confident",
]

_SUMMARY_HEADER = [
    "scan_id", "created_at",
    "detected_count", "total_assigned", "missing_slots",
    "L0", "L1", "L2", "L3", "L4", "image_url",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_creds(service_account_file: str):
    try:
        from google.oauth2 import service_account as sa
        if not Path(service_account_file).exists():
            logger.debug("sheets: {} not found — skipping", service_account_file)
            return None
        return sa.Credentials.from_service_account_file(service_account_file, scopes=_SCOPES)
    except Exception as e:
        logger.debug("sheets: credential load failed: {}", e)
        return None


def _build_service(service_account_file: str):
    try:
        from googleapiclient.discovery import build
        creds = _get_creds(service_account_file)
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


def _ensure_header(service, spreadsheet_id: str, tab: str, header: list[str]) -> None:
    """Write the header row to the tab only if the tab is currently empty."""
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=f"{tab}!A1:A1",
        ).execute()
        if result.get("values"):
            return  # already has data
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f"{tab}!A1",
            valueInputOption="RAW",
            body={"values": [header]},
        ).execute()
    except Exception as e:
        logger.debug("sheets: _ensure_header failed for {}: {}", tab, e)


def _apply_hex_backgrounds(
    service,
    spreadsheet_id: str,
    tab: str,
    start_row_1based: int,
    rows: list[list],
    hex_col_index: int,
) -> None:
    """Apply cell background color to rows that have a hex value in hex_col_index."""
    requests = []
    sheet_id = _get_sheet_id(service, spreadsheet_id, tab)
    for i, row in enumerate(rows):
        hex_color = row[hex_col_index] if hex_col_index < len(row) else ""
        if hex_color and isinstance(hex_color, str) and hex_color.startswith("#") and len(hex_color) == 7:
            abs_row = start_row_1based + i - 1  # 0-based for API
            requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": abs_row,
                        "endRowIndex": abs_row + 1,
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def append_detail_to_sheet(
    scan_result: dict,
    spreadsheet_id: str,
    tab: str,
    service_account_file: str = "credentials.json",
) -> None:
    """Append one row per bottle (samples + references) to the Detail tab."""
    try:
        service = _build_service(service_account_file)
        if service is None:
            return

        sid       = scan_result.get("scan_id", "")
        ts        = scan_result.get("timestamp", "")
        grid_cols = scan_result.get("grid_cols", 15)

        def _row_col(ci):
            if ci is None:
                return "", ""
            return (ci - 1) // grid_cols + 1, (ci - 1) % grid_cols + 1

        rows: list[list] = []

        for slot_id, d in scan_result.get("slots", {}).items():
            ci  = d.get("cell_index")
            lab = d.get("lab") or [None, None, None]
            r, c = _row_col(ci)
            rows.append([
                sid, ts,
                False, "",
                slot_id, ci if ci is not None else "", r, c,
                lab[0] if lab[0] is not None else "",
                lab[1] if lab[1] is not None else "",
                lab[2] if lab[2] is not None else "",
                d.get("hex") or "",
                d.get("concentration_index") if d.get("concentration_index") is not None else "",
                d.get("path_distance")       if d.get("path_distance")       is not None else "",
                d.get("hist_bhatt")          if d.get("hist_bhatt")          is not None else "",
                d.get("delta_e")             if d.get("delta_e")             is not None else "",
                d.get("color_level")         if d.get("color_level")         is not None else "",
                bool(d.get("best_fit")),
                bool(d.get("force_l0")),
                bool(d.get("detected")),
                bool(d.get("confident")),
            ])

        for ref_level_str, refs in scan_result.get("reference_labs", {}).items():
            ref_level = int(ref_level_str)
            for i, ref in enumerate(refs):
                lab = ref.get("lab") or [None, None, None]
                ci  = ref.get("cell_index")
                r, c = _row_col(ci)
                rows.append([
                    sid, ts,
                    True, ref_level,
                    f"REF_L{ref_level}_{i}",
                    ci if ci is not None else "", r, c,
                    lab[0] if lab[0] is not None else "",
                    lab[1] if lab[1] is not None else "",
                    lab[2] if lab[2] is not None else "",
                    ref.get("hex") or "",
                    "", "", "", "",
                    ref_level, False, False, True, True,
                ])

        if not rows:
            return

        _ensure_header(service, spreadsheet_id, tab, _DETAIL_HEADER)

        append_resp = service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=f"{tab}!A1",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": rows},
        ).execute()

        # Color the hex column (index 11 = column L)
        updated_range = append_resp["updates"]["updatedRange"]
        start_str = updated_range.split("!")[1].split(":")[0]
        start_row = int(re.search(r"\d+", start_str).group())
        _apply_hex_backgrounds(service, spreadsheet_id, tab, start_row, rows, hex_col_index=11)

        logger.info("sheets: Detail tab — appended {} rows for scan {}", len(rows), sid)
    except Exception as e:
        logger.warning("sheets: append_detail_to_sheet failed: {}", e)


def append_summary_to_sheet(
    scan_result: dict,
    spreadsheet_id: str,
    tab: str,
    service_account_file: str = "credentials.json",
    drive_file_id: str | None = None,
) -> None:
    """Append one summary row to the Summary tab.

    drive_file_id: Google Drive file ID of the annotated image. When provided,
    the image_url column is written as a clickable =HYPERLINK() formula.
    """
    try:
        service = _build_service(service_account_file)
        if service is None:
            return

        sid     = scan_result.get("scan_id", "")
        ts      = scan_result.get("timestamp", "")
        summary = scan_result.get("summary", {})

        if drive_file_id:
            image_url = f'=HYPERLINK("https://drive.google.com/file/d/{drive_file_id}/view","View Image")'
        else:
            image_url = ""

        row = [
            sid, ts,
            scan_result.get("detected_count", 0),
            scan_result.get("total_assigned", 0),
            ",".join(scan_result.get("missing_slots", [])),
            summary.get("L0", 0), summary.get("L1", 0),
            summary.get("L2", 0), summary.get("L3", 0),
            summary.get("L4", 0),
            image_url,
        ]

        _ensure_header(service, spreadsheet_id, tab, _SUMMARY_HEADER)

        service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=f"{tab}!A1",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": [row]},
        ).execute()

        logger.info("sheets: Summary tab — appended scan {}", sid)
    except Exception as e:
        logger.warning("sheets: append_summary_to_sheet failed: {}", e)


def write_slot_config_to_sheet(
    cfg,  # SlotConfig
    spreadsheet_id: str,
    tab: str,
    service_account_file: str = "credentials.json",
) -> None:
    """Clear and rewrite the SlotAssignment tab with current slot config."""
    try:
        service = _build_service(service_account_file)
        if service is None:
            return

        header = [
            "cell_index", "slot_id", "is_reference", "ref_level",
            "row", "col", "grid_rows", "grid_cols", "is_white_reference",
        ]
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
                cfg.rows,
                cfg.cols,
                str(cell.is_white_reference),
            ])

        sheet = service.spreadsheets()
        sheet.values().clear(spreadsheetId=spreadsheet_id, range=f"{tab}!A:Z").execute()
        sheet.values().update(
            spreadsheetId=spreadsheet_id,
            range=f"{tab}!A1",
            valueInputOption="RAW",
            body={"values": rows},
        ).execute()
        logger.info("sheets: SlotAssignment tab updated ({} cells)", len(cfg.cells))
    except Exception as e:
        logger.warning("sheets: write_slot_config failed: {}", e)


def read_slot_config_from_sheet(
    spreadsheet_id: str,
    tab: str,
    service_account_file: str = "credentials.json",
):
    """
    Read the SlotAssignment tab and return a SlotConfig.
    Returns None if the sheet is unreachable or empty.
    Column order: cell_index | slot_id | is_reference | ref_level | row | col | grid_rows | grid_cols
    """
    try:
        from server.slot_config import SlotConfig, CellConfig

        service = _build_service(service_account_file)
        if service is None:
            return None

        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=f"{tab}!A:I",
        ).execute()
        rows = result.get("values", [])
        if len(rows) < 2:
            return None

        cells: dict[int, CellConfig] = {}
        cols_set: set[int] = set()
        grid_rows_stored: int | None = None
        grid_cols_stored: int | None = None

        for row in rows[1:]:
            if len(row) < 4:
                continue
            try:
                cell_idx  = int(row[0])
                slot_id   = str(row[1])
                is_ref    = row[2].strip().lower() == "true"
                ref_level = int(row[3]) if row[3].strip() else None
                if len(row) > 5 and row[5]:
                    cols_set.add(int(row[5]))
                if len(row) > 6 and row[6] and grid_rows_stored is None:
                    grid_rows_stored = int(row[6])
                if len(row) > 7 and row[7] and grid_cols_stored is None:
                    grid_cols_stored = int(row[7])
                is_wb = len(row) > 8 and row[8].strip().lower() == "true"
                cells[cell_idx] = CellConfig(
                    slot_id=slot_id,
                    is_reference=is_ref,
                    ref_level=ref_level,
                    is_white_reference=is_wb,
                )
            except (ValueError, IndexError):
                continue

        if not cells:
            return None

        grid_cols = grid_cols_stored or (max(cols_set) if cols_set else 15)
        max_cell  = max(cells.keys())
        grid_rows = grid_rows_stored or ((max_cell - 1) // grid_cols + 1)

        logger.info("sheets: loaded {} cells from {} (grid {}×{})", len(cells), tab, grid_rows, grid_cols)
        return SlotConfig(rows=grid_rows, cols=grid_cols, cells=cells)
    except Exception as e:
        logger.warning("sheets: read_slot_config failed: {}", e)
        return None
