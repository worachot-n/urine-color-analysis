"""
Google Sheets integration — two tabs in a single spreadsheet.

Tab "SlotAssignment": cleared and rewritten whenever POST /api/slots saves.
Tab "Results":        one batch appended per successful /analyze call.

Both calls are fire-and-forget; any failure is logged but never surfaces to
the caller.

Config (configs/config.toml [google]):
    spreadsheet_id       = "..."
    slots_tab            = "SlotAssignment"
    results_tab          = "Results"
    service_account_file = "credentials.json"
"""

from __future__ import annotations

from pathlib import Path
from loguru import logger

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
]


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
    service_account_file: str = "credentials.json",
) -> None:
    """Append one summary row + N detail rows to the Results tab."""
    try:
        service = _build_service(service_account_file)
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
            "", "", "", "", "", "", "", "", "", "",  # align with 15 detail columns
        ]

        detail_header = [
            "scan_id", "slot_id", "cell_index", "is_reference", "ref_level",
            "detected", "color_level", "delta_e", "confident", "L", "a", "b", "hex",
            "hist_bhatt", "combined",
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
                d.get("hist_bhatt", "") if d.get("hist_bhatt") is not None else "",
                d.get("combined", "") if d.get("combined") is not None else "",
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
                is_wb = (
                    len(row) > 8
                    and row[8].strip().lower() == "true"
                )
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

        # Prefer explicit dimensions stored in the sheet; fall back to inference
        grid_cols = grid_cols_stored or (max(cols_set) if cols_set else 15)
        max_cell  = max(cells.keys())
        grid_rows = grid_rows_stored or ((max_cell - 1) // grid_cols + 1)

        logger.info("sheets: loaded {} cells from {} (grid {}×{})", len(cells), tab, grid_rows, grid_cols)
        return SlotConfig(rows=grid_rows, cols=grid_cols, cells=cells)
    except Exception as e:
        logger.warning("sheets: read_slot_config failed: {}", e)
        return None


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
