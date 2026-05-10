"""
Slot configuration — load/save server/slot_config.json and expose
cell-level query helpers used by the pipeline.

Schema:
    {
      "rows": 13, "cols": 15,
      "cells": {
        "1":  {"slot_id": "REF_L0", "is_reference": true,  "ref_level": 0,
               "is_white_reference": false},
        "6":  {"slot_id": "WB1",    "is_reference": false, "ref_level": null,
               "is_white_reference": true},
        "16": {"slot_id": "A01",    "is_reference": false, "ref_level": null,
               "is_white_reference": false}
      }
    }

Cell index is 1-based row-major: cell_index = (row - 1) * cols + col.
Cells absent from "cells" are ignored by the pipeline entirely.

Cell types are mutually exclusive — `is_reference` and `is_white_reference`
must not both be true; load_slot_config raises ValueError if they are.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_DEFAULT_PATH = Path(__file__).parent / "slot_config.json"


@dataclass
class CellConfig:
    slot_id: str
    is_reference: bool
    ref_level: Optional[int]
    is_white_reference: bool = False


@dataclass
class SlotConfig:
    rows: int
    cols: int
    cells: dict[int, CellConfig] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "cells": {
                str(k): {
                    "slot_id": v.slot_id,
                    "is_reference": v.is_reference,
                    "ref_level": v.ref_level,
                    "is_white_reference": v.is_white_reference,
                }
                for k, v in self.cells.items()
            },
        }


def load_slot_config(path: "str | Path | None" = None) -> SlotConfig:
    p = Path(path) if path else _DEFAULT_PATH
    try:
        data = json.loads(p.read_text())
        cells: dict[int, CellConfig] = {}
        for k, v in data.get("cells", {}).items():
            is_ref = bool(v.get("is_reference", False))
            is_wb  = bool(v.get("is_white_reference", False))
            if is_ref and is_wb:
                raise ValueError(
                    f"slot_config cell {k}: is_reference and is_white_reference "
                    "cannot both be true"
                )
            cells[int(k)] = CellConfig(
                slot_id=v["slot_id"],
                is_reference=is_ref,
                ref_level=v.get("ref_level"),
                is_white_reference=is_wb,
            )
        return SlotConfig(rows=int(data.get("rows", 13)), cols=int(data.get("cols", 15)), cells=cells)
    except FileNotFoundError:
        return SlotConfig(rows=13, cols=15)
    except ValueError:
        raise
    except Exception:
        return SlotConfig(rows=13, cols=15)


def save_slot_config(cfg: SlotConfig, path: "str | Path | None" = None) -> None:
    p = Path(path) if path else _DEFAULT_PATH
    p.write_text(json.dumps(cfg.to_dict(), indent=2))


def active_cell_indices(cfg: SlotConfig) -> set[int]:
    """All assigned cell indices (reference, sample, or white-reference)."""
    return set(cfg.cells.keys())


def reference_cells(cfg: SlotConfig) -> dict[int, int]:
    """Map cell_index → ref_level for all colour-reference cells."""
    return {
        k: v.ref_level
        for k, v in cfg.cells.items()
        if v.is_reference and v.ref_level is not None
    }


def white_reference_cells(cfg: SlotConfig) -> set[int]:
    """All cell indices flagged as white-reference (neutral patch for WB)."""
    return {k for k, v in cfg.cells.items() if v.is_white_reference}


def sample_cells(cfg: SlotConfig) -> dict[int, str]:
    """
    Map cell_index → slot_id for sample (non-reference, non-WB) cells only.

    Both `is_reference` and `is_white_reference` are excluded so the pipeline
    doesn't try to classify a paper patch as a urine sample.
    """
    return {
        k: v.slot_id
        for k, v in cfg.cells.items()
        if not v.is_reference and not v.is_white_reference
    }
