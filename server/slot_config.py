"""
Slot configuration — load/save server/slot_config.json and expose
cell-level query helpers used by the pipeline.

Schema:
    {
      "rows": 13, "cols": 15,
      "cells": {
        "1":  {"slot_id": "REF_L0", "is_reference": true,  "ref_level": 0},
        "16": {"slot_id": "A01",    "is_reference": false, "ref_level": null}
      }
    }

Cell index is 1-based row-major: cell_index = (row - 1) * cols + col.
Cells absent from "cells" are ignored by the pipeline entirely.
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
            cells[int(k)] = CellConfig(
                slot_id=v["slot_id"],
                is_reference=bool(v["is_reference"]),
                ref_level=v.get("ref_level"),
            )
        return SlotConfig(rows=int(data.get("rows", 13)), cols=int(data.get("cols", 15)), cells=cells)
    except FileNotFoundError:
        return SlotConfig(rows=13, cols=15)
    except Exception:
        return SlotConfig(rows=13, cols=15)


def save_slot_config(cfg: SlotConfig, path: "str | Path | None" = None) -> None:
    p = Path(path) if path else _DEFAULT_PATH
    p.write_text(json.dumps(cfg.to_dict(), indent=2))


def active_cell_indices(cfg: SlotConfig) -> set[int]:
    """All assigned cell indices (both reference and sample)."""
    return set(cfg.cells.keys())


def reference_cells(cfg: SlotConfig) -> dict[int, int]:
    """Map cell_index → ref_level for all reference cells."""
    return {
        k: v.ref_level
        for k, v in cfg.cells.items()
        if v.is_reference and v.ref_level is not None
    }


def sample_cells(cfg: SlotConfig) -> dict[int, str]:
    """Map cell_index → slot_id for all sample (non-reference) cells."""
    return {k: v.slot_id for k, v in cfg.cells.items() if not v.is_reference}
