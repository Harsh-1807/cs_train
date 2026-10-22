"""Load AMI xlsx data with strict label/GT column blocklist."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

SHEETS_USED = [
    "AMI_Comm_Health",
    "Consumer",
    "Consumer_Monthly_Readings",
    "DCU_Daily_Status",
    "Consumer_Event_Log",
    "Asset_Comm_Health",
    "Network_Topology",
]

SHEETS_OMITTED = [
    "README",
    "Summary",
    "Event_Class",
    "Power_Plant",
    "Transmission_Grid",
    "Grid_Substation",
    "Distribution_Substation",
    "Power_Transformer",
    "Feeder_11kV",
    "DTC",
    "LT_Line",
    "Feeder_Monthly_Audit",
    "Asset_Event_Log",
]

COLUMN_BLOCKLIST_EXACT = {
    "Is_Defect",
    "Defect_Category",
    "Defect_Type",
    "Defect_Start_Month",
    "Record_Label",
    "Consumer_Is_Defect",
    "Linked_Cause_GT",
}

COLUMN_BLOCKLIST_PATTERNS = [
    re.compile(r"^GT_.*"),
    re.compile(r".*_GT$"),
]


def is_blocked_column(name: str) -> bool:
    if name in COLUMN_BLOCKLIST_EXACT:
        return True
    return any(p.match(name) for p in COLUMN_BLOCKLIST_PATTERNS)


COLUMN_BLOCKLIST = sorted(COLUMN_BLOCKLIST_EXACT | {"^GT_.*", ".*_GT$"})


def _strip_blocked_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if not is_blocked_column(c)]
    dropped = [c for c in df.columns if is_blocked_column(c)]
    if dropped:
        pass  # silent strip — documented in METHODOLOGY.md
    return df[keep].copy()


def load_ami_data(path: str, sheets: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load training-safe dataframes (no GT/label columns)."""
    use_sheets = sheets or SHEETS_USED
    xl = pd.ExcelFile(path)
    available = set(xl.sheet_names)
    missing = [s for s in use_sheets if s not in available]
    if missing:
        raise ValueError(f"Missing required sheets: {missing}")

    data: dict[str, pd.DataFrame] = {}
    for sheet in use_sheets:
        df = pd.read_excel(path, sheet_name=sheet)
        data[sheet] = _strip_blocked_columns(df)
    return data


def load_ami_data_with_gt(path: str) -> dict[str, pd.DataFrame]:
    """Load all used sheets WITH GT columns — evaluate.py only."""
    xl = pd.ExcelFile(path)
    data: dict[str, pd.DataFrame] = {}
    for sheet in SHEETS_USED:
        if sheet in xl.sheet_names:
            data[sheet] = pd.read_excel(path, sheet_name=sheet)
    return data


def get_gt_columns(path: str) -> dict[str, list[str]]:
    """Return GT columns per sheet for evaluation."""
    xl = pd.ExcelFile(path)
    result: dict[str, list[str]] = {}
    for sheet in SHEETS_USED:
        if sheet not in xl.sheet_names:
            continue
        df = pd.read_excel(path, sheet_name=sheet, nrows=0)
        result[sheet] = [c for c in df.columns if is_blocked_column(c)]
    return result
