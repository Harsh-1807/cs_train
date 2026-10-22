"""Map ground-truth Defect_Type to 5-class root cause taxonomy (evaluate only)."""

from __future__ import annotations


def defect_type_to_root_cause(defect_type: str) -> str:
    if not defect_type or str(defect_type).strip().lower() in ("none", "nan", ""):
        return "Normal"
    dt = str(defect_type).lower()

    if "tamper" in dt:
        return "Tamper"
    if any(k in dt for k in ("poor rf", "cellular", "sim", "backhaul", "coverage", "network")):
        return "Network coverage"
    if any(k in dt for k in ("firmware", "upgrade", "mismatch")):
        return "Firmware"
    if any(k in dt for k in ("supply", "outage", "tripping", "transient", "power")):
        return "Transient"
    if any(k in dt for k in ("nic", "hardware", "fault", "dcu", "gateway", "hes mapping", "malfunction")):
        return "Hardware"
    return "Unknown"
