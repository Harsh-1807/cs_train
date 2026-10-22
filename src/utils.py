"""Shared utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(config: dict[str, Any]) -> torch.device:
    pref = config.get("device", "auto")
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
