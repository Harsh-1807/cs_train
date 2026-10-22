"""Feeder-level holdout split for inductive evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _feeder_comm_rates(ami: pd.DataFrame) -> pd.DataFrame:
    """Compute per-feeder rate of Non-Communicating + Intermittent meters."""
    bad = ami["Comm_Status"].astype(str).str.contains(
        "Non-Communicating|Intermittent", case=False, na=False
    )
    tmp = ami.assign(_bad=bad.astype(int))
    agg = (
        tmp.groupby("Feeder_Code", as_index=False)
        .agg(bad_rate=("_bad", "mean"), n=("Consumer_Code", "count"))
    )
    return agg


def create_feeder_holdout_split(
    ami: pd.DataFrame,
    holdout_ratio: float = 0.20,
    seed: int = 2026,
) -> dict[str, Any]:
    """
    Stratified feeder holdout: bin feeders by bad comm rate, sample holdout
    proportionally from each bin.
    """
    rates = _feeder_comm_rates(ami)
    feeders = rates["Feeder_Code"].astype(str).tolist()
    n_holdout = max(1, int(round(len(feeders) * holdout_ratio)))

    # Stratify into 4 bins by bad_rate
    rates = rates.copy()
    rates["bin"] = pd.qcut(rates["bad_rate"], q=min(4, len(rates)), duplicates="drop")
    rng = np.random.default_rng(seed)

    test_feeders: list[str] = []
    for _, grp in rates.groupby("bin", observed=True):
        n_bin_test = max(1, int(round(len(grp) / len(feeders) * n_holdout)))
        chosen = rng.choice(grp["Feeder_Code"].astype(str).tolist(), size=min(n_bin_test, len(grp)), replace=False)
        test_feeders.extend(chosen)

    test_feeders = list(dict.fromkeys(test_feeders))[:n_holdout]
    train_feeders = [f for f in feeders if f not in test_feeders]

    ami_codes = ami.copy()
    ami_codes["Feeder_Code"] = ami_codes["Feeder_Code"].astype(str)
    ami_codes["Consumer_Code"] = ami_codes["Consumer_Code"].astype(str)

    train_mask = ami_codes["Feeder_Code"].isin(train_feeders)
    test_mask = ami_codes["Feeder_Code"].isin(test_feeders)

    manifest = {
        "mode": "split",
        "strategy": "feeder_holdout",
        "seed": seed,
        "holdout_ratio": holdout_ratio,
        "train_feeders": train_feeders,
        "test_feeders": test_feeders,
        "train_consumers": int(train_mask.sum()),
        "test_consumers": int(test_mask.sum()),
        "train_consumer_codes": ami_codes.loc[train_mask, "Consumer_Code"].tolist(),
        "test_consumer_codes": ami_codes.loc[test_mask, "Consumer_Code"].tolist(),
    }
    return manifest


def save_split_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_split_manifest(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def consumer_train_test_masks(
    consumer_codes: pd.Series | np.ndarray,
    manifest: dict[str, Any] | None,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_mask, test_mask) boolean arrays aligned to consumer_codes."""
    codes = np.asarray(consumer_codes).astype(str)
    n = len(codes)
    if mode == "full" or manifest is None:
        train = np.ones(n, dtype=bool)
        test = np.zeros(n, dtype=bool)
        return train, test

    train_set = set(str(c) for c in manifest.get("train_consumer_codes", []))
    test_set = set(str(c) for c in manifest.get("test_consumer_codes", []))
    train = np.array([c in train_set for c in codes])
    test = np.array([c in test_set for c in codes])
    return train, test
