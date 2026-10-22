"""Temporal DCU daily sequences for DTC-level encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


DCU_SEQ_COLS = [
    "Daily_Read_pct",
    "LP_Completeness_pct",
    "Packet_Loss_pct",
    "DCU_Up",
    "Backhaul_Up",
]
SEQ_LEN = 90


def build_dcu_sequences(dcu_daily: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """
    Build per-DTC daily sequences [SEQ_LEN, n_features].
    Returns dtc_sequences dict and dtc_code -> index mapping.
    """
    dcu = dcu_daily.copy()
    dcu["DTC_Code"] = dcu["DTC_Code"].astype(str)
    if "Date" in dcu.columns:
        dcu["Date"] = pd.to_datetime(dcu["Date"], errors="coerce")
        dcu = dcu.sort_values(["DTC_Code", "Date"])

    for c in DCU_SEQ_COLS:
        if c in dcu.columns:
            dcu[c] = pd.to_numeric(dcu[c], errors="coerce").fillna(0)

    sequences: dict[str, np.ndarray] = {}
    for dtc, grp in dcu.groupby("DTC_Code"):
        feat = grp[DCU_SEQ_COLS].values.astype(np.float32)
        if len(feat) >= SEQ_LEN:
            seq = feat[-SEQ_LEN:]
        else:
            pad = np.zeros((SEQ_LEN - len(feat), len(DCU_SEQ_COLS)), dtype=np.float32)
            seq = np.vstack([pad, feat]) if len(feat) > 0 else pad
        sequences[str(dtc)] = seq

    dtc_codes = sorted(sequences.keys())
    code_to_idx = {c: i for i, c in enumerate(dtc_codes)}
    return sequences, code_to_idx


class DCUTemporalEncoder(nn.Module):
    """GRU encoder for 90-day DCU daily status per DTC."""

    def __init__(self, input_dim: int = len(DCU_SEQ_COLS), hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.1)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, T, F]
        _, h = self.gru(seq)
        return self.proj(h[-1])


def encode_dcu_sequences(
    sequences: dict[str, np.ndarray],
    device: torch.device,
    hidden_dim: int = 64,
) -> dict[str, np.ndarray]:
    """Run temporal encoder on all DTC sequences (inference / feature prep)."""
    if not sequences:
        return {}
    encoder = DCUTemporalEncoder(out_dim=hidden_dim).to(device)
    encoder.eval()
    result: dict[str, np.ndarray] = {}
    codes = list(sequences.keys())
    batch = torch.tensor(np.stack([sequences[c] for c in codes]), dtype=torch.float32, device=device)
    with torch.no_grad():
        emb = encoder(batch).cpu().numpy()
    for i, code in enumerate(codes):
        result[code] = emb[i]
    return result


def augment_consumer_with_dtc_temporal(
    X: pd.DataFrame,
    ami: pd.DataFrame,
    dtc_embeddings: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Concatenate DTC temporal embedding to each consumer row."""
    ami = ami.copy()
    ami["Consumer_Code"] = ami["Consumer_Code"].astype(str)
    ami["DTC_Code"] = ami["DTC_Code"].astype(str)
    if not dtc_embeddings:
        return X
    dim = next(iter(dtc_embeddings.values())).shape[0]
    cols = [f"dtc_temporal_{i}" for i in range(dim)]
    rows = []
    for code in X.index.astype(str):
        dtc = ami.loc[ami["Consumer_Code"] == code, "DTC_Code"]
        dtc_code = str(dtc.iloc[0]) if len(dtc) else ""
        vec = dtc_embeddings.get(dtc_code, np.zeros(dim, dtype=np.float32))
        rows.append(vec)
    temporal_df = pd.DataFrame(rows, index=X.index, columns=cols)
    return pd.concat([X, temporal_df], axis=1)
