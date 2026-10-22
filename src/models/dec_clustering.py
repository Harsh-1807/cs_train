"""Deep Embedded Clustering (DEC) for root-cause discovery."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

ROOT_CAUSE_CLASSES = [
    "Tamper",
    "Network coverage",
    "Hardware",
    "Firmware",
    "Transient",
]


class DECClustering(nn.Module):
    """DEC with learnable soft cluster assignment (Student-t kernel)."""

    def __init__(self, n_clusters: int = 5, embedding_dim: int = 128, alpha: float = 1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.centroids = nn.Parameter(torch.randn(n_clusters, embedding_dim))

    def soft_assign(self, z: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(z, self.centroids) ** 2
        q = (1.0 + dist / self.alpha) ** (-(self.alpha + 1) / 2)
        return q / q.sum(dim=1, keepdim=True)

    def target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        weight = q ** 2 / q.sum(dim=0, keepdim=True)
        return weight / weight.sum(dim=1, keepdim=True)

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        q = self.soft_assign(z)
        p = self.target_distribution(q).detach()
        return F.kl_div(q.log(), p, reduction="batchmean")

    def init_centroids(self, z: np.ndarray) -> None:
        km = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=2026)
        km.fit(z)
        self.centroids.data = torch.tensor(km.cluster_centers_, dtype=torch.float32)


class RootCauseAnalyzerV2:
    """DEC-based root cause with signature mapping on cluster centroids."""

    def __init__(self, n_clusters: int = 5, contamination: float = 0.35):
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.dec: DECClustering | None = None
        self.cluster_class_map: dict[int, str] = {}
        self._combined_scores: np.ndarray | None = None

    def fit(
        self,
        embeddings: np.ndarray,
        recon_errors: np.ndarray,
        feature_df: pd.DataFrame,
        meta: pd.DataFrame,
        dec_module: DECClustering,
    ) -> "RootCauseAnalyzerV2":
        self.dec = dec_module
        combined = np.hstack([embeddings, recon_errors.reshape(-1, 1)])
        self._combined_scores = combined.mean(axis=1)

        q = dec_module.soft_assign(torch.tensor(embeddings, dtype=torch.float32)).detach().numpy()
        labels = q.argmax(axis=1)

        for cid in range(self.n_clusters):
            mask = labels == cid
            if mask.sum() == 0:
                self.cluster_class_map[cid] = ROOT_CAUSE_CLASSES[cid % len(ROOT_CAUSE_CLASSES)]
                continue
            sub_feat = feature_df.iloc[mask]
            sub_meta = meta.iloc[mask] if not meta.empty else pd.DataFrame()
            self.cluster_class_map[cid] = self._infer_class(sub_feat, sub_meta)
        return self

    @staticmethod
    def _infer_class(feat: pd.DataFrame, meta: pd.DataFrame) -> str:
        scores = {c: 0.0 for c in ROOT_CAUSE_CLASSES}
        if "event_tamper_before_nic" in feat.columns:
            scores["Tamper"] += feat["event_tamper_before_nic"].mean() * 3
        if "event_tamper" in feat.columns:
            scores["Tamper"] += feat["event_tamper"].mean() * 2
        if "event_nic_removal" in feat.columns:
            scores["Hardware"] += feat["event_nic_removal"].mean() * 1.5
        rssi_cols = [c for c in feat.columns if "RSSI" in c or "rssi" in c.lower()]
        if rssi_cols and feat[rssi_cols].mean().mean() < -85:
            scores["Network coverage"] += 2.5
        loss_cols = [c for c in feat.columns if "Packet_Loss" in c or "loss" in c.lower()]
        if loss_cols and feat[loss_cols].mean().mean() > 15:
            scores["Network coverage"] += 2
        if "event_firmware" in feat.columns:
            scores["Firmware"] += feat["event_firmware"].mean() * 3
        if "dcu_up_frac" in feat.columns and feat["dcu_up_frac"].mean() < 0.5:
            scores["Hardware"] += 2
        if not meta.empty and "NIC_Status" in meta.columns:
            scores["Hardware"] += meta["NIC_Status"].astype(str).str.contains(
                "Faulty|Removed", case=False
            ).mean() * 2
        outage_cols = [c for c in feat.columns if "Outage" in c or "Power_Failure" in c]
        if outage_cols:
            scores["Transient"] += feat[outage_cols].mean().mean() / 100
        if not meta.empty and "Comm_Status" in meta.columns:
            scores["Transient"] += meta["Comm_Status"].astype(str).str.contains(
                "Intermittent", case=False
            ).mean() * 2
        return max(scores, key=scores.get)

    def predict(
        self,
        embeddings: np.ndarray,
        recon_errors: np.ndarray,
        feature_df: pd.DataFrame,
        meta: pd.DataFrame,
    ) -> pd.DataFrame:
        q = self.dec.soft_assign(torch.tensor(embeddings, dtype=torch.float32)).detach().numpy()
        labels = q.argmax(axis=1)
        max_q = q.max(axis=1)

        anomaly_score = recon_errors + (1 - max_q)
        threshold = np.quantile(anomaly_score, 1 - self.contamination)

        root_cause = np.array(["Normal"] * len(embeddings), dtype=object)
        for i, cid in enumerate(labels):
            if anomaly_score[i] >= threshold:
                root_cause[i] = self.cluster_class_map.get(int(cid), ROOT_CAUSE_CLASSES[int(cid) % 5])

        nic_rc = np.array([""] * len(embeddings), dtype=object)
        nic_mask = self._nic_mask(feature_df, meta)
        nic_rc[nic_mask] = root_cause[nic_mask]

        return pd.DataFrame({
            "anomaly_score": anomaly_score,
            "is_anomaly": anomaly_score >= threshold,
            "cluster_id": labels,
            "cluster_confidence": max_q,
            "root_cause_class": root_cause,
            "nic_removal_root_cause": nic_rc,
        }, index=feature_df.index)

    @staticmethod
    def _nic_mask(feat: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        mask = np.zeros(len(feat), dtype=bool)
        if "event_nic_removal" in feat.columns:
            mask |= feat["event_nic_removal"].values.astype(bool)
        if not meta.empty and "NIC_Status" in meta.columns:
            mask |= meta["NIC_Status"].astype(str).str.contains("Removed", case=False).values
        return mask

    def attribution_summary(
        self,
        predictions: pd.DataFrame,
        meta: pd.DataFrame,
        readings: pd.DataFrame,
    ) -> pd.DataFrame:
        pred = predictions.copy()
        if "Missed_Daily_Reads_Q1" in meta.columns:
            pred["missed_reads_q1"] = pd.to_numeric(
                meta["Missed_Daily_Reads_Q1"], errors="coerce"
            ).fillna(0).values
        else:
            pred["missed_reads_q1"] = 0
        r = readings.copy()
        r["Consumer_Code"] = r["Consumer_Code"].astype(str)
        rna = r.groupby("Consumer_Code")["Billing_Basis"].apply(
            lambda s: int(s.astype(str).str.contains("RNA|Assessed", case=False, na=False).any())
        )
        pred["has_rna_bill"] = pred.index.map(lambda c: rna.get(str(c), 0))
        rows = []
        for cause in ROOT_CAUSE_CLASSES + ["Normal"]:
            sub = pred[pred["root_cause_class"] == cause]
            rows.append({
                "root_cause_class": cause,
                "consumer_count": len(sub),
                "total_missed_reads_q1": sub["missed_reads_q1"].sum(),
                "rna_bill_count": sub["has_rna_bill"].sum(),
            })
        return pd.DataFrame(rows)
