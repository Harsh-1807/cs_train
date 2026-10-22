"""Unsupervised root-cause clustering and 5-class signature mapping."""

from __future__ import annotations

from typing import Any

import hdbscan
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


ROOT_CAUSE_CLASSES = [
    "Tamper",
    "Network coverage",
    "Hardware",
    "Firmware",
    "Transient",
]


class RootCauseAnalyzer:
    """Anomaly detection + HDBSCAN + signature-based class assignment."""

    def __init__(
        self,
        contamination: float = 0.35,
        min_cluster_size: int = 15,
        min_samples: int = 5,
        random_state: int = 2026,
    ):
        self.contamination = contamination
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.iso_forest: IsolationForest | None = None
        self.clusterer: hdbscan.HDBSCAN | None = None
        self.cluster_class_map: dict[int, str] = {}

    def fit(
        self,
        embeddings: np.ndarray,
        recon_errors: np.ndarray,
        feature_df: pd.DataFrame,
        meta: pd.DataFrame,
    ) -> "RootCauseAnalyzer":
        combined = np.hstack([
            self.scaler.fit_transform(embeddings),
            recon_errors.reshape(-1, 1),
        ])
        self.iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        iso_scores = -self.iso_forest.fit(combined).score_samples(combined)

        # Anomaly = top contamination by combined score
        threshold = np.quantile(iso_scores, 1 - self.contamination)
        anomaly_mask = iso_scores >= threshold

        anom_emb = combined[anomaly_mask]
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
        )
        cluster_labels = np.full(len(combined), -1, dtype=int)
        if len(anom_emb) >= self.min_cluster_size:
            anom_clusters = self.clusterer.fit_predict(anom_emb)
            cluster_labels[anomaly_mask] = anom_clusters

        self._build_cluster_map(cluster_labels, feature_df, meta, anomaly_mask)
        self._combined = combined
        self._iso_scores = iso_scores
        self._cluster_labels = cluster_labels
        self._anomaly_mask = anomaly_mask
        return self

    def _build_cluster_map(
        self,
        cluster_labels: np.ndarray,
        feature_df: pd.DataFrame,
        meta: pd.DataFrame,
        anomaly_mask: np.ndarray,
    ) -> None:
        anom_indices = np.where(anomaly_mask)[0]
        anom_clusters = cluster_labels[anomaly_mask]
        for cid in set(anom_clusters) - {-1}:
            local = anom_clusters == cid
            global_idx = anom_indices[local]
            sub_idx = feature_df.index[global_idx]
            sub_feat = feature_df.loc[sub_idx]
            sub_meta = meta.loc[sub_idx] if not meta.empty else pd.DataFrame()
            self.cluster_class_map[int(cid)] = self._infer_class(sub_feat, sub_meta)

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
        if rssi_cols:
            rssi_mean = feat[rssi_cols].mean().mean()
            if rssi_mean < -85:
                scores["Network coverage"] += 2.5
        hop_cols = [c for c in feat.columns if "Hop" in c]
        if hop_cols:
            if feat[hop_cols].mean().mean() > 5:
                scores["Network coverage"] += 1.5
        loss_cols = [c for c in feat.columns if "Packet_Loss" in c or "loss" in c.lower()]
        if loss_cols:
            if feat[loss_cols].mean().mean() > 15:
                scores["Network coverage"] += 1.5

        if "event_firmware" in feat.columns:
            scores["Firmware"] += feat["event_firmware"].mean() * 3
        if "dcu_up_frac" in feat.columns:
            if feat["dcu_up_frac"].mean() < 0.5:
                scores["Hardware"] += 2

        if not meta.empty and "NIC_Status" in meta.columns:
            nic = meta["NIC_Status"].astype(str)
            scores["Hardware"] += nic.str.contains("Faulty|Removed", case=False).mean() * 2

        outage_cols = [c for c in feat.columns if "Outage" in c or "Power_Failure" in c]
        if outage_cols:
            scores["Transient"] += feat[outage_cols].mean().mean() / 100

        if not meta.empty and "Comm_Status" in meta.columns:
            comm = meta["Comm_Status"].astype(str)
            scores["Transient"] += comm.str.contains("Intermittent", case=False).mean() * 2

        return max(scores, key=scores.get)

    def predict(
        self,
        embeddings: np.ndarray,
        recon_errors: np.ndarray,
        feature_df: pd.DataFrame,
        meta: pd.DataFrame,
    ) -> pd.DataFrame:
        combined = np.hstack([
            self.scaler.transform(embeddings),
            recon_errors.reshape(-1, 1),
        ])
        iso_scores = -self.iso_forest.score_samples(combined)
        threshold = np.quantile(self._iso_scores, 1 - self.contamination)
        anomaly_mask = iso_scores >= threshold

        root_cause = np.array(["Normal"] * len(combined), dtype=object)
        anom_indices = np.where(anomaly_mask)[0]

        for i in anom_indices:
            root_cause[i] = self._infer_class(
                feature_df.iloc[[i]],
                meta.iloc[[i]] if not meta.empty else pd.DataFrame(),
            )

        if self.clusterer is not None and hasattr(self.clusterer, "labels_") and len(anom_indices) > 0:
            anom_emb = combined[anomaly_mask]
            test_labels, _ = hdbscan.approximate_predict(self.clusterer, anom_emb)
            for j, i in enumerate(anom_indices):
                cid = int(test_labels[j])
                if cid != -1 and cid in self.cluster_class_map:
                    root_cause[i] = self.cluster_class_map[cid]

        # NIC removal sub-classification
        nic_rc = np.array([""] * len(combined), dtype=object)
        nic_mask = self._nic_removal_mask(feature_df, meta)
        nic_rc[nic_mask] = root_cause[nic_mask]

        return pd.DataFrame({
            "anomaly_score": iso_scores,
            "is_anomaly": anomaly_mask,
            "root_cause_class": root_cause,
            "nic_removal_root_cause": nic_rc,
        }, index=feature_df.index)

    @staticmethod
    def _nic_removal_mask(feat: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
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
        """Summarize missed reads and RNA bills by root cause."""
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
        for cause in ROOT_CAUSE_CLASSES + ["Normal", "Unknown"]:
            sub = pred[pred["root_cause_class"] == cause]
            rows.append({
                "root_cause_class": cause,
                "consumer_count": len(sub),
                "total_missed_reads_q1": sub["missed_reads_q1"].sum(),
                "rna_bill_count": sub["has_rna_bill"].sum(),
            })
        return pd.DataFrame(rows)
