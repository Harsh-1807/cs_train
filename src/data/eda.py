"""Exploratory data analysis plots before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from .features import build_consumer_feature_matrix
from .loader import load_ami_data


def run_eda(input_path: str, output_dir: str, config_path: str | None = None) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = load_ami_data(input_path)
    ami = data["AMI_Comm_Health"]
    events = data["Consumer_Event_Log"]
    readings = data["Consumer_Monthly_Readings"]

    sns.set_theme(style="whitegrid")

    # Histograms — key comm metrics
    hist_cols = [
        "Avg_RSSI_dBm", "Avg_Read_Success_pct", "Avg_Packet_Loss_pct",
        "Missed_Daily_Reads_Q1", "Days_Since_Last_Comm", "Hop_Count",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, hist_cols):
        if col in ami.columns:
            vals = pd.to_numeric(ami[col], errors="coerce").dropna()
            ax.hist(vals, bins=50, color="steelblue", edgecolor="white")
            ax.set_title(col)
    plt.tight_layout()
    plt.savefig(out / "histograms_comm_metrics.png", dpi=120)
    plt.close()

    # Comm status distribution
    if "Comm_Status" in ami.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ami["Comm_Status"].value_counts().plot(kind="bar", ax=ax, color="coral")
        ax.set_title("Comm Status Distribution")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(out / "comm_status_bar.png", dpi=120)
        plt.close()

    # NIC status
    if "NIC_Status" in ami.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ami["NIC_Status"].value_counts().plot(kind="bar", ax=ax, color="seagreen")
        ax.set_title("NIC Status Distribution")
        plt.tight_layout()
        plt.savefig(out / "nic_status_bar.png", dpi=120)
        plt.close()

    # Billing basis
    if "Billing_Basis" in readings.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        readings["Billing_Basis"].value_counts().plot(kind="bar", ax=ax, color="mediumpurple")
        ax.set_title("Billing Basis (Monthly Readings)")
        plt.tight_layout()
        plt.savefig(out / "billing_basis_bar.png", dpi=120)
        plt.close()

    # Event class frequency
    if "Event_Class" in events.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        events["Event_Class"].value_counts().sort_index().plot(kind="bar", ax=ax)
        ax.set_title("Event Class Frequency")
        plt.tight_layout()
        plt.savefig(out / "event_class_frequency.png", dpi=120)
        plt.close()

    # Correlation heatmap (numeric AMI features)
    X, _, _ = build_consumer_feature_matrix(data)
    num_X = X.select_dtypes(include=[np.number])
    if num_X.shape[1] > 30:
        # top variance columns
        num_X = num_X[num_X.var().sort_values(ascending=False).head(30).index]
    corr = num_X.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, xticklabels=True, yticklabels=True)
    ax.set_title("Feature Correlation Heatmap (top 30)")
    plt.tight_layout()
    plt.savefig(out / "correlation_heatmap.png", dpi=120)
    plt.close()

    # Summary JSON
    summary = {
        "n_consumers": len(ami),
        "n_events": len(events),
        "n_monthly_readings": len(readings),
        "comm_status": ami["Comm_Status"].value_counts().to_dict() if "Comm_Status" in ami.columns else {},
        "nic_status": ami["NIC_Status"].value_counts().to_dict() if "NIC_Status" in ami.columns else {},
    }
    with open(out / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="AMI EDA")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/eda")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_eda(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
