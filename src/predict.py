"""Run inference on new AMI xlsx (same schema)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from src.data.features import build_consumer_feature_matrix, build_jan_feb_features
from src.data.graph_builder import build_hetero_graph
from src.data.loader import load_ami_data
from src.models.graph_mae import GraphMAE, per_node_reconstruction_error
from src.models.hgt_encoder import HGTEncoder, consumer_embeddings
from src.utils import ensure_dir, get_device, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict AMI root cause and silent risk")
    p.add_argument("--input", required=True)
    p.add_argument("--artifacts", default="artifacts/")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output", default="outputs/predictions")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    artifacts = Path(args.artifacts)
    device = get_device(config)
    out = ensure_dir(args.output)

    data = load_ami_data(args.input)
    X, meta, _ = build_consumer_feature_matrix(data, config)
    X_jan_feb = build_jan_feb_features(X)

    train_mask = np.ones(len(X), dtype=bool)
    graph, graph_meta = build_hetero_graph(data, X, train_mask=train_mask)

    with open(artifacts / "graph_meta.json") as f:
        saved_graph_meta = json.load(f)

    mae_cfg = config["training"]["graph_mae"]
    metadata = HGTEncoder.metadata_from_hetero(graph)
    in_channels = HGTEncoder.in_channels_from_hetero(graph)

    model = GraphMAE(
        metadata=metadata,
        in_channels=in_channels,
        hidden_channels=mae_cfg["hidden_dim"],
        num_layers=mae_cfg["num_hgt_layers"],
        num_heads=mae_cfg["num_heads"],
    )
    model.load_state_dict(torch.load(artifacts / "graph_mae.pt", map_location=device, weights_only=True))
    model = model.to(device).eval()

    rc_analyzer = joblib.load(artifacts / "root_cause.pkl")
    silent_model = joblib.load(artifacts / "silent_predictor.pkl")

    embeddings = consumer_embeddings(model.encoder, graph, device).numpy()
    recon_errors = per_node_reconstruction_error(model, graph, device).numpy()

    rc_pred = rc_analyzer.predict(embeddings, recon_errors, X, meta)
    silent_scores = silent_model.predict_proba(X_jan_feb, embeddings)

    out_df = rc_pred.copy()
    out_df.index.name = "consumer_code"
    out_df["silent_risk_score"] = silent_scores
    out_df["silent_risk_rank"] = (-out_df["silent_risk_score"]).rank(method="min").astype(int)
    if "Missed_Daily_Reads_Q1" in meta.columns:
        out_df["missed_reads_q1"] = meta["Missed_Daily_Reads_Q1"].values
    out_df["estimated_bill_attributed_cause"] = out_df["root_cause_class"]
    out_df.reset_index(inplace=True)
    out_df.rename(columns={"index": "consumer_code"}, inplace=True)

    out_df.to_csv(out / "consumer_predictions.csv", index=False)

    attr = rc_analyzer.attribution_summary(rc_pred, meta, data["Consumer_Monthly_Readings"])
    attr.to_csv(out / "attribution_summary.csv", index=False)

    print(f"Predictions: {out / 'consumer_predictions.csv'}")


if __name__ == "__main__":
    main()
