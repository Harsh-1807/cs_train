"""Run inference on new AMI xlsx — auto-detects v1 or v2 pipeline."""

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
from src.data.temporal_features import (
    DCUTemporalEncoder,
    augment_consumer_with_dtc_temporal,
    build_dcu_sequences,
)
from src.models.ami_gnn_v2 import AMIGnnV2, consumer_embeddings_v2, per_node_reconstruction_error_v2
from src.models.graph_mae import GraphMAE, per_node_reconstruction_error
from src.models.hgt_encoder import HGTEncoder, consumer_embeddings
from src.utils import ensure_dir, get_device, load_config


def _detect_version(artifacts: Path) -> str:
    meta_path = artifacts / "feature_meta.pkl"
    if meta_path.exists():
        meta = joblib.load(meta_path)
        if meta.get("pipeline_version") == "v2":
            return "v2"
    tm = artifacts / "training_meta.json"
    if tm.exists():
        with open(tm) as f:
            if json.load(f).get("pipeline_version") == "v2":
                return "v2"
    return "v1"


def predict_v1(args, config, artifacts, device, out) -> None:
    data = load_ami_data(args.input)
    X, meta, _ = build_consumer_feature_matrix(data, config)
    X_jan_feb = build_jan_feb_features(X)
    graph, _ = build_hetero_graph(data, X, train_mask=np.ones(len(X), dtype=bool))

    mae_cfg = config["training"]["graph_mae"]
    metadata = HGTEncoder.metadata_from_hetero(graph)
    in_channels = HGTEncoder.in_channels_from_hetero(graph)
    model = GraphMAE(
        metadata=metadata, in_channels=in_channels,
        hidden_channels=mae_cfg["hidden_dim"],
        num_layers=mae_cfg["num_hgt_layers"], num_heads=mae_cfg["num_heads"],
    )
    model.load_state_dict(torch.load(artifacts / "graph_mae.pt", map_location=device, weights_only=True))
    model = model.to(device).eval()
    rc_analyzer = joblib.load(artifacts / "root_cause.pkl")
    silent_model = joblib.load(artifacts / "silent_predictor.pkl")
    embeddings = consumer_embeddings(model.encoder, graph, device).numpy()
    recon_errors = per_node_reconstruction_error(model, graph, device).numpy()
    _write_outputs(rc_analyzer, silent_model, embeddings, recon_errors, X, X_jan_feb, meta, data, out)


def predict_v2(args, config, artifacts, device, out) -> None:
    data = load_ami_data(args.input)
    ami = data["AMI_Comm_Health"]
    sequences, _ = build_dcu_sequences(data["DCU_Daily_Status"])
    tcfg = config["training"]["temporal"]
    dcu_encoder = DCUTemporalEncoder(out_dim=tcfg["hidden_dim"]).to(device)
    dcu_encoder.load_state_dict(
        torch.load(artifacts / "dcu_temporal_encoder.pt", map_location=device, weights_only=True)
    )
    dcu_encoder.eval()
    codes = list(sequences.keys())
    dtc_temporal = {}
    if codes:
        batch = torch.tensor(np.stack([sequences[c] for c in codes]), dtype=torch.float32, device=device)
        with torch.no_grad():
            for i, c in enumerate(codes):
                dtc_temporal[c] = dcu_encoder(batch[i : i + 1]).cpu().numpy().flatten()

    X, meta, _ = build_consumer_feature_matrix(data, config)
    X = augment_consumer_with_dtc_temporal(X, ami, dtc_temporal)
    X_jan_feb = build_jan_feb_features(X)
    graph, _ = build_hetero_graph(
        data, X, train_mask=np.ones(len(X), dtype=bool), dtc_temporal=dtc_temporal
    )

    v2cfg = config["training"]["ami_gnn_v2"]
    metadata = HGTEncoder.metadata_from_hetero(graph)
    in_channels = HGTEncoder.in_channels_from_hetero(graph)
    model = AMIGnnV2(
        metadata=metadata, in_channels=in_channels,
        hidden_channels=v2cfg["hidden_dim"],
        num_layers=v2cfg["num_hgt_layers"], num_heads=v2cfg["num_heads"],
        n_clusters=v2cfg["n_clusters"],
    )
    model.load_state_dict(torch.load(artifacts / "ami_gnn_v2.pt", map_location=device, weights_only=True))
    model = model.to(device).eval()
    rc_analyzer = joblib.load(artifacts / "root_cause_v2.pkl")
    silent_model = joblib.load(artifacts / "silent_predictor.pkl")
    embeddings = consumer_embeddings_v2(model, graph, device).numpy()
    recon_errors = per_node_reconstruction_error_v2(model, graph, device).numpy()
    _write_outputs(rc_analyzer, silent_model, embeddings, recon_errors, X, X_jan_feb, meta, data, out)


def _write_outputs(rc_analyzer, silent_model, embeddings, recon_errors, X, X_jan_feb, meta, data, out) -> None:
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
    out_df.to_csv(out / "consumer_predictions.csv", index=False)
    attr = rc_analyzer.attribution_summary(rc_pred, meta, data["Consumer_Monthly_Readings"])
    attr.to_csv(out / "attribution_summary.csv", index=False)
    print(f"Predictions: {out / 'consumer_predictions.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict AMI root cause and silent risk")
    p.add_argument("--input", required=True)
    p.add_argument("--artifacts", default="artifacts/")
    p.add_argument("--config", default=None, help="Auto: config.v2.yaml for v2, config.yaml for v1")
    p.add_argument("--output", default="outputs/predictions")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = Path(args.artifacts)
    version = _detect_version(artifacts)
    config_path = args.config or ("config.v2.yaml" if version == "v2" else "config.yaml")
    config = load_config(config_path)
    device = get_device(config)
    out = ensure_dir(args.output)
    print(f"Pipeline: {version}")
    if version == "v2":
        predict_v2(args, config, artifacts, device, out)
    else:
        predict_v1(args, config, artifacts, device, out)


if __name__ == "__main__":
    main()
