"""Train AMI GNN v2 — Graph MAE + contrastive + DEC + temporal DCU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from tqdm import tqdm

from src.data.features import (
    build_consumer_feature_matrix,
    build_jan_feb_features,
    build_silent_targets,
)
from src.data.graph_builder import build_hetero_graph
from src.data.loader import load_ami_data
from src.data.split_manager import create_feeder_holdout_split, save_split_manifest
from src.data.temporal_features import (
    DCUTemporalEncoder,
    augment_consumer_with_dtc_temporal,
    build_dcu_sequences,
    encode_dcu_sequences,
)
from src.explain.shap_analysis import run_shap_analysis
from src.models.ami_gnn_v2 import (
    AMIGnnV2,
    consumer_embeddings_v2,
    per_node_reconstruction_error_v2,
    train_ami_gnn_v2,
)
from src.models.dec_clustering import RootCauseAnalyzerV2
from src.models.hgt_encoder import HGTEncoder
from src.models.silent_predictor import SilentMeterPredictor
from src.utils import ensure_dir, get_device, load_config


def pretrain_dcu_encoder(
    sequences: dict[str, np.ndarray],
    device: torch.device,
    epochs: int = 15,
    hidden_dim: int = 64,
) -> DCUTemporalEncoder:
    """Self-supervised next-step prediction on DCU daily sequences."""
    if not sequences:
        enc = DCUTemporalEncoder(out_dim=hidden_dim)
        return enc.to(device)

    encoder = DCUTemporalEncoder(out_dim=hidden_dim).to(device)
    decoder = nn.Linear(hidden_dim, sequences[next(iter(sequences))].shape[1]).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    codes = list(sequences.keys())
    data = torch.tensor(np.stack([sequences[c] for c in codes]), dtype=torch.float32, device=device)

    for _ in tqdm(range(epochs), desc="DCU temporal pretrain"):
        encoder.train()
        h = encoder(data)
        # predict last day features from sequence encoding
        target = data[:, -1, :]
        pred = decoder(h)
        loss = nn.functional.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return encoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AMI GNN v2 (SOTA)")
    p.add_argument("--input", required=True)
    p.add_argument("--config", default="config.v2.yaml")
    p.add_argument("--artifacts", default="artifacts/")
    p.add_argument("--mode", choices=["full", "split"], default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    mode = args.mode or config["training"]["mode"]
    artifacts = ensure_dir(args.artifacts)
    device = get_device(config)

    print(f"AMI GNN v2 | Device: {device} | Mode: {mode}")

    data = load_ami_data(args.input)
    ami = data["AMI_Comm_Health"]

    # Temporal DCU encoding (90-day sequences per DTC)
    tcfg = config["training"]["temporal"]
    sequences, _ = build_dcu_sequences(data["DCU_Daily_Status"])
    dcu_encoder = pretrain_dcu_encoder(
        sequences, device,
        epochs=tcfg["pretrain_epochs"],
        hidden_dim=tcfg["hidden_dim"],
    )
    dcu_encoder.eval()
    codes = list(sequences.keys())
    if codes:
        batch = torch.tensor(
            np.stack([sequences[c] for c in codes]), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            dtc_temporal = {
                codes[i]: dcu_encoder(batch[i : i + 1]).cpu().numpy().flatten()
                for i in range(len(codes))
            }
    else:
        dtc_temporal = {}

    X, meta, feature_names = build_consumer_feature_matrix(data, config)
    X = augment_consumer_with_dtc_temporal(X, ami, dtc_temporal)
    feature_names = X.columns.tolist()
    X_jan_feb = build_jan_feb_features(X)
    silent_y = build_silent_targets(ami, data["Consumer_Monthly_Readings"])

    manifest = None
    if mode == "split":
        manifest = create_feeder_holdout_split(
            ami,
            holdout_ratio=config["training"]["split"]["holdout_ratio"],
            seed=config["training"]["split"]["seed"],
        )
        save_split_manifest(manifest, artifacts / "split_manifest.json")
        print(f"Split: train={manifest['train_consumers']} test={manifest['test_consumers']}")
    else:
        save_split_manifest({"mode": "full"}, artifacts / "split_manifest.json")

    train_codes = set(manifest["train_consumer_codes"]) if manifest and mode == "split" else set(X.index)
    train_mask = np.array([str(c) in train_codes for c in X.index.astype(str)])
    test_mask = ~train_mask if mode == "split" else np.zeros(len(X), dtype=bool)

    graph, graph_meta = build_hetero_graph(
        data, X, train_mask=train_mask, test_mask=test_mask, dtc_temporal=dtc_temporal
    )

    metadata = HGTEncoder.metadata_from_hetero(graph)
    in_channels = HGTEncoder.in_channels_from_hetero(graph)
    v2cfg = config["training"]["ami_gnn_v2"]

    model = AMIGnnV2(
        metadata=metadata,
        in_channels=in_channels,
        hidden_channels=v2cfg["hidden_dim"],
        num_layers=v2cfg["num_hgt_layers"],
        num_heads=v2cfg["num_heads"],
        n_clusters=v2cfg["n_clusters"],
    )

    train_consumer_mask = torch.tensor(train_mask, dtype=torch.bool) if mode == "split" else None
    train_metrics = train_ami_gnn_v2(
        model, graph, device,
        epochs=v2cfg["epochs"],
        lr=v2cfg["lr"],
        mask_ratio=v2cfg["mask_ratio"],
        contrastive_weight=v2cfg["contrastive_weight"],
        dec_weight=v2cfg["dec_weight"],
        dec_warmup=v2cfg["dec_warmup_epochs"],
        edge_drop_rate=v2cfg["edge_drop_rate"],
        train_mask=train_consumer_mask,
    )
    print(f"v2 final loss: {train_metrics['final_loss']:.6f}")

    embeddings = consumer_embeddings_v2(model, graph, device).numpy()
    recon_errors = per_node_reconstruction_error_v2(model, graph, device).numpy()

    rc_cfg = config["training"]["anomaly"]
    rc_analyzer = RootCauseAnalyzerV2(
        n_clusters=v2cfg["n_clusters"],
        contamination=rc_cfg["contamination"],
    )
    fit_mask = train_mask if mode == "split" else np.ones(len(X), dtype=bool)
    rc_analyzer.fit(
        embeddings[fit_mask], recon_errors[fit_mask],
        X.iloc[fit_mask], meta.iloc[fit_mask], model.dec,
    )
    rc_pred = rc_analyzer.predict(embeddings, recon_errors, X, meta)

    sp_cfg = config["training"]["silent_predictor"]
    silent_model = SilentMeterPredictor(
        n_estimators=sp_cfg["n_estimators"],
        learning_rate=sp_cfg["learning_rate"],
        max_depth=sp_cfg["max_depth"],
    )
    silent_metrics = silent_model.fit(
        X_jan_feb, silent_y, embeddings,
        train_mask=train_mask if mode == "split" else None,
    )
    silent_scores = silent_model.predict_proba(X_jan_feb, embeddings)

    unsup_metrics: dict = {"ami_gnn_v2": train_metrics}
    try:
        emb_train = embeddings[fit_mask]
        if len(emb_train) > 500:
            idx = np.random.default_rng(42).choice(len(emb_train), 500, replace=False)
            sample = emb_train[idx]
            labels = rc_pred.iloc[fit_mask]["cluster_id"].values[idx]
            unsup_metrics["silhouette"] = float(silhouette_score(sample, labels, metric="cosine"))
            unsup_metrics["davies_bouldin"] = float(davies_bouldin_score(sample, labels))
            unsup_metrics["calinski_harabasz"] = float(calinski_harabasz_score(sample, labels))
    except Exception as e:
        unsup_metrics["embedding_metrics_error"] = str(e)
    unsup_metrics["silent_predictor"] = silent_metrics

    attr = rc_analyzer.attribution_summary(rc_pred, meta, data["Consumer_Monthly_Readings"])
    report_dir = ensure_dir(config["paths"]["outputs"]) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    attr.to_csv(report_dir / "attribution_summary.csv", index=False)

    shap_paths = run_shap_analysis(
        silent_model, X_jan_feb.iloc[fit_mask], embeddings[fit_mask], output_dir=report_dir
    )

    torch.save(model.state_dict(), artifacts / "ami_gnn_v2.pt")
    torch.save(dcu_encoder.state_dict(), artifacts / "dcu_temporal_encoder.pt")
    joblib.dump(rc_analyzer, artifacts / "root_cause_v2.pkl")
    joblib.dump(silent_model, artifacts / "silent_predictor.pkl")
    joblib.dump({
        "feature_names": feature_names,
        "jan_feb_cols": X_jan_feb.columns.tolist(),
        "pipeline_version": "v2",
    }, artifacts / "feature_meta.pkl")

    with open(artifacts / "graph_meta.json", "w") as f:
        json.dump({k: v for k, v in graph_meta.items() if k != "code_to_idx"}, f, indent=2, default=str)

    training_meta = {
        "pipeline_version": "v2",
        "mode": mode,
        "device": str(device),
        "n_consumers": len(X),
        "metrics": unsup_metrics,
        "shap": shap_paths,
        "architecture": "GraphMAE+GRACE+DEC+TemporalDCU+HGT-3L",
    }
    with open(artifacts / "training_meta.json", "w") as f:
        json.dump(training_meta, f, indent=2, default=str)

    pred_out = ensure_dir(config["paths"]["outputs"]) / "predictions"
    out_df = rc_pred.copy()
    out_df.index.name = "consumer_code"
    out_df["silent_risk_score"] = silent_scores
    out_df["silent_risk_rank"] = (-out_df["silent_risk_score"]).rank(method="min").astype(int)
    out_df.reset_index(inplace=True)
    out_df.to_csv(pred_out / "consumer_predictions.csv", index=False)

    print(f"v2 artifacts saved to {artifacts}")


if __name__ == "__main__":
    main()
