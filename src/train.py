"""Train AMI unsupervised GNN pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.data.features import (
    build_consumer_feature_matrix,
    build_jan_feb_features,
    build_silent_targets,
)
from src.data.graph_builder import build_hetero_graph
from src.data.loader import load_ami_data
from src.data.split_manager import create_feeder_holdout_split, save_split_manifest
from src.explain.shap_analysis import run_shap_analysis
from src.models.graph_mae import GraphMAE, per_node_reconstruction_error, train_graph_mae
from src.models.hgt_encoder import HGTEncoder, consumer_embeddings
from src.models.root_cause import RootCauseAnalyzer
from src.models.silent_predictor import SilentMeterPredictor
from src.utils import ensure_dir, get_device, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AMI GNN pipeline")
    p.add_argument("--input", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--artifacts", default="artifacts/")
    p.add_argument("--mode", choices=["full", "split"], default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    mode = args.mode or config["training"]["mode"]
    artifacts = ensure_dir(args.artifacts)
    device = get_device(config)

    print(f"Device: {device} | Mode: {mode}")

    data = load_ami_data(args.input)
    X, meta, feature_names = build_consumer_feature_matrix(data, config)
    X_jan_feb = build_jan_feb_features(X)
    silent_y = build_silent_targets(data["AMI_Comm_Health"], data["Consumer_Monthly_Readings"])

    manifest = None
    if mode == "split":
        manifest = create_feeder_holdout_split(
            data["AMI_Comm_Health"],
            holdout_ratio=config["training"]["split"]["holdout_ratio"],
            seed=config["training"]["split"]["seed"],
        )
        manifest_path = config["training"]["split"]["manifest_path"]
        save_split_manifest(manifest, artifacts / Path(manifest_path).name)
        print(f"Split: train={manifest['train_consumers']} test={manifest['test_consumers']} feeders")
    else:
        manifest_path = None
        save_split_manifest({"mode": "full"}, artifacts / "split_manifest.json")

    train_codes = set(manifest["train_consumer_codes"]) if manifest and mode == "split" else set(X.index)
    train_mask = np.array([c in train_codes for c in X.index.astype(str)])
    test_mask = ~train_mask if mode == "split" else np.zeros(len(X), dtype=bool)

    graph, graph_meta = build_hetero_graph(data, X, train_mask=train_mask, test_mask=test_mask)

    metadata = HGTEncoder.metadata_from_hetero(graph)
    in_channels = HGTEncoder.in_channels_from_hetero(graph)

    mae_cfg = config["training"]["graph_mae"]
    model = GraphMAE(
        metadata=metadata,
        in_channels=in_channels,
        hidden_channels=mae_cfg["hidden_dim"],
        num_layers=mae_cfg["num_hgt_layers"],
        num_heads=mae_cfg["num_heads"],
    )

    train_consumer_mask = torch.tensor(train_mask, dtype=torch.bool) if mode == "split" else None
    mae_metrics = train_graph_mae(
        model,
        graph,
        device,
        epochs=mae_cfg["epochs"],
        lr=mae_cfg["lr"],
        mask_ratio=mae_cfg["mask_ratio"],
        train_mask=train_consumer_mask,
    )
    print(f"Graph MAE final loss: {mae_metrics['final_loss']:.6f}")

    embeddings = consumer_embeddings(model.encoder, graph, device).numpy()
    recon_errors = per_node_reconstruction_error(model, graph, device).numpy()

    # Root cause
    rc_cfg = config["training"]["anomaly"]
    rc_analyzer = RootCauseAnalyzer(
        contamination=rc_cfg["contamination"],
        min_cluster_size=rc_cfg["hdbscan_min_cluster_size"],
        min_samples=rc_cfg["hdbscan_min_samples"],
    )
    fit_mask = train_mask if mode == "split" else np.ones(len(X), dtype=bool)
    rc_analyzer.fit(
        embeddings[fit_mask],
        recon_errors[fit_mask],
        X.iloc[fit_mask],
        meta.iloc[fit_mask],
    )
    rc_pred = rc_analyzer.predict(embeddings, recon_errors, X, meta)

    # Silent meter
    sp_cfg = config["training"]["silent_predictor"]
    silent_model = SilentMeterPredictor(
        n_estimators=sp_cfg["n_estimators"],
        learning_rate=sp_cfg["learning_rate"],
        max_depth=sp_cfg["max_depth"],
    )
    silent_metrics = silent_model.fit(
        X_jan_feb,
        silent_y,
        embeddings,
        train_mask=train_mask if mode == "split" else None,
    )
    silent_scores = silent_model.predict_proba(X_jan_feb, embeddings)

    # Unsupervised embedding metrics
    unsup_metrics: dict = {"graph_mae": mae_metrics}
    try:
        emb_train = embeddings[fit_mask]
        if len(emb_train) > 500:
            sample_idx = np.random.default_rng(42).choice(len(emb_train), 500, replace=False)
            emb_sample = emb_train[sample_idx]
            unsup_metrics["silhouette"] = float(silhouette_score(emb_sample, metric="cosine"))
            unsup_metrics["davies_bouldin"] = float(davies_bouldin_score(emb_sample))
            unsup_metrics["calinski_harabasz"] = float(calinski_harabasz_score(emb_sample, np.zeros(len(emb_sample))))
    except Exception as e:
        unsup_metrics["embedding_metrics_error"] = str(e)

    unsup_metrics["silent_predictor"] = silent_metrics

    # Attribution
    attr = rc_analyzer.attribution_summary(rc_pred, meta, data["Consumer_Monthly_Readings"])
    attr_path = ensure_dir(config["paths"]["outputs"]) / "reports"
    attr_path.mkdir(parents=True, exist_ok=True)
    attr.to_csv(attr_path / "attribution_summary.csv", index=False)

    # SHAP
    shap_paths = run_shap_analysis(
        silent_model, X_jan_feb.iloc[fit_mask], embeddings[fit_mask],
        output_dir=attr_path,
    )

    # Save artifacts
    torch.save(model.state_dict(), artifacts / "graph_mae.pt")
    torch.save(model.encoder.state_dict(), artifacts / "hgt_encoder.pt")
    joblib.dump(rc_analyzer, artifacts / "root_cause.pkl")
    joblib.dump(silent_model, artifacts / "silent_predictor.pkl")
    joblib.dump({"feature_names": feature_names, "jan_feb_cols": X_jan_feb.columns.tolist()}, artifacts / "feature_meta.pkl")

    with open(artifacts / "graph_meta.json", "w") as f:
        json.dump({k: v for k, v in graph_meta.items() if k != "code_to_idx"}, f, indent=2, default=str)

    training_meta = {
        "mode": mode,
        "device": str(device),
        "n_consumers": len(X),
        "n_features": len(feature_names),
        "embedding_dim": mae_cfg["hidden_dim"],
        "metrics": unsup_metrics,
        "shap": shap_paths,
    }
    with open(artifacts / "training_meta.json", "w") as f:
        json.dump(training_meta, f, indent=2, default=str)

    # Save training predictions for evaluate convenience
    pred_out = ensure_dir(config["paths"]["outputs"]) / "predictions"
    pred_out.mkdir(parents=True, exist_ok=True)
    out_df = rc_pred.copy()
    out_df.index.name = "consumer_code"
    out_df["silent_risk_score"] = silent_scores
    out_df["silent_risk_rank"] = (-out_df["silent_risk_score"]).rank(method="min").astype(int)
    out_df.reset_index(inplace=True)
    out_df.rename(columns={"index": "consumer_code"}, inplace=True)
    out_df.to_csv(pred_out / "consumer_predictions.csv", index=False)

    print(f"Artifacts saved to {artifacts}")
    print(f"Predictions saved to {pred_out / 'consumer_predictions.csv'}")


if __name__ == "__main__":
    main()
