"""SHAP explainability for silent predictor and root-cause features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def run_shap_analysis(
    silent_model: Any,
    X: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: str | Path,
    max_samples: int = 2000,
) -> dict[str, str]:
    """TreeSHAP on LightGBM silent predictor."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, index=X.index, columns=emb_cols)
    full = pd.concat([X, emb_df], axis=1)
    full = full.reindex(columns=silent_model.feature_names, fill_value=0)

    if len(full) > max_samples:
        full = full.sample(max_samples, random_state=2026)

    explainer = shap.TreeExplainer(silent_model.model)
    shap_values = explainer.shap_values(full.values)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        full,
        feature_names=full.columns.tolist(),
        show=False,
        max_display=20,
    )
    summary_path = out / "shap_summary_silent.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=120, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    bar_df = pd.DataFrame({
        "feature": full.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    bar_path = out / "shap_bar.csv"
    bar_df.to_csv(bar_path, index=False)

    return {"summary_plot": str(summary_path), "bar_csv": str(bar_path)}
