"""LightGBM silent meter risk predictor on GNN embeddings + tabular features."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


class SilentMeterPredictor:
    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 8,
        random_state: int = 2026,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model: lgb.LGBMClassifier | None = None
        self.feature_names: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        embeddings: np.ndarray,
        train_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
        emb_df = pd.DataFrame(embeddings, index=X.index, columns=emb_cols)
        full = pd.concat([X, emb_df], axis=1)
        self.feature_names = full.columns.tolist()

        if train_mask is not None:
            full = full.loc[train_mask]
            y = y.loc[full.index]

        X_arr = full.values.astype(np.float32)
        y_arr = y.values.astype(int)

        metrics: dict[str, Any] = {}
        if len(np.unique(y_arr)) > 1 and len(y_arr) > 100:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_arr, y_arr, test_size=0.2, random_state=self.random_state, stratify=y_arr
            )
            self.model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbose=-1,
            )
            self.model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
            )
            proba = self.model.predict_proba(X_va)[:, 1]
            metrics["val_auroc"] = float(roc_auc_score(y_va, proba))
            metrics["val_auprc"] = float(average_precision_score(y_va, proba))
            metrics["val_f1"] = float(f1_score(y_va, (proba >= 0.5).astype(int)))
        else:
            self.model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbose=-1,
            )
            self.model.fit(X_arr, y_arr)

        return metrics

    def predict_proba(
        self,
        X: pd.DataFrame,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
        emb_df = pd.DataFrame(embeddings, index=X.index, columns=emb_cols)
        full = pd.concat([X, emb_df], axis=1)
        full = full.reindex(columns=self.feature_names, fill_value=0)
        return self.model.predict_proba(full.values.astype(np.float32))[:, 1]
