"""Adaptive evaluation — uses GT only here, never during training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.data.loader import load_ami_data_with_gt
from src.data.split_manager import load_split_manifest
from src.evaluate_mapping import defect_type_to_root_cause
from src.utils import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate AMI predictions vs GT")
    p.add_argument("--input", required=True)
    p.add_argument("--predictions", default="outputs/predictions")
    p.add_argument("--artifacts", default="artifacts/")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output", default="outputs/reports")
    return p.parse_args()


def _eval_subset(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    subset_name: str,
) -> dict:
    merged = pred.merge(gt, on="consumer_code", how="inner")
    if merged.empty:
        return {"subset": subset_name, "error": "no overlap"}

    y_true = merged["gt_root_cause"].values
    y_pred = merged["root_cause_class"].values
    mask = y_true != "Normal"
    if mask.sum() == 0:
        return {"subset": subset_name, "error": "no defective GT rows"}

    report = classification_report(
        y_true[mask], y_pred[mask],
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true[mask], y_pred[mask], labels=sorted(set(y_true[mask]) | set(y_pred[mask])))

    # NIC removal subset
    nic_mask = merged["NIC_Status"].astype(str).str.contains("Removed", case=False, na=False)
    nic_metrics = {}
    if nic_mask.sum() > 0:
        nic_m = merged[nic_mask]
        nic_metrics = {
            "n_nic_removed": int(nic_mask.sum()),
            "macro_f1": float(f1_score(
                nic_m["gt_root_cause"], nic_m["nic_removal_root_cause"],
                average="macro", zero_division=0,
            )),
        }

    return {
        "subset": subset_name,
        "n_evaluated": int(len(merged)),
        "n_defective": int(mask.sum()),
        "macro_f1": report.get("macro avg", {}).get("f1-score", 0),
        "weighted_f1": report.get("weighted avg", {}).get("f1-score", 0),
        "classification_report": report,
        "confusion_matrix_shape": list(cm.shape),
        "nic_removal": nic_metrics,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out = ensure_dir(args.output)
    artifacts = Path(args.artifacts)

    pred = pd.read_csv(Path(args.predictions) / "consumer_predictions.csv")
    pred["consumer_code"] = pred["consumer_code"].astype(str)

    gt_data = load_ami_data_with_gt(args.input)
    ami_gt = gt_data["AMI_Comm_Health"].copy()
    ami_gt["consumer_code"] = ami_gt["Consumer_Code"].astype(str)
    ami_gt["gt_root_cause"] = ami_gt.get("Defect_Type", pd.Series(dtype=str)).map(defect_type_to_root_cause)
    gt_cols = ["consumer_code", "gt_root_cause"]
    if "NIC_Status" in ami_gt.columns:
        gt_cols.append("NIC_Status")
    if "Defect_Type" in ami_gt.columns:
        gt_cols.append("Defect_Type")
    gt = ami_gt[gt_cols].drop_duplicates("consumer_code")

    manifest = load_split_manifest(artifacts / "split_manifest.json")
    mode = manifest.get("mode", "full") if manifest else "full"

    results: dict = {"mode": mode, "evaluations": []}

    if mode == "split" and manifest:
        test_codes = set(str(c) for c in manifest.get("test_consumer_codes", []))
        train_codes = set(str(c) for c in manifest.get("train_consumer_codes", []))
        pred_test = pred[pred["consumer_code"].isin(test_codes)]
        pred_train = pred[pred["consumer_code"].isin(train_codes)]
        results["evaluations"].append(_eval_subset(pred_test, gt, "held_out_test_feeders"))
        results["evaluations"].append(_eval_subset(pred_train, gt, "train_feeders"))
    else:
        results["evaluations"].append(_eval_subset(pred, gt, "full_dataset"))

    # Silent meter proxy eval (Comm_Status based, observable GT)
    if "Comm_Status" in ami_gt.columns:
        ami_gt["silent_obs"] = ami_gt["Comm_Status"].astype(str).str.contains(
            "Non-Communicating|Intermittent", case=False, na=False
        ).astype(int)
        m = pred.merge(ami_gt[["consumer_code", "silent_obs"]], on="consumer_code")
        if len(m) > 0 and "silent_risk_score" in m.columns:
            from sklearn.metrics import roc_auc_score
            try:
                results["silent_auroc_observable"] = float(
                    roc_auc_score(m["silent_obs"], m["silent_risk_score"])
                )
            except ValueError:
                results["silent_auroc_observable"] = None

    # Load training meta if present
    meta_path = artifacts / "training_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            results["training_metrics"] = json.load(f)

    report_path = out / "metrics_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Evaluation report: {report_path}")
    for ev in results["evaluations"]:
        if "macro_f1" in ev:
            print(f"  [{ev['subset']}] macro-F1={ev['macro_f1']:.4f} n={ev.get('n_evaluated', 0)}")


if __name__ == "__main__":
    main()
