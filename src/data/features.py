"""Feature engineering from observable AMI columns only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TAMPER_EVENTS = {6, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 28}
FIRMWARE_EVENTS = {8, 30}
NIC_REMOVAL_EVENT = 9


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _one_hot(series: pd.Series, prefix: str, top_n: int = 10) -> pd.DataFrame:
    vals = series.fillna("Unknown").astype(str)
    top = vals.value_counts().head(top_n).index.tolist()
    vals = vals.where(vals.isin(top), "Other")
    dummies = pd.get_dummies(vals, prefix=prefix)
    return dummies


def build_event_features(events: pd.DataFrame, consumer_codes: pd.Series) -> pd.DataFrame:
    """Aggregate event log to per-consumer features."""
    ev = events.copy()
    ev["Consumer_Code"] = ev["Consumer_Code"].astype(str)
    ev["Event_Class"] = pd.to_numeric(ev["Event_Class"], errors="coerce")

    base = pd.DataFrame({"Consumer_Code": consumer_codes.astype(str).unique()})
    if ev.empty:
        base["event_total"] = 0
        base["event_nic_removal"] = 0
        base["event_tamper"] = 0
        base["event_firmware"] = 0
        return base.set_index("Consumer_Code")

    grp = ev.groupby("Consumer_Code")
    agg = grp.agg(
        event_total=("Event_Class", "count"),
        event_distinct=("Event_Class", "nunique"),
        event_duration_min=("Duration_in_min", lambda x: pd.to_numeric(x, errors="coerce").sum()),
    )
    agg["event_nic_removal"] = grp.apply(
        lambda g: int((g["Event_Class"] == NIC_REMOVAL_EVENT).any())
    )
    agg["event_tamper"] = grp.apply(
        lambda g: int(g["Event_Class"].isin(TAMPER_EVENTS).sum())
    )
    agg["event_firmware"] = grp.apply(
        lambda g: int(g["Event_Class"].isin(FIRMWARE_EVENTS).sum())
    )
    agg["event_tamper_before_nic"] = grp.apply(_tamper_before_nic)
    return agg


def _tamper_before_nic(g: pd.DataFrame) -> int:
    g = g.sort_values("First_Occurrence", na_position="last")
    classes = g["Event_Class"].tolist()
    nic_idx = next((i for i, c in enumerate(classes) if c == NIC_REMOVAL_EVENT), None)
    if nic_idx is None:
        return 0
    prior = classes[:nic_idx]
    return int(any(c in TAMPER_EVENTS and c != NIC_REMOVAL_EVENT for c in prior))


def build_dcu_features(dcu_daily: pd.DataFrame, ami: pd.DataFrame) -> pd.DataFrame:
    """Aggregate DCU daily stats per consumer via DCU_ID / DTC_Code."""
    ami = ami.copy()
    ami["Consumer_Code"] = ami["Consumer_Code"].astype(str)
    ami["DTC_Code"] = ami["DTC_Code"].astype(str)

    dcu = dcu_daily.copy()
    dcu["DTC_Code"] = dcu["DTC_Code"].astype(str)
    num_cols = ["Daily_Read_pct", "LP_Completeness_pct", "Packet_Loss_pct", "DCU_Up"]
    for c in num_cols:
        if c in dcu.columns:
            dcu[c] = pd.to_numeric(dcu[c], errors="coerce")

    dtc_agg = dcu.groupby("DTC_Code").agg(
        dcu_read_pct_mean=("Daily_Read_pct", "mean"),
        dcu_read_pct_std=("Daily_Read_pct", "std"),
        dcu_lp_mean=("LP_Completeness_pct", "mean"),
        dcu_loss_mean=("Packet_Loss_pct", "mean"),
        dcu_up_frac=("DCU_Up", "mean"),
    )
    out = ami[["Consumer_Code", "DTC_Code"]].merge(dtc_agg, on="DTC_Code", how="left")
    return out.set_index("Consumer_Code").drop(columns=["DTC_Code"])


def build_monthly_features(readings: pd.DataFrame, consumer_codes: pd.Series) -> pd.DataFrame:
    """Pivot monthly readings + RNA flags."""
    r = readings.copy()
    r["Consumer_Code"] = r["Consumer_Code"].astype(str)
    r["Month"] = r["Month"].astype(str)

    rna = r.groupby("Consumer_Code")["Billing_Basis"].apply(
        lambda s: int(s.astype(str).str.contains("RNA|Assessed", case=False, na=False).any())
    ).rename("has_rna_bill")

    pivot_cols = ["Read_Success_pct", "Heartbeat_pct", "Packet_Loss_pct", "Outage_Minutes"]
    available = [c for c in pivot_cols if c in r.columns]
    for c in available:
        r[c] = pd.to_numeric(r[c], errors="coerce")

    pieces = [rna]
    for col in available:
        p = r.pivot_table(index="Consumer_Code", columns="Month", values=col, aggfunc="mean")
        p.columns = [f"{col}_{m}" for m in p.columns]
        pieces.append(p)

    pq = r.groupby("Consumer_Code")["PQ_Data_Status"].apply(
        lambda s: int(s.astype(str).str.contains("Not received", case=False, na=False).any())
    ).rename("pq_not_received")

    pieces.append(pq)
    feat = pd.concat(pieces, axis=1)
    feat = feat.reindex(consumer_codes.astype(str).unique())
    return feat


def build_asset_features(assets: pd.DataFrame, ami: pd.DataFrame) -> pd.DataFrame:
    """Upstream asset health aggregated to feeder/substation level."""
    ami = ami.copy()
    ami["Consumer_Code"] = ami["Consumer_Code"].astype(str)
    ami["Feeder_Code"] = ami["Feeder_Code"].astype(str)

    a = assets.copy()
    if "Feeder_Code" not in a.columns:
        return pd.DataFrame(index=ami["Consumer_Code"].unique())

    num_cols = [
        "Meter_Heartbeat_pct", "Load_Profile_Completeness_pct",
        "Meter_Packet_Loss_pct", "DCU_Avg_RF_Meter_Read_pct",
    ]
    for c in num_cols:
        if c in a.columns:
            a[c] = pd.to_numeric(a[c], errors="coerce")

    agg_cols = [c for c in num_cols if c in a.columns]
    if not agg_cols:
        return pd.DataFrame(index=ami["Consumer_Code"].unique())

    feeder_agg = a.groupby("Feeder_Code")[agg_cols].mean()
    feeder_agg.columns = [f"asset_{c}" for c in feeder_agg.columns]
    out = ami[["Consumer_Code", "Feeder_Code"]].merge(feeder_agg, on="Feeder_Code", how="left")
    return out.set_index("Consumer_Code").drop(columns=["Feeder_Code"])


def build_consumer_feature_matrix(
    data: dict[str, pd.DataFrame],
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Returns:
        X: feature matrix indexed by Consumer_Code
        meta: metadata (Comm_Status, NIC_Status, Feeder_Code, etc.)
        feature_names: column names
    """
    ami = data["AMI_Comm_Health"].copy()
    ami["Consumer_Code"] = ami["Consumer_Code"].astype(str)

    numeric_ami = [
        "Distance_from_DTC_m", "Heartbeat_Jan_pct", "Heartbeat_Feb_pct", "Heartbeat_Mar_pct",
        "Avg_Heartbeat_pct", "Read_Success_Jan_pct", "Read_Success_Feb_pct", "Read_Success_Mar_pct",
        "Avg_Read_Success_pct", "Missed_Daily_Reads_Q1", "Load_Profile_Completeness_pct",
        "Data_Gap_Days_Q1", "Frozen_Register_Days_Q1", "Avg_RSSI_dBm",
        "Packet_Loss_Jan_pct", "Packet_Loss_Feb_pct", "Packet_Loss_Mar_pct",
        "Avg_Packet_Loss_pct", "Avg_Latency_ms", "Hop_Count",
        "V_R_Avg", "V_Y_Avg", "V_B_Avg", "I_R_Avg_A", "I_Y_Avg_A", "I_B_Avg_A",
        "I_Neutral_Avg_A", "Days_Since_Last_Comm", "Outage_Minutes_Q1",
        "Power_Failure_Events_Q1", "Avg_Supply_Voltage_V", "Avg_Frequency_Hz",
    ]
    ami = _to_numeric(ami, numeric_ami)

    event_feat = build_event_features(data["Consumer_Event_Log"], ami["Consumer_Code"])
    dcu_feat = build_dcu_features(data["DCU_Daily_Status"], ami)
    monthly_feat = build_monthly_features(data["Consumer_Monthly_Readings"], ami["Consumer_Code"])
    asset_feat = build_asset_features(data["Asset_Comm_Health"], ami)

    cat_frames = [
        _one_hot(ami["Comm_Technology"], "commtech"),
        _one_hot(ami["NIC_Status"], "nic"),
        _one_hot(ami["DCU_Status"], "dcu_st"),
        _one_hot(ami["Backhaul_Status"], "backhaul"),
        _one_hot(ami["Comm_Status"], "commst"),
        _one_hot(ami["HES_Registration"], "hesreg"),
    ]

    base = ami.set_index("Consumer_Code")[numeric_ami]
    X = pd.concat([base, event_feat, dcu_feat, monthly_feat, asset_feat] + cat_frames, axis=1)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    meta_cols = ["Feeder_Code", "DTC_Code", "Comm_Status", "NIC_Status", "Missed_Daily_Reads_Q1"]
    meta = ami.set_index("Consumer_Code")[[c for c in meta_cols if c in ami.columns]]

    feature_names = X.columns.tolist()
    return X, meta, feature_names


def build_silent_targets(ami: pd.DataFrame, readings: pd.DataFrame) -> pd.Series:
    """
    Temporal proxy target: at-risk in Mar (Non-Communicating or Intermittent).
    Features for prediction should use Jan-Feb only (handled in silent_predictor).
    """
    ami = ami.copy()
    ami["Consumer_Code"] = ami["Consumer_Code"].astype(str)
    bad = ami["Comm_Status"].astype(str).str.contains(
        "Non-Communicating|Intermittent", case=False, na=False
    )
    return bad.astype(int).rename("silent_target")


ASSET_LEVEL_MAP = {
    "DTC": "dtc",
    "DTC Meter": "dtc",
    "DCU": "dtc",
    "11kV Feeder": "feeder",
    "Feeder": "feeder",
    "Feeder Meter": "feeder",
    "Power Transformer": "transformer",
    "PT": "transformer",
    "Distribution Substation": "substation",
    "LT Line": "lt_line",
}


def build_infrastructure_node_features(
    data: dict[str, pd.DataFrame],
    ami: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Real observable features for every infrastructure node type in the graph.
    Covers feeder/DTC/substation/transformer/LT-line failure scenarios (POC multi-level).
    """
    ami = ami.copy()
    ami["Consumer_Code"] = ami["Consumer_Code"].astype(str)
    ami["Feeder_Code"] = ami["Feeder_Code"].astype(str)
    ami["DTC_Code"] = ami["DTC_Code"].astype(str)

    infra: dict[str, dict[str, list[float]]] = {}

    def _add(ntype: str, code: str, values: dict[str, float]) -> None:
        if ntype not in infra:
            infra[ntype] = {}
        if code not in infra[ntype]:
            infra[ntype][code] = {}
        infra[ntype][code].update(values)

    # --- Asset_Comm_Health: direct meter/DCU/feeder/PT measurements ---
    assets = data.get("Asset_Comm_Health", pd.DataFrame())
    if not assets.empty:
        a = assets.copy()
        num_cols = [
            "Meter_Heartbeat_pct", "Meter_Read_Jan_pct", "Meter_Read_Feb_pct",
            "Meter_Read_Mar_pct", "Load_Profile_Completeness_pct",
            "DCU_Packet_Loss_pct", "Meter_Packet_Loss_pct", "Avg_Latency_ms",
            "Avg_RSSI_dBm", "DCU_Avg_RF_Meter_Read_pct", "RF_Meters_Under_DCU",
            "Outage_Minutes_Q1", "Avg_Supply_Voltage_V", "Avg_Frequency_Hz",
        ]
        for c in num_cols:
            if c in a.columns:
                a[c] = pd.to_numeric(a[c], errors="coerce")

        for _, row in a.iterrows():
            level = str(row.get("Asset_Level", ""))
            ntype = ASSET_LEVEL_MAP.get(level)
            code = str(row.get("Asset_Code", ""))
            if not ntype or not code or code == "nan":
                # fallback: map via feeder/substation codes
                if pd.notna(row.get("Feeder_Code")):
                    ntype, code = "feeder", str(row["Feeder_Code"])
                elif pd.notna(row.get("Substation_Code")):
                    ntype, code = "substation", str(row["Substation_Code"])
                else:
                    continue
            vals = {c: float(row[c]) if c in a.columns and pd.notna(row.get(c)) else 0.0 for c in num_cols if c in a.columns}
            _add(ntype, code, vals)

    # --- DCU_Daily_Status: DTC-level time series (DCU failure, backhaul) ---
    dcu = data.get("DCU_Daily_Status", pd.DataFrame())
    if not dcu.empty:
        dcu = dcu.copy()
        dcu["DTC_Code"] = dcu["DTC_Code"].astype(str)
        for c in ["Daily_Read_pct", "LP_Completeness_pct", "Packet_Loss_pct", "DCU_Up", "Backhaul_Up"]:
            if c in dcu.columns:
                dcu[c] = pd.to_numeric(dcu[c], errors="coerce")
        dtc_dcu = dcu.groupby("DTC_Code").agg(
            dcu_daily_read_mean=("Daily_Read_pct", "mean"),
            dcu_daily_read_min=("Daily_Read_pct", "min"),
            dcu_daily_lp_mean=("LP_Completeness_pct", "mean"),
            dcu_daily_loss_mean=("Packet_Loss_pct", "mean"),
            dcu_up_frac=("DCU_Up", "mean"),
            backhaul_up_frac=("Backhaul_Up", "mean"),
        )
        for code, row in dtc_dcu.iterrows():
            _add("dtc", str(code), row.to_dict())

    # --- Consumer aggregates rolled up to feeder / DTC / LT line ---
    consumer = data.get("Consumer", pd.DataFrame())
    c_src = consumer if not consumer.empty else ami
    c_src = c_src.copy()
    c_src["Consumer_Code"] = c_src["Consumer_Code"].astype(str)

    roll_cols = [
        "Avg_Read_Success_pct", "Avg_Heartbeat_pct", "Avg_Packet_Loss_pct",
        "Missed_Daily_Reads_Q1", "Load_Profile_Completeness_pct",
    ]
    for c in roll_cols:
        if c in c_src.columns:
            c_src[c] = pd.to_numeric(c_src[c], errors="coerce")

    avail_roll = [c for c in roll_cols if c in c_src.columns]
    if avail_roll:
        if "Feeder_Code" in c_src.columns:
            for code, grp in c_src.groupby("Feeder_Code"):
                _add("feeder", str(code), {f"child_{c}_mean": grp[c].mean() for c in avail_roll})
                _add("feeder", str(code), {f"child_{c}_min": grp[c].min() for c in avail_roll})
        if "DTC_Code" in c_src.columns:
            for code, grp in c_src.groupby("DTC_Code"):
                _add("dtc", str(code), {f"child_{c}_mean": grp[c].mean() for c in avail_roll})
        if "LT_Line_Code" in c_src.columns:
            for code, grp in c_src.groupby("LT_Line_Code"):
                _add("lt_line", str(code), {f"child_{c}_mean": grp[c].mean() for c in avail_roll})

    # --- Feeder outage from monthly readings ---
    readings = data.get("Consumer_Monthly_Readings", pd.DataFrame())
    if not readings.empty and "Feeder_Code" in readings.columns:
        r = readings.copy()
        r["Feeder_Code"] = r["Feeder_Code"].astype(str)
        if "Outage_Minutes" in r.columns:
            r["Outage_Minutes"] = pd.to_numeric(r["Outage_Minutes"], errors="coerce")
            for code, grp in r.groupby("Feeder_Code"):
                _add("feeder", str(code), {
                    "feeder_outage_mean": grp["Outage_Minutes"].mean(),
                    "feeder_outage_max": grp["Outage_Minutes"].max(),
                })

    # --- Hierarchy roll-up via Network_Topology (parent nodes inherit child comm stats) ---
    topology = data.get("Network_Topology", pd.DataFrame())
    if not topology.empty and avail_roll:
        child_stats: dict[str, dict[str, list[float]]] = {}
        if "LT_Line_Code" in c_src.columns and "Consumer_Code" in c_src.columns:
            lt_map = c_src.set_index("Consumer_Code")["LT_Line_Code"].astype(str).to_dict()
            for code, grp in c_src.groupby("Consumer_Code"):
                lt = lt_map.get(str(code), "")
                if lt:
                    key = f"lt_line:{lt}"
                    child_stats.setdefault(key, {}).setdefault("codes", []).append(str(code))
        # Map consumer metrics to LT lines already done; walk topology for higher levels
        level_map = {
            "Consumer": "consumer", "LT Line": "lt_line", "DTC": "dtc",
            "11kV Feeder": "feeder", "Power Transformer": "transformer",
            "Distribution Substation": "substation", "Grid Substation": "grid_ss",
            "Transmission Grid": "transmission", "Power Plant": "power_plant",
        }
        ami_roll = ami.set_index("Consumer_Code") if "Consumer_Code" in ami.columns else c_src.set_index("Consumer_Code")
        for _, edge in topology.iterrows():
            plevel = level_map.get(str(edge["Parent_Level"]), str(edge["Parent_Level"]).lower())
            clevel = level_map.get(str(edge["Child_Level"]), str(edge["Child_Level"]).lower())
            pcode, ccode = str(edge["Parent_Code"]), str(edge["Child_Code"])
            if clevel == "consumer" and ccode in ami_roll.index:
                vals = {f"hier_{c}": float(ami_roll.loc[ccode, c]) for c in avail_roll if c in ami_roll.columns}
                _add(plevel, pcode, vals)
            elif clevel != "consumer":
                # link child node stats to parent (accumulate via topology presence)
                pass

    # Convert to aligned numpy vectors per node type
    result: dict[str, dict[str, np.ndarray]] = {}
    for ntype, nodes in infra.items():
        result[ntype] = {}
        all_keys = sorted({k for v in nodes.values() for k in v})
        for code, vals in nodes.items():
            result[ntype][code] = np.array([vals.get(k, 0.0) for k in all_keys], dtype=np.float32)
        result[ntype]["__feature_names__"] = np.array(all_keys)  # type: ignore[assignment]
    return result


def build_jan_feb_features(X: pd.DataFrame) -> pd.DataFrame:
    """Subset columns that represent Jan-Feb temporal signal for silent prediction."""
    jan_feb_cols = [c for c in X.columns if "_Jan" in c or "_Feb" in c or "Jan" in c or "Feb" in c]
    static_cols = [c for c in X.columns if c not in jan_feb_cols and "_Mar" not in c]
    # Exclude Mar-month comm columns from temporal train features
    exclude_mar = [c for c in static_cols if "Mar" not in c]
    return X[exclude_mar]
