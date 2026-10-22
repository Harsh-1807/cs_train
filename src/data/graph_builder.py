"""Build heterogeneous PyG graph from AMI topology and consumer features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .features import build_infrastructure_node_features

LEVEL_TO_TYPE = {
    "Power Plant": "power_plant",
    "Transmission Grid": "transmission",
    "Grid Substation": "grid_ss",
    "Distribution Substation": "substation",
    "Power Transformer": "transformer",
    "11kV Feeder": "feeder",
    "DTC": "dtc",
    "LT Line": "lt_line",
    "Consumer": "consumer",
}


def _level_type(level: str) -> str:
    return LEVEL_TO_TYPE.get(str(level).strip(), str(level).lower().replace(" ", "_"))


def build_hetero_graph(
    data: dict[str, pd.DataFrame],
    X: pd.DataFrame,
    train_mask: np.ndarray | None = None,
    test_mask: np.ndarray | None = None,
    infra_feature_dim: int = 16,
) -> tuple[HeteroData, dict[str, Any]]:
    """
    Build HeteroData with consumer nodes and infrastructure nodes carrying
    real observable features from Asset_Comm_Health, DCU_Daily_Status,
    and rolled-up consumer aggregates (feeder/DTC/LT-line scenarios).
    """
    topology = data["Network_Topology"].copy()
    ami = data["AMI_Comm_Health"].copy()
    ami["Consumer_Code"] = ami["Consumer_Code"].astype(str)

    consumer_codes = X.index.astype(str).tolist()
    code_to_idx = {c: i for i, c in enumerate(consumer_codes)}

    # Collect infrastructure node codes per type
    infra_nodes: dict[str, dict[str, int]] = {}
    edge_src: dict[tuple[str, str, str], list[int]] = {}
    edge_dst: dict[tuple[str, str, str], list[int]] = {}

    def _ensure_node(ntype: str, code: str) -> int:
        if ntype not in infra_nodes:
            infra_nodes[ntype] = {}
        if code not in infra_nodes[ntype]:
            infra_nodes[ntype][code] = len(infra_nodes[ntype])
        return infra_nodes[ntype][code]

    for _, row in topology.iterrows():
        plevel = _level_type(row["Parent_Level"])
        clevel = _level_type(row["Child_Level"])
        pcode = str(row["Parent_Code"])
        ccode = str(row["Child_Code"])

        etype = ("feeds", plevel, clevel)
        if etype not in edge_src:
            edge_src[etype] = []
            edge_dst[etype] = []

        if clevel == "consumer" and ccode in code_to_idx:
            # infrastructure -> consumer edge via direct consumer attachment
            pidx = _ensure_node(plevel, pcode)
            edge_src[etype].append(pidx)
            edge_dst[etype].append(code_to_idx[ccode])
        elif plevel == "consumer":
            continue
        else:
            pidx = _ensure_node(plevel, pcode)
            cidx = _ensure_node(clevel, ccode)
            edge_src[etype].append(pidx)
            edge_dst[etype].append(cidx)

    # Consumer -> DTC / Feeder edges
    ami_idx = ami.set_index("Consumer_Code")
    for code in consumer_codes:
        if code not in ami_idx.index:
            continue
        ci = code_to_idx[code]
        dtc = str(ami_idx.loc[code, "DTC_Code"])
        feeder = str(ami_idx.loc[code, "Feeder_Code"])
        dtc_i = _ensure_node("dtc", dtc)
        feeder_i = _ensure_node("feeder", feeder)
        et_dtc = ("served_by", "consumer", "dtc")
        et_feeder = ("on_feeder", "consumer", "feeder")
        for et, dst_i in [(et_dtc, dtc_i), (et_feeder, feeder_i)]:
            if et not in edge_src:
                edge_src[et] = []
                edge_dst[et] = []
            edge_src[et].append(ci)
            edge_dst[et].append(dst_i)

    hetero = HeteroData()

    x_consumer = torch.tensor(X.values, dtype=torch.float32)
    hetero["consumer"].x = x_consumer
    hetero["consumer"].num_nodes = x_consumer.size(0)

    if train_mask is not None:
        hetero["consumer"].train_mask = torch.tensor(train_mask, dtype=torch.bool)
    if test_mask is not None:
        hetero["consumer"].test_mask = torch.tensor(test_mask, dtype=torch.bool)

    # Infrastructure nodes: real multi-level features (feeder/DTC/SS/PT/LT)
    infra_feats = build_infrastructure_node_features(data, ami)
    rng = np.random.default_rng(42)
    for ntype, mapping in infra_nodes.items():
        n = len(mapping)
        code_list = sorted(mapping.keys(), key=lambda c: mapping[c])
        type_feats = infra_feats.get(ntype, {})
        feat_names = type_feats.get("__feature_names__", None)
        fdim = len(feat_names) if feat_names is not None else infra_feature_dim
        fdim = max(fdim, infra_feature_dim)
        rows = []
        for code in code_list:
            if code in type_feats and isinstance(type_feats[code], np.ndarray):
                v = type_feats[code]
                if len(v) < fdim:
                    v = np.pad(v, (0, fdim - len(v)))
                rows.append(v[:fdim])
            else:
                rows.append(rng.normal(0, 0.01, fdim).astype(np.float32))
        feat = torch.tensor(np.stack(rows), dtype=torch.float32)
        hetero[ntype].x = feat
        hetero[ntype].num_nodes = n

    for etype, src_list in edge_src.items():
        if not src_list:
            continue
        src_t = torch.tensor(src_list, dtype=torch.long)
        dst_t = torch.tensor(edge_dst[etype], dtype=torch.long)
        hetero[etype].edge_index = torch.stack([src_t, dst_t], dim=0)

    # Reverse edges for message passing
    for etype in list(hetero.edge_types):
        rev_name = "rev_" + etype[0]
        rev = (rev_name, etype[2], etype[1])
        ei = hetero[etype].edge_index
        hetero[rev].edge_index = torch.stack([ei[1], ei[0]], dim=0)

    meta = {
        "consumer_codes": consumer_codes,
        "code_to_idx": code_to_idx,
        "infra_nodes": {k: list(v.keys()) for k, v in infra_nodes.items()},
        "feature_dim": X.shape[1],
        "infra_feature_dim": infra_feature_dim,
    }
    return hetero, meta


def add_reverse_edges(hetero: HeteroData) -> HeteroData:
    """Ensure all edge types have reverse edges."""
    for etype in list(hetero.edge_types):
        if etype[0].startswith("rev_"):
            continue
        rev = ("rev_" + etype[0], etype[2], etype[1])
        if rev not in hetero.edge_types:
            ei = hetero[etype].edge_index
            hetero[rev].edge_index = torch.stack([ei[1], ei[0]], dim=0)
    return hetero
