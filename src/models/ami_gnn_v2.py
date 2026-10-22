"""AMI GNN v2: Graph MAE + contrastive learning + DEC joint training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from tqdm import tqdm

from .contrastive import ContrastiveProjector, augment_x_dict, edge_dropout, nt_xent_loss
from .dec_clustering import DECClustering
from .graph_mae import _recon_loss
from .hgt_encoder import HGTEncoder


class AMIGnnV2(nn.Module):
    """Unified SOTA model: HGT encoder + MAE decoders + contrastive projector + DEC."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_channels: dict[str, int],
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        n_clusters: int = 5,
        proj_dim: int = 64,
    ):
        super().__init__()
        self.encoder = HGTEncoder(metadata, in_channels, hidden_channels, num_layers, num_heads)
        self.decoders = nn.ModuleDict(
            {nt: nn.Linear(hidden_channels, in_channels[nt]) for nt in in_channels}
        )
        self.projector = ContrastiveProjector(hidden_channels, hidden_channels, proj_dim)
        self.dec = DECClustering(n_clusters, hidden_channels)
        self.hidden_channels = hidden_channels

    def encode(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return self.encoder(x_dict, edge_index_dict)

    def decode(self, h_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {nt: self.decoders[nt](h_dict[nt]) for nt in h_dict if nt in self.decoders}

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        mask_dict: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if mask_dict:
            x_in = {}
            for nt, x in x_dict.items():
                m = mask_dict.get(nt)
                if m is not None:
                    x_in[nt] = x.clone()
                    x_in[nt][m] = 0.0
                else:
                    x_in[nt] = x
        else:
            x_in = x_dict
        h = self.encode(x_in, edge_index_dict)
        recon = self.decode(h)
        return h, recon

    def mask_features(
        self,
        data: HeteroData,
        mask_ratio: float = 0.3,
    ) -> dict[str, torch.Tensor]:
        mask_dict: dict[str, torch.Tensor] = {}
        for nt in data.node_types:
            n = data[nt].x.size(0)
            n_mask = max(1, int(n * mask_ratio))
            perm = torch.randperm(n, device=data[nt].x.device)[:n_mask]
            m = torch.zeros(n, dtype=torch.bool, device=data[nt].x.device)
            m[perm] = True
            mask_dict[nt] = m
        return mask_dict


def train_ami_gnn_v2(
    model: AMIGnnV2,
    data: HeteroData,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    mask_ratio: float = 0.3,
    contrastive_weight: float = 0.5,
    dec_weight: float = 0.3,
    dec_warmup: int = 25,
    edge_drop_rate: float = 0.15,
    train_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Joint training: MAE + GRACE contrastive + DEC clustering."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}

    history: dict[str, list[float]] = {"total": [], "mae": [], "cl": [], "dec": []}

    for epoch in tqdm(range(epochs), desc="AMI GNN v2"):
        model.train()

        # View 1: MAE masking
        mask_dict = model.mask_features(data, mask_ratio)
        if train_mask is not None and "consumer" in mask_dict:
            mask_dict["consumer"] = mask_dict["consumer"] & train_mask.to(device)

        h1, recon = model(x_dict, edge_index_dict, mask_dict)
        loss_mae = _recon_loss(x_dict, recon, mask_dict)

        # View 2: contrastive augmentation
        x_view2 = augment_x_dict(x_dict, mask_ratio * 0.7)
        edge_view2 = edge_dropout(edge_index_dict, edge_drop_rate)
        h2 = model.encode(x_view2, edge_view2)

        z1 = model.projector(h1["consumer"])
        z2 = model.projector(h2["consumer"])
        if train_mask is not None:
            z1 = z1[train_mask]
            z2 = z2[train_mask]
        loss_cl = nt_xent_loss(z1, z2) if z1.size(0) > 1 else torch.tensor(0.0, device=device)

        # DEC on consumer embeddings
        z_dec = h1["consumer"]
        if train_mask is not None:
            z_dec = z_dec[train_mask]
        loss_dec = torch.tensor(0.0, device=device)
        if epoch >= dec_warmup and z_dec.size(0) > model.dec.n_clusters:
            if epoch == dec_warmup:
                model.dec.init_centroids(z_dec.detach().cpu().numpy())
            loss_dec = model.dec.loss(z_dec)

        loss = loss_mae + contrastive_weight * loss_cl + dec_weight * loss_dec

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        history["total"].append(float(loss.item()))
        history["mae"].append(float(loss_mae.item()))
        history["cl"].append(float(loss_cl.item()) if loss_cl.dim() == 0 else float(loss_cl))
        history["dec"].append(float(loss_dec.item()) if loss_dec.dim() == 0 else float(loss_dec))

    return {
        "final_loss": history["total"][-1] if history["total"] else 0.0,
        "history": history,
    }


def consumer_embeddings_v2(model: AMIGnnV2, data: HeteroData, device: torch.device) -> torch.Tensor:
    model.eval()
    x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
    with torch.no_grad():
        h, _ = model(x_dict, edge_index_dict, mask_dict=None)
    return h["consumer"].cpu()


def per_node_reconstruction_error_v2(
    model: AMIGnnV2,
    data: HeteroData,
    device: torch.device,
    node_type: str = "consumer",
) -> torch.Tensor:
    model.eval()
    x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
    with torch.no_grad():
        _, recon = model(x_dict, edge_index_dict, mask_dict=None)
    x = x_dict[node_type]
    r = recon[node_type]
    return ((x - r) ** 2).mean(dim=1).cpu()
