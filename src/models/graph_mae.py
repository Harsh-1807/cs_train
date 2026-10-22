"""Graph Masked Autoencoder with HGT backbone."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from tqdm import tqdm

from .hgt_encoder import HGTEncoder


class GraphMAE(nn.Module):
    """Mask node features, encode with HGT, decode to reconstruct."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_channels: dict[str, int],
        hidden_channels: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.encoder = HGTEncoder(metadata, in_channels, hidden_channels, num_layers, num_heads)
        self.decoders = nn.ModuleDict(
            {nt: nn.Linear(hidden_channels, in_channels[nt]) for nt in in_channels}
        )
        self.hidden_channels = hidden_channels

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        mask_dict: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if mask_dict:
            x_masked = {}
            for nt, x in x_dict.items():
                m = mask_dict.get(nt)
                if m is not None:
                    x_masked[nt] = x.clone()
                    x_masked[nt][m] = 0.0
                else:
                    x_masked[nt] = x
        else:
            x_masked = x_dict

        h = self.encoder(x_masked, edge_index_dict)
        recon = {nt: self.decoders[nt](h[nt]) for nt in h if nt in self.decoders}
        return h, recon

    def mask_features(
        self,
        data: HeteroData,
        mask_ratio: float = 0.3,
        rng: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        mask_dict: dict[str, torch.Tensor] = {}
        for nt in data.node_types:
            n = data[nt].x.size(0)
            n_mask = max(1, int(n * mask_ratio))
            perm = torch.randperm(n, generator=rng)[:n_mask]
            m = torch.zeros(n, dtype=torch.bool)
            m[perm] = True
            mask_dict[nt] = m
        return mask_dict


def _recon_loss(
    x_dict: dict[str, torch.Tensor],
    recon: dict[str, torch.Tensor],
    mask_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    losses = []
    for nt, x in x_dict.items():
        if nt not in recon or nt not in mask_dict:
            continue
        m = mask_dict[nt]
        if m.sum() == 0:
            continue
        losses.append(nn.functional.mse_loss(recon[nt][m], x[m]))
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def train_graph_mae(
    model: GraphMAE,
    data: HeteroData,
    device: torch.device,
    epochs: int = 80,
    lr: float = 1e-3,
    mask_ratio: float = 0.3,
    train_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Train Graph MAE; optionally restrict loss to train consumer nodes."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}

    history: list[float] = []
    for epoch in tqdm(range(epochs), desc="Graph MAE"):
        model.train()
        mask_dict = model.mask_features(data, mask_ratio)
        mask_dict = {k: v.to(device) for k, v in mask_dict.items()}

        if train_mask is not None and "consumer" in mask_dict:
            # Only mask train consumers for split mode
            cm = train_mask.to(device)
            consumer_mask = mask_dict["consumer"]
            consumer_mask = consumer_mask & cm
            mask_dict["consumer"] = consumer_mask

        h, recon = model(x_dict, edge_index_dict, mask_dict)
        loss = _recon_loss(x_dict, recon, mask_dict)

        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.item()))

    return {"final_loss": history[-1] if history else 0.0, "loss_history": history}


def per_node_reconstruction_error(
    model: GraphMAE,
    data: HeteroData,
    device: torch.device,
    node_type: str = "consumer",
) -> torch.Tensor:
    """MSE reconstruction error per node (full feature vector)."""
    model.eval()
    x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
    with torch.no_grad():
        _, recon = model(x_dict, edge_index_dict, mask_dict=None)
    x = x_dict[node_type]
    r = recon[node_type]
    err = ((x - r) ** 2).mean(dim=1)
    return err.cpu()
