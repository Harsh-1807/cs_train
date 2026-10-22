"""Heterogeneous Graph Transformer (HGT) encoder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear


class HGTEncoder(nn.Module):
    """Multi-layer HGT encoder with per-node-type input projection."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_channels: dict[str, int],
        hidden_channels: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.node_types = metadata[0]
        self.hidden_channels = hidden_channels

        self.input_proj = nn.ModuleDict(
            {nt: Linear(in_channels[nt], hidden_channels) for nt in self.node_types if nt in in_channels}
        )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                hidden_channels,
                hidden_channels,
                metadata,
                num_heads,
            )
            self.convs.append(conv)

        self.norms = nn.ModuleList(
            [nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in self.node_types}) for _ in range(num_layers)]
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        h = {}
        for nt, x in x_dict.items():
            if nt in self.input_proj:
                h[nt] = self.input_proj[nt](x)
            else:
                h[nt] = x

        for conv, norm_dict in zip(self.convs, self.norms):
            h_new = conv(h, edge_index_dict)
            for nt in h_new:
                h[nt] = norm_dict[nt](h_new[nt] + h.get(nt, 0))
                h[nt] = torch.relu(h[nt])
        return h

    @staticmethod
    def metadata_from_hetero(data: HeteroData) -> tuple[list[str], list[tuple[str, str, str]]]:
        node_types = list(data.node_types)
        edge_types = list(data.edge_types)
        return node_types, edge_types

    @staticmethod
    def in_channels_from_hetero(data: HeteroData) -> dict[str, int]:
        return {nt: data[nt].x.size(-1) for nt in data.node_types}


def consumer_embeddings(
    encoder: HGTEncoder,
    data: HeteroData,
    device: torch.device,
) -> torch.Tensor:
    """Return consumer node embeddings [N, hidden]."""
    encoder.eval()
    x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
    with torch.no_grad():
        h = encoder(x_dict, edge_index_dict)
    return h["consumer"].cpu()
