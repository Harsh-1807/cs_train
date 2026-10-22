"""Graph contrastive learning utilities (GRACE-style)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData


class ContrastiveProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Normalized temperature-scaled cross entropy (SimCLR/GRACE)."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    n = z1.size(0)
    reps = torch.cat([z1, z2], dim=0)
    sim = torch.mm(reps, reps.t()) / temperature
    mask = torch.eye(2 * n, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(n)], device=sim.device)
    return F.cross_entropy(sim, targets)


def edge_dropout(
    edge_index_dict: dict[tuple, torch.Tensor],
    drop_rate: float = 0.15,
) -> dict[tuple, torch.Tensor]:
    """Randomly drop edges for augmented graph view."""
    out = {}
    for et, ei in edge_index_dict.items():
        if et[0].startswith("rev_"):
            continue
        e = ei.size(1)
        if e == 0:
            out[et] = ei
            continue
        keep = torch.rand(e, device=ei.device) > drop_rate
        if keep.sum() == 0:
            keep[0] = True
        out[et] = ei[:, keep]
        rev = ("rev_" + et[0], et[2], et[1])
        if rev in edge_index_dict:
            rev_ei = edge_index_dict[rev][:, keep]
            out[rev] = rev_ei
    # copy any rev edges not yet added
    for et, ei in edge_index_dict.items():
        if et not in out:
            out[et] = ei
    return out


def augment_x_dict(
    x_dict: dict[str, torch.Tensor],
    mask_ratio: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Feature masking augmentation."""
    out = {}
    for nt, x in x_dict.items():
        x_aug = x.clone()
        n = x.size(0)
        n_mask = max(1, int(n * mask_ratio))
        idx = torch.randperm(n, device=x.device)[:n_mask]
        x_aug[idx] = 0.0
        out[nt] = x_aug
    return out
