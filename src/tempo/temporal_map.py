"""The learned temporal distance F_phi and its training loss.

F_phi re-embeds a frozen world-model latent so that Euclidean distance counts environment steps: for two frames of the
same trajectory k steps apart, ||F(z_t) - F(z_{t+k})|| ~= k / kappa. With kappa = 25 (one plan), a 25-step displacement
has unit length. Pairs farther apart than `max_steps` are only pushed to at least max_steps / kappa (hinge), so the map is
calibrated up to three plans and only orders states beyond that.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, out_dim: int, hidden: int, layers: int) -> nn.Sequential:
    """`layers` hidden layers of width `hidden` with SiLU activations, then a linear output."""
    mods, d = [], in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class TemporalMap(nn.Module):
    """F_phi: R^latent_dim -> R^out_dim, trained so that map distance is time-to-reach in units of `kappa` steps."""

    def __init__(self, latent_dim: int, out_dim: int = 64, hidden: int = 256, layers: int = 2, kappa: float = 25.0):
        super().__init__()
        self.latent_dim, self.out_dim, self.kappa = int(latent_dim), int(out_dim), float(kappa)
        self.net = mlp(self.latent_dim, self.out_dim, hidden, layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def dist(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        return (self(za) - self(zb)).norm(dim=-1)

    def sqdist(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        return ((self(za) - self(zb)) ** 2).sum(-1)


def temporal_distance_loss(
    fmap: TemporalMap, near: dict, far: Optional[dict] = None, max_steps: int = 75, far_weight: float = 0.5
) -> torch.Tensor:
    """L_F = smooth-L1(||F(z_t) - F(z_{t+k})||, k / kappa)  for k <= max_steps
           + far_weight * relu(max_steps / kappa - ||F(z_t) - F(z_{t+d})||)  for d > max_steps.

    `near` / `far` hold tensors `z` (n, D), `z_future` (n, D) and `delta` (n,) of same-episode pairs; `far` may be None
    when the stored episodes are too short to contain pairs beyond `max_steps`."""
    d = fmap.dist(near["z"], near["z_future"])
    loss = F.smooth_l1_loss(d, near["delta"].float() / fmap.kappa)
    if far is not None:
        dn = fmap.dist(far["z"], far["z_future"])
        loss = loss + far_weight * F.relu(max_steps / fmap.kappa - dn).mean()
    return loss
