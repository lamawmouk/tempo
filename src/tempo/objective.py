"""TEMPO as a stable-worldmodel planning Objective.

The CEM solver rolls every candidate action sequence through the frozen world model and scores the rolled-out info dict:

    J(a) = (1 - w) * own(z_hat_{t+H}, G) + w * d_H^2 * ||F(z_hat_{t+H}) - F(G)||^2

`own` is the world model's published cost (GoalMSE for LeWM / PLDM, the per-source mean MSE for DINO-WM), so the w = 0 arm
is exactly the baseline planner. d_H^2 (`scale`) is the median squared H-step latent displacement of the training data
(for DINO-WM: the median of its own cost over H-step pairs), which puts both terms in the same units.
"""

from __future__ import annotations

import copy
from typing import Optional, Sequence

import torch
import torch.nn as nn
from stable_worldmodel.planning import GoalMSE
from stable_worldmodel.planning.objective import WeightedSum

from tempo.temporal_map import TemporalMap
from tempo.world_models import pool_grid


class TemporalDistanceCost(nn.Module):
    """scale * ||F(z_hat_{t+H}) - F(G)||^2 on the terminal predicted latent.

    Single-latent models (LeWM, PLDM): z_hat = predicted_emb[:, :, -1], G = goal_emb[:, -1].
    DINO-WM: the latent is a patch grid that also carries the tiled action embedding after a rollout, so the vector F_phi was
    trained on, [patch-mean pixel embedding, extra embeddings], is rebuilt from the per-source predictions and goal embeddings.
    `fmap` may be updated online; a frozen copy on the candidates' device is refreshed whenever `version()` changes."""

    def __init__(self, fmap: TemporalMap, scale: float, dino_keys: Optional[Sequence[str]] = None, version=lambda: 0):
        super().__init__()
        self.fmap, self.scale = fmap, float(scale)
        self.dino_keys = list(dino_keys) if dino_keys is not None else None
        self.version = version
        self._copy, self._copy_key = None, None

    def _map_on(self, device) -> TemporalMap:
        if next(self.fmap.parameters()).device == device:
            return self.fmap
        key = (str(device), self.version())
        if self._copy_key != key:
            self._copy, self._copy_key = copy.deepcopy(self.fmap).to(device).eval(), key
        return self._copy

    def _terminal_and_goal(self, info: dict):
        if self.dino_keys is None:
            return info["predicted_emb"][:, :, -1].float(), info["goal_emb"][:, -1].float()
        z = [pool_grid(info["predicted_pixels_emb"][:, :, -1].float())]
        z += [info[f"predicted_{k}_emb"][:, :, -1].float() for k in self.dino_keys]
        g = [pool_grid(info["pixels_goal_emb"][:, -1].float())]
        g += [info[f"{k}_goal_emb"][:, -1].float() for k in self.dino_keys]
        return torch.cat(z, -1), torch.cat(g, -1)

    def forward(self, info: dict) -> torch.Tensor:
        z_T, goal = self._terminal_and_goal(info)  # (B, S, D), (B, D)
        fmap = self._map_on(z_T.device)
        with torch.no_grad():
            d2 = ((fmap(z_T) - fmap(goal.to(z_T.device))[:, None]) ** 2).sum(-1)
        return self.scale * d2


def dinowm_own_cost(keys: Sequence[str] = ()) -> WeightedSum:
    """DINO-WM's published planning cost: mean-MSE on the predicted pixel embedding plus mean-MSE on every extra source.
    The reduction must be 'mean' (GoalMSE defaults to 'sum', which makes the 256x384 pixel grid dominate any extra source)."""
    terms = [(1.0, GoalMSE(pred_key="predicted_pixels_emb", goal_key="pixels_goal_emb", reduction="mean"))]
    terms += [(1.0, GoalMSE(pred_key=f"predicted_{k}_emb", goal_key=f"{k}_goal_emb", reduction="mean")) for k in keys]
    return WeightedSum(terms)


def make_objective(
    own: nn.Module, fmap: Optional[TemporalMap], weight: float, scale: float, dino_keys: Optional[Sequence[str]] = None, version=lambda: 0
) -> nn.Module:
    """(1 - w) * own + w * TemporalDistanceCost; returns `own` unchanged when w = 0 (the baseline planner)."""
    if weight == 0 or fmap is None:
        return own
    tempo = TemporalDistanceCost(fmap, scale, dino_keys=dino_keys, version=version)
    if weight == 1:
        return tempo
    return WeightedSum([(1.0 - weight, own), (weight, tempo)])
