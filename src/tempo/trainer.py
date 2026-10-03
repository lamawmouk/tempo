"""Gradient updates of F_phi from the latent buffer (offline pre-training and the online rounds share one optimizer)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from tempo.buffer import LatentBuffer
from tempo.config import MapConfig
from tempo.temporal_map import TemporalMap, temporal_distance_loss


class MapTrainer:
    def __init__(self, fmap: TemporalMap, buffer: LatentBuffer, cfg: MapConfig, seed: int = 0):
        self.fmap, self.buffer, self.cfg = fmap, buffer, cfg
        self.opt = torch.optim.Adam(fmap.parameters(), lr=cfg.lr)
        self.rng = np.random.default_rng(seed)
        self.n_updates = 0

    def step(self) -> Optional[float]:
        try:
            near = self.buffer.sample_near(self.cfg.batch_size, self.cfg.max_steps, self.rng)
        except ValueError:
            return None
        try:
            far = self.buffer.sample_far(self.cfg.batch_size, self.cfg.max_steps, self.rng)
        except ValueError:
            far = None
        loss = temporal_distance_loss(self.fmap, near, far, self.cfg.max_steps, self.cfg.far_weight)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.fmap.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.n_updates += 1
        return float(loss.detach())

    def train(self, n_steps: int) -> float:
        """n_steps updates; returns the mean loss (nan when the buffer is empty)."""
        losses = [x for x in (self.step() for _ in range(n_steps)) if x is not None]
        return float(np.mean(losses)) if losses else float("nan")

    def state_dict(self) -> dict:
        return dict(fmap=self.fmap.state_dict(), opt=self.opt.state_dict(), n_updates=self.n_updates)

    def load_state_dict(self, sd: dict) -> None:
        self.fmap.load_state_dict(sd["fmap"])
        if "opt" in sd:
            self.opt.load_state_dict(sd["opt"])
        self.n_updates = int(sd.get("n_updates", 0))
