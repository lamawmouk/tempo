"""Latent trajectories for training F_phi: offline expert windows plus the planner's own online episodes.

Only the time index matters to F_phi, so an episode is just its sequence of frozen latents (T + 1, D). Rewards, actions and
success labels are never stored.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


class LatentBuffer:
    def __init__(self, latent_dim: int, capacity: int = 200_000):
        self.latent_dim, self.capacity = int(latent_dim), int(capacity)
        self.episodes: List[np.ndarray] = []
        self._open: Dict[int, List[np.ndarray]] = {}
        self._index = None  # (episode id, t, T) per stored transition, rebuilt lazily

    # ------------------------------------------------------------------ writing
    def add_episode(self, z: np.ndarray) -> None:
        z = np.asarray(z, np.float32)
        assert z.ndim == 2 and z.shape[1] == self.latent_dim, z.shape
        if len(z) < 2:
            return
        self.episodes.append(z)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)
        self._index = None

    def start(self, env: int, z0: np.ndarray) -> None:
        self._open[env] = [np.asarray(z0, np.float32)]

    def append(self, env: int, z: np.ndarray) -> None:
        self._open[env].append(np.asarray(z, np.float32))

    def is_open(self, env: int) -> bool:
        return env in self._open

    def open_length(self, env: int) -> int:
        """Transitions recorded so far in the open episode of `env` (0 when none is open)."""
        return len(self._open[env]) - 1 if env in self._open else 0

    def close(self, env: int) -> None:
        frames = self._open.pop(env, None)
        if frames is not None:
            self.add_episode(np.stack(frames))

    def discard(self, env: int) -> None:
        self._open.pop(env, None)

    # ------------------------------------------------------------------ sampling
    @property
    def n_steps(self) -> int:
        return sum(len(e) - 1 for e in self.episodes)

    def _flat(self):
        if self._index is None:
            ep = np.concatenate([np.full(len(e) - 1, i) for i, e in enumerate(self.episodes)]) if self.episodes else np.zeros(0, int)
            t = np.concatenate([np.arange(len(e) - 1) for e in self.episodes]) if self.episodes else np.zeros(0, int)
            T = np.concatenate([np.full(len(e) - 1, len(e) - 1) for e in self.episodes]) if self.episodes else np.zeros(0, int)
            self._index = (ep, t, T)
        return self._index

    def _gather(self, ep, t, delta) -> Dict[str, torch.Tensor]:
        z = np.stack([self.episodes[e][s] for e, s in zip(ep, t)])
        zf = np.stack([self.episodes[e][s + d] for e, s, d in zip(ep, t, delta)])
        return dict(z=torch.as_tensor(z), z_future=torch.as_tensor(zf), delta=torch.as_tensor(delta))

    def sample_near(self, n: int, max_steps: int, rng: np.random.Generator) -> Dict[str, torch.Tensor]:
        """Same-episode pairs (z_t, z_{t+k}) with k ~ U[1, min(max_steps, T - t)]."""
        ep, t, T = self._flat()
        if len(ep) == 0:
            raise ValueError("empty buffer")
        idx = rng.integers(0, len(ep), size=n)
        delta = rng.integers(1, np.minimum(max_steps, T[idx] - t[idx]) + 1)
        return self._gather(ep[idx], t[idx], delta)

    def sample_far(self, n: int, max_steps: int, rng: np.random.Generator) -> Dict[str, torch.Tensor]:
        """Same-episode pairs (z_t, z_{t+d}) with d > max_steps; raises ValueError when no episode is long enough."""
        ep, t, T = self._flat()
        valid = np.nonzero(T - t > max_steps + 1)[0]
        if len(valid) == 0:
            raise ValueError(f"no pairs farther than {max_steps} steps")
        idx = valid[rng.integers(0, len(valid), size=n)]
        delta = np.array([rng.integers(max_steps + 1, T_ - s + 1) for s, T_ in zip(t[idx], T[idx])])
        return self._gather(ep[idx], t[idx], delta)
