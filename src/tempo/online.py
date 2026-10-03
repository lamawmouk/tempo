"""Running the planner on dataset windows, and recording its own trajectories for the online rounds of F_phi."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from tempo.buffer import LatentBuffer
from tempo.config import EnvSpec, TrainConfig
from tempo.planner import TempoPolicy
from tempo.trainer import MapTrainer


class Recorder:
    """Wraps world.envs.step / reset so every transition of a World-driven run is stored as frozen latents, and F_phi
    takes `updates_per_plan` steps after every plan and `updates_per_episode` steps after every finished episode."""

    def __init__(self, world, policy: TempoPolicy, buffer: LatentBuffer, trainer: MapTrainer, cfg: TrainConfig):
        self.world, self.policy, self.buffer, self.trainer, self.cfg = world, policy, buffer, trainer, cfg
        self._step, self._reset = world.envs.step, world.envs.reset
        world.envs.step, world.envs.reset = self.step, self.reset

    def detach(self) -> None:
        self.world.envs.step, self.world.envs.reset = self._step, self._reset

    def reset(self, seed=None, options=None, mask=None):
        out = self._reset(seed=seed, options=options, mask=mask)
        for i in range(self.world.num_envs) if mask is None else np.nonzero(mask)[0]:
            self.buffer.discard(int(i))
        return out

    def step(self, actions, mask=None):
        active = range(self.world.num_envs) if mask is None else np.nonzero(mask)[0]
        z = self.policy.last_z.numpy()
        for i in active:
            if not self.buffer.is_open(int(i)):
                self.buffer.start(int(i), z[i])
        if self.policy.replanned.any():
            self.trainer.train(self.cfg.updates_per_plan)
        out = self._step(actions, mask=mask)
        _, _, term, trunc, infos = out
        z_next = self.policy.encoder(infos).numpy()
        for i in active:
            self.buffer.append(int(i), z_next[i])
            if term[i] or trunc[i]:
                self._close(int(i))
        return out

    def _close(self, i: int) -> None:
        self.buffer.close(i)
        self.trainer.train(self.cfg.updates_per_episode)

    def close_open(self) -> None:
        for i in range(self.world.num_envs):
            if self.buffer.open_length(i) > 0:
                self._close(i)
            self.buffer.discard(i)


def run_windows(world, ds, spec: EnvSpec, episodes, starts, goal_offset: int, budget: int, recorder: Optional[Recorder] = None) -> dict:
    """Start every env from a dataset frame and plan toward the frame `goal_offset` steps later, for `budget` steps."""
    world.policy.reset_episodes()
    metrics = world.evaluate(
        dataset=ds,
        episodes_idx=[int(e) for e in episodes],
        start_steps=[int(s) for s in starts],
        goal_offset=goal_offset,
        eval_budget=budget,
        callables=spec.callables,
    )
    if recorder is not None:
        recorder.close_open()
    return metrics


def seed_everything(planner_seed: int, group: int, policy: TempoPolicy) -> None:
    policy.solver.torch_gen.manual_seed(planner_seed)
    torch.manual_seed(planner_seed + group)
    np.random.seed(planner_seed + group)
