"""The CEM planner with the TEMPO objective, as a stable-worldmodel policy."""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import stable_worldmodel as swm
import torch
from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
from stable_worldmodel.planning.evaluator import flat_goal_encode, split_goal_encode
from stable_worldmodel.planning.solver import CEMSolver
from stable_worldmodel.policy import WorldModelPolicy

from tempo.config import EnvSpec, PlannerConfig
from tempo.objective import dinowm_own_cost, make_objective
from tempo.temporal_map import TemporalMap
from tempo.world_models import ChunkedRollout, dinowm_emb_keys, image_transform, is_dinowm, state_encoder


class TempoPolicy(WorldModelPolicy):
    """swm's world-model policy that also exposes the latent of the current frame (`last_z`) and flags the envs that start a
    new plan (`replanned`), so the online rounds can store the planner's own trajectories for F_phi."""

    def __init__(self, solver, config, encoder: Callable[[dict], torch.Tensor], **kwargs):
        super().__init__(solver, config, **kwargs)
        self.encoder = encoder
        self.plan_steps = config.receding_horizon * config.action_block  # env steps executed per plan
        self.last_z: Optional[torch.Tensor] = None
        self.plan_time = 0.0

    def set_env(self, env):
        super().set_env(env)
        self._since = np.zeros(env.num_envs, dtype=int)
        self.replanned = np.zeros(env.num_envs, dtype=bool)

    def reset_episodes(self) -> None:
        """Drop action queues, warm starts and frame history between World runs that reuse this policy."""
        n = self.env.num_envs
        for i in range(n):
            self._action_buffer[i].clear()
        self._next_init = None
        if self._history_buffer is not None:
            self._history_buffer.reset(list(range(n)))
        self._since[:] = 0
        self.plan_time = 0.0

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        flush = info_dict.get("_needs_flush")
        if flush is not None:
            self._since[np.asarray(flush, bool)] = 0
        self.last_z = self.encoder(info_dict)
        self.replanned = self._since % self.plan_steps == 0
        self._since += 1
        t0 = time.perf_counter()
        act = super().get_action(info_dict, **kwargs)
        self.plan_time += time.perf_counter() - t0
        scaler = (self.process or {}).get("action")
        if scaler is not None and hasattr(scaler, "var_"):
            const = np.asarray(scaler.var_).reshape(-1) <= 1e-12  # dims the data never varied: CEM noise otherwise
            if const.any():
                act = np.asarray(act, np.float32).copy()
                act[..., const] = np.asarray(scaler.mean_, np.float32)[const]
        return act


def build_planner(
    model,
    spec: EnvSpec,
    cfg: PlannerConfig,
    process: dict,
    fmap: Optional[TemporalMap],
    scale: float,
    num_envs: int,
    device: str,
    seed: int,
    map_version=lambda: 0,
) -> TempoPolicy:
    """LeWM / PLDM: own cost = GoalMSE on the single latent; all envs share one CEM solve.
    DINO-WM: own cost = per-source mean MSE, one env per CEM solve (the platform default) and chunked rollouts."""
    tf = {"pixels": image_transform(), "goal": image_transform()}
    dino = is_dinowm(model)
    keys = dinowm_emb_keys(model) if dino else []
    own = dinowm_own_cost(keys) if dino else GoalMSE()
    objective = make_objective(own, fmap, cfg.cost_weight, scale, dino_keys=keys if dino else None, version=map_version)
    evaluator = ShootingCostEvaluator(
        ChunkedRollout(model) if dino else model, objective, encode_goal=split_goal_encode if dino else flat_goal_encode
    )
    solver = CEMSolver(
        evaluator,
        batch_size=1 if dino else num_envs,
        num_samples=cfg.num_samples,
        n_steps=cfg.n_steps,
        topk=cfg.topk,
        var_scale=1.0,
        device=device,
        seed=seed,
    )
    plan = swm.PlanConfig(
        horizon=cfg.horizon,
        receding_horizon=cfg.receding,
        action_block=cfg.action_block,
        history_len=cfg.history_len,
        warm_start=cfg.warm_start,
    )
    history_keys = ("pixels", *keys) if dino else spec.history_keys
    return TempoPolicy(
        solver, plan, state_encoder(model, tf["pixels"], device, process), process=process, transform=tf, history_keys=history_keys
    )
