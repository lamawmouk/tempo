"""swm policy that drives PointMazeEnv worlds with the environment's own scripted expert (`expert_action`).
The World marks a new episode with `_needs_flush` in the info dict (popped here, as swm's BasePolicy does); the env's phase memory is
reset by its own reset(), so nothing else is needed per episode. Records how many collection goals each env reached (`goals_reached`).
"""

from __future__ import annotations

import numpy as np
from stable_worldmodel.policy import BasePolicy


class ScriptedExpertPolicy(BasePolicy):
    def __init__(self, noise: float = 0.1, p_random: float = 0.03, seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.type = "expert"
        self.noise, self.p_random, self.seed = float(noise), float(p_random), int(seed)
        self.goals_reached = 0
        self._prev_goal = None

    def set_env(self, env) -> None:
        super().set_env(env)
        for i in range(env.num_envs):
            self._wrapper(i)._rng = np.random.default_rng(self.seed + 7919 * i)
        self._prev_goal = [None] * env.num_envs

    def _wrapper(self, i: int):
        return self.env.envs[i].unwrapped

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        info_dict.pop("_needs_flush", None)
        n = self.env.num_envs
        acts = np.zeros((n, int(np.prod(self.env.single_action_space.shape))), np.float32)
        for i in range(n):
            w = self._wrapper(i)
            g_before = tuple(np.round(np.asarray(getattr(w, "goal_xy", getattr(w, "goal_xyz", lambda: np.zeros(1)))(), np.float32), 4))
            acts[i] = w.expert_action(noise=self.noise, p_random=self.p_random)
            g_after = tuple(np.round(np.asarray(getattr(w, "goal_xy", getattr(w, "goal_xyz", lambda: np.zeros(1)))(), np.float32), 4))
            if self._prev_goal[i] is not None and g_after != g_before:
                self.goals_reached += 1
            self._prev_goal[i] = g_after
        return acts
