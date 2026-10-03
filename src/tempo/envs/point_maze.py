"""OGBench point maze as a goal-conditioned pixel environment for TEMPO.

Wraps stable-worldmodel's `MazeEnv` (`swm/OGBMaze-v0`, loco 'point', ob_type 'pixels'); the agent is a 2-D point mass whose action
(2-d in [-1, 1]) is added to its position (0.2 units per unit action). Observation = the rendered frame; the goal is NOT drawn in the
pixels (OGBench convention: the goal is given as an image), so a goal frame is just the frame at the goal position.

  * infos per step: state = xy (2), proprio = xy, qpos / qvel (the MuJoCo state, 2 + 2), goal_xy, success;
  * reset(options={'init_state': xy, 'goal_state': xy}) replays a dataset window (World._evaluate_from_dataset via
    reset_options_from_dataset); the episode is `terminated` (= success) when ||xy - goal|| <= goal_tol (OGBench point tol 1.0);
    without a goal_state (data collection) the episode never terminates on its own: the expert re-samples a goal cell on arrival;
  * goal_rendering(goal_state) renders the goal position without disturbing the episode.
Importing this module registers tempo/PointMazeMedium-v0 and tempo/PointMazeLarge-v0 (EnvSpec import_hook).
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _make_platform_maze(maze_type: str, image_size: int):
    import inspect

    from stable_worldmodel.envs.ogbench.maze_env import MazeEnv

    kw = dict(loco_env_type="point", maze_env_type="maze", maze_type=maze_type, ob_type="pixels", terminate_at_goal=False)
    sig = inspect.signature(MazeEnv.__init__).parameters
    for k, v in (("width", image_size), ("height", image_size), ("render_mode", "rgb_array")):
        if k in sig or "kwargs" in sig:
            kw[k] = v
    try:
        return MazeEnv(**kw)
    except TypeError:  # constructor without render kwargs
        for k in ("width", "height", "render_mode"):
            kw.pop(k, None)
        return MazeEnv(**kw)


class PointMazeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self, maze_type: str = "medium", image_size: int = 224, goal_tol: Optional[float] = None, render_mode: str = "rgb_array", **_
    ):
        super().__init__()
        self.maze_type, self.image_size, self.render_mode = maze_type, int(image_size), render_mode
        self.env = _make_platform_maze(maze_type, self.image_size)
        self.inner = self.env.env.unwrapped  # OGBench maze env: get_xy / set_xy / set_goal / cur_goal_xy / get_oracle_subgoal / maze_map
        self.goal_tol = float(goal_tol) if goal_tol is not None else float(getattr(self.inner, "_goal_tol", 1.0))
        # observation = the frame CHANNELS-FIRST: swm's dataset evaluation injects the dataset's decoded 'observation' column (CHW)
        # into the live info buffers, so the live observation must have the same layout; 'pixels' (HWC, from render) is handled by swm.
        self.observation_space = spaces.Box(0, 255, (3, self.image_size, self.image_size), np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._goal: Optional[np.ndarray] = None  # evaluation goal (dataset window); None during collection
        self._steps = 0
        self._rng = np.random.default_rng(0)
        self._free_ij = [
            (i, j)
            for i in range(self.inner.maze_map.shape[0])
            for j in range(self.inner.maze_map.shape[1])
            if self.inner.maze_map[i, j] == 0
        ]

    # ------------------------------------------------------------------ state helpers
    def xy(self) -> np.ndarray:
        return np.asarray(self.inner.get_xy(), np.float32).reshape(-1)[:2]

    def get_state(self) -> np.ndarray:
        return np.concatenate([np.asarray(self.inner.data.qpos, np.float32).ravel(), np.asarray(self.inner.data.qvel, np.float32).ravel()])

    def set_state(self, vec: np.ndarray) -> None:
        vec = np.asarray(vec, np.float64).reshape(-1)
        nq = self.inner.model.nq
        if vec.size >= nq + self.inner.model.nv:
            self.inner.set_state(vec[:nq], vec[nq : nq + self.inner.model.nv])
        else:  # xy only
            self.inner.set_xy(vec[:2])

    def set_goal_state(self, vec: Optional[np.ndarray]) -> None:
        if vec is None:
            self._goal = None
            return
        g = np.asarray(vec, np.float64).reshape(-1)[:2]
        self.inner.set_goal(goal_xy=g)
        self._goal = g.astype(np.float32)

    def goal_xy(self) -> np.ndarray:
        return np.asarray(self.inner.cur_goal_xy, np.float32).reshape(-1)[:2]

    def goal_distance(self) -> float:
        return float(np.linalg.norm(self.xy() - self.goal_xy()))

    def goal_success(self) -> bool:
        return self.goal_distance() <= self.goal_tol

    def _ensure_renderer(self) -> None:
        """OGBench's pixel maze owns a mujoco.Renderer whose EGL context is bound to the thread that created it; swm's lance writer steps
        the world from a background thread, and the inner env renders inside its own reset()/step(), so a renderer created in another
        thread raises EGL_BAD_ACCESS. Drop it whenever the calling thread changes; get_ob() re-creates it lazily in the current thread."""
        import threading

        tid = threading.get_ident()
        if getattr(self, "_render_tid", None) != tid:
            self.inner.custom_renderer = None
            self._render_tid = tid

    def _resize(self, img) -> np.ndarray:
        img = np.asarray(img)
        if img.ndim == 3 and (img.shape[0] != self.image_size or img.shape[1] != self.image_size):
            from PIL import Image

            img = np.array(Image.fromarray(img.astype(np.uint8)).resize((self.image_size, self.image_size), Image.BILINEAR))
        return img.astype(np.uint8)

    @staticmethod
    def _chw(img: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.transpose(img, (2, 0, 1)))

    def _frame(self) -> np.ndarray:
        self._ensure_renderer()
        return self._resize(self.inner.get_ob() if getattr(self.inner, "_ob_type", "pixels") == "pixels" else self.inner.render())

    def _info(self, terminated: bool) -> dict:
        xy = self.xy()
        return {
            "state": xy.copy(),
            "proprio": xy.copy(),
            "qpos": np.asarray(self.inner.data.qpos, np.float32).ravel().copy(),
            "qvel": np.asarray(self.inner.data.qvel, np.float32).ravel().copy(),
            "goal_xy": self.goal_xy().copy(),
            "success": bool(terminated),
            "goal_distance": np.float32(self.goal_distance()),
        }

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        options = dict(options or {})
        goal = options.pop("goal_state", None)
        init = options.pop("init_state", None)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._ensure_renderer()
        obs, _ = self.env.reset(seed=seed)  # OGBench: random task (init cell / goal cell) with position noise; renders internally
        if init is not None:
            self.set_state(init)
        self.set_goal_state(goal)
        self._steps = 0
        frame = self._frame() if init is not None else self._resize(obs)
        return self._chw(frame), self._info(False)

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32).reshape(-1)[:2], -1.0, 1.0)
        self._ensure_renderer()
        obs, _, _, _, _ = self.env.step(a)  # renders internally (ob_type pixels)
        self._steps += 1
        terminated = self.goal_success() if self._goal is not None else False
        return self._chw(self._resize(obs)), float(terminated), bool(terminated), False, self._info(terminated)

    def render(self):
        return self._frame()

    def close(self):
        self.env.close()

    # ------------------------------------------------------------------ swm protocol hooks (World._evaluate_from_dataset)
    def reset_options_from_dataset(self, init_row: dict, goal_row: dict) -> dict:
        qpos, qvel = init_row.get("qpos"), init_row.get("qvel")
        init = (
            np.concatenate([np.asarray(qpos, np.float32).ravel(), np.asarray(qvel, np.float32).ravel()])
            if qpos is not None and qvel is not None
            else np.asarray(init_row["state"], np.float32).ravel()
        )
        opts = {"init_state": init}
        g = goal_row.get("goal_state", goal_row.get("goal_proprio"))
        if g is not None:
            opts["goal_state"] = np.asarray(g, np.float32).reshape(-1)[:2]
        return opts

    def goal_rendering(self, goal_state: np.ndarray) -> np.ndarray:
        cur = self.get_state()
        try:
            self.inner.set_xy(np.asarray(goal_state, np.float64).reshape(-1)[:2])
            return self._frame()
        finally:
            self.set_state(cur)

    # ------------------------------------------------------------------ scripted expert (data collection)
    def resample_goal(self) -> None:
        """New random free cell as the collection goal (the OGBench 'navigate' data-generation behaviour)."""
        cur = self.inner.xy_to_ij(self.xy())
        for _ in range(50):
            ij = self._free_ij[int(self._rng.integers(len(self._free_ij)))]
            if tuple(ij) != tuple(cur):
                break
        self.inner.set_goal(goal_ij=ij)

    def expert_action(self, noise: float = 0.15, p_random: float = 0.02) -> np.ndarray:
        """Oracle waypoint controller: BFS subgoal from OGBench, full-speed P-step towards it, Gaussian noise; re-samples the goal on arrival."""
        xy = self.xy()
        if self._goal is None and self.goal_distance() <= 0.8 * self.goal_tol:
            self.resample_goal()
        sub, _ = self.inner.get_oracle_subgoal(xy, self.goal_xy())
        a = (np.asarray(sub, np.float32) - xy) / 0.2
        a = np.clip(a + self._rng.normal(0.0, noise, 2), -1.0, 1.0)
        if self._rng.random() < p_random:
            a = self._rng.uniform(-1.0, 1.0, 2)
        return a.astype(np.float32)


MAZE_IDS = {"tempo/PointMazeMedium-v0": "medium", "tempo/PointMazeLarge-v0": "large"}


def register_maze_envs() -> None:
    for sid, mt in MAZE_IDS.items():
        if sid not in gym.registry:
            gym.register(id=sid, entry_point="tempo.envs.point_maze:PointMazeEnv", kwargs={"maze_type": mt})


register_maze_envs()
