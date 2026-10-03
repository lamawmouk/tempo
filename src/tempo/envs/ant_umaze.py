"""D4RL Ant-U-Maze as a goal-conditioned pixel environment for TEMPO (PLDM's Ant-U setting, from pixels).

Simulator: gymnasium-robotics `AntMazeEnv(maze_map=maps.U_MAZE, xml_file=<D4RL ant.xml>)` -- the maintained port of the D4RL antmaze
model (maze_size_scaling 4, frame_skip 5) running D4RL's OWN ant model (timestep 0.02, 30 a torques; see PHYSICS below), built through the
supported `maze_map` / `xml_file` arguments, so no patching of OGBench's factory-defined class.
Rendering: AntMazeEnv's own render() segfaults under headless EGL, so frames come from a direct `mujoco.Renderer` on the
ant model with a PLDM-style top-down free camera (D4RLAntMazeDrawer: elevation -90; distance tightened from 30 to 26 so the 20-unit
maze fills the 224 px frame). The goal is NOT drawn: the goal is an image, as in our other mazes.

Coordinates: D4RL's umaze puts the free cells at x, y in {0, 4, 8} (reset (0,0), goal (0,8)); gymnasium-robotics centres the same
5x5 map at the origin with cells at {-4, 0, 4}. U_MAZE is symmetric under row reversal, so the two frames differ by a PURE
TRANSLATION of -4 on both axes -- no reflection, hence the ant's orientation, velocities and actions from the D4RL dataset stay
consistent when its states are replayed here (convert_d4rl_antumaze.py).

  * infos per step: state = xy (2), proprio = the ant's own state without xy (qpos[2:15] + qvel, 27), qpos (15), qvel (14), goal_xy,
    success, goal_distance;
  * reset(options={'init_state': qpos+qvel or xy, 'goal_state': xy}) replays a dataset window (reset_options_from_dataset);
    the episode is `terminated` (= success) when ||xy - goal|| <= goal_tol, PLDM's 0.5 (gymnasium-robotics uses 0.45);
  * goal_rendering(goal_state) renders the goal position without disturbing the episode.
Importing this module registers tempo/AntUMaze-v0 (EnvSpec import_hook).
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

GOAL_TOL = 0.5  # PLDM ant_draw.CustomMazeEnv._is_goal_reached threshold
D4RL_OFFSET = np.array([4.0, 4.0], np.float64)  # gymnasium-robotics xy = D4RL xy - D4RL_OFFSET
CAM_DISTANCE = 26.0  # PLDM used 30; 26 fills the frame at fovy 45 for a 20-unit maze
N_QPOS, N_QVEL = 15, 14
# Physics (root cause of the open-loop replay drift): D4RL's ant.xml runs timestep 0.02 x frame_skip 5 = 0.10 s per
# step with motors ctrlrange [-30, 30] (NormalizedBoxEnv maps a in [-1, 1] -> 30 a; contact solref .02 1 / solimp .8 .8 .01), while gymnasium's
# ant.xml runs 0.01 x 5 = 0.05 s with ctrlrange [-1, 1] and gear 150: the dataset transitions are 0.10 s / 30 a physics. "d4rl" (default) loads
# D4RL's ant.xml (tempo/envs/assets/d4rl_ant.xml, verbatim from the D4RL repo) through AntMazeEnv's xml_file argument and scales the
# [-1, 1] actions by 30 like NormalizedBoxEnv; "gym" keeps the gymnasium model (only for the A/B replay audit).
PHYSICS = os.environ.get("TEMPO_ANTU_PHYSICS", "d4rl")
D4RL_ANT_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "d4rl_ant.xml")
D4RL_CTRL_SCALE = 30.0


def d4rl_to_local_xy(xy) -> np.ndarray:
    return np.asarray(xy, np.float64).reshape(-1)[:2] - D4RL_OFFSET


def local_to_d4rl_xy(xy) -> np.ndarray:
    return np.asarray(xy, np.float64).reshape(-1)[:2] + D4RL_OFFSET


def _top_down_camera():
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.0]
    cam.distance, cam.elevation, cam.azimuth = CAM_DISTANCE, -90.0, 90.0
    return cam


class AntUMazeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, image_size: int = 224, goal_tol: float = GOAL_TOL, render_mode: str = "rgb_array", physics: str = PHYSICS, **_):
        super().__init__()
        from gymnasium_robotics.envs.maze import maps
        from gymnasium_robotics.envs.maze.ant_maze_v5 import AntMazeEnv

        self.image_size, self.goal_tol, self.render_mode = int(image_size), float(goal_tol), render_mode
        assert physics in ("d4rl", "gym"), physics
        self.physics = physics
        kw = {"xml_file": D4RL_ANT_XML} if physics == "d4rl" else {}
        self._ctrl_scale = D4RL_CTRL_SCALE if physics == "d4rl" else 1.0
        self.env = AntMazeEnv(maze_map=maps.U_MAZE, continuing_task=True, reset_target=False, **kw)  # no render_mode: no library renderer
        if physics == "d4rl":
            assert (
                abs(self.env.ant_env.model.opt.timestep - 0.02) < 1e-9
                and abs(float(self.env.ant_env.model.actuator_ctrlrange[0, 1]) - 30.0) < 1e-6
            ), (self.env.ant_env.model.opt.timestep, self.env.ant_env.model.actuator_ctrlrange[0])
        self.ant = self.env.ant_env
        self.maze_map = np.asarray(maps.U_MAZE, np.int64)
        assert self.ant.model.nq == N_QPOS and self.ant.model.nv == N_QVEL, (self.ant.model.nq, self.ant.model.nv)
        # observation = the frame CHANNELS-FIRST (swm injects the dataset's CHW 'observation' into live buffers); 'pixels' (HWC) via render()
        self.observation_space = spaces.Box(0, 255, (3, self.image_size, self.image_size), np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, (8,), np.float32)
        self._goal: Optional[np.ndarray] = None
        self._steps = 0
        self._rng = np.random.default_rng(0)
        self._renderers: dict = {}  # thread id -> mujoco.Renderer: an EGL context is bound to the thread that created it
        self._cam = _top_down_camera()
        import mujoco

        self._vopt = mujoco.MjvOption()
        self._vopt.sitegroup[:] = (
            0  # hide gymnasium-robotics' red "target" site: the goal is an image, and the site's position is random per reset
        )
        self._free_ij = [(i, j) for i in range(self.maze_map.shape[0]) for j in range(self.maze_map.shape[1]) if self.maze_map[i, j] == 0]

    # ------------------------------------------------------------------ state helpers
    def xy(self) -> np.ndarray:
        return np.asarray(self.ant.data.qpos[:2], np.float32).copy()

    def get_state(self) -> np.ndarray:
        return np.concatenate([np.asarray(self.ant.data.qpos, np.float32).ravel(), np.asarray(self.ant.data.qvel, np.float32).ravel()])

    def set_state(self, vec: np.ndarray) -> None:
        """Full MuJoCo state (qpos 15 + qvel 14) or xy only (keeps the current pose, zeroes velocity)."""
        vec = np.asarray(vec, np.float64).reshape(-1)
        if vec.size >= N_QPOS + N_QVEL:
            self.ant.set_state(vec[:N_QPOS], vec[N_QPOS : N_QPOS + N_QVEL])
        else:
            qpos = np.array(self.ant.data.qpos, np.float64)
            qpos[:2] = vec[:2]
            self.ant.set_state(qpos, np.zeros(N_QVEL))

    def set_goal_state(self, vec: Optional[np.ndarray]) -> None:
        if vec is None:
            self._goal = None
            return
        g = np.asarray(vec, np.float64).reshape(-1)[:2]
        self.env.goal = g.copy()
        self._goal = g.astype(np.float32)

    def goal_xy(self) -> np.ndarray:
        return np.asarray(self.env.goal if self._goal is None else self._goal, np.float32).reshape(-1)[:2]

    def goal_distance(self) -> float:
        return float(np.linalg.norm(self.xy() - self.goal_xy()))

    def goal_success(self) -> bool:
        return self.goal_distance() <= self.goal_tol

    # ------------------------------------------------------------------ rendering
    def _renderer(self):
        import mujoco

        tid = threading.get_ident()
        r = self._renderers.get(tid)
        if r is None:
            r = mujoco.Renderer(self.ant.model, height=self.image_size, width=self.image_size)
            self._renderers[tid] = r
        return r

    def _frame(self) -> np.ndarray:
        r = self._renderer()
        r.update_scene(self.ant.data, camera=self._cam, scene_option=self._vopt)
        return np.asarray(r.render(), np.uint8)

    @staticmethod
    def _chw(img: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.transpose(img, (2, 0, 1)))

    def _info(self, terminated: bool) -> dict:
        qpos = np.asarray(self.ant.data.qpos, np.float32).ravel().copy()
        qvel = np.asarray(self.ant.data.qvel, np.float32).ravel().copy()
        return {
            "state": qpos[:2].copy(),
            "proprio": np.concatenate([qpos[2:], qvel]),
            "qpos": qpos,
            "qvel": qvel,
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
        self.env.reset(seed=seed)  # random free cell for the ant, goal drawn by gymnasium-robotics
        if init is not None:
            self.set_state(init)
        self.set_goal_state(goal)
        self._steps = 0
        return self._chw(self._frame()), self._info(False)

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32).reshape(-1)[:8], -1.0, 1.0)
        self.env.step(a * self._ctrl_scale)  # frame_skip 5; d4rl physics: D4RL ant.xml + NormalizedBoxEnv's 30 a torques
        self._steps += 1
        terminated = self.goal_success() if self._goal is not None else False
        return self._chw(self._frame()), float(terminated), bool(terminated), False, self._info(terminated)

    def render(self):
        return self._frame()

    def close(self):
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:
                pass
        self._renderers.clear()
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
        g = goal_row.get("goal_state")  # (goal_proprio[:2] would be z + quat_w, never xy)
        if g is not None:
            opts["goal_state"] = np.asarray(g, np.float32).reshape(-1)[:2]
        return opts

    def goal_rendering(self, goal_state: np.ndarray) -> np.ndarray:
        """Frame of the goal: the ant standing at the goal position in its rest pose (the goal is an image, not a marker)."""
        cur = self.get_state()
        try:
            g = np.asarray(goal_state, np.float64).reshape(-1)
            if g.size >= N_QPOS + N_QVEL:
                self.set_state(g)
            else:
                qpos = np.array(self.ant.init_qpos, np.float64)
                qpos[:2] = g[:2]
                self.ant.set_state(qpos, np.zeros(N_QVEL))
            return self._frame()
        finally:
            self.set_state(cur)


def register_antumaze_env() -> None:
    if "tempo/AntUMaze-v0" not in gym.registry:
        gym.register(id="tempo/AntUMaze-v0", entry_point="tempo.envs.ant_umaze:AntUMazeEnv")


register_antumaze_env()
