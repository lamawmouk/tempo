"""WallRandom: DINO-WM's generalisation variant of Wall (Sec. 4.5) as a stable-worldmodel env.

Wall (swm/Wall-v0, p3_swm.envs.wall_env.WallEnv) keeps ONE wall / door layout. WallRandom re-samples the layout at every reset from
the layout table DINO-WM's WallDatasetConfig generates (wall x in [wall_padding, img - wall_padding), door y in
[door_padding, img - door_padding)), split into TRAIN layouts and HELD-OUT layouts exactly as their `exclude_*_train` /
`only_*_val` codes do: a "min-max" range of wall x values (and door y values) is excluded from training and used only at test.

    tempo/WallRandom-v0        split='train'  -> layouts outside the excluded ranges (data collection, boards)
    tempo/WallRandomVal-v0     split='val'    -> ONLY the excluded ranges (unseen wall AND door positions)

Reset options (all optional): 'wall_location' / 'door_location' pin a layout, 'state' / 'goal_state' pin start and goal; without a
start the env samples start and goal on opposite sides of the door (DotWall.generate_random_state), i.e. the cross-wall task.
info adds 'wall_location', 'door_location' (scalars) so the collected dataset records the layout of every frame.

DINO-WM WallRandom protocol: 10,240 random trajectories x 50 steps on random layouts; test = random start on one side to a random
goal on the other with wall/door positions never seen in training (their Fig. / Table: DINO-WM 0.82 SR). Default held-out codes
here: wall x 28-36 (of 20..44) and door y 28-36 (of 10..54) — a centred band of each, so train and test layouts are disjoint in
BOTH coordinates while the train set still spans the whole arena.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium.envs.registration import register, registry
from p3_swm.envs.wall.data.wall import WallDatasetConfig
from p3_swm.envs.wall.envs.wall import DotWall
from p3_swm.envs.wall_env import DEFAULTS, WallEnv

HELDOUT_WALL = "28-36"
HELDOUT_DOOR = "28-36"


def _code_range(code: str):
    vals = [int(x) for x in code.split("-") if x]
    return list(range(vals[0], vals[-1] + 1)) if vals else []


def split_layouts(cfg, heldout_wall: str, heldout_door: str):
    """Train / held-out layout tables with DINO-WM's semantics (wall_utils.generate_wall_layouts): every (wall x, door y) with
    wall x in [wall_padding, img - wall_padding) and door y in [door_padding, img - door_padding); a layout is HELD OUT when
    BOTH its wall x is in `heldout_wall` and its door y is in `heldout_door` (their exclude_*_train / only_*_val codes), train = the rest."""
    hw, hd = set(_code_range(heldout_wall)), set(_code_range(heldout_door))
    train, val = {}, {}
    for w in range(cfg.wall_padding, cfg.img_size - cfg.wall_padding):
        for d in range(cfg.door_padding, cfg.img_size - cfg.door_padding):
            lay = {"type": "v", "wall_pos": w, "door_pos": d}
            (val if (w in hw and d in hd) else train)[f"v_wall{w}_door{d}"] = lay
    return train, val


class WallRandomEnv(WallEnv):
    ACTION_SCALE = 2.0  # PINNED: DotWall (DINO-WM's released planning env) moves the dot by action*2; the WallRandom datasets
    #                     were recorded at this scale, so evaluation must use it whatever WallEnv's default becomes.

    def __init__(
        self,
        render_mode: str = "rgb_array",
        split: str = "train",
        heldout_wall: str = HELDOUT_WALL,
        heldout_door: str = HELDOUT_DOOR,
        success_dist: float = 4.5,
        action_scale: float | None = None,
        **kwargs,
    ):
        # build the fixed-layout parent first (sets spaces, rng, dot), then swap in the multi-layout DotWall
        super().__init__(
            render_mode=render_mode,
            success_dist=success_dist,
            action_scale=self.ACTION_SCALE if action_scale is None else action_scale,
            **kwargs,
        )
        assert split in ("train", "val"), split
        self.split = split
        cfg = dict(DEFAULTS)
        cfg.update(
            fix_wall=False,
            fix_wall_location=None,
            fix_door_location=None,
            exclude_wall_train=heldout_wall,
            exclude_door_train=heldout_door,
            only_wall_val=heldout_wall,
            only_door_val=heldout_door,
        )
        self.cfg = WallDatasetConfig(**cfg)
        train_layouts, val_layouts = split_layouts(self.cfg, heldout_wall, heldout_door)
        self.layouts = train_layouts if split == "train" else val_layouts
        assert self.layouts, f"no {split} layouts for heldout wall {heldout_wall} / door {heldout_door}"
        self.dot = DotWall(np.random.default_rng(0), self.cfg, fix_wall=False, cross_wall=False, device="cpu")
        self.dot.layouts = self.layouts
        self.env_name = "WallRandom" if split == "train" else "WallRandomVal"
        self._resample_layout(np.random.default_rng(0))
        self.dot.reset(location=torch.zeros(2))

    # ---------------- layouts ----------------
    def _resample_layout(self, rng: np.random.Generator) -> None:
        code = list(self.layouts.keys())[int(rng.integers(len(self.layouts)))]
        lay = self.layouts[code]
        self.dot.wall_x = torch.tensor(int(lay["wall_pos"]))
        self.dot.hole_y = torch.tensor(int(lay["door_pos"]))
        self.dot.left_wall_x = self.dot.wall_x - self.cfg.wall_width // 2
        self.dot.right_wall_x = self.dot.wall_x + self.cfg.wall_width // 2
        self.dot.wall_img = self.dot._render_walls(self.dot.wall_x, self.dot.hole_y)
        self._goal_img = None

    def _set_layout(self, wall_location=None, door_location=None):
        """Pin a layout (dataset replay / evaluation trials); values outside the split's table are allowed."""
        if wall_location is not None:
            self.dot.wall_x = torch.tensor(int(round(float(np.asarray(wall_location).reshape(-1)[0]))))
        if door_location is not None:
            self.dot.hole_y = torch.tensor(int(round(float(np.asarray(door_location).reshape(-1)[0]))))
        self.dot.left_wall_x = self.dot.wall_x - self.cfg.wall_width // 2
        self.dot.right_wall_x = self.dot.wall_x + self.cfg.wall_width // 2
        self.dot.wall_img = self.dot._render_walls(self.dot.wall_x, self.dot.hole_y)
        self._goal_img = None

    # ---------------- gym API ----------------
    def reset(self, seed=None, options=None):
        options = options or {}
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if "wall_location" in options or "door_location" in options:
            self._set_layout(options.get("wall_location"), options.get("door_location"))
        else:
            self._resample_layout(self._rng)
        # WallEnv.reset re-renders the wall image from dot.wall_x / hole_y and samples a cross-wall start / goal unless given
        opts = {k: v for k, v in options.items() if k not in ("wall_location", "door_location")}
        return super().reset(seed=None, options=opts)

    def _get_info(self):
        info = super()._get_info()
        info["wall_location"] = np.array([float(self.dot.wall_x)], dtype=np.float32)
        info["door_location"] = np.array([float(self.dot.hole_y)], dtype=np.float32)
        return info


class RandomWalkPolicy:
    """DINO-WM's Wall data-collection actions (WallDatasetConfig): a heading that drifts by `angle_noise` rad per step and a step
    length ~ N(step_mean, step_std) clipped to [lower, upper]; the released wall_single.lance has |action| = 1.00 on average, and
    WallEnv passes the action to DotWall unchanged, so the action IS the per-step displacement."""

    def __init__(
        self,
        seed: int = 0,
        angle_noise: float = 0.2,
        step_mean: float = 1.0,
        step_std: float = 0.4,
        lower: float = 0.2,
        upper: float = 1.8,
        action_scale: float = 2.0,
    ):
        self.rng = np.random.default_rng(seed)
        self.angle_noise, self.step_mean, self.step_std, self.lower, self.upper, self.action_scale = (
            angle_noise,
            step_mean,
            step_std,
            lower,
            upper,
            action_scale,
        )
        self.heading = None

    def set_env(self, env):
        self.env = env

    def reset_headings(self, n: int) -> None:
        self.heading = self.rng.uniform(0, 2 * np.pi, size=n)

    def get_action(self, info_dict, **kwargs):
        n = int(np.asarray(info_dict["pixels"]).shape[0]) if "pixels" in info_dict else self.env.num_envs
        if self.heading is None or len(self.heading) != n:
            self.reset_headings(n)
        self.heading = self.heading + self.rng.normal(0.0, self.angle_noise, size=n)
        step = np.clip(self.rng.normal(self.step_mean, self.step_std, size=n), self.lower, self.upper)
        act = (
            np.stack([np.cos(self.heading), np.sin(self.heading)], -1) * step[:, None]
        )  # released wall data: |action| ~ 1.0 arena unit per step
        return act.astype(np.float32)


for _id, _split in (("tempo/WallRandom-v0", "train"), ("tempo/WallRandomVal-v0", "val")):
    if _id not in registry:
        register(id=_id, entry_point="tempo.envs.wall_random:WallRandomEnv", max_episode_steps=300, kwargs={"split": _split})
