"""Environments: stable-worldmodel's built-in tasks plus the custom ones in this package.

tempo.envs.point_maze   tempo/PointMazeMedium-v0, tempo/PointMazeLarge-v0  (OGBench point maze from pixels)
tempo.envs.ant_umaze    tempo/AntUMaze-v0                                (D4RL Ant-U-Maze from pixels)
tempo.envs.push_obj     tempo/PushTObj-v0                                (Push-T with per-window block shapes)
tempo.envs.wall_random  tempo/WallRandom-v0, tempo/WallRandomVal-v0      (DINO-WM Wall with random layouts; needs p3_swm)
"""

from __future__ import annotations

import importlib
import inspect
from copy import deepcopy

import numpy as np

from tempo.config import EnvSpec

_CAST = {"int": int, "float": float, "bool": bool}


def _apply_callables_with_cast(env, callables, init_state):
    """swm's World._apply_callables plus a `cast` key ({'value': col, 'cast': 'int'}): Scene's button states are stored as
    float arrays but its set_state indexes with them."""
    for spec in callables:
        if not hasattr(env, spec["method"]):
            continue
        prepared = {}
        for name, data in spec.get("args", {}).items():
            if data.get("in_dataset", True):
                if data.get("value") not in init_state:
                    continue
                prepared[name] = deepcopy(init_state[data["value"]])
            else:
                prepared[name] = data.get("value")
            if data.get("cast"):
                v = np.asarray(prepared[name]).reshape(-1)
                cast = _CAST[data["cast"]]
                prepared[name] = cast(v[0]) if v.size == 1 else np.asarray(v, dtype=cast)
        getattr(env, spec["method"])(**prepared)


def _install_callable_cast() -> None:
    from stable_worldmodel.world import world as w

    if "cast" not in inspect.getsource(w._apply_callables):
        w._apply_callables = _apply_callables_with_cast


def make_world(spec: EnvSpec, num_envs: int, max_episode_steps: int):
    import stable_worldmodel as swm

    if spec.import_hook:
        importlib.import_module(spec.import_hook)
    _install_callable_cast()
    return swm.World(
        spec.env_id,
        num_envs=num_envs,
        image_shape=(224, 224),
        max_episode_steps=max_episode_steps,
        goal_conditioned=True,
        **spec.world_kwargs,
    )
