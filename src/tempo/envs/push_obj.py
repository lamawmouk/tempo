"""PushObj evaluation env: Push-T whose block SHAPE is restored from the dataset row of every evaluation window.

swm.World._evaluate_from_dataset prefers `env.reset_options_from_dataset(init_row, goal_row)` when the env defines it, and resets each
env with the returned options in one batched reset. PushT.reset consumes 'variation_values' (block.shape is set BEFORE the pymunk block is
built), 'state' (7-d: agent xy, block xy, block angle, agent velocity) and 'goal_state'. The PushObj shards
(scripts/data/collect_pushobj.py) carry a `block_shape` column, so windows from a held-out-shape shard evaluate on that shape.

    tempo/PushTObj-v0   registered on import (tempo.envs.push_obj); env config configs/envs/pushobj.yaml.
"""

from __future__ import annotations

import numpy as np
from gymnasium.envs.registration import register, registry
from stable_worldmodel.envs.pusht.env import PushT


class PushTObjEnv(PushT):
    def reset_options_from_dataset(self, init_row: dict, goal_row: dict) -> dict:
        opts = {"variation": ()}
        if "block_shape" in init_row:
            opts["variation_values"] = {"block.shape": int(np.asarray(init_row["block_shape"]).reshape(-1)[0])}
        if "state" in init_row:
            opts["state"] = np.asarray(init_row["state"], dtype=np.float64).reshape(-1)
        g = goal_row.get("goal_state")  # the goal frame's `state` column, remapped by _extract_init_goal
        if g is not None:
            opts["goal_state"] = np.asarray(g, dtype=np.float64).reshape(-1)
        return opts


if "tempo/PushTObj-v0" not in registry:
    register(id="tempo/PushTObj-v0", entry_point="tempo.envs.push_obj:PushTObjEnv")
