"""Collect DINO-WM's PushObj data on the platform: Push-T with the BLOCK SHAPE pinned per shard.

    python scripts/data/collect_pushobj.py --shapes T,o,square,small_tee --episodes-per-shape 5000 --num-envs 32 --max-steps 100
    python scripts/data/collect_pushobj.py --shapes L,Z,I,+ --episodes-per-shape 500 --tag test          (held-out shapes, eval windows)
    --smoke : 2 shapes x 8 episodes, 4 envs, <out>/smoke/

DINO-WM (Sec. 4.5): 20,000 random trajectories of 100 steps over four training shapes; test on unseen shapes (Tetris-like, '+');
success = agent AND block at their targets (the platform's eval_state: position error < 20 px, block angle < pi/9 modulo symmetry).
Policy: the platform's Push-T collection policy (WeakPolicy: random agent targets constrained to a box around the block).
Columns: pixels 224x224, action, pos_agent, vel_agent, block_pose, goal_*, plus `state` (7-d: agent xy, block xy, block angle,
agent velocity) and `proprio` (4-d: agent xy + velocity) added here so the shards match pusht_expert_train.lance and the harness callables
(_set_state / _set_goal_state). One lance per shape: <out>/<tag>_shape_<name>.lance; merge with merge_lance_shards.py.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np


class PushTStateInfo(gym.Wrapper):
    """Adds the 7-d `state` and 4-d `proprio` of the released Push-T dataset to every info dict."""

    def _aug(self, info):
        u = self.env.unwrapped
        pa = np.asarray(u.agent.position, dtype=np.float32)
        va = np.asarray(u.agent.velocity, dtype=np.float32)
        bp = np.array(list(u.block.position) + [u.block.angle], dtype=np.float32)
        info = dict(info)
        info["state"] = np.concatenate([pa, bp, va]).astype(np.float32)
        info["proprio"] = np.concatenate([pa, va]).astype(np.float32)
        info["block_shape"] = np.array([int(u.variation_space["block"]["shape"].value)], dtype=np.int64)
        return info

    def reset(self, **kw):
        o, i = self.env.reset(**kw)
        return o, self._aug(i)

    def step(self, a):
        o, r, t, tr, i = self.env.step(a)
        return o, r, t, tr, self._aug(i)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="T,o,square,small_tee")
    p.add_argument("--episodes-per-shape", type=int, default=5000)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--tag", default="train")
    p.add_argument("--out", default=None, help="directory (default $STABLEWM_HOME/datasets/pusht_obj)")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import stable_worldmodel as swm
    from stable_worldmodel.envs.pusht import WeakPolicy

    probe = gym.make("swm/PushT-v1").unwrapped
    shapes_all = list(probe.shapes)
    probe.close()
    home = os.environ.get("STABLEWM_HOME", str(Path.home()))
    out = Path(a.out or f"{home}/datasets/pusht_obj") / ("smoke" if a.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    shapes = [s for s in a.shapes.split(",") if s]
    if a.smoke:
        shapes = shapes[:2]
    episodes, num_envs = (a.episodes_per_shape, a.num_envs) if not a.smoke else (8, 4)
    rng = np.random.default_rng(a.seed)
    for shape in shapes:
        assert shape in shapes_all, (shape, shapes_all)
        idx = shapes_all.index(shape)
        path = out / f"{a.tag}_shape_{shape.replace('+', 'plus')}.lance"  # lance table names allow [A-Za-z0-9_.-] only
        if path.exists():
            print(f"[collect] skip {path} (exists)")
            continue
        t0 = time.time()
        world = swm.World(
            "swm/PushT-v1",
            num_envs=num_envs,
            image_shape=(224, 224),
            max_episode_steps=a.max_steps,
            goal_conditioned=True,
            extra_wrappers=[PushTStateInfo],
        )
        world.set_policy(WeakPolicy(dist_constraint=100, seed=int(rng.integers(1 << 31))))
        opts = {"variation": ("agent.start_position", "block.start_position", "block.angle"), "variation_values": {"block.shape": idx}}
        print(f"[collect] shape={shape} (idx {idx}) episodes={episodes} envs={num_envs} steps={a.max_steps} -> {path}", flush=True)
        world.collect(str(path), episodes=episodes, seed=int(rng.integers(1 << 31)), options=opts)
        import lance

        ds = lance.dataset(str(path))
        t = ds.to_table(columns=["episode_idx", "block_shape"]).to_pandas()
        print(
            f"[collect] rows {ds.count_rows()} episodes {t.episode_idx.nunique()} shapes seen {sorted(t.block_shape.astype(int).unique().tolist())} "
            f"cols {ds.schema.names} in {time.time() - t0:.0f}s",
            flush=True,
        )
        world.close()


if __name__ == "__main__":
    main()
