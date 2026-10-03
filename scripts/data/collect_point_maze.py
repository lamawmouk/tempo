"""Collect the point-maze datasets with the environment's scripted expert (noisy waypoint following).

python scripts/data/collect_point_maze.py --maze medium --episodes 3000 --max-steps 200 --num-envs 8
python scripts/data/collect_point_maze.py --maze large  --episodes 3000 --max-steps 200 --num-envs 8
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

ENV_IDS = {"medium": "tempo/PointMazeMedium-v0", "large": "tempo/PointMazeLarge-v0"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maze", required=True, choices=list(ENV_IDS))
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--noise", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--out", default=os.path.join(os.environ.get("STABLEWM_HOME", str(Path.home() / ".stable_worldmodel")), "datasets"))
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    np.random.seed(a.seed)
    import lance
    import stable_worldmodel as swm

    import tempo.envs.point_maze  # noqa: F401  registers the maze ids
    from tempo.envs.scripted_expert import ScriptedExpertPolicy
    from tempo.lance_utils import NanSafeLanceWriter, frame_stats, frames_ok, sample_rows

    if a.smoke:
        a.episodes, a.num_envs, a.max_steps = 4, 2, 60
    dest = Path(a.out) / f"ogbench/pointmaze_{a.maze}_expert{'_smoke' if a.smoke else ''}.lance"
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{dest} exists (remove it to re-collect)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    world = swm.World(ENV_IDS[a.maze], num_envs=a.num_envs, max_episode_steps=a.max_steps, image_shape=(224, 224), goal_conditioned=False)
    world.set_policy(ScriptedExpertPolicy(noise=a.noise, seed=a.seed))
    world.collect(writer=NanSafeLanceWriter(dest), episodes=a.episodes, seed=a.seed)
    world.close()

    ds = lance.dataset(str(dest))
    ep = ds.to_table(columns=["episode_idx"]).column(0).to_numpy()
    _, counts = np.unique(ep, return_counts=True)
    print(f"[collect] {ds.count_rows()} rows, {len(counts)} episodes (length {counts.min()}-{counts.max()}), {time.time() - t0:.0f}s")
    frames = [r["pixels"] for r in ds.take(sample_rows(ds.count_rows()), columns=["pixels"]).to_pylist()]
    if not frames_ok(frame_stats(frames)):
        raise SystemExit("[collect] frame check failed: constant or black frames (broken renderer); do not use this dataset")
    print(f"[collect] wrote {dest}")


if __name__ == "__main__":
    main()
