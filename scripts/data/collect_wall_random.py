"""Collect DINO-WM's WallRandom training data on the platform: random-walk trajectories on random wall / door layouts (train split).

    python scripts/data/collect_wall_random.py --episodes 10240 --num-envs 32 --max-steps 50 --out $STABLEWM_HOME/datasets/wall/wall_random.lance
    --split val --episodes 512 --out .../wall/wall_random_val.lance      (held-out layouts, for eval windows / sanity)
    --smoke                                                              (8 episodes, 4 envs, <name>_smoke.lance)

DINO-WM (Sec. 4.5 / App.): 10,240 random trajectories of 50 steps with randomised wall and door positions; test on unseen positions.
Columns: pixels (224x224 via the world's pixel wrapper), action, proprio, state, goal_state, wall_location, door_location.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--episodes", type=int, default=10240)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--out", default=None, help="lance path (default $STABLEWM_HOME/datasets/wall/wall_random[_val].lance)")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    import stable_worldmodel as swm

    import tempo.envs.wall_random as W  # registers tempo/WallRandom-v0 and tempo/WallRandomVal-v0

    env_id = "tempo/WallRandom-v0" if a.split == "train" else "tempo/WallRandomVal-v0"
    home = os.environ.get("STABLEWM_HOME", str(Path.home()))
    out = Path(a.out or f"{home}/datasets/wall/wall_random{'_val' if a.split == 'val' else ''}.lance")
    if a.smoke:
        out = out.with_name(out.stem + "_smoke.lance")
    episodes, num_envs = (a.episodes, a.num_envs) if not a.smoke else (8, 4)
    assert not out.exists(), f"{out} exists"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    world = swm.World(env_id, num_envs=num_envs, image_shape=(224, 224), max_episode_steps=a.max_steps, goal_conditioned=True)
    world.set_policy(W.RandomWalkPolicy(seed=a.seed))
    print(f"[collect] {env_id} episodes={episodes} envs={num_envs} steps={a.max_steps} seed={a.seed} -> {out}", flush=True)
    world.collect(str(out), episodes=episodes, seed=a.seed)
    import lance

    ds = lance.dataset(str(out))
    t = ds.to_table(columns=["episode_idx", "wall_location", "door_location"]).to_pandas()
    lay = t.groupby("episode_idx")[["wall_location", "door_location"]].first()
    print(
        f"[collect] rows {ds.count_rows()} episodes {t.episode_idx.nunique()} distinct layouts {len(lay.astype(str).drop_duplicates())} "
        f"cols {ds.schema.names} in {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
