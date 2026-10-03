"""D4RL antmaze-umaze-diverse -> stable-worldmodel lance: D4RL's ACTION SEQUENCES RE-SIMULATED in our environment, rendered from pixels.

Why re-simulation: D4RL's recorded states come from mujoco-py 2.1 with D4RL's ant.xml; even with the same
xml loaded into MuJoCo 3 (antumaze_env PHYSICS=d4rl) the recorded actions reproduce the recorded trajectory only approximately (one frameskip-5
step off by ~50 % of its displacement, recorded goal frames reachable in 15-30 % of non-trivial windows), so a dataset of RECORDED states is
scored against goals our simulator cannot produce. Here every D4RL episode is cut into --resim-chunk (100) step chunks; at each chunk start the
recorded qpos/qvel is written into OUR simulator (translated by -4 on x, y into gymnasium-robotics' frame, see antumaze_env.py), the recorded
actions are rolled open-loop and OUR states/frames are stored. Each chunk becomes one dataset episode: dynamics in the data == dynamics at eval,
and every goal frame t+k is reachable from row t by construction (deterministic simulator; see audit_antumaze_physics.py --determinism).
Chunks that START with the ant flipped (torso up-vector z < 0; D4RL's ant lies on its back in 28 % of its own steps) or that get stuck (net
displacement < 0.2 m while the recorded chunk moved > 1 m) are DROPPED; chunks that flip mid-way are kept (per-row `upright` column); all counted in <out>.meta.json together with the source episode ids (needed for a leak-free source-level split: --part train|val selects a
deterministic --val-frac of SOURCE episodes with --val-seed, so chunks of one source episode never straddle the split).
Columns as our other maze sets: state (xy), proprio (qpos[2:]+qvel, 27), qpos, qvel, goal_xy (D4RL's goal, reference only), success,
goal_distance, pixels, observation (both HWC uint8 -> JPEG), reward, terminated, truncated, action (8), id, render_time, plus
source_episode, chunk, d4rl_state (the recorded xy at the same step, for drift audits).
Sharded: --shard k --nshards N takes every N-th source episode of the part; merge with tempo.lance_utils.merge_lance_shards.
--resim-chunk 0 restores the old recorded-state rendering (audits only).

usage: convert_d4rl_antumaze.py --h5 <file> --out <dir.lance> [--part train|val|all --val-frac 0.1 --val-seed 0] [--shard 0 --nshards 8]
       [--resim-chunk 100] [--episodes-limit 5] [--image-size 224]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def episode_bounds(timeouts, terminals):
    ends = np.where(timeouts | terminals)[0]
    if len(ends) == 0 or ends[-1] != len(timeouts) - 1:
        ends = np.append(ends, len(timeouts) - 1)
    starts = np.concatenate([[0], ends[:-1] + 1])
    return list(zip(starts.tolist(), (ends + 1).tolist()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--episodes-limit", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--min-len", type=int, default=30, help="drop fragments shorter than this (a 1-step terminal episode is useless)")
    p.add_argument(
        "--resim-chunk", type=int, default=100, help="re-simulate D4RL's actions in chunks of this many steps (0 = render recorded states)"
    )
    p.add_argument("--part", choices=["all", "train", "val"], default="all")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--val-seed", type=int, default=0)
    p.add_argument("--stuck-tol", type=float, default=0.2)
    p.add_argument("--stuck-rec", type=float, default=1.0)
    a = p.parse_args()

    import gymnasium as gym
    import h5py

    import tempo.envs.ant_umaze as A
    from tempo.lance_utils import NanSafeLanceWriter

    f = h5py.File(a.h5, "r")
    n = len(f["actions"])
    bounds = episode_bounds(f["timeouts"][:], f["terminals"][:])
    bounds = [b for b in bounds if b[1] - b[0] >= a.min_len]
    # source-level split: a deterministic --val-frac of SOURCE episodes (by position in `bounds`) is the held-out part
    perm = np.random.default_rng(a.val_seed).permutation(len(bounds))
    n_val = int(round(a.val_frac * len(bounds)))
    val_src = set(perm[:n_val].tolist())
    src_ids = [i for i in range(len(bounds)) if a.part == "all" or (i in val_src) == (a.part == "val")]
    src_ids = src_ids[a.shard :: a.nshards]
    if a.episodes_limit > 0:
        src_ids = src_ids[: a.episodes_limit]
    mine = [bounds[i] for i in src_ids]
    print(
        f"[convert] {a.h5}: {n} transitions, {len(bounds)} episodes >= {a.min_len} steps; part {a.part} ({len(val_src)} val sources); "
        f"shard {a.shard}/{a.nshards} takes {len(mine)} source episodes, resim chunk {a.resim_chunk} -> {a.out}",
        flush=True,
    )

    env = gym.make("tempo/AntUMaze-v0", image_size=a.image_size).unwrapped
    env.reset(seed=0)
    qpos_all, qvel_all, act_all, goal_all = f["infos/qpos"], f["infos/qvel"], f["actions"], f["infos/goal"]

    def render_episode(s, e, ep_id):
        T = e - s
        qpos = np.asarray(qpos_all[s:e], np.float64)
        qvel = np.asarray(qvel_all[s:e], np.float64)
        act = np.asarray(act_all[s:e], np.float32)
        goal = np.asarray(goal_all[s:e], np.float64) - A.D4RL_OFFSET
        qpos[:, :2] -= A.D4RL_OFFSET
        px, obs, st, pr, gx, suc, gd, rt = [], [], [], [], [], [], [], []
        for t in range(T):
            t0 = time.perf_counter()
            env.set_state(np.concatenate([qpos[t], qvel[t]]))
            fr = env.render()
            rt.append(np.float32(time.perf_counter() - t0))
            px.append(fr)
            obs.append(fr)  # BOTH HWC uint8: the lance writer JPEG-encodes `pixels` by name but detects any other image
            # column only by shape (uint8, ndim 3, last dim 1/3); a CHW array is stored as a 150,528-float list (600 KB/row, 620 GB
            # for this set -- the 5-episode smoke caught it). swm decodes both blobs to CHW on load, as for our other mazes.
            xy = qpos[t, :2].astype(np.float32)
            st.append(xy)
            pr.append(np.concatenate([qpos[t, 2:], qvel[t]]).astype(np.float32))
            d = np.float32(np.linalg.norm(xy - goal[t]))
            gx.append(goal[t].astype(np.float32))
            gd.append(np.array([d], np.float32))
            suc.append(np.array([float(d <= A.GOAL_TOL)], np.float32))
        return {
            "state": st,
            "proprio": pr,
            "qpos": [q.astype(np.float32) for q in qpos],
            "qvel": [v.astype(np.float32) for v in qvel],
            "goal_xy": gx,
            "success": suc,
            "goal_distance": gd,
            "render_time": [np.array([r], np.float32) for r in rt],
            "pixels": px,
            "observation": obs,
            "reward": suc,
            "terminated": [np.zeros(1, np.float32)] * T,
            "truncated": [np.array([float(t == T - 1)], np.float32) for t in range(T)],
            "action": [x for x in act],
            "id": [np.array([float(ep_id)], np.float32)] * T,
        }

    def resim_chunks(s, e, src_id):
        """Roll the recorded actions of source episode [s, e) in OUR simulator, one chunk at a time, each chunk anchored at the recorded state.
        Returns (episodes, stats): episodes = list of row dicts (one per kept chunk), stats = dict(chunks, flipped, stuck)."""
        CH = a.resim_chunk
        T = e - s
        qpos = np.asarray(qpos_all[s:e], np.float64)
        qvel = np.asarray(qvel_all[s:e], np.float64)
        act = np.asarray(act_all[s:e], np.float32)
        goal = np.asarray(goal_all[s:e], np.float64) - A.D4RL_OFFSET
        qpos[:, :2] -= A.D4RL_OFFSET
        eps, stats = [], {"chunks": 0, "start_flipped": 0, "flipped_mid": 0, "stuck": 0}
        for ci, c in enumerate(range(0, T - CH + 1, CH)):
            stats["chunks"] += 1
            # D4RL's own recorded ant lies flipped (torso up-vector z < 0) in 28 % of all steps: once it flips it stays on its back for the
            # rest of the episode (28 % of episodes are > 50 % flipped). A chunk that STARTS flipped is 100 frames of a dead ant -> dropped
            # (counted as start_flipped). Chunks that flip mid-way are KEPT (our sim flips in 7 % of 100-step chunks from upright starts,
            # D4RL's own trajectories in 8 %: flipping is part of the dynamics the world model must learn) and counted as flipped_mid.
            up0 = 1.0 - 2.0 * (qpos[c, 4] ** 2 + qpos[c, 5] ** 2)
            if up0 < 0.0:
                stats["start_flipped"] += 1
                continue
            env.set_state(np.concatenate([qpos[c], qvel[c]]))
            px, obs, st, pr, qp, qv, gx, suc, gd, rt, rec, upr = [], [], [], [], [], [], [], [], [], [], [], []
            flipped = False
            for t in range(CH):
                t0 = time.perf_counter()
                fr = env.render()
                rt.append(np.float32(time.perf_counter() - t0))
                full = env.get_state().astype(np.float64)  # OUR qpos (15) + qvel (14)
                xy = full[:2].astype(np.float32)
                px.append(fr)
                obs.append(fr)
                st.append(xy)
                pr.append(full[2:].astype(np.float32))
                qp.append(full[:15].astype(np.float32))
                qv.append(full[15:].astype(np.float32))
                rec.append(qpos[c + t, :2].astype(np.float32))
                d = np.float32(np.linalg.norm(xy - goal[c + t]))
                gx.append(goal[c + t].astype(np.float32))
                gd.append(np.array([d], np.float32))
                suc.append(np.array([float(d <= A.GOAL_TOL)], np.float32))
                upr.append(np.array([float(1.0 - 2.0 * (full[4] ** 2 + full[5] ** 2) >= 0.0)], np.float32))  # 1 = torso upright at this row
                env.step(act[c + t])
                w_, x_, y_, _ = env.ant.data.qpos[3:7]
                if 1.0 - 2.0 * (x_ * x_ + y_ * y_) < 0.0:  # torso up-vector points down: the ant flipped
                    flipped = True
            net = float(np.linalg.norm(st[-1].astype(np.float64) - st[0].astype(np.float64)))
            rec_net = float(np.linalg.norm(qpos[c + CH - 1, :2] - qpos[c, :2]))
            if flipped:
                stats["flipped_mid"] += 1  # kept (see above)
            if net < a.stuck_tol and rec_net > a.stuck_rec:
                stats["stuck"] += 1
                continue
            ep_id = src_id * 16 + ci
            eps.append(
                {
                    "state": st,
                    "proprio": pr,
                    "qpos": qp,
                    "qvel": qv,
                    "d4rl_state": rec,
                    "upright": upr,
                    "goal_xy": gx,
                    "success": suc,
                    "goal_distance": gd,
                    "render_time": [np.array([r], np.float32) for r in rt],
                    "pixels": px,
                    "observation": obs,
                    "reward": suc,
                    "terminated": [np.zeros(1, np.float32)] * CH,
                    "truncated": [np.array([float(t == CH - 1)], np.float32) for t in range(CH)],
                    "action": [x for x in act[c : c + CH]],
                    "id": [np.array([float(ep_id)], np.float32)] * CH,
                    "source_episode": [np.array([float(src_id)], np.float32)] * CH,
                    "chunk": [np.array([float(ci)], np.float32)] * CH,
                }
            )
        return eps, stats

    t_start = time.time()
    frames = 0
    n_eps = 0
    tot = {"chunks": 0, "start_flipped": 0, "flipped_mid": 0, "stuck": 0}
    with NanSafeLanceWriter(a.out) as w:
        for k, ((s, e), src_id) in enumerate(zip(mine, src_ids)):
            if a.resim_chunk > 0:
                eps, stt = resim_chunks(s, e, src_id)
                for key in tot:
                    tot[key] += stt[key]
            else:
                eps = [render_episode(s, e, src_id)]
            if eps:
                w.write_episodes(eps)
            frames += sum(len(x["state"]) for x in eps)
            n_eps += len(eps)
            if k % 10 == 0 or k == len(mine) - 1:
                el = time.time() - t_start
                print(
                    f"[convert] source {k + 1}/{len(mine)} -> {n_eps} episodes, frames {frames} {frames / max(el, 1e-6):.0f} fps elapsed {el:.0f}s "
                    f"| chunks {tot['chunks']} start_flipped(dropped) {tot['start_flipped']} flipped_mid(kept) {tot['flipped_mid']} stuck(dropped) {tot['stuck']}",
                    flush=True,
                )
    env.close()
    import json

    meta = {
        "part": a.part,
        "val_frac": a.val_frac,
        "val_seed": a.val_seed,
        "shard": a.shard,
        "nshards": a.nshards,
        "resim_chunk": a.resim_chunk,
        "physics": env.physics,
        "source_episodes": src_ids,
        "episodes": n_eps,
        "frames": frames,
        **tot,
    }
    json.dump(meta, open(a.out.rstrip("/") + ".meta.json", "w"))
    print(
        f"[convert] wrote {a.out}: {frames} frames, {n_eps} episodes from {len(mine)} source episodes in {time.time() - t_start:.0f}s | {tot}",
        flush=True,
    )
    print("=== DONE exit=0 ===", flush=True)


if __name__ == "__main__":
    main()
