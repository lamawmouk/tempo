"""Reachability sidecar for the Ant-U eval set: the stored anchors are float32 and contact chaos occasionally amplifies the
rounding, so replaying the recorded actions from the stored row misses the stored goal row in ~0 / 0.7 / 2 % of windows at g25/50/75. Instead of
an oracle-ceiling caveat, make it an ARM-INDEPENDENT window filter: for every row r and offset g (same episode), replay the recorded actions from the
stored anchor and record reach[g][r] = ||xy - state[r+g]|| <= GOAL_TOL. tempo.data.eligible_windows (EnvSpec.reach_sidecar) ANDs it into the eligibility
mask before the seeded draw, identical for every planner and cost.
Physics only (renderer stubbed): usage  build_antumaze_reach_mask.py <split.lance> --shard k --nshards N  (writes <split>.reach.part<k>.npz)
                                        build_antumaze_reach_mask.py <split.lance> --merge N            (writes <split>.reach.npz)"""

import argparse
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("dataset")
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshards", type=int, default=1)
p.add_argument("--merge", type=int, default=0)
p.add_argument("--offsets", default="25,50,75")
a = p.parse_args()
offs = [int(x) for x in a.offsets.split(",")]
base = a.dataset.rstrip("/")
import lance

ds = lance.dataset(a.dataset)
n = ds.count_rows()
if a.merge:
    out = {}
    for g in offs:
        m = np.zeros(n, bool)
        done = np.zeros(n, bool)
        for k in range(a.merge):
            z = np.load(f"{base}.reach.part{k}.npz")
            m[z["rows"]] = z[f"g{g}"]
            done[z["rows"]] = True
        assert done.all(), f"g{g}: {(~done).sum()} rows missing across parts"
        out[f"g{g}"] = m
        print(f"g{g}: reachable {m.sum()} / {n} rows (unreachable {(~m).sum()}; rows without a same-episode goal count as unreachable)")
    # fingerprint = sha1 of the (episode_idx, step_idx, state) columns of the file the mask was computed from: nontrivial_window_mask recomputes it,
    # so a re-rendered split10 cannot silently reuse a stale mask
    import hashlib

    t = ds.to_table(columns=["episode_idx", "step_idx", "state"]).to_pandas()
    fp = hashlib.sha1(
        np.ascontiguousarray(np.stack(t.state.values).astype(np.float32)).tobytes()
        + t.episode_idx.values.astype(np.int64).tobytes()
        + t.step_idx.values.astype(np.int64).tobytes()
    ).hexdigest()
    np.savez(f"{base}.reach.npz", n=n, offsets=np.asarray(offs), fingerprint=np.array(fp), dataset_version=np.array(int(ds.version)), **out)
    print("->", f"{base}.reach.npz", "fingerprint", fp, "lance version", ds.version)
    sys.exit(0)
import tempo.envs.ant_umaze as A

A.AntUMazeEnv._frame = lambda self: np.zeros((self.image_size, self.image_size, 3), np.uint8)
t = ds.to_table(columns=["episode_idx", "step_idx", "state", "qpos", "qvel", "action"]).to_pandas()
ep = t.episode_idx.values
st = np.stack(t.state.values).astype(np.float64)
qp = np.stack(t.qpos.values).astype(np.float64)
qv = np.stack(t.qvel.values).astype(np.float64)
ac = np.stack(t.action.values).astype(np.float32)
assert (np.diff(t.step_idx.values)[np.diff(ep) == 0] == 1).all(), "rows not step-contiguous"
eps = np.unique(ep)
mine = eps[a.shard :: a.nshards]
rows = np.nonzero(np.isin(ep, mine))[0]
env = A.AntUMazeEnv()
env.reset(seed=0)
res = {g: np.zeros(len(rows), bool) for g in offs}
G = max(offs)
for i, r in enumerate(rows):
    # EXACTLY the evaluator's path (swm World._evaluate_from_dataset -> reset(options=reset_options_from_dataset(init_row, goal_row))): a library
    # reset (mj_resetData: qacc_warmstart = 0) followed by set_state from the float32 row, so the replay here and the live eval share the same
    # solver starting state; a bare set_state on a stepped env would carry the previous window's warmstart and can diverge under contact chaos.
    env.reset(
        seed=0,
        options=env.reset_options_from_dataset(
            {"qpos": qp[r].astype(np.float32), "qvel": qv[r].astype(np.float32), "state": st[r]},
            {"goal_state": st[min(r + G, n - 1)].astype(np.float32)},
        ),
    )
    for k in range(G):
        if r + k + 1 >= n or ep[r + k + 1] != ep[r]:
            break
        env.step(ac[r + k])
        if (k + 1) in res:
            res[k + 1][i] = np.linalg.norm(env.xy().astype(np.float64) - st[r + k + 1]) <= A.GOAL_TOL
    if i % 5000 == 0:
        print(f"[reach] shard {a.shard}: {i}/{len(rows)} rows", flush=True)
np.savez(f"{base}.reach.part{a.shard}.npz", rows=rows, **{f"g{g}": res[g] for g in offs})
print(
    f"[reach] shard {a.shard}: {len(rows)} rows -> {base}.reach.part{a.shard}.npz | reachable "
    + ", ".join(f"g{g} {res[g].sum()}" for g in offs),
    flush=True,
)
