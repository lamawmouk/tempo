"""Datasets, the latent cache, evaluation windows and offline training windows."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn import preprocessing

from tempo.buffer import LatentBuffer
from tempo.config import EnvSpec
from tempo.world_models import dinowm_emb_keys, dinowm_pool, image_transform, is_dinowm


def load_dataset(name: str, process_keys=()):
    import stable_worldmodel as swm

    return swm.data.load_dataset(name, cache_dir=swm.data.utils.get_cache_dir(), keys_to_cache=list(process_keys))


@dataclass
class EpisodeTable:
    ep: np.ndarray  # episode id per row
    st: np.ndarray  # step index per row
    ids: np.ndarray  # unique episode ids
    start_row: np.ndarray  # first row of each episode
    length: np.ndarray  # frames per episode


def episode_table(ds) -> EpisodeTable:
    names = set(ds.column_names) | set(getattr(ds, "_schema_names", ()))
    ep = np.asarray(ds.get_col_data("episode_idx" if "episode_idx" in names else "ep_idx")).reshape(-1).astype(np.int64)
    st = np.asarray(ds.get_col_data("step_idx")).reshape(-1).astype(np.int64)
    ids, start_row = np.unique(ep, return_index=True)
    return EpisodeTable(ep, st, ids, start_row, np.bincount(ep)[ids])


def select_eval_rows(tab: EpisodeTable, seed: int, n: int, goal_offset: int, eligible: Optional[np.ndarray] = None):
    """Start rows of n evaluation windows: rng(seed).choice over rows with step <= length - offset - 1 (and `eligible`)."""
    max_start = dict(zip(tab.ids.tolist(), (tab.length - goal_offset - 1).tolist()))
    ok = tab.st <= np.array([max_start[int(e)] for e in tab.ep])
    if eligible is not None:
        ok &= np.asarray(eligible, bool).reshape(-1)
    valid = np.nonzero(ok)[0]
    rows = np.sort(valid[np.random.default_rng(seed).choice(len(valid), size=n, replace=False)])
    return tab.ep[rows], tab.st[rows], rows


def eligible_windows(ds, tab: EpisodeTable, spec: EnvSpec, goal_offset: int, dataset: str) -> Optional[np.ndarray]:
    """Per-row mask of admissible evaluation windows, identical for every method; None when the spec sets no filter.
    trivial_tol: the task state must change by more than tol over the window (otherwise it is solved at the start).
    window_flag_col: a 0/1 column that must be 1 at the start and goal rows (e.g. Ant-U `upright`).
    reach_sidecar: replaying the recorded actions from the stored start must reach the stored goal row."""
    if spec.trivial_tol is None and spec.window_flag_col is None and not spec.reach_sidecar:
        return None
    n = len(tab.ep)
    same = np.zeros(n, bool)
    same[:-goal_offset] = tab.ep[goal_offset:] == tab.ep[:-goal_offset]
    idx = np.nonzero(same)[0]
    mask = np.zeros(n, bool)
    mask[idx] = True
    if spec.trivial_tol is not None:
        st = np.asarray(ds.get_col_data(spec.state_col), np.float64).reshape(n, -1)
        if spec.state_dims is not None:
            st = st[:, list(spec.state_dims)]
        mask[idx] &= np.linalg.norm(st[idx + goal_offset] - st[idx], axis=1) > float(spec.trivial_tol)
    if spec.window_flag_col is not None:
        flag = np.asarray(ds.get_col_data(spec.window_flag_col), np.float64).reshape(n, -1)[:, 0] > 0.5
        mask[idx] &= flag[idx] & flag[idx + goal_offset]
    if spec.reach_sidecar:
        import stable_worldmodel as swm

        base = os.path.join(swm.data.utils.get_cache_dir(), "datasets", dataset)
        path = next((c for c in (base + ".reach.npz", os.path.realpath(base) + ".reach.npz") if os.path.exists(c)), None)
        assert path is not None, f"reachability sidecar missing for {dataset} (scripts/data/build_antumaze_reach_mask.py)"
        z = np.load(path)
        fp = hashlib.sha1(
            np.ascontiguousarray(np.asarray(ds.get_col_data(spec.state_col), np.float32).reshape(n, -1)).tobytes()
            + tab.ep.tobytes()
            + tab.st.tobytes()
        ).hexdigest()
        assert str(z["fingerprint"]) == fp, f"{path} was computed from a different dataset; rebuild it"
        mask[idx] &= np.asarray(z[f"g{goal_offset}"], bool)[idx]
    return mask


def fit_scalers(ds, spec: EnvSpec, extra_cols=()) -> Dict[str, preprocessing.StandardScaler]:
    """Training-set StandardScalers for the planner's inputs (and goal_<col> aliases); never fit on evaluation data."""
    process = {}
    for col in list(spec.process_keys) + [c for c in extra_cols if c not in spec.process_keys]:
        data = np.asarray(ds.get_col_data(col))
        data = data.reshape(data.shape[0], -1)
        process[col] = preprocessing.StandardScaler().fit(data[~np.isnan(data).any(axis=1)])
        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


@torch.no_grad()
def encode_dataset(
    model,
    ds,
    tab: EpisodeTable,
    out_path: Path,
    device: str,
    H: int,
    process=None,
    episodes_per_chunk: int = 32,
    frame_batch: int = 256,
    limit_episodes: Optional[int] = None,
) -> Path:
    """Frozen latent of every training frame -> npz (z fp16, ep, st). For DINO-WM also `own_cost_H`, the median of its own
    planning cost between frames H apart, which replaces d_H^2 as the scale of the TEMPO term."""
    if out_path.exists():
        return out_path
    tf = image_transform()
    ids = tab.ids if limit_episodes is None else tab.ids[:limit_episodes]
    lens = tab.length[: len(ids)]
    dino = is_dinowm(model)
    keys = dinowm_emb_keys(model) if dino else []
    Z, EP, ST, own = [], [], [], []
    t0 = time.time()
    for c in range(0, len(ids), episodes_per_chunk):
        cid, cl = ids[c : c + episodes_per_chunk], lens[c : c + episodes_per_chunk]
        for eid, ch in zip(cid, ds.load_chunk(np.asarray(cid), np.zeros(len(cid), int), np.asarray(cl))):
            px, n = ch["pixels"], ch["pixels"].shape[0]
            zs, pix = [], []
            for b in range(0, n, frame_batch):
                info = {"pixels": torch.stack([tf(f) for f in px[b : b + frame_batch]]).to(device)[:, None]}
                if not dino:
                    zs.append(model.encode(info)["emb"][:, 0].float().cpu())
                    continue
                for k in keys:
                    v = np.asarray(ch[k][b : b + frame_batch], np.float32).reshape(len(info["pixels"]), -1)
                    info[k] = torch.as_tensor(process[k].transform(v), dtype=torch.float32, device=device)[:, None]
                out = model.encode(info, emb_keys=keys)
                zs.append(dinowm_pool(out, keys)[:, 0].cpu())
                pix.append({s: out[f"{s}_emb"][:, 0].float() for s in ["pixels", *keys]})
            Z.append(torch.cat(zs).numpy().astype(np.float16))
            EP.append(np.full(n, int(eid)))
            ST.append(np.arange(n))
            if dino and n > H:  # DINO-WM's own cost: mean-MSE per source, summed over sources
                a = torch.arange(n - H)
                cost = 0
                for s in ["pixels", *keys]:
                    e = torch.cat([p[s] for p in pix])
                    cost = cost + ((e[a + H] - e[a]) ** 2).flatten(1).mean(1)
                own.append(cost.cpu())
        print(f"[encode] {c + len(cid)}/{len(ids)} episodes, {time.time() - t0:.0f}s", flush=True)
    extra = {"own_cost_H": np.array(float(torch.cat(own).median()))} if own else {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, z=np.concatenate(Z), ep=np.concatenate(EP), st=np.concatenate(ST), **extra)
    os.replace(tmp, out_path)
    return out_path


def median_displacement(z: np.ndarray, ep: np.ndarray, k: int, rng: np.random.Generator, n: int = 20_000) -> float:
    """Median ||z_{t+k} - z_t|| over random same-episode pairs (d_H for k = H)."""
    idx = rng.integers(0, len(z) - k - 1, size=n)
    idx = idx[ep[idx] == ep[idx + k]]
    return float(np.median(np.linalg.norm(z[idx + k].astype(np.float32) - z[idx].astype(np.float32), axis=1)))


def add_offline_windows(
    buffer: LatentBuffer,
    z: np.ndarray,
    ep: np.ndarray,
    n_windows: int,
    length: int,
    rng: np.random.Generator,
    exclude_rows=(),
    exclude_episodes=(),
) -> int:
    """Expert windows z[t0 : t0 + length + 1] of the training data. `exclude_rows` drops the evaluation START rows (the
    standard protocol); `exclude_episodes` drops whole episodes (held-out protocol)."""
    ids, first = np.unique(ep, return_index=True)
    lengths = np.bincount(ep)[ids]
    ok = (lengths > length + 1) & ~np.isin(ids, np.asarray(list(exclude_episodes), np.int64))
    first, lengths = first[ok], lengths[ok]
    assert len(first) > 0, "no episodes left for the offline windows"
    excl = set(int(r) for r in exclude_rows)
    added = 0
    while added < n_windows:
        k = int(rng.integers(0, len(first)))
        row = int(first[k]) + int(rng.integers(0, lengths[k] - length))
        if row in excl:
            continue
        buffer.add_episode(z[row : row + length + 1].astype(np.float32))
        added += 1
    return added
