"""Small helpers around swm's lance writer.

NanSafeLanceWriter: swm's EverythingToInfoWrapper reports a NaN-filled placeholder action on the reset frame and World.collect
rotates it to the last row of every episode; lancedb >= 0.38 refuses to store a float vector column containing NaN. The wrapper
replaces NaN actions (and any NaN in other float vector columns) with 0 before handing the episode to the real writer — the
placeholder row has no successor frame, so no transition is affected.
merge_lance_shards: concatenate sharded collections into one dataset with re-indexed, unique episode_idx.
frame_stats / frames_ok: constant or dark frames mean a broken renderer, never a valid dataset.
"""

from __future__ import annotations

import hashlib
import io
from typing import Iterable, List, Sequence

import numpy as np


class NanSafeLanceWriter:
    def __init__(self, path, fmt: str = "lance", nan_cols: Iterable[str] = ("action",)):
        from stable_worldmodel.data.format import get_format

        self._cm = get_format(fmt).open_writer(str(path))
        self._w = None
        self.nan_cols = tuple(nan_cols)
        self.n_fixed = 0

    def __enter__(self):
        self._w = self._cm.__enter__()
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)

    def _fix(self, ep: dict) -> dict:
        for col in self.nan_cols:
            if col in ep:
                fixed = []
                for a in ep[col]:
                    arr = np.asarray(a, np.float32)
                    if np.isnan(arr).any():
                        arr = np.nan_to_num(arr, nan=0.0)
                        self.n_fixed += 1
                    fixed.append(arr)
                ep[col] = fixed
        return ep

    def write_episodes(self, episodes) -> None:
        assert self._w is not None, "use inside a `with` block"
        self._w.write_episodes(self._fix(ep) for ep in episodes)


def merge_lance_shards(
    shards: List[str], dest: str, batch_rows: int = 4096, drop_failed: bool = True, drop_unsuccessful: bool = False
) -> dict:
    """Append every shard's rows to `dest`; episodes with any expert_failed=True row are dropped when drop_failed, episodes whose
    task_success never became true (timeouts) when drop_unsuccessful. The KEPT episodes are renumbered contiguously 0..K-1 across shards:
    swm's LanceDataset indexes its per-episode offsets by episode_idx, so gaps (dropped episodes) break load_chunk with an IndexError
    (seen on the first Obstruction3D merge: max id 17995 for 5,238 episodes). Returns dict(episodes=kept, dropped=n_failed, dropped_unsuccessful=n_timeouts)."""
    import lance
    import pyarrow as pa
    import pyarrow.compute as pc

    offset, kept, dropped, dropped_unsucc = 0, 0, 0, 0  # offset = number of episodes already written (= next contiguous id)
    first = True
    for s in shards:
        ds = lance.dataset(str(s))
        ep = ds.to_table(columns=["episode_idx"]).column(0).to_numpy()
        bad = set()
        if drop_failed and "expert_failed" in ds.schema.names:
            t = ds.to_table(columns=["episode_idx", "expert_failed"]).to_pandas()
            t["expert_failed"] = t["expert_failed"].map(lambda v: bool(np.asarray(v).reshape(-1)[0]))
            bad = set(int(e) for e in t.loc[t["expert_failed"], "episode_idx"].unique())
        if drop_unsuccessful and "task_success" in ds.schema.names:
            t = ds.to_table(columns=["episode_idx", "task_success"]).to_pandas()
            t["task_success"] = t["task_success"].map(lambda v: bool(np.asarray(v).reshape(-1)[0]))
            ok = t.groupby("episode_idx")["task_success"].max()
            unsucc = set(int(e) for e in ok.index[~ok.values]) - bad
            dropped_unsucc += len(unsucc)
            bad |= unsucc
        keep_ids = sorted(set(np.unique(ep).tolist()) - bad)
        remap = {int(e): offset + i for i, e in enumerate(keep_ids)}  # contiguous new ids for this shard's kept episodes
        for batch in ds.to_batches(batch_size=batch_rows):
            tbl = pa.Table.from_batches([batch])
            if bad:
                mask = pc.invert(
                    pc.is_in(tbl.column("episode_idx"), value_set=pa.array(sorted(bad), type=tbl.schema.field("episode_idx").type))
                )
                tbl = tbl.filter(mask)
                if tbl.num_rows == 0:
                    continue
            idx = tbl.schema.get_field_index("episode_idx")
            old_ids = tbl.column("episode_idx").to_numpy()
            new = pa.array(np.array([remap[int(e)] for e in old_ids]), type=tbl.schema.field("episode_idx").type)
            tbl = tbl.set_column(idx, "episode_idx", new)
            lance.write_dataset(tbl, str(dest), mode="overwrite" if first else "append")
            first = False
        offset += len(keep_ids)
        kept += len(keep_ids)
        dropped += len(bad)
    return dict(episodes=kept, dropped=dropped - dropped_unsucc, dropped_unsuccessful=dropped_unsucc)


# ---------------------------------------------------------------------------------------------------- frame checks


def frame_stats(frames: Sequence) -> dict:
    """frames: JPEG/PNG bytes or HWC uint8 arrays. Returns distinct count, mean intensity, mean encoded size (KB, bytes inputs only)."""
    from PIL import Image

    hashes, means, sizes = set(), [], []
    for b in frames:
        if isinstance(b, (bytes, bytearray)):
            hashes.add(hashlib.md5(bytes(b)).hexdigest())
            arr = np.asarray(Image.open(io.BytesIO(bytes(b))))
            sizes.append(len(b))
        else:
            arr = np.asarray(b)
            hashes.add(hashlib.md5(arr.tobytes()).hexdigest())
        means.append(float(arr.mean()))
    return dict(
        n=len(frames),
        distinct=len(hashes),
        mean=float(np.mean(means)) if means else float("nan"),
        min_mean=float(np.min(means)) if means else float("nan"),
        kb=float(np.mean(sizes)) / 1e3 if sizes else float("nan"),
    )


def frames_ok(stats: dict, min_distinct_frac: float = 0.75, min_mean: float = 10.0, min_frame_mean: float = 8.0) -> bool:
    """The darkest sampled frame must also be non-black and >= 75 % of the sample must be distinct
    (a 70 %-frozen or 75 %-black sample used to pass)."""
    return (
        stats["distinct"] >= max(3, int(stats["n"] * min_distinct_frac))
        and stats["mean"] >= min_mean
        and stats["min_mean"] >= min_frame_mean
    )


def sample_rows(n_rows: int, k: int = 60):
    return sorted(set(np.linspace(0, max(n_rows - 1, 0), min(k, max(n_rows, 1))).astype(int).tolist()))
