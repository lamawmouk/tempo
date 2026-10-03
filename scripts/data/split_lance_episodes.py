"""Episode-level train/val split of an swm lance dataset (DINO-WM-style world-model-level held-out evaluation).

Rows stay episode-contiguous in ascending order and episode ids are renumbered 0..N-1 inside each split, because swm's LanceDataset
derives per-episode offsets positionally from the episode_idx column. All other columns are copied unchanged (ep_idx, the float
duplicate of episode_idx, is remapped too). A JSON manifest with the original episode ids of each split is written next to the outputs.
Usage: split_lance_episodes.py <src.lance> <dest_dir> <stem> [--val-frac 0.1] [--seed 0]
"""

import argparse
import json
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("dest_dir")
p.add_argument("stem")
p.add_argument("--val-frac", type=float, default=0.1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--batch-rows", type=int, default=8192)
p.add_argument("--train-name", default="train")
p.add_argument("--val-name", default="val")
a = p.parse_args()
t0 = time.time()
ds = lance.dataset(a.src)
ep_all = ds.to_table(columns=["episode_idx"]).column(0).to_numpy()
ids = np.unique(ep_all)
assert (np.diff(ep_all) >= 0).all(), "source is not episode-contiguous ascending"
rng = np.random.default_rng(a.seed)
perm = rng.permutation(ids)
n_val = int(round(a.val_frac * len(ids)))
splits = {a.val_name: np.sort(perm[:n_val]), a.train_name: np.sort(perm[n_val:])}
dest = Path(a.dest_dir)
dest.mkdir(parents=True, exist_ok=True)
manifest = {
    "source": a.src,
    "seed": a.seed,
    "val_frac": a.val_frac,
    "num_episodes": int(len(ids)),
    "splits": {k: [int(x) for x in v] for k, v in splits.items()},
}
# episode-scoped side table ('<src>_episodes.lance': episode_idx + model_xml/ep_meta for RoboCasa replay): swm looks for '<table>_episodes.lance'
# next to each split table, with the RENUMBERED ids -> filter + remap it per split (outside the exists-skip so a re-run repairs an old set).
side_src = Path(str(a.src).rstrip("/")[: -len(".lance")] + "_episodes.lance") if str(a.src).endswith(".lance") else None
side = lance.dataset(str(side_src)).to_table() if side_src is not None and side_src.exists() else None
for name, sel in splits.items():
    out = dest / f"{a.stem}_{name}.lance"
    remap = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
    remap[sel] = np.arange(len(sel))
    if side is not None:
        sout = dest / f"{a.stem}_{name}_episodes.lance"
        if not sout.exists():
            m = pc.is_in(side.column("episode_idx"), value_set=pa.array(sel.astype(side.column("episode_idx").type.to_pandas_dtype())))
            st = side.filter(m)
            old_ids = st.column("episode_idx").to_numpy()
            new_ids = remap[old_ids]
            assert (new_ids >= 0).all()
            st = st.set_column(st.schema.get_field_index("episode_idx"), "episode_idx", pa.array(new_ids.astype(old_ids.dtype)))
            st = st.take(pa.array(np.argsort(new_ids)))
            lance.write_dataset(st, str(sout), mode="create")
            print(f"{name}: side table {st.num_rows} episodes -> {sout}", flush=True)
        assert lance.dataset(str(sout)).count_rows() == len(sel), f"side table rows != episodes for {name}"
    if out.exists():
        print(f"exists, skipping: {out}")
        continue
    sel_set = pa.array(sel.astype(np.int32))
    written = 0
    first = True
    for batch in ds.to_batches(batch_size=a.batch_rows):
        tbl = pa.Table.from_batches([batch])
        mask = pc.is_in(tbl.column("episode_idx"), value_set=sel_set)
        tbl = tbl.filter(mask)
        if tbl.num_rows == 0:
            continue
        old = tbl.column("episode_idx").to_numpy()
        new = remap[old]
        assert (new >= 0).all()
        tbl = tbl.set_column(tbl.schema.get_field_index("episode_idx"), "episode_idx", pa.array(new.astype(np.int32)))
        if "ep_idx" in tbl.schema.names:
            i = tbl.schema.get_field_index("ep_idx")
            tbl = tbl.set_column(i, "ep_idx", pa.FixedSizeListArray.from_arrays(pa.array(new.astype(np.float32)), 1))
        lance.write_dataset(tbl, str(out), mode="create" if first else "append")
        first = False
        written += tbl.num_rows
    chk = lance.dataset(str(out))
    e = chk.to_table(columns=["episode_idx"]).column(0).to_numpy()
    u = np.unique(e)
    assert (np.diff(e) >= 0).all() and (u == np.arange(len(u))).all(), "output not contiguous"
    print(f"{name}: {len(u)} episodes, {chk.count_rows()} rows -> {out} ({time.time() - t0:.0f}s)", flush=True)
(dest / f"{a.stem}_split_manifest.json").write_text(json.dumps(manifest))
print("manifest ->", dest / f"{a.stem}_split_manifest.json")
