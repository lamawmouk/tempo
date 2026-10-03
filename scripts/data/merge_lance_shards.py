"""Merge several lance shards (same schema) into one dataset with globally unique episode_idx.

    python scripts/data/merge_lance_shards.py --out $STABLEWM_HOME/datasets/pusht_obj/train.lance $STABLEWM_HOME/datasets/pusht_obj/train_shape_*.lance
Episodes are re-numbered consecutively in shard order; every other column is copied as is (pixels bytes, action, state, ...).
swm's lance format keeps EPISODE-SCOPED data (e.g. RoboCasa's model_xml / ep_meta) in a `<table>_episodes.lance` side table next to
the main table (one row per episode: episode_idx + one column per key); those are merged too, with the same episode_idx remap, into
`<out>_episodes.lance`. Without that the merged set silently loses the scene XML and every eval replay runs in the wrong scene.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("shards", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-rows", type=int, default=20000)
    a = p.parse_args()
    out = Path(a.out)
    assert not out.exists(), f"{out} exists"
    t0 = time.time()
    offset = 0
    total = 0
    mode = "create"
    ep_out = side_table_path(out)
    ep_mode = "create"
    ep_total = 0
    ep_expected = 0
    for sh in a.shards:
        ds = lance.dataset(sh)
        eps = ds.to_table(columns=["episode_idx"]).column("episode_idx").to_numpy()
        uniq = np.unique(eps)
        remap = {int(e): offset + i for i, e in enumerate(uniq)}
        side = side_table_path(Path(sh))
        if side.exists():
            et = lance.dataset(str(side)).to_table()
            e = et.column("episode_idx").to_numpy()
            missing = sorted(set(int(x) for x in e) - set(remap))
            assert not missing, f"{side}: episode ids {missing[:5]} have no rows in the main table"
            new_e = pa.array([remap[int(x)] for x in e], type=et.schema.field("episode_idx").type)
            et = et.set_column(et.schema.get_field_index("episode_idx"), "episode_idx", new_e)
            lance.write_dataset(et, str(ep_out), mode=ep_mode)
            ep_mode = "append"
            ep_total += et.num_rows
            print(
                f"[merge] {side.name}: {et.num_rows} episode rows, columns {[c for c in et.schema.names if c != 'episode_idx']}", flush=True
            )
        ep_expected += len(uniq)
        for batch in ds.to_batches(batch_size=a.batch_rows):
            tb = pa.Table.from_batches([batch])
            e = tb.column("episode_idx").to_numpy()
            new_e = pa.array([remap[int(x)] for x in e], type=tb.schema.field("episode_idx").type)
            tb = tb.set_column(tb.schema.get_field_index("episode_idx"), "episode_idx", new_e)
            lance.write_dataset(tb, str(out), mode=mode)
            mode = "append"
            total += tb.num_rows
        print(
            f"[merge] {sh}: {len(uniq)} episodes -> ids {offset}..{offset + len(uniq) - 1} ({ds.count_rows()} rows) {time.time() - t0:.0f}s",
            flush=True,
        )
        offset += len(uniq)
    ds = lance.dataset(str(out))
    print(f"[merge] wrote {out}: {ds.count_rows()} rows, {offset} episodes, cols {ds.schema.names}")
    if ep_out.exists():
        assert ep_total == ep_expected, f"episode side table has {ep_total} rows for {ep_expected} episodes"
        print(f"[merge] wrote {ep_out}: {ep_total} episode rows (episode-scoped data)")


def side_table_path(main: Path) -> Path:
    """`<dir>/<stem>_episodes.lance` for a `<dir>/<stem>.lance` main table (swm LanceWriter's episodes side table)."""
    main = Path(main)
    return main.with_name(main.stem + "_episodes.lance")


if __name__ == "__main__":
    main()
