import numpy as np

from tempo.buffer import LatentBuffer
from tempo.data import EpisodeTable, add_offline_windows, median_displacement, select_eval_rows


def _table(lengths):
    ep = np.concatenate([np.full(n, i) for i, n in enumerate(lengths)])
    st = np.concatenate([np.arange(n) for n in lengths])
    ids, start = np.unique(ep, return_index=True)
    return EpisodeTable(ep, st, ids, start, np.asarray(lengths))


def test_eval_rows_are_deterministic_and_leave_room_for_the_goal():
    tab = _table([40, 100, 60])
    e1, s1, r1 = select_eval_rows(tab, seed=42, n=20, goal_offset=25)
    e2, s2, r2 = select_eval_rows(tab, seed=42, n=20, goal_offset=25)
    assert np.array_equal(r1, r2)
    assert np.all(s1 + 25 < tab.length[e1])
    eligible = np.zeros(len(tab.ep), bool)
    eligible[tab.ep == 1] = True
    e3, _, _ = select_eval_rows(tab, seed=0, n=10, goal_offset=25, eligible=eligible)
    assert set(e3.tolist()) == {1}


def test_offline_windows_respect_exclusions():
    lengths = [80, 80, 80]
    tab = _table(lengths)
    z = np.stack([tab.ep * 1000 + tab.st] * 3, 1).astype(np.float32)
    buf = LatentBuffer(3)
    add_offline_windows(
        buf, z, tab.ep, n_windows=300, length=25, rng=np.random.default_rng(0), exclude_rows=[0, 1, 2], exclude_episodes=[2]
    )
    assert len(buf.episodes) == 300 and all(len(w) == 26 for w in buf.episodes)
    first = np.array([w[0, 0] for w in buf.episodes])
    assert not np.isin(first, [0, 1, 2]).any()  # excluded start rows
    assert (first // 1000 != 2).all()  # excluded episode
    assert all(np.all(np.diff(w[:, 0]) == 1) for w in buf.episodes)  # contiguous frames of one episode


def test_median_displacement():
    tab = _table([200] * 10)
    z = (tab.st[:, None] * np.ones((1, 4))).astype(np.float32)  # moves 2 units per step in norm
    d = median_displacement(z, tab.ep, 25, np.random.default_rng(0), n=2000)
    assert abs(d - 25 * 2) < 1e-4
