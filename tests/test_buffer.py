import numpy as np
import pytest

from tempo.buffer import LatentBuffer


def _episode(T, D=4, start=0.0):
    return np.arange(start, start + T + 1, dtype=np.float32)[:, None].repeat(D, 1)  # z_t encodes t


def test_near_pairs_stay_in_episode_and_range():
    buf = LatentBuffer(4)
    buf.add_episode(_episode(30))
    buf.add_episode(_episode(10, start=1000))
    b = buf.sample_near(2000, max_steps=25, rng=np.random.default_rng(0))
    k = (b["z_future"][:, 0] - b["z"][:, 0]).numpy()
    assert np.array_equal(k, b["delta"].numpy())  # same episode: frame gap == delta
    assert k.min() >= 1 and k.max() <= 25


def test_far_pairs_need_long_episodes():
    buf = LatentBuffer(4)
    buf.add_episode(_episode(50))
    with pytest.raises(ValueError):
        buf.sample_far(8, max_steps=75, rng=np.random.default_rng(0))
    buf.add_episode(_episode(200))
    b = buf.sample_far(500, max_steps=75, rng=np.random.default_rng(0))
    assert b["delta"].min() > 75


def test_online_episodes_and_capacity():
    buf = LatentBuffer(2, capacity=2)
    buf.start(0, np.zeros(2))
    assert buf.open_length(0) == 0
    for t in range(3):
        buf.append(0, np.full(2, t + 1))
    assert buf.open_length(0) == 3
    buf.close(0)
    assert len(buf.episodes) == 1 and buf.n_steps == 3
    buf.start(1, np.zeros(2))
    buf.close(1)  # single frame: nothing to learn from
    assert len(buf.episodes) == 1
    buf.add_episode(_episode(5, D=2))
    buf.add_episode(_episode(6, D=2))
    assert len(buf.episodes) == 2 and buf.n_steps == 11  # oldest dropped
