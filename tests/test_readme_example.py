"""The 'Use TEMPO with your own world model' snippet from the README, on random latents."""

import numpy as np
import pytest
import torch

pytest.importorskip("stable_worldmodel")


def test_readme_library_example():
    from stable_worldmodel.planning import GoalMSE

    from tempo import TemporalMap
    from tempo.buffer import LatentBuffer
    from tempo.config import MapConfig
    from tempo.objective import make_objective
    from tempo.trainer import MapTrainer

    rng = np.random.default_rng(0)
    trajectories = [np.cumsum(rng.normal(size=(100, 192)), 0) for _ in range(8)]  # (T + 1, D) frozen latents each
    d_H = 25.0

    buffer = LatentBuffer(latent_dim=192)
    for z in trajectories:
        buffer.add_episode(z)
    fmap = TemporalMap(latent_dim=192)
    MapTrainer(fmap, buffer, MapConfig()).train(20)
    cost = make_objective(GoalMSE(), fmap, weight=0.5, scale=d_H**2)

    info = {"predicted_emb": torch.randn(2, 300, 8, 192), "goal_emb": torch.randn(2, 1, 192)}
    assert cost(info).shape == (2, 300)
