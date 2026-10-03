import numpy as np
import torch

from tempo.buffer import LatentBuffer
from tempo.config import MapConfig
from tempo.temporal_map import TemporalMap, temporal_distance_loss
from tempo.trainer import MapTrainer


def test_shapes_and_distance():
    f = TemporalMap(latent_dim=12, out_dim=4, hidden=16)
    za, zb = torch.randn(7, 12), torch.randn(7, 12)
    assert f(za).shape == (7, 4)
    assert torch.allclose(f.dist(za, zb) ** 2, f.sqdist(za, zb), atol=1e-5)
    assert torch.allclose(f.dist(za, za), torch.zeros(7), atol=1e-6)


def test_loss_terms():
    f = TemporalMap(latent_dim=3, out_dim=2, hidden=8, kappa=25.0)
    z = torch.randn(5, 3)
    near = dict(z=z, z_future=z, delta=torch.zeros(5))
    assert temporal_distance_loss(f, near).item() == 0.0  # identical frames, zero steps apart
    far = dict(z=z, z_future=z, delta=torch.full((5,), 100))
    hinge = temporal_distance_loss(f, near, far, max_steps=75, far_weight=0.5).item()
    assert abs(hinge - 0.5 * 75 / 25) < 1e-6  # coincident far pair costs the full margin


def test_map_learns_to_count_steps():
    """Episodes are noisy walks along a hidden direction: F must recover time, not the latent norm."""
    rng = np.random.default_rng(0)
    buf = LatentBuffer(latent_dim=8)
    direction = rng.normal(size=8)
    direction /= np.linalg.norm(direction)
    for _ in range(64):
        t = np.arange(120)[:, None]
        buf.add_episode(rng.normal(size=8) + 0.05 * t * direction + 0.01 * rng.normal(size=(120, 8)))
    cfg = MapConfig(dim=8, hidden=64, batch_size=128, lr=3e-3)
    torch.manual_seed(0)
    fmap = TemporalMap(8, cfg.dim, cfg.hidden, cfg.layers, cfg.kappa)
    trainer = MapTrainer(fmap, buf, cfg, seed=0)
    first, last = trainer.train(5), trainer.train(1500)
    assert last < 0.25 * first
    ep = torch.as_tensor(buf.episodes[0])
    with torch.no_grad():
        d25, d50 = fmap.dist(ep[10], ep[35]).item(), fmap.dist(ep[10], ep[60]).item()
    assert 0.6 < d25 < 1.4 and 1.4 < d50 < 2.6  # one / two units for 25 / 50 steps
