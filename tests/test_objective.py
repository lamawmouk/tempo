import pytest
import torch

pytest.importorskip("stable_worldmodel")
from stable_worldmodel.planning import GoalMSE  # noqa: E402

from tempo.objective import TemporalDistanceCost, make_objective  # noqa: E402
from tempo.temporal_map import TemporalMap  # noqa: E402


def _info(B=2, S=5, T=6, D=8):
    return {"predicted_emb": torch.randn(B, S, T, D), "goal_emb": torch.randn(B, 1, D)}


def test_tempo_term_matches_definition():
    torch.manual_seed(0)
    f = TemporalMap(8, 4, 16)
    info = _info()
    cost = TemporalDistanceCost(f, scale=3.0)(info)
    with torch.no_grad():
        ref = 3.0 * ((f(info["predicted_emb"][:, :, -1]) - f(info["goal_emb"][:, -1])[:, None]) ** 2).sum(-1)
    assert cost.shape == (2, 5) and torch.allclose(cost, ref, atol=1e-5)


def test_blend_and_baseline():
    torch.manual_seed(0)
    f, own, info = TemporalMap(8, 4, 16), GoalMSE(), _info()
    assert make_objective(own, f, 0.0, 1.0) is own  # w = 0: the world model's own planner
    blend = make_objective(own, f, 0.5, 2.0)(info)
    ref = 0.5 * own(info) + 0.5 * TemporalDistanceCost(f, 2.0)(info)
    assert torch.allclose(blend, ref, atol=1e-5)


def test_dinowm_pooled_latent():
    torch.manual_seed(0)
    f = TemporalMap(16 + 3, 4, 16)
    B, S, T, P = 2, 4, 3, 9
    info = {
        "predicted_pixels_emb": torch.randn(B, S, T, P, 16),
        "predicted_proprio_emb": torch.randn(B, S, T, 3),
        "pixels_goal_emb": torch.randn(B, 1, P, 16),
        "proprio_goal_emb": torch.randn(B, 1, 3),
    }
    cost = TemporalDistanceCost(f, 1.0, dino_keys=["proprio"])(info)
    z = torch.cat([info["predicted_pixels_emb"][:, :, -1].mean(-2), info["predicted_proprio_emb"][:, :, -1]], -1)
    g = torch.cat([info["pixels_goal_emb"][:, -1].mean(-2), info["proprio_goal_emb"][:, -1]], -1)
    with torch.no_grad():
        ref = ((f(z) - f(g)[:, None]) ** 2).sum(-1)
    assert torch.allclose(cost, ref, atol=1e-5)
