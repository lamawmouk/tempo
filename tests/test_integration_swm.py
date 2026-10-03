"""End-to-end run of one cell on a real stable-worldmodel World (TwoRoom) and a real lance dataset, with a tiny stand-in world
model in place of a trained checkpoint: CEM with the TEMPO objective acts, the online rounds update F_phi, eval.json is written."""

import json

import pytest
import torch
import torch.nn as nn

swm = pytest.importorskip("stable_worldmodel")


D = 8


class ToyWorldModel(nn.Module):
    """LeWM interface: encode(info)['emb'] (B, T, D); rollout(info, actions)['predicted_emb'] (B, S, H + T, D)."""

    def __init__(self, action_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.proj = nn.Linear(3 * 16 * 16, D)
        self.act = nn.Linear(action_dim, D, bias=False)

    def encode(self, info):
        px = info["pixels"].float()
        B, T = px.shape[:2]
        feats = 100 * nn.functional.adaptive_avg_pool2d(px.flatten(0, 1), 16).flatten(1)
        info["emb"] = self.proj(feats).reshape(B, T, D)
        return info

    def rollout(self, info, action_sequence):
        B, S, T = action_sequence.shape[:3]
        init = self.encode({"pixels": info["pixels"][:, 0]})["emb"]  # (B, H, D)
        ctx = init[:, None].expand(B, S, -1, -1)
        steps = torch.cumsum(self.act(action_sequence.to(self.proj.weight.dtype)), dim=2)
        info["predicted_emb"] = torch.cat([ctx, ctx[:, :, -1:] + steps], dim=2)
        return info


@pytest.fixture(scope="module")
def tworoom_dataset(tmp_path_factory):
    """A few random TwoRoom episodes written with swm's own lance writer."""
    pytest.importorskip("lance")
    path = tmp_path_factory.mktemp("data") / "tworoom_random.lance"
    world = swm.World("swm/TwoRoom-v1", num_envs=2, image_shape=(224, 224), max_episode_steps=60, goal_conditioned=False)
    world.set_policy(swm.policy.RandomPolicy(seed=0))
    world.collect(path=str(path), episodes=6, seed=0, progress=False)
    world.close()
    return path


def test_full_cell_smoke(tworoom_dataset, tmp_path, monkeypatch):
    """encode -> offline -> online -> eval through run_cell, with the stand-in model in place of a checkpoint."""
    import tempo.pipeline as pipeline
    from tempo.config import EnvSpec, load_config

    monkeypatch.setattr(pipeline, "load_world_model", lambda name, device: ToyWorldModel(2 * 5))
    spec = EnvSpec(
        name="tworoom_toy",
        env_id="swm/TwoRoom-v1",
        dataset=str(tworoom_dataset),
        cell="toy",
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
        ],
        history_keys=("pixels", "proprio"),
        process_keys=("action", "proprio"),
    )
    cfg = load_config(overrides=["planner.history_len=1", "eval.goal_offset=10"])
    cell = pipeline.Cell(env=spec, cfg=cfg, checkpoint="toy_lewm_s0/weights_epoch_10.pt", seed=0, out_dir=tmp_path, smoke=True)
    results = pipeline.run_cell(cell)
    assert set(results) == {"group0"} and len(results["group0"]["episode_successes"]) == 2
    out = cell.run_dir()
    assert (out / "eval.json").exists() and (out / "fmap.pt").exists()
    assert torch.load(out / "fmap.pt")["n_updates"] > cfg.train.offline_steps  # online rounds updated F_phi
    summary = json.loads((out / "eval.json").read_text())
    assert summary["scale"] > 0  # the TEMPO term is active
