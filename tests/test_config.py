import pytest

from tempo.config import TempoConfig, available_envs, load_config, load_env_spec


def test_defaults_match_paper():
    cfg = load_config()
    assert cfg.H == 25 and cfg.map.kappa == 25 and cfg.map.max_steps == 75
    assert (cfg.planner.num_samples, cfg.planner.n_steps, cfg.planner.topk) == (300, 30, 30)
    assert cfg.planner.cost_weight == 0.5 and cfg.train.online_rounds == 8


def test_overrides():
    cfg = load_config(overrides=["eval.goal_offset=75", "planner.cost_weight=0"])
    assert cfg.eval.goal_offset == 75 and cfg.planner.cost_weight == 0.0
    with pytest.raises(KeyError):
        load_config(overrides=["planner.nope=1"])
    assert isinstance(TempoConfig().to_dict()["map"], dict)


@pytest.mark.parametrize("name", available_envs())
def test_every_env_config_loads(name):
    spec = load_env_spec(name)
    assert spec.env_id and spec.dataset and spec.cell
    assert spec.checkpoint("lewm", 42) == f"{spec.cell}_lewm_s42/weights_epoch_10.pt"


def test_extends():
    base, split = load_env_spec("tworoom"), load_env_spec("tworoom_split")
    assert split.callables == base.callables and split.env_id == base.env_id
    assert split.dataset != base.dataset and split.eval_dataset and split.cell == "two10v"
