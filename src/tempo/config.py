"""Experiment configuration (configs/tempo.yaml) and environment specs (configs/envs/<name>.yaml)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class MapConfig:
    """F_phi architecture and loss."""

    dim: int = 64  # embedding width
    hidden: int = 256  # hidden width
    layers: int = 2  # hidden layers
    kappa: float = 25.0  # environment steps per unit of map distance (one plan)
    max_steps: int = 75  # calibrated range; farther same-episode pairs are only pushed apart (hinge)
    far_weight: float = 0.5  # weight of the hinge term
    lr: float = 1.0e-3
    batch_size: int = 256
    grad_clip: float = 10.0


@dataclass
class PlannerConfig:
    """CEM planner shared by every method (the published LeWM / PLDM / DINO-WM protocol)."""

    num_samples: int = 300
    n_steps: int = 30
    topk: int = 30
    horizon: int = 5  # plan length in action blocks
    action_block: int = 5  # environment steps per block -> H = 25
    receding: int = 5  # blocks executed before replanning (5 = execute the whole plan)
    history_len: int = 3
    warm_start: bool = True
    cost_weight: float = 0.5  # w in J = (1 - w) own + w * d_H^2 * ||F(z_hat_H) - F(G)||^2; 0 = the plain planner


@dataclass
class TrainConfig:
    offline_windows: int = 20_000
    offline_steps: int = 3_000
    online_rounds: int = 8
    updates_per_plan: int = 5  # F_phi steps after every executed plan (any env)
    updates_per_episode: int = 50  # F_phi steps after every finished episode


@dataclass
class EvalConfig:
    goal_offset: int = 25  # goal frame this many steps after the start frame
    budget_factor: int = 2  # episode budget = budget_factor x goal_offset
    groups: int = 5  # fixed groups of windows per seed
    windows: int = 50  # windows per group (= parallel environments)
    planner_seed: int = 42


@dataclass
class TempoConfig:
    map: MapConfig = field(default_factory=MapConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def H(self) -> int:
        return self.planner.horizon * self.planner.action_block

    def to_dict(self) -> dict:
        return asdict(self)


def _update(obj, values: Dict[str, Any]):
    names = {f.name: f for f in fields(obj)}
    for k, v in values.items():
        if k not in names:
            raise KeyError(f"unknown config key '{k}' for {type(obj).__name__}")
        cur = getattr(obj, k)
        if is_dataclass(cur):
            _update(cur, v)
        else:
            setattr(obj, k, type(cur)(v) if cur is not None and not isinstance(cur, bool) else v)
    return obj


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> TempoConfig:
    """Defaults <- YAML file <- dotted `section.key=value` overrides (e.g. eval.goal_offset=75)."""
    cfg = TempoConfig()
    p = Path(path) if path else CONFIG_DIR / "tempo.yaml"
    if p.exists():
        _update(cfg, yaml.safe_load(p.read_text()) or {})
    for item in overrides or []:
        key, _, raw = item.partition("=")
        section, _, name = key.partition(".")
        _update(getattr(cfg, section), {name: yaml.safe_load(raw)})
    return cfg


# ---------------------------------------------------------------------------------------------------------------- envs
@dataclass
class EnvSpec:
    name: str
    env_id: str  # gymnasium id passed to swm.World
    dataset: str  # training dataset under $STABLEWM_HOME/datasets
    cell: str  # checkpoint prefix: <cell>_<wm>_s<seed>/weights_epoch_10.pt
    callables: List[dict] = field(default_factory=list)  # swm World.evaluate reset hooks (dataset row -> env state)
    history_keys: Tuple[str, ...] = ("pixels", "proprio")
    process_keys: Tuple[str, ...] = ("action", "proprio")  # columns normalised with training-set StandardScalers
    state_col: str = "state"  # dataset column describing the task state (eval-window filters)
    state_dims: Optional[List[int]] = None
    world_kwargs: dict = field(default_factory=dict)
    import_hook: Optional[str] = None  # module whose import registers the env
    eval_dataset: Optional[str] = None  # held-out evaluation windows (world-model-level split / unseen configurations)
    trivial_tol: Optional[float] = None  # drop eval windows already solved at the start row
    window_flag_col: Optional[str] = None  # per-row 0/1 column required at the start and goal rows
    reach_sidecar: bool = False  # require '<dataset>.reach.npz' (scripts/data/build_antumaze_reach_mask.py)

    def checkpoint(self, wm: str, seed: int) -> str:
        return f"{self.cell}_{wm}_s{seed}/weights_epoch_10.pt"


def load_env_spec(name_or_path: str) -> EnvSpec:
    """Load configs/envs/<name>.yaml (or a path). A spec may `extends: <base>` and override any field."""
    p = Path(name_or_path)
    if not p.suffix:
        p = CONFIG_DIR / "envs" / f"{name_or_path}.yaml"
    raw = yaml.safe_load(p.read_text()) or {}
    base = raw.pop("extends", None)
    if base:
        merged = asdict(load_env_spec(base))
        merged.update(raw)
        raw = merged
    raw.setdefault("name", p.stem)
    for k in ("history_keys", "process_keys"):
        if k in raw:
            raw[k] = tuple(raw[k])
    return EnvSpec(**raw)


def available_envs() -> List[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "envs").glob("*.yaml"))
