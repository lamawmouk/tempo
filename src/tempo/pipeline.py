"""One evaluation cell: (environment, world model, seed, goal offset) -> eval.json.

    1. encode   frozen latent of every training frame (cached per checkpoint)
    2. offline  F_phi on expert windows of the training data
    3. online   F_phi keeps learning from the TEMPO planner's own episodes on training windows
    4. eval     fixed groups of dataset windows, everything frozen

With planner.cost_weight = 0 the cell is the plain planner of the world model (stages 2-3 are skipped).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from tempo.buffer import LatentBuffer
from tempo.config import EnvSpec, TempoConfig
from tempo.data import (
    add_offline_windows,
    eligible_windows,
    encode_dataset,
    episode_table,
    fit_scalers,
    load_dataset,
    median_displacement,
    select_eval_rows,
)
from tempo.envs import make_world
from tempo.online import Recorder, run_windows, seed_everything
from tempo.planner import build_planner
from tempo.temporal_map import TemporalMap
from tempo.trainer import MapTrainer
from tempo.world_models import dinowm_emb_keys, is_dinowm, load_world_model


@dataclass
class Cell:
    env: EnvSpec
    cfg: TempoConfig
    checkpoint: str
    seed: int = 42
    out_dir: Path = Path("runs")
    eval_dataset: Optional[str] = None  # overrides env.eval_dataset
    holdout_episodes: bool = False  # drop every episode that contributes an eval window from all training data
    stages: tuple = ("encode", "offline", "online", "eval")
    smoke: bool = False
    tag: str = field(default="")

    def run_dir(self) -> Path:
        name = self.tag or (
            f"{self.env.name}_{Path(self.checkpoint).parts[0]}_g{self.cfg.eval.goal_offset}_w{self.cfg.planner.cost_weight:g}"
        )
        return self.out_dir / name


def run_cell(cell: Cell) -> dict:
    cfg, spec, ev = cell.cfg, cell.env, cell.cfg.eval
    if cell.smoke:  # tiny end-to-end run for checking an installation
        cfg.train.offline_windows, cfg.train.offline_steps, cfg.train.online_rounds = 64, 5, 1
        cfg.planner.num_samples, cfg.planner.n_steps, cfg.planner.topk = 8, 1, 4
        ev.groups, ev.windows = 1, 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cell.seed)
    np.random.seed(cell.seed)
    rng = np.random.default_rng(cell.seed)
    out = cell.run_dir()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    use_map = cfg.planner.cost_weight > 0

    model = load_world_model(cell.checkpoint, device)
    dino = is_dinowm(model)
    keys = dinowm_emb_keys(model) if dino else []
    ds = load_dataset(spec.dataset, spec.process_keys)
    tab = episode_table(ds)
    eval_name = cell.eval_dataset or spec.eval_dataset
    ds_eval = load_dataset(eval_name, spec.process_keys) if eval_name else ds
    tab_eval = episode_table(ds_eval) if eval_name else tab
    process = fit_scalers(ds, spec, extra_cols=keys)  # training-set statistics only
    print(f"[setup] {spec.name} | {cell.checkpoint} | {len(tab.ids)} train episodes | eval on {eval_name or spec.dataset}", flush=True)

    # ---- evaluation windows (fixed per planner seed, identical for every method)
    eligible = eligible_windows(ds_eval, tab_eval, spec, ev.goal_offset, eval_name or spec.dataset)
    groups = [select_eval_rows(tab_eval, ev.planner_seed + g, ev.windows, ev.goal_offset, eligible) for g in range(ev.groups)]
    same_ds = ds_eval is ds
    eval_rows = np.concatenate([g[2] for g in groups]) if same_ds else np.zeros(0, np.int64)
    eval_eps = set(int(e) for g in groups for e in g[0]) if (cell.holdout_episodes and same_ds) else set()

    # ---- 1. latents + the scale of the TEMPO term
    lat_path = cell.out_dir / "latents" / f"{Path(cell.checkpoint).parts[0]}{'_smoke' if cell.smoke else ''}.npz"
    if "encode" in cell.stages or not lat_path.exists():
        encode_dataset(model, ds, tab, lat_path, device, cfg.H, process=process, limit_episodes=5 if cell.smoke else None)
    lat = dict(np.load(lat_path))
    d_H = median_displacement(lat["z"], lat["ep"], cfg.H, rng)
    scale = float(lat["own_cost_H"]) if dino else d_H**2
    print(f"[latents] {lat['z'].shape}, median {cfg.H}-step displacement {d_H:.2f}, TEMPO scale {scale:.4f}", flush=True)

    fmap = TemporalMap(lat["z"].shape[1], cfg.map.dim, cfg.map.hidden, cfg.map.layers, cfg.map.kappa) if use_map else None
    buffer = LatentBuffer(lat["z"].shape[1])
    trainer = MapTrainer(fmap, buffer, cfg.map, seed=cell.seed) if use_map else None
    map_path = out / "fmap.pt"

    # ---- 2. offline
    if use_map and "offline" in cell.stages:
        n = add_offline_windows(
            buffer, lat["z"], lat["ep"], cfg.train.offline_windows, ev.goal_offset, rng, exclude_rows=eval_rows, exclude_episodes=eval_eps
        )
        loss = trainer.train(cfg.train.offline_steps)
        print(f"[offline] {n} windows, {cfg.train.offline_steps} steps, loss {loss:.4f}, {time.time() - t0:.0f}s", flush=True)

    budget = ev.budget_factor * ev.goal_offset
    world = make_world(spec, ev.windows, max_episode_steps=2 * budget)
    policy = build_planner(
        model,
        spec,
        cfg.planner,
        process,
        fmap,
        scale,
        ev.windows,
        device,
        ev.planner_seed,
        map_version=lambda: trainer.n_updates if trainer else 0,
    )
    world.set_policy(policy)

    # ---- 3. online rounds on training windows (never an evaluation start row / held-out episode)
    if use_map and "online" in cell.stages:
        rec = Recorder(world, policy, buffer, trainer, cfg.train)
        excl = set(int(r) for r in eval_rows)
        for r in range(cfg.train.online_rounds):
            _, _, rows = select_eval_rows(tab, 10_000 + cell.seed * 100 + r, ev.windows * 2, ev.goal_offset)
            rows = np.array([x for x in rows if int(x) not in excl and int(tab.ep[x]) not in eval_eps])[: ev.windows]
            m = run_windows(world, ds, spec, tab.ep[rows], tab.st[rows], ev.goal_offset, budget, recorder=rec)
            print(
                f"[online] round {r + 1}/{cfg.train.online_rounds}: success {m['success_rate']:.1f}%, "
                f"F updates {trainer.n_updates}, {time.time() - t0:.0f}s",
                flush=True,
            )
        rec.detach()
    if use_map:
        if "offline" in cell.stages or "online" in cell.stages:
            torch.save(trainer.state_dict(), map_path)
        else:
            trainer.load_state_dict(torch.load(map_path))

    # ---- 4. evaluation
    results = {}
    if "eval" in cell.stages:
        for g, (episodes, starts, _) in enumerate(groups):
            seed_everything(ev.planner_seed, g, policy)
            m = run_windows(world, ds_eval, spec, episodes, starts, ev.goal_offset, budget)
            results[f"group{g}"] = dict(
                success_rate=float(m["success_rate"]),
                episode_successes=[bool(x) for x in m["episode_successes"]],
                episodes=[int(e) for e in episodes],
                starts=[int(s) for s in starts],
                plan_time_s=float(policy.plan_time),
            )
            print(f"[eval] group {g}: success {m['success_rate']:.1f}%", flush=True)
        rates = [r["success_rate"] for r in results.values()]
        summary = dict(
            env=spec.name,
            checkpoint=cell.checkpoint,
            seed=cell.seed,
            eval_dataset=eval_name or spec.dataset,
            success_rate_mean=float(np.mean(rates)),
            success_rate_std=float(np.std(rates)),
            d_H=d_H,
            scale=scale,
            config=cfg.to_dict(),
            results=results,
            elapsed_s=time.time() - t0,
        )
        (out / "eval.json").write_text(json.dumps(summary, indent=1))
        print(f"[done] {out.name}: {np.mean(rates):.1f} +- {np.std(rates):.1f} % -> {out / 'eval.json'}", flush=True)
    world.close()
    return results
