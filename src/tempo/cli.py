"""Command line entry point.

tempo run --env tworoom --wm lewm --seed 42                              # TEMPO (w = 0.5)
tempo run --env tworoom --wm lewm --seed 42 planner.cost_weight=0        # the plain LeWM planner
tempo run --env pusht --wm pldm --seed 43 eval.goal_offset=75            # longer horizon
tempo envs                                                               # list environment configs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tempo.config import available_envs, load_config, load_env_spec


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="tempo")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="train F_phi and evaluate one cell")
    r.add_argument("--env", required=True, help="name in configs/envs/ or a path to an env YAML")
    r.add_argument("--wm", default="lewm", help="checkpoint family: lewm | pldm | dwpix (DINO-WM, pixels only) | ...")
    r.add_argument("--seed", type=int, default=42, help="world-model seed (selects the checkpoint)")
    r.add_argument("--checkpoint", default=None, help="explicit checkpoint name (default: <cell>_<wm>_s<seed>/weights_epoch_10.pt)")
    r.add_argument("--config", default=None, help="experiment YAML (default: configs/tempo.yaml)")
    r.add_argument("--eval-dataset", default=None, help="evaluate on windows of another dataset (held-out split)")
    r.add_argument("--holdout-episodes", action="store_true", help="no training window from any evaluation episode")
    r.add_argument("--stages", default="encode,offline,online,eval")
    r.add_argument("--out", default="runs")
    r.add_argument("--tag", default="", help="run directory name (default derived from env, checkpoint, offset, weight)")
    r.add_argument("--smoke", action="store_true", help="tiny end-to-end run")
    r.add_argument("overrides", nargs="*", help="section.key=value, e.g. eval.goal_offset=75 planner.cost_weight=0")
    sub.add_parser("envs", help="list environment configs")
    a = p.parse_args(argv)

    if a.cmd == "envs":
        print("\n".join(available_envs()))
        return
    from tempo.pipeline import Cell, run_cell  # heavy imports (stable-worldmodel) only when running

    spec = load_env_spec(a.env)
    cell = Cell(
        env=spec,
        cfg=load_config(a.config, a.overrides),
        checkpoint=a.checkpoint or spec.checkpoint(a.wm, a.seed),
        seed=a.seed,
        out_dir=Path(a.out),
        eval_dataset=a.eval_dataset,
        holdout_episodes=a.holdout_episodes,
        stages=tuple(a.stages.split(",")),
        smoke=a.smoke,
        tag=a.tag,
    )
    run_cell(cell)


if __name__ == "__main__":
    main()
