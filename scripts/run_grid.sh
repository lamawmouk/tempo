#!/bin/bash
# Submit the main grid: TEMPO and the plain planner, every environment x world model x seed x goal offset.
#   ENVS="tworoom pusht" WMS="lewm pldm" bash scripts/run_grid.sh
set -euo pipefail
ENVS=${ENVS:-"tworoom pusht reacher cube scene pointmaze pointmaze_large antumaze_split pushobj"}
WMS=${WMS:-"lewm pldm dwpix"}
SEEDS=${SEEDS:-"42 43 44"}
OFFSETS=${OFFSETS:-"25 50 75"}
for env in $ENVS; do for wm in $WMS; do for s in $SEEDS; do for g in $OFFSETS; do
  for w in 0.5 0; do
    sbatch scripts/slurm/tempo.sbatch "$env" "$wm" "$s" eval.goal_offset=$g planner.cost_weight=$w
  done
done; done; done; done
