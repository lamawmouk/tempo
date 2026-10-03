#!/bin/bash
# One TEMPO cell on a single GPU.
#   sbatch scripts/slurm/tempo.sbatch <env> <wm> <seed> [overrides...]
#   sbatch scripts/slurm/tempo.sbatch tworoom lewm 42 eval.goal_offset=75
#SBATCH -J tempo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/tempo_%j.out
set -euo pipefail
ENV=$1; WM=$2; SEED=$3; shift 3
export MUJOCO_GL=${MUJOCO_GL:-egl}
: "${STABLEWM_HOME:?set STABLEWM_HOME to the directory holding datasets/ and checkpoints/}"
mkdir -p logs
tempo run --env "$ENV" --wm "$WM" --seed "$SEED" --out "${TEMPO_OUT:-runs}" "$@"
