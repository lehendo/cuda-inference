#!/bin/bash
# Autotuning runs: GPU_AUTOTUNE_MODE=tune|exp-b|clear
#SBATCH --job-name=gie_tune
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eu
source "${GPU_REPO:-$HOME/cuda-inference}/scripts/env.sh"
"$GPU_REPO/scripts/build.sh" >/dev/null
"$GPU_REPO/scripts/run_autotune.sh" "${GPU_AUTOTUNE_MODE:-tune}"
