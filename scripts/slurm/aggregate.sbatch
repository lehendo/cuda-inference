#!/bin/bash
# CPU-only: aggregate raw results and regenerate figures. GPU_RESULTS_DIR=<dir with results.jsonl files>
# Submit with:  GPU_SLURM_GRES=none scripts/submit.sh scripts/slurm/aggregate.sbatch
#SBATCH --job-name=gie_agg
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eu
source "${GPU_REPO:-$HOME/cuda-inference}/scripts/env.sh"
cd "$GPU_REPO/analysis"; "$GPU_PYENV/bin/python" aggregate_results.py --raw "${GPU_RESULTS_DIR:?}" --out "$GPU_REPO/results/processed"
"$GPU_PYENV/bin/python" make_figures.py --raw "${GPU_RESULTS_DIR:?}" --figures "$GPU_REPO/results/figures"
