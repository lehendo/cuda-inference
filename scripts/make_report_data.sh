#!/bin/bash
# Aggregate all collected campaigns (results/raw/*) into results/processed and results/figures.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$GPU_REPO/analysis"
RAW=()
for d in "$GPU_REPO"/results/raw/*/; do
  for s in tuning_set heldout sweeps precision_bf16_tf32; do [ -d "$d$s" ] && RAW+=(--raw "$d$s"); done
done
"$GPU_PYENV/bin/python" aggregate_results.py "${RAW[@]}" --out "$GPU_REPO/results/processed"
"$GPU_PYENV/bin/python" make_figures.py "${RAW[@]}" --figures "$GPU_REPO/results/figures"
for d in "$GPU_REPO"/results/raw/*/experiment_b; do
  [ -d "$d" ] && "$GPU_PYENV/bin/python" analyze_experiment_b.py --in "$d" --out "$GPU_REPO/results/processed/experiment_b_$(basename "$(dirname "$d")")" --figures "$GPU_REPO/results/figures" || true
done
