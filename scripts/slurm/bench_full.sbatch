#!/bin/bash
# Full benchmark suite: tuning set + held-out set, FP32 and FP16, fused and unfused, warm and cold L2.
#SBATCH --job-name=gie_full
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eu
source "${GPU_REPO:-$HOME/cuda-inference}/scripts/env.sh"
"$GPU_REPO/scripts/build.sh" >/dev/null
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
STAMP=$(date +%Y%m%d)
BASE="${GPU_OUT_DIR:-$GPU_REPO/results/raw/${STAMP}_${GPU_NAME}}"
mkdir -p "$BASE"
for SET in workloads heldout_workloads; do
  GPU_BENCH_WORKLOADS="$GPU_REPO/benchmarks/$SET.json" GPU_OUT_DIR="$BASE/$SET" GPU_BENCH_TAG="${STAMP}_${GPU_NAME}_$SET" \
    "$GPU_REPO/scripts/run_benchmarks.sh" --mode both --fusion both --e2e "$@"
done
# Cold-L2 protocol on the tuning set (fused only) to show cache-residency sensitivity.
GPU_BENCH_WORKLOADS="$GPU_REPO/benchmarks/workloads.json" GPU_OUT_DIR="$BASE/coldL2" GPU_BENCH_TAG="${STAMP}_${GPU_NAME}_coldL2" \
  "$GPU_REPO/scripts/run_benchmarks.sh" --flush-l2 "$@"
