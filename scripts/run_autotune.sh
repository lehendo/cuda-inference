#!/bin/bash
# Autotune workload sets and record search cost / winners (see tools gie_tune).
#   scripts/run_autotune.sh tune      -> tune the tuning set into the cache
#   scripts/run_autotune.sh clear     -> clear the tuning cache
#   scripts/run_autotune.sh exp-b     -> Experiment B: fixed vs exhaustive vs model-guided search
set -euo pipefail
source "$(dirname "$0")/env.sh"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
OUT="${GPU_OUT_DIR:-$GPU_REPO/results/raw/autotune_${GPU_NAME}}"
case "${1:-tune}" in
  clear) "$GPU_BUILD_DIR/benchmarks/gie_tune" --clear-cache ;;
  tune) "$GPU_BUILD_DIR/benchmarks/gie_tune" tune --workloads "$GPU_REPO/benchmarks/workloads.json" --precisions "${GPU_BENCH_PRECISIONS:-fp32,fp16}" --out "$OUT" ;;
  exp-b) "$GPU_BUILD_DIR/benchmarks/gie_tune" experiment-b --tuning-set "$GPU_REPO/benchmarks/workloads.json" \
            --heldout-set "$GPU_REPO/benchmarks/heldout_workloads.json" --precisions "${GPU_BENCH_PRECISIONS:-fp32,fp16}" --out "$OUT" ;;
  *) echo "usage: $0 tune|clear|exp-b" >&2; exit 2 ;;
esac
