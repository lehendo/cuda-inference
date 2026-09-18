#!/bin/bash
# Run gie_bench with env-var driven defaults.
#   GPU_BENCH_WORKLOADS (benchmarks/workloads.json)   GPU_BENCH_REPS (30 samples)
#   GPU_BENCH_PRECISIONS (fp32,fp16)   GPU_BENCH_BACKENDS   GPU_OUT_DIR   GPU_BENCH_TAG
# Extra arguments are passed to gie_bench, e.g.  scripts/run_benchmarks.sh --fusion both --flush-l2
set -euo pipefail
source "$(dirname "$0")/env.sh"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
TAG="${GPU_BENCH_TAG:-$(date +%Y%m%d_%H%M%S)_${GPU_NAME}}"
OUT="${GPU_OUT_DIR:-$GPU_REPO/results/raw/$TAG}"
ARGS=(--workloads "${GPU_BENCH_WORKLOADS:-$GPU_REPO/benchmarks/workloads.json}"
      --precisions "${GPU_BENCH_PRECISIONS:-fp32,fp16}"
      --samples "${GPU_BENCH_REPS:-30}" --out "$OUT" --tag "$TAG")
[ -n "${GPU_BENCH_BACKENDS:-}" ] && ARGS+=(--backends "$GPU_BENCH_BACKENDS")
echo "gie_bench -> $OUT"
"$GPU_BUILD_DIR/benchmarks/gie_bench" "${ARGS[@]}" "$@"
