#!/bin/bash
# Nsight Systems timeline of a benchmark run (launch pattern, sync, memcpy, graphs).
#   scripts/run_nsys.sh <workload-substring> <backend> [precision]
set -uo pipefail
source "$(dirname "$0")/env.sh"
WL="${1:?workload substring}"; BE="${2:?backend}"; PREC="${3:-fp16}"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
OUT="${GPU_OUT_DIR:-$GPU_REPO/results/raw/nsys_${GPU_NAME}}"; mkdir -p "$OUT"
nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node --sample=none --force-overwrite=true \
     -o "$OUT/nsys_${WL}_${BE}_${PREC}" \
     "$GPU_BUILD_DIR/benchmarks/gie_bench" --workloads "$GPU_REPO/benchmarks/workloads.json" --filter "$WL" \
     --backends "$BE" --precisions "$PREC" --mode "${GPU_NSYS_MODE:-fixed}" --warmup 3 --samples 5 --e2e --no-validate \
     --out "$OUT/bench_${WL}_${BE}_${PREC}"
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum --format csv "$OUT/nsys_${WL}_${BE}_${PREC}.nsys-rep" \
     > "$OUT/nsys_${WL}_${BE}_${PREC}_stats.csv" 2>&1 || true
echo "wrote $OUT/nsys_${WL}_${BE}_${PREC}.nsys-rep"
