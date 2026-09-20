#!/bin/bash
# Re-measure ONE backend (default cutlass) over a collected campaign with exactly the campaign's options.
#   scripts/rerun_backend.sh <results/raw/<name>> [backend]       (run on a GPU node of the same GPU model)
set -u
source "$(dirname "$0")/env.sh"
R="$1"; BE="${2:-cutlass}"; B="$GPU_BUILD_DIR/benchmarks"; W="$GPU_REPO/benchmarks"; S="${GPU_BENCH_REPS:-30}"
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
for d in tuning_set heldout sweeps precision_bf16_tf32 cold_l2; do python3 "$GPU_REPO/scripts/strip_backend_rows.py" "$R/$d" "$BE"; done
"$B/gie_bench" --workloads "$W/workloads.json" --backends "$BE" --precisions fp32,fp16 --mode both --fusion fused --e2e --skip-transfers --samples "$S" --out "$R/tuning_set" --tag "${NAME}_tuning"
"$B/gie_bench" --workloads "$W/heldout_workloads.json" --backends "$BE" --precisions fp32,fp16 --mode both --fusion fused --samples "$S" --out "$R/heldout" --tag "${NAME}_heldout"
"$B/gie_bench" --workloads "$W/sweeps.json" --backends "$BE" --precisions fp32,fp16 --mode tuned --fusion fused --samples "$S" --out "$R/sweeps" --tag "${NAME}_sweeps"
"$B/gie_bench" --workloads "$W/workloads.json" --category throughput --backends "$BE" --precisions bf16,tf32 --mode tuned --fusion fused --samples "$S" --out "$R/precision_bf16_tf32" --tag "${NAME}_prec"
"$B/gie_bench" --workloads "$W/workloads.json" --backends "$BE" --precisions fp32,fp16 --mode tuned --fusion fused --flush-l2 --samples "$S" --out "$R/cold_l2" --tag "${NAME}_coldL2"
echo RERUN_DONE
