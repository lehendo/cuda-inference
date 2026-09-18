#!/bin/bash
# Full measurement campaign on ONE GPU (run on a GPU node; see scripts/slurm/bench_full.sbatch for the
# batch version). Order = importance, so a walltime cut-off still leaves the most valuable data.
#   GPU_OUT_DIR   result root (default $GPU_REPO/results/raw/<date>_<gpu>)
#   GPU_BENCH_REPS samples per measurement (default 30)
set -u
source "$(dirname "$0")/env.sh"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
BASE="${GPU_OUT_DIR:-$GPU_REPO/results/raw/$(date +%Y%m%d)_${GPU_NAME}}"
mkdir -p "$BASE"
B="$GPU_BUILD_DIR/benchmarks"
S="${GPU_BENCH_REPS:-30}"
W="$GPU_REPO/benchmarks"
step() { echo "=== [$(date +%H:%M:%S)] $*"; }

step "device ceilings"; "$GPU_BUILD_DIR/gie_info" --measure-ceilings | tee "$BASE/gie_info.txt"
step "main: tuning set, fp32+fp16, fixed+tuned, fused+unfused, e2e"
"$B/gie_bench" --workloads "$W/workloads.json" --precisions fp32,fp16 --mode both --fusion both --e2e --samples "$S" --out "$BASE/tuning_set" --tag "${GPU_NAME}_tuning"
step "held-out set"
"$B/gie_bench" --workloads "$W/heldout_workloads.json" --precisions fp32,fp16 --mode both --fusion both --samples "$S" --out "$BASE/heldout" --tag "${GPU_NAME}_heldout"
step "Experiment B (search strategies)"
"$B/gie_tune" experiment-b --tuning-set "$W/workloads.json" --heldout-set "$W/heldout_workloads.json" --precisions fp32,fp16 --out "$BASE/experiment_b"
step "Experiment D sweeps"
"$B/gie_bench" --workloads "$W/sweeps.json" --precisions fp32,fp16 --mode tuned --fusion fused --samples "$S" --out "$BASE/sweeps" --tag "${GPU_NAME}_sweeps"
step "Experiment F: CUDA Graphs / streams"
"$B/gie_graph_bench" --out "$BASE/graphs" --precision fp16
"$B/gie_graph_bench" --out "$BASE/graphs" --precision fp32
step "RMSNorm extension"
"$B/gie_rmsnorm_bench" "$BASE/rmsnorm"
step "Experiment C: bf16 / tf32 on throughput workloads"
"$B/gie_bench" --workloads "$W/workloads.json" --category throughput --precisions bf16,tf32 --mode tuned --fusion fused --samples "$S" --out "$BASE/precision_bf16_tf32" --tag "${GPU_NAME}_prec"
step "cold-L2 protocol, tuning set"
"$B/gie_bench" --workloads "$W/workloads.json" --precisions fp32,fp16 --mode tuned --fusion fused --flush-l2 --samples "$S" --out "$BASE/cold_l2" --tag "${GPU_NAME}_coldL2"
step "CAMPAIGN_DONE"
