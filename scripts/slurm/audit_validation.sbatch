#!/bin/bash
# Audit validation pipeline for one GPU, run detached so a dropped login session cannot interrupt it.
# Stages (each logged under $AUDIT_LOGS, failures do not stop later stages):
#   1 incremental rebuild of the audited tree + native CTest
#   2 FP32 investigation (only if AUDIT_FP32_BUILD is set): ablations, occupancy/interleave variants, cuDNN GEMM-family baseline
#   3 representative benchmarks (tuning set + held-out set) into NEW result directories
#   4 reduced compute-sanitizer sweep (scripts/run_sanitizers.sh)
# Required env: AUDIT_TAG (a40|h100). Optional: AUDIT_FP32_BUILD (build dir that has -DGIE_ENABLE_ABLATIONS=ON), GPU_CUDA_ARCH.
# Nothing under an existing results directory is overwritten: every output path contains AUDIT_TAG and a stage name.
#SBATCH --job-name=gie_audit
#SBATCH --output=%x_%j.out
set -uo pipefail
source "${GPU_REPO:-$HOME/cuda-inference}/scripts/env.sh"
TAG="${AUDIT_TAG:?AUDIT_TAG}"
export GPU_BUILD_DIR="${GPU_BUILD_DIR:-$GPU_WORK/build_audit}"
LOGS="$GPU_WORK/logs/audit_${TAG}_v2"; mkdir -p "$LOGS"
R="$GPU_REPO/results/raw"
stage() { echo "[$(date -Is)] stage $1 start" | tee -a "$LOGS/stages.log"; }
done_stage() { echo "[$(date -Is)] stage $1 end (exit $2)" | tee -a "$LOGS/stages.log"; }
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | tee "$LOGS/gpu.txt"

stage 1
(cd "$GPU_REPO" && GPU_CUDA_ARCH="${GPU_CUDA_ARCH:-80;86;89;90}" scripts/build.sh) > "$LOGS/build.log" 2>&1
ctest --test-dir "$GPU_BUILD_DIR" --output-on-failure > "$LOGS/ctest.log" 2>&1
done_stage 1 $?

B="$GPU_BUILD_DIR/benchmarks/gie_bench"
if [ -n "${AUDIT_FP32_BUILD:-}" ]; then
  stage 2
  F="$AUDIT_FP32_BUILD/benchmarks/gie_bench"
  ABL=implicit_gemm_fp32_64x64x8_t4x4,ablation1_implicit_gemm_fp32_64x64x8_t4x4,ablation2_implicit_gemm_fp32_64x64x8_t4x4,ablation3_implicit_gemm_fp32_64x64x8_t4x4,implicit_gemm_fp32_128x128x8_t8x8,ablation1_implicit_gemm_fp32_128x128x8_t8x8,ablation2_implicit_gemm_fp32_128x128x8_t8x8,ablation3_implicit_gemm_fp32_128x128x8_t8x8
  VAR=implicit_gemm_fp32_128x128x8_t8x8,implicit_gemm_fp32_128x128x8_t8x8_mb2,implicit_gemm_fp32_128x128x8_t8x8_il,implicit_gemm_fp32_128x128x8_t8x8_mb2_il,implicit_gemm_fp32_128x128x16_t8x8_mb2_il,implicit_gemm_fp32_128x64x8_t8x4,implicit_gemm_fp32_128x64x8_t8x4_mb3,implicit_gemm_fp32_128x64x8_t8x4_mb3_il,implicit_gemm_fp32_64x128x8_t4x8,implicit_gemm_fp32_64x128x8_t4x8_il,implicit_gemm_fp32_64x128x8_t4x8_mb3_il,implicit_gemm_fp32_64x64x8_t4x4
  "$F" --workloads "$GPU_REPO/benchmarks/workloads.json" --filter thr_b --backends cudnn --precisions fp32 --mode tuned --fusion fused \
       --kernels "$ABL" --no-validate --samples 30 --out "$R/audit_fp32_ablation_${TAG}" --tag "abl_${TAG}" > "$LOGS/fp32_ablation.log" 2>&1
  "$F" --workloads "$GPU_REPO/benchmarks/workloads.json" --filter thr_b --backends cudnn --precisions fp32 --mode tuned --fusion fused \
       --kernels "$VAR" --samples 30 --out "$R/audit_fp32_variants_${TAG}" --tag "var_${TAG}" > "$LOGS/fp32_variants.log" 2>&1
  "$F" --workloads "$GPU_REPO/benchmarks/workloads.json" --category throughput --backends cudnn,cudnn_gemm,implicit_gemm --precisions fp32 \
       --mode both --fusion fused --samples 30 --out "$R/audit_fp32_cudnn_gemm_${TAG}" --tag "cudnngemm_${TAG}" > "$LOGS/fp32_cudnn_gemm.log" 2>&1
  done_stage 2 $?
fi

stage 3
export GIE_TUNE_CACHE="$GPU_WORK/tune_cache_audit_${TAG}.json"
for ws in workloads heldout_workloads; do
  "$B" --workloads "$GPU_REPO/benchmarks/${ws}.json" --backends implicit_gemm,tensor_core,dispatcher,cudnn,cudnn_gemm,cutlass \
       --precisions fp32,fp16 --mode both --fusion fused --e2e --samples 30 \
       --out "$R/audit_repr_${TAG}/${ws}" --tag "repr_${TAG}_${ws}" > "$LOGS/repr_${ws}.log" 2>&1
done
done_stage 3 $?

stage 4
"$GPU_REPO/scripts/run_sanitizers.sh" "$LOGS/sanitizer" > "$LOGS/sanitizer_sweep.log" 2>&1
done_stage 4 $?
echo AUDIT_PIPELINE_DONE | tee -a "$LOGS/stages.log"
