#!/bin/bash
# Follow-up audit job for one GPU (detached): validates the cuDNN baseline re-timing change and the sanitizer-script fixes.
#   1 incremental rebuild + native CTest
#   2 cuDNN / cuDNN-GEMM-family rows on the tuning and held-out sets, run twice (rep1, rep2) to measure algorithm-choice stability
#   3 synccheck on test_direct_conv with the cuBLAS-backed backend excluded (coverage-check fix)
#   4 gie_graph_bench (graph replay + stream concurrency) with no profiler attached, fp16 and fp32
# Required env: AUDIT_TAG (a40|h100). Outputs go to new directories only.
#SBATCH --job-name=gie_audit2
#SBATCH --output=%x_%j.out
set -uo pipefail
source "${GPU_REPO:-$HOME/cuda-inference}/scripts/env.sh"
TAG="${AUDIT_TAG:?AUDIT_TAG}"
export GPU_BUILD_DIR="${GPU_BUILD_DIR:-$GPU_WORK/build_audit}"
LOGS="$GPU_WORK/logs/audit_${TAG}_v3"; mkdir -p "$LOGS"
R="$GPU_REPO/results/raw/audit_cudnn_retime_${TAG}"
echo "stage 1 $(date -Is)" | tee -a "$LOGS/stages.log"
(cd "$GPU_REPO" && GPU_CUDA_ARCH="${GPU_CUDA_ARCH:-80;86;89;90}" scripts/build.sh) > "$LOGS/build.log" 2>&1
ctest --test-dir "$GPU_BUILD_DIR" --output-on-failure > "$LOGS/ctest.log" 2>&1
B="$GPU_BUILD_DIR/benchmarks/gie_bench"
echo "stage 2 $(date -Is)" | tee -a "$LOGS/stages.log"
for rep in rep1 rep2; do
  for ws in workloads heldout_workloads; do
    "$B" --workloads "$GPU_REPO/benchmarks/${ws}.json" --backends cudnn,cudnn_gemm --precisions fp32,fp16 --mode both --fusion fused \
         --samples 30 --out "$R/$rep/$ws" --tag "retime_${TAG}_${rep}_${ws}" > "$LOGS/cudnn_${rep}_${ws}.log" 2>&1
  done
done
echo "stage 3 $(date -Is)" | tee -a "$LOGS/stages.log"
export GIE_UNDER_SANITIZER=1 GIE_TEST_REDUCED=1 GIE_TEST_SKIP_CUBLAS=1
compute-sanitizer --tool synccheck --error-exitcode 99 "$GPU_BUILD_DIR/tests/test_direct_conv" > "$LOGS/synccheck_direct_conv.log" 2>&1
echo "synccheck test_direct_conv exit=$?" | tee -a "$LOGS/stages.log"
echo "stage 4 $(date -Is)" | tee -a "$LOGS/stages.log"
# CUDA Graph / stream-concurrency benchmark WITHOUT a profiler attached (the Nsight Systems run perturbs launch-bound timings)
for prec in fp16 fp32; do
  "$GPU_BUILD_DIR/benchmarks/gie_graph_bench" --out "$GPU_REPO/results/raw/audit_graphs_noprof_${TAG}" --precision $prec > "$LOGS/graphs_${prec}.log" 2>&1
done
echo AUDIT2_DONE | tee -a "$LOGS/stages.log"
