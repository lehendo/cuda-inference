#!/bin/bash
# Nsight Compute profile of one benchmark workload/backend.
#   scripts/run_ncu.sh <workload-substring> <backend> [precision] [section-set]
# Section sets: basic (default) | full | roofline | tensor | memory
# Requires GPU performance counter access. If the driver restricts counters to administrators
# (ERR_NVGPUCTRPERM) the run is reported as blocked -- no privilege workaround is attempted.
set -uo pipefail
source "$(dirname "$0")/env.sh"
WL="${1:?workload substring}"; BE="${2:?backend}"; PREC="${3:-fp16}"; SET="${4:-basic}"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
OUT="${GPU_OUT_DIR:-$GPU_REPO/results/raw/ncu_${GPU_NAME}}"; mkdir -p "$OUT"
case "$SET" in
  basic)    SECTIONS=(--section SpeedOfLight --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis) ;;
  roofline) SECTIONS=(--section SpeedOfLight --section SpeedOfLight_RooflineChart) ;;
  tensor)   SECTIONS=(--section SpeedOfLight --section ComputeWorkloadAnalysis --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_tensor_op_hmma.sum) ;;
  memory)   SECTIONS=(--section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables) ;;
  full)     SECTIONS=(--set full) ;;
  *) echo "unknown set $SET" >&2; exit 2 ;;
esac
# --samples 3 + --warmup 1 keeps the profiled launch count tiny; ncu replays each kernel several times.
LOG="$OUT/ncu_${WL}_${BE}_${PREC}_${SET}.log"
ncu "${SECTIONS[@]}" --kernel-name-base demangled --launch-skip 2 --launch-count 1 --target-processes all \
    -o "$OUT/ncu_${WL}_${BE}_${PREC}_${SET}" --force-overwrite \
    "$GPU_BUILD_DIR/benchmarks/gie_bench" --workloads "$GPU_REPO/benchmarks/workloads.json" --filter "$WL" \
    --backends "$BE" --precisions "$PREC" --mode fixed --warmup 1 --samples 1 --min-sample-ms 0 --no-validate \
    --out "$OUT/bench_${WL}_${BE}_${PREC}" 2>&1 | tee "$LOG"
if grep -q "ERR_NVGPUCTRPERM" "$LOG"; then
  echo "BLOCKED: GPU performance counters are not accessible to this user (ERR_NVGPUCTRPERM)." | tee -a "$LOG"
  exit 3
fi
