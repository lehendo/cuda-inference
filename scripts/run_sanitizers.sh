#!/bin/bash
# compute-sanitizer sweep over the correctness tests on the reduced shape set (GIE_TEST_REDUCED=1).
# racecheck is ~100x slower, so it only runs the representative subset below.
#   scripts/run_sanitizers.sh [out_dir]      (run on a GPU node)
source "$(dirname "$0")/env.sh"
OUT="${1:-$GPU_WORK/logs/sanitizer}"; mkdir -p "$OUT"
export GIE_UNDER_SANITIZER=1 GIE_TEST_REDUCED=1 GIE_TUNE_CACHE="${GIE_TUNE_CACHE:-/tmp/gie_tune_sanitizer.json}"
T="$GPU_BUILD_DIR/tests"
run() {  # tool test [filter]
  local tool="$1" t="$2" f="${3:-}"
  compute-sanitizer --tool "$tool" --error-exitcode 99 --print-limit 5 "$T/$t" $f > "$OUT/${tool}_${t}${f:+_$f}.log" 2>&1
  echo "$tool $t ${f:+(filter $f) }exit=$? $(grep -hE 'ERROR SUMMARY|RACECHECK SUMMARY|tests,' "$OUT/${tool}_${t}${f:+_$f}.log" | tr '\n' ' ')"
}
for t in test_validation test_implicit_gemm test_tensor_core test_direct_conv test_fusion test_rmsnorm test_runtime_concurrency test_cuda_graph; do run memcheck "$t"; done
for t in test_validation test_implicit_gemm test_tensor_core test_direct_conv; do run racecheck "$t"; done
run racecheck test_fusion relu_zeroes
# concurrency / graph tests: racecheck is too slow for the whole files, so one representative case each
run racecheck test_runtime_concurrency create_then_use_on_fresh_stream_is_race_free
run racecheck test_runtime_concurrency one_plan_concurrent_streams_changing_inputs
run racecheck test_cuda_graph graph_replay_with_changing_inputs_matches_reference
run racecheck test_rmsnorm
for t in test_validation test_implicit_gemm test_tensor_core test_direct_conv test_fusion test_rmsnorm test_runtime_concurrency test_cuda_graph; do run initcheck "$t"; done
# synccheck: our own kernels only. On Hopper it flags a divergent barrier inside a cuBLAS nvJet kernel (library code,
# used by the im2col baseline) and aborts the process, so cuBLAS-backed kernels are excluded from this one tool.
export GIE_TEST_SKIP_CUBLAS=1
for t in test_validation test_implicit_gemm test_tensor_core test_direct_conv test_fusion test_rmsnorm test_runtime_concurrency test_cuda_graph; do run synccheck "$t"; done
unset GIE_TEST_SKIP_CUBLAS
echo SWEEP_DONE
