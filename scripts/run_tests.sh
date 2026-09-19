#!/bin/bash
# Build (if needed) and run the full correctness suite on the current GPU node.
#   scripts/run_tests.sh                run all tests via CTest
#   SANITIZE=memcheck scripts/run_tests.sh    run a compact test subset under compute-sanitizer
#                                              (memcheck | racecheck | initcheck | synccheck)
set -euo pipefail
source "$(dirname "$0")/env.sh"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
[ -f "$GPU_BUILD_DIR/CMakeCache.txt" ] || "$GPU_REPO/scripts/build.sh"
cmake --build "$GPU_BUILD_DIR" -j "$(nproc)"
if [ -n "${SANITIZE:-}" ]; then
  # Sanitizers slow kernels by 10-100x: run each backend's suite on the reduced shape list.
  export GIE_TEST_REDUCED=1
  for t in test_validation test_direct_conv test_implicit_gemm test_tensor_core test_fusion; do
    echo "== compute-sanitizer --tool $SANITIZE : $t"
    compute-sanitizer --tool "$SANITIZE" --error-exitcode 99 "$GPU_BUILD_DIR/tests/$t"
  done
else
  ctest --test-dir "$GPU_BUILD_DIR" --output-on-failure "$@"
fi
