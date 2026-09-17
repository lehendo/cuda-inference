#!/bin/bash
# Configure and build (Release by default).
#   scripts/build.sh [extra cmake -D flags...]
# Env: GPU_CUDA_ARCH (default "80;86;89;90"), GPU_BUILD_TYPE (Release|Debug|RelWithDebInfo), JOBS.
set -euo pipefail
source "$(dirname "$0")/env.sh"
JOBS="${JOBS:-$(nproc)}"
BUILD_TYPE="${GPU_BUILD_TYPE:-Release}"
EXTRA=()
[ -n "${CUDNN_ROOT:-}" ] && [ -d "$CUDNN_ROOT" ] && EXTRA+=("-DCUDNN_ROOT=$CUDNN_ROOT")
[ -d "${CUTLASS_DIR:-/nonexistent}/include/cutlass" ] && EXTRA+=("-DCUTLASS_DIR=$CUTLASS_DIR")
cmake -S "$GPU_REPO" -B "$GPU_BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" "${EXTRA[@]}" "$@"
cmake --build "$GPU_BUILD_DIR" -j "$JOBS"
