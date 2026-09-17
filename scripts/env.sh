#!/bin/bash
# Source this file to get a build/run environment on the Illinois Campus Cluster (or any
# system with Lmod + a CUDA module). Everything is overridable through environment variables.
#   GPU_CUDA_MODULE  CUDA module to load            (default cuda/12.8)
#   GPU_WORK         scratch work dir (builds, results, logs)   (default /scratch/$USER/cuda-inference-work)
#   GPU_PYENV        python env with numpy/pandas/matplotlib   (default /scratch/$USER/envs/ci)
#   GPU_CUDA_ARCH    CMake CUDA architectures, e.g. "90" or "80;86;89;90"
if ! type module >/dev/null 2>&1; then
  for f in /usr/share/lmod/lmod/init/bash /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
    [ -f "$f" ] && source "$f" && break
  done
fi
export GPU_CUDA_MODULE="${GPU_CUDA_MODULE:-cuda/12.8}"
if type module >/dev/null 2>&1; then module load "$GPU_CUDA_MODULE" >/dev/null 2>&1; fi
export GPU_WORK="${GPU_WORK:-/scratch/$USER/cuda-inference-work}"
export GPU_PYENV="${GPU_PYENV:-/scratch/$USER/envs/ci}"
export GPU_REPO="${GPU_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export GPU_BUILD_DIR="${GPU_BUILD_DIR:-$GPU_WORK/build}"
export CUDNN_ROOT="${CUDNN_ROOT:-$(ls -d "$GPU_PYENV"/lib/python*/site-packages/nvidia/cudnn 2>/dev/null | head -1)}"
export CUTLASS_DIR="${CUTLASS_DIR:-$GPU_WORK/deps/cutlass}"
export GIE_TUNE_CACHE="${GIE_TUNE_CACHE:-$GPU_WORK/tune_cache.json}"
export GIE_CEILING_DIR="${GIE_CEILING_DIR:-$GPU_WORK/ceilings}"
mkdir -p "$GPU_WORK" "$GPU_BUILD_DIR" 2>/dev/null
