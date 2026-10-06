#!/bin/bash
# Nsight Compute: GPU_NCU_WORKLOAD=<substring> GPU_NCU_BACKEND=<backend> [GPU_NCU_PRECISION GPU_NCU_SET]
#SBATCH --job-name=gie_ncu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -u
source "${GPU_REPO:-$HOME/cuda-inference}/scripts/env.sh"
"$GPU_REPO/scripts/build.sh" >/dev/null
"$GPU_REPO/scripts/run_ncu.sh" "${GPU_NCU_WORKLOAD:?}" "${GPU_NCU_BACKEND:?}" "${GPU_NCU_PRECISION:-fp16}" "${GPU_NCU_SET:-basic}"
