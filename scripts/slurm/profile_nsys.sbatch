#!/bin/bash
# Nsight Systems: GPU_NSYS_WORKLOAD=<substring> GPU_NSYS_BACKEND=<backend> [GPU_NSYS_PRECISION]
#SBATCH --job-name=gie_nsys
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -u
source "${GPU_REPO:-$HOME/cuda-inference}/scripts/env.sh"
"$GPU_REPO/scripts/build.sh" >/dev/null
"$GPU_REPO/scripts/run_nsys.sh" "${GPU_NSYS_WORKLOAD:?}" "${GPU_NSYS_BACKEND:?}" "${GPU_NSYS_PRECISION:-fp16}"
