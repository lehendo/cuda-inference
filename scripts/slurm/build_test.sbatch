#!/bin/bash
# Build the project and run the complete CTest suite on one GPU.
#SBATCH --job-name=gie_test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eu
source "${GPU_REPO:-$HOME/cuda-inference}/scripts/env.sh"
"$GPU_REPO/scripts/build.sh"
"$GPU_REPO/scripts/run_tests.sh"
"$GPU_REPO/scripts/check_tensor_core_sass.sh" || true
