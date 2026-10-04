#!/bin/bash
# Benchmark ONE backend:   GPU_BENCH_BACKENDS=implicit_gemm scripts/submit.sh scripts/slurm/bench_backend.sbatch
#SBATCH --job-name=gie_bench1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eu
source "${GPU_REPO:-$HOME/cuda-inference}/scripts/env.sh"
"$GPU_REPO/scripts/build.sh" >/dev/null
export GPU_BENCH_BACKENDS="${GPU_BENCH_BACKENDS:?set GPU_BENCH_BACKENDS to one backend name}"
"$GPU_REPO/scripts/run_benchmarks.sh" "$@"
