#!/bin/bash
# Submit one of the scripts in scripts/slurm/ with resources taken from environment variables.
#   GPU_SLURM_ACCOUNT   Slurm account (-A). Empty => omitted. No default is assumed.
#   GPU_SLURM_PARTITION partition        (default: secondary)
#   GPU_SLURM_GRES      gpu request     (default: gpu:1  -- one GPU unless you ask for more)
#   GPU_SLURM_TIME      wall time       (default: 02:00:00; always request more than you expect to need)
#   GPU_SLURM_CPUS / GPU_SLURM_MEM      (default: 8 / 48G)
#   GPU_SLURM_CONSTRAINT optional --constraint (e.g. "A100|H100")
#   GPU_OUT_DIR  result directory       GPU_CUDA_ARCH  CMake CUDA arch list    GPU_BENCH_REPS  samples per measurement
# Usage: scripts/submit.sh scripts/slurm/<job>.sbatch [extra sbatch flags]
set -euo pipefail
source "$(dirname "$0")/env.sh"
SCRIPT="$1"; shift
ARGS=(--chdir "$GPU_WORK/logs" --export=ALL
      --partition="${GPU_SLURM_PARTITION:-secondary}"
      --time="${GPU_SLURM_TIME:-02:00:00}"
      --cpus-per-task="${GPU_SLURM_CPUS:-8}"
      --mem="${GPU_SLURM_MEM:-48G}")
[ "${GPU_SLURM_GRES:-gpu:1}" != "none" ] && ARGS+=(--gres="${GPU_SLURM_GRES:-gpu:1}")
[ -n "${GPU_SLURM_ACCOUNT:-}" ] && ARGS+=(--account="$GPU_SLURM_ACCOUNT")
[ -n "${GPU_SLURM_CONSTRAINT:-}" ] && ARGS+=(--constraint="$GPU_SLURM_CONSTRAINT")
mkdir -p "$GPU_WORK/logs"
export GPU_REPO
exec sbatch "${ARGS[@]}" "$@" "$GPU_REPO/$SCRIPT"
