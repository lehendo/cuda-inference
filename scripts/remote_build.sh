#!/bin/bash
# Helper for development: build on an existing Slurm allocation (CPU or GPU) from the login node.
#   scripts/remote_build.sh <jobid> [arch] [build_dir_name] [extra cmake flags...]  -> log in $GPU_WORK/logs/build_<name>.log
set -euo pipefail
JOB="$1"; ARCH="${2:-90}"; NAME="${3:-build$ARCH}"; shift 3 || true
EXTRA="$*"
HOST="${GPU_CLUSTER_HOST:-cc-login.campuscluster.illinois.edu}"
LOGDIR='/scratch/$USER/cuda-inference-work/logs'
ssh "$HOST" "mkdir -p $LOGDIR; nohup srun --jobid=$JOB --overlap bash -c 'cd ~/${GPU_REMOTE_REPO:-cuda-inference} && export GPU_CUDA_ARCH=\"$ARCH\" GPU_BUILD_DIR=/scratch/\$USER/cuda-inference-work/$NAME && scripts/build.sh $EXTRA' > $LOGDIR/build_$NAME.log 2>&1 &"
echo "build started: log at $LOGDIR/build_$NAME.log"
