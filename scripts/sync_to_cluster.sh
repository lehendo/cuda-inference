#!/bin/bash
# Mirror the local working tree to the cluster checkout (source only; builds and
# results live under $GPU_WORK on scratch). Local tree is the source of truth.
set -euo pipefail
HOST="${GPU_CLUSTER_HOST:-cc-login.campuscluster.illinois.edu}"
REMOTE_DIR="${GPU_REMOTE_REPO:-cuda-inference}"
cd "$(dirname "$0")/.."
rsync -az --delete --exclude '.git/' --exclude '.claude/' --exclude 'build*/' \
    --exclude 'results/' --exclude '__pycache__/' ./ "${HOST}:${REMOTE_DIR}/"
