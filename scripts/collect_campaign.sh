#!/bin/bash
# Copy one campaign's raw outputs into the repository's results/raw/<name>/ and (re)build processed tables and
# figures for everything under results/raw. Run on the cluster (or anywhere with the python env).
#   scripts/collect_campaign.sh <campaign_dir> <name>      e.g.  scripts/collect_campaign.sh $GPU_WORK/campaign_v3_a40 a40
set -euo pipefail
source "$(dirname "$0")/env.sh"
SRC="$1"; NAME="$2"
DST="$GPU_REPO/results/raw/$NAME"
mkdir -p "$DST"
for d in tuning_set heldout sweeps precision_bf16_tf32 cold_l2 graphs rmsnorm experiment_b; do
  [ -d "$SRC/$d" ] && rm -rf "$DST/$d" && cp -r "$SRC/$d" "$DST/$d"
done
cp -f "$SRC/gie_info.txt" "$SRC/tune_cache.json" "$SRC/campaign.log" "$DST/" 2>/dev/null || true
echo "collected $SRC -> $DST"
