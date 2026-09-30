#!/usr/bin/env python3
"""(1) Warm-L2 vs cold-L2 protocol: slowdown of each variant when L2 is flushed before every launch.
(2) Measurement noise: spread of per-sample latencies as (p90 - p10) / median, per GPU and variant family."""
import sys

import numpy as np
import pandas as pd

from common import geomean, load_results, ok_rows

root = (sys.argv[1] if len(sys.argv) > 1 else "results") + "/raw"
pd.set_option("display.width", 200)
warm = ok_rows(load_results([f"{root}/a40/tuning_set", f"{root}/h100/tuning_set"]))
cold = ok_rows(load_results([f"{root}/a40/cold_l2", f"{root}/h100/cold_l2"]))
k = ["gpu", "workload", "precision", "variant"]
w = warm[(warm.fusion == "fused")][k + ["median_us"]].drop_duplicates(k)
c = cold[k + ["median_us"]].drop_duplicates(k)
m = w.merge(c, on=k, suffixes=("_warm", "_cold"))
m["cold_over_warm"] = m.median_us_cold / m.median_us_warm
print("## cold-L2 / warm-L2 latency (geomean over workloads); >1 = slower when inputs are not L2-resident")
print(m.groupby(["gpu", "precision", "variant"]).cold_over_warm.agg(n="count", geomean=geomean, max="max").round(2).to_string())
warm = warm[warm.samples >= 20]
warm = warm.assign(spread=(warm.p90_us - warm.p10_us) / warm.median_us, p99_over_med=warm.p99_us / warm.median_us)
print("\n## per-sample spread (p90-p10)/median and p99/median, tuning set (fused), by GPU")
print(warm.groupby("gpu")[["spread", "p99_over_med"]].describe(percentiles=[0.5, 0.9]).round(3).T.to_string())
short = warm[warm.median_us < 15]
print("\n## kernels shorter than 15 us:", len(short), " median spread", round(float(short.spread.median()), 3))
