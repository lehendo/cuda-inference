#!/usr/bin/env python3
"""Experiment C (precision): timing and accuracy of FP32 / TF32 / FP16 / BF16 on the throughput workloads, per GPU.
Timing is each precision's best available implementation (dispatcher, cache-backed); accuracy is the measured error of
that implementation against the double-precision reference. Different precisions are NOT numerically equivalent."""
import sys

import numpy as np
import pandas as pd

from common import geomean

P = (sys.argv[1] if len(sys.argv) > 1 else "results") + "/processed"
pd.set_option("display.width", 220)
al = pd.read_csv(f"{P}/all_results.csv")
t = al[(al.status == "ok") & (al.fusion == "fused") & (al.variant == "dispatcher:auto") & (al.category == "throughput") & (al.split == "tune")]
w = t.pivot_table(index=["gpu", "workload"], columns="precision", values="median_us", aggfunc="first")
print("## dispatcher latency (us) on throughput workloads"); print(w.round(0).to_string())
rel = pd.DataFrame({f"{p}_vs_fp32_speedup": w["fp32"] / w[p] for p in ("tf32", "fp16", "bf16") if p in w})
print("\n## geomean speedup over strict FP32 (same workload, different numerics!)")
print(rel.groupby(level=0).agg(geomean).round(2).to_string())
e = t.groupby(["gpu", "precision"]).agg(max_abs_err=("max_abs_err", "max"), mean_abs_err=("mean_abs_err", "median"), rms_err=("rms_err", "median"),
                                        worst_tol_ratio=("worst_tol_ratio", "max"), kernel=("kernel", lambda s: ", ".join(sorted(set(s))[:3])))
print("\n## accuracy of the selected implementation vs double-precision reference"); print(e.to_string())
