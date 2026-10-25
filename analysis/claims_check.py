#!/usr/bin/env python3
"""Numbers cited in the README and reports that need a direct lookup (sweep trends, wins vs im2col, search cost, calibration)."""
import sys

import numpy as np
import pandas as pd

from common import geomean

root = sys.argv[1] if len(sys.argv) > 1 else "results"
P = root + "/processed"
pd.set_option("display.width", 220)
al = pd.read_csv(f"{P}/all_results.csv")
ok = al[(al.status == "ok") & (al.fusion == "fused") & (al.precision == "fp16")]

print("## implicit_gemm:tuned vs im2col_gemm (fused), wins/total per GPU/precision/split")
sp = pd.read_csv(f"{P}/speedup_vs_im2col_gemm.csv")
sp = sp[(sp.fusion == "fused") & (sp.variant == "implicit_gemm:tuned")]
print(sp.groupby(["gpu", "precision", "split"]).apply(lambda g: f"{(g.speedup > 1).sum()}/{len(g)}  geomean {geomean(g.speedup):.2f}").to_string())

sw = {"sweep_batch": "B", "sweep_kernel": "kernel_area", "sweep_channels": "C", "sweep_spatial": "H", "sweep_irregular": "H"}
for gpu in sorted(ok.gpu.unique()):
    for cat, x in sw.items():
        s = ok[(ok.gpu == gpu) & (ok.category == cat)]
        t = s.pivot_table(index=x, columns="variant", values="tflops", aggfunc="first")
        cols = [c for c in ["tensor_core:tuned", "implicit_gemm:tuned", "im2col_gemm", "cudnn:library", "cutlass:library", "dispatcher:auto"] if c in t]
        print(f"\n## {gpu} {cat} FP16 effective TFLOP/s vs {x}")
        print(t[cols].round(1).to_string())

for g in ("a40", "h100"):
    d = pd.read_csv(f"{P}/experiment_b_{g}/experiment_b_summary.csv")
    print(f"\n## exp B {g}: search cost and calibration")
    cols = [c for c in d.columns if c in ("split", "precision", "exhaustive_search_s_median", "candidates_measured_median", "prior_k3_geomean_slowdown", "calibrated_k3_geomean_slowdown", "prior_k1_geomean_slowdown", "calibrated_k1_geomean_slowdown", "prior_k5_geomean_slowdown", "calibrated_k5_geomean_slowdown", "spearman_prior_median", "spearman_calibrated_median")]
    print(d[cols].round(3).to_string(index=False))
