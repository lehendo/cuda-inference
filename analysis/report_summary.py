#!/usr/bin/env python3
"""Prints the headline tables used in the results write-up, all derived from results/processed + results/raw."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from common import geomean, load_results

root = sys.argv[1] if len(sys.argv) > 1 else "results"
P = os.path.join(root, "processed")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 80)

print("## environments"); print(pd.read_csv(f"{P}/environments.csv").drop_duplicates(["gpu"]).to_string(index=False))
st = pd.read_csv(f"{P}/status_counts.csv"); print("\n## status counts"); print(st.groupby(["gpu"]).sum(numeric_only=True).to_string())

a = pd.read_csv(f"{P}/aggregate_speedups.csv")
print("\n## geomean speedup vs cuDNN (fused rows)")
x = a[(a.baseline == "cudnn:library") & (a.fusion == "fused")]
print(x[["gpu", "precision", "split", "variant", "n_workloads", "geomean_speedup", "median_speedup", "min_speedup", "max_speedup", "wins", "losses"]].round(2).to_string(index=False))
print("\n## geomean speedup vs naive direct CUDA")
x = a[(a.baseline == "direct") & (a.fusion == "fused") & (a.split == "tune")]
print(x[["gpu", "precision", "variant", "n_workloads", "geomean_speedup", "wins", "losses"]].round(1).to_string(index=False))

print("\n## autotune vs fixed")
t = pd.read_csv(f"{P}/autotune_vs_fixed.csv")
print(t.groupby(["gpu", "precision", "backend"]).tuned_over_fixed_speedup.agg(n="count", geomean=geomean, median="median", max="max").round(2).to_string())

print("\n## fusion (unfused / fused time)")
f = pd.read_csv(f"{P}/fusion.csv")
print(f.groupby(["gpu", "precision", "variant"]).fusion_speedup.agg(n="count", geomean=geomean, median="median", max="max").round(2).to_string())

print("\n## tensor core vs SIMT implicit GEMM (tuned)")
tc = pd.read_csv(f"{P}/tensor_core_vs_simt_tuned.csv")
print(tc.groupby(["gpu", "precision"]).tc_over_simt_speedup.agg(n="count", geomean=geomean, median="median", min="min", max="max").round(2).to_string())

print("\n## workspace memory (bytes, median / max across workloads)")
print(pd.read_csv(f"{P}/workspace_memory.csv").to_string(index=False))

print("\n## im2col accounting (median fraction of pipeline)")
im = pd.read_csv(f"{P}/im2col_accounting.csv")
im["gemm_frac"] = im.gemm_only_us / im.pipeline_us
im["im2col_frac"] = im.im2col_us / im.pipeline_us
print(im.groupby(["gpu", "precision"])[["gemm_frac", "im2col_frac"]].median().round(2).to_string())

print("\n## cutlass layout accounting (layout-inclusive / kernel-only)")
c = pd.read_csv(f"{P}/cutlass_layout_accounting.csv"); c["ratio"] = c.layout_inclusive_us / c.median_us
print(c.groupby(["gpu", "precision"]).ratio.agg(n="count", median="median", max="max").round(2).to_string())

print("\n## dispatcher selection (cache-backed) regret vs best custom")
d = pd.read_csv(f"{P}/dispatcher_selection.csv")
print(d.groupby(["gpu", "precision"]).regret.agg(n="count", geomean=geomean, max="max").round(3).to_string())

print("\n## win counts (fused, tuning set)")
w = pd.read_csv(f"{P}/win_counts.csv"); w = w[(w.fusion == "fused") & (w.split == "tune")]
print(w.pivot_table(index=["gpu", "precision"], columns="winner", values="workloads_won", aggfunc="sum", fill_value=0).to_string())

fl = pd.read_csv(f"{P}/failures.csv")
print("\n## non-ok rows"); print(fl.groupby(["gpu", "status", "backend"]).size().to_string())
fv = fl[fl.status == "failed_validation"]
print("\nfailed_validation rows by backend/precision:"); print(fv.groupby(["gpu", "backend", "precision"]).size().to_string() if len(fv) else "none")

print("\n## sweeps: Tensor Core/dispatcher best backend by category")
al = pd.read_csv(f"{P}/all_results.csv")
ok = al[(al.status == "ok") & (al.fusion == "fused")]
ws = pd.read_csv(f"{P}/winners.csv"); ws = ws[(ws.fusion == "fused")]
print(ws.merge(al[["workload", "category"]].drop_duplicates(), on="workload").groupby(["gpu", "precision", "category", "winner"]).size().unstack(fill_value=0).to_string())

print("\n## experiments from raw")
for g in sorted(glob.glob(f"{root}/raw/*/")):
    name = os.path.basename(g.rstrip("/"))
    for fn in glob.glob(f"{g}graphs/graphs.jsonl"):
        r = pd.DataFrame([json.loads(l) for l in open(fn)])
        gr = r[r.experiment == "F_cuda_graphs"]
        print(f"\n[{name}] CUDA Graphs (device_speedup eager/graph, host launch us eager->graph, capture+instantiate us)")
        print(gr.groupby(["precision", "layers"]).agg(device_speedup=("device_speedup", "median"), eager_host=("eager_host_launch_us", "median"), graph_host=("graph_host_launch_us", "median"),
                                                     capture=("capture_us", "median"), inst=("instantiate_us", "median"), identical=("bitwise_identical_replays", "all")).round(2).to_string())
        sc = r[r.experiment == "F_stream_concurrency"]
        print(sc.groupby(["kernel", "independent_convs"]).speedup.median().round(2).to_string() if len(sc) else "")
    for fn in glob.glob(f"{g}rmsnorm/rmsnorm.jsonl"):
        r = pd.DataFrame([json.loads(l) for l in open(fn)])
        print(f"\n[{name}] RMSNorm fused-kernel speedup over unfused baseline; bandwidth GB/s (median over shapes)")
        print(r.groupby(["dtype", "fused_add"]).agg(speedup=("speedup", "median"), gbs=("fused_gbs", "median")).round(2).to_string())

print("\n## tensor_core:tuned vs CUTLASS (kernel-only and layout-inclusive); speedup = CUTLASS time / my time")
al = pd.read_csv(f"{P}/all_results.csv")
ok = al[(al.status == "ok") & (al.fusion == "fused") & (al.precision.isin(["fp16", "bf16"]))]
tcm = ok[ok.variant == "tensor_core:tuned"][["gpu", "workload", "precision", "split", "median_us"]].rename(columns={"median_us": "mine_us"})
cu = ok[ok.variant == "cutlass:library"][["gpu", "workload", "precision", "split", "median_us", "layout_inclusive_us"]]
m = tcm.merge(cu, on=["gpu", "workload", "precision", "split"])
m["vs_cutlass_kernel_only"] = m.median_us / m.mine_us
m["vs_cutlass_layout_incl"] = m.layout_inclusive_us / m.mine_us
print(m.groupby(["gpu", "precision", "split"])[["vs_cutlass_kernel_only", "vs_cutlass_layout_incl"]].agg(geomean).round(2).to_string())
print("workloads where my tuned TC (incl. its own transposes) beats CUTLASS layout-inclusive:",
      m.groupby(["gpu", "split"]).apply(lambda g: f"{(g.vs_cutlass_layout_incl > 1).sum()}/{len(g)}").to_dict())
