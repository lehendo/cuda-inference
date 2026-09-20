#!/usr/bin/env python3
"""Pivot table of median latency (us) per workload and backend/mode from gie_bench results.jsonl files.
usage: compare_variants.py [--precision fp16] [--fusion fused] results.jsonl [...]"""
import argparse
import json
from collections import OrderedDict

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--precision", default="fp16")
ap.add_argument("--fusion", default="fused")
a = ap.parse_args()
rows = []
for f in a.files:
    rows += [json.loads(l) for l in open(f) if l.strip()]
table, kern, cols = OrderedDict(), {}, []
for r in rows:
    if r["precision"] != a.precision or r["fusion"] != a.fusion or r["status"] != "ok":
        continue
    v = r["backend"] + ((":" + r["mode"]) if r["mode"] in ("tuned", "fixed") and r["backend"] in ("simt_tiled", "implicit_gemm", "tensor_core") else "")
    if v not in cols:
        cols.append(v)
    table.setdefault(r["workload"], {})[v] = r["median_us"]
    kern[(r["workload"], v)] = r.get("kernel", "")
print(f"{'workload':32s}" + "".join(f"{c[:14]:>15s}" for c in cols))
for w, d in table.items():
    print(f"{w[:31]:32s}" + "".join(f"{d[c]:15.1f}" if c in d else f"{'--':>15s}" for c in cols))
print()
for w in table:
    for c in cols:
        if c.startswith("tensor_core") and (w, c) in kern:
            print(f"{w[:31]:32s} {c:18s} {kern[(w, c)]}")
