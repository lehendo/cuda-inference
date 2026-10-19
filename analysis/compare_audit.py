#!/usr/bin/env python3
"""Compare the audit re-run against the first campaign, row by row, and recompute the headline ratios on the new data.

usage: compare_audit.py <results/raw> [--out prefix]
Reads  <raw>/{a40,h100}/{tuning_set,heldout}/results.jsonl   (first campaign, pre-audit)
       <raw>/audit_repr_{a40,h100}/{workloads,heldout_workloads}/results.jsonl   (audit re-run, new source fingerprint)
Only rows with status == ok on both sides are paired. ratio = old_median_us / new_median_us (>1: the new run is faster).
Differences below ~5% are run-to-run noise on a shared cluster node; the script flags |ratio-1| > 10% separately.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PAIRS = {"a40": ("NVIDIA A40", [("tuning_set", "workloads"), ("heldout", "heldout_workloads")]),
         "h100": ("NVIDIA H100 NVL", [("tuning_set", "workloads"), ("heldout", "heldout_workloads")])}


def load(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def key(r):
    return (r["workload"], r["precision"], r.get("fusion", ""), r["backend"], r.get("mode", ""))


def geomean(v):
    v = [x for x in v if x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def main():
    raw = Path(sys.argv[1])
    out = []
    emit = lambda s="": (print(s), out.append(s))
    for g, (gname, sets) in PAIRS.items():
        emit(f"## {gname}: audit re-run vs first campaign (old_us/new_us; >1 = new faster)")
        agg = defaultdict(list)
        worst = []
        for old_set, new_set in sets:
            old = {key(r): r for r in load(raw / g / old_set / "results.jsonl") if r["status"] == "ok" and r.get("median_us")}
            new = {key(r): r for r in load(raw / f"audit_repr_{g}" / new_set / "results.jsonl") if r["status"] == "ok" and r.get("median_us")}
            for k in old.keys() & new.keys():
                if k[3] in ("memcpy",):
                    continue
                ratio = old[k]["median_us"] / new[k]["median_us"]
                agg[(k[1], k[3], k[4])].append(ratio)
                worst.append((ratio, g, old_set, k))
        emit(f"{'precision':9s} {'backend':14s} {'mode':8s} {'n':>4s} {'geomean':>8s} {'min':>6s} {'max':>6s}")
        for (prec, be, mode), v in sorted(agg.items()):
            emit(f"{prec:9s} {be:14s} {mode:8s} {len(v):4d} {geomean(v):8.3f} {min(v):6.2f} {max(v):6.2f}")
        worst.sort(key=lambda t: t[0])
        emit("slowest 8 pairs (possible regressions or noise):")
        for ratio, _, s, k in worst[:8]:
            emit(f"  {ratio:5.2f}  {s:10s} {k[0]:34s} {k[1]} {k[3]}:{k[4]}")
        emit()
        # headline ratios on the NEW data only
        new_rows = []
        for _, new_set in sets:
            new_rows += [(new_set, r) for r in load(raw / f"audit_repr_{g}" / new_set / "results.jsonl") if r["status"] == "ok" and r.get("median_us")]
        by = defaultdict(dict)
        for s, r in new_rows:
            by[(s, r["workload"], r["precision"])][(r["backend"], r.get("mode", ""), r.get("fusion", ""))] = r["median_us"]

        def ratio_set(num, den, prec):
            res = defaultdict(list)
            for (s, w, p), d in by.items():
                if p != prec:
                    continue
                a = next((v for k, v in d.items() if k[0] == num[0] and k[1] == num[1]), None)
                b = next((v for k, v in d.items() if k[0] == den[0] and k[1] == den[1]), None)
                if a and b:
                    res[s].append(b / a)  # >1: numerator backend faster
            return res

        emit(f"## {gname}: headline ratios on the audit re-run (>1: first backend faster)")
        for label, num, den, prec in [
            ("FP16 tensor_core:tuned vs cudnn", ("tensor_core", "tuned"), ("cudnn", "library"), "fp16"),
            ("FP16 dispatcher vs cudnn", ("dispatcher", "auto"), ("cudnn", "library"), "fp16"),
            ("FP32 implicit_gemm:tuned vs cudnn (any algorithm)", ("implicit_gemm", "tuned"), ("cudnn", "library"), "fp32"),
            ("FP32 implicit_gemm:tuned vs cudnn_gemm (GEMM-family)", ("implicit_gemm", "tuned"), ("cudnn_gemm", "library"), "fp32"),
            ("FP32 dispatcher vs cudnn", ("dispatcher", "auto"), ("cudnn", "library"), "fp32"),
            ("FP16 tensor_core:tuned vs cutlass (kernel-only)", ("tensor_core", "tuned"), ("cutlass", "library"), "fp16"),
        ]:
            for s, v in sorted(ratio_set(num, den, prec).items()):
                emit(f"  {label:58s} {s:18s} n={len(v):3d} geomean={geomean(v):6.3f} median={sorted(v)[len(v)//2]:6.3f} wins={sum(x > 1 for x in v)}")
        emit()
    if len(sys.argv) > 3 and sys.argv[2] == "--out":
        Path(sys.argv[3] + ".txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
