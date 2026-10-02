#!/usr/bin/env python3
"""Independent recomputation of every headline claim directly from results/raw/*.

This script deliberately does NOT import the aggregation code (common.py / aggregate_results.py) and does not read
results/processed: it re-derives the numbers from the raw JSONL rows so that a bug in the aggregation pipeline cannot
silently propagate. It also audits the raw data itself: duplicate rows, missing rows, rows from kernels that no longer
exist, synthetic fixtures, held-out/tuning leakage hazards and provenance metadata.

usage: verify_claims.py [results_root] [--registry kernels.txt] [--out results/verification/<name>]
"""
import argparse
import collections
import glob
import json
import math
import os
import sys

import numpy as np

SETS = ["tuning_set", "heldout", "sweeps", "precision_bf16_tf32"]


def load_rows(root):
    rows = []
    for gdir in sorted(glob.glob(os.path.join(root, "raw", "*"))):
        gname = os.path.basename(gdir)
        for s in SETS:
            f = os.path.join(gdir, s, "results.jsonl")
            if not os.path.exists(f):
                continue
            with open(f) as fh:
                for ln, line in enumerate(fh):
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    r["_dir"], r["_set"], r["_line"] = gname, s, ln
                    rows.append(r)
    return rows


def variant(r):
    b, m = r["backend"], r.get("mode", "n/a")
    if b in ("simt_tiled", "implicit_gemm", "tensor_core", "cudnn", "cutlass", "dispatcher") and m not in ("n/a", "host"):
        return f"{b}:{m}"
    return b


def geomean(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")


def ratio_table(rows, num_variant, den_variant, split_filter=None, precision=None, fusion="fused", key_extra=("gpu",)):
    """speedup of `num_variant` over `den_variant` := t(den)/t(num) on identical (gpu, workload, precision, fusion)."""
    idx = {}
    for r in rows:
        if r["status"] != "ok" or r.get("median_us") is None or r["fusion"] != fusion:
            continue
        if precision and r["precision"] != precision:
            continue
        if split_filter and r["_set"] not in split_filter:
            continue
        idx.setdefault((r["gpu"], r["_set"], r["workload"], r["precision"]), {})[variant(r)] = r
    out = []
    for k, d in idx.items():
        if num_variant in d and den_variant in d:
            out.append((k, d[den_variant]["median_us"] / d[num_variant]["median_us"]))
    return out


def summarize(vals):
    v = np.array([x for _, x in vals])
    if len(v) == 0:
        return "n=0"
    return f"n={len(v)} geomean={geomean(v):.3f} median={np.median(v):.3f} min={v.min():.3f} max={v.max():.3f} wins={(v > 1).sum()} losses={(v < 1).sum()}"


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")

    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j)
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results")
    ap.add_argument("--registry", help="text file with one registered kernel name per line (from gie_info --list-kernels)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines = []

    def P(*args):
        s = " ".join(str(x) for x in args)
        print(s)
        lines.append(s)

    rows = load_rows(a.root)
    P(f"# independent verification of {a.root}: {len(rows)} raw rows")
    gpus = sorted({r["gpu"] for r in rows})

    # ---------------------------------------------------------------- data integrity
    P("\n## 1. raw-data integrity")
    ident = collections.Counter()
    for r in rows:
        ident[(r["_dir"], r["_set"], r["workload"], r["precision"], r["fusion"], r["backend"], r.get("mode"))] += 1
    dups = {k: v for k, v in ident.items() if v > 1}
    P(f"duplicate (dir,set,workload,precision,fusion,backend,mode) keys: {len(dups)}")
    for k, v in list(dups.items())[:5]:
        P("   dup:", k, v)
    synth = [r for r in rows if "SYNTH" in str(r.get("gpu", "")).upper()]
    P(f"rows from synthetic fixtures: {len(synth)}")
    for p in glob.glob(os.path.join(a.root, "**", "*SYNTHETIC*"), recursive=True):
        P("   synthetic artifact on disk:", p)
    status = collections.Counter((r["gpu"], r["status"]) for r in rows)
    for k in sorted(status):
        P(f"status {k[0]:16s} {k[1]:18s} {status[k]}")
    P("failed_validation rows:", sum(1 for r in rows if r["status"] == "failed_validation"))
    if a.registry and os.path.exists(a.registry):
        reg = set(l.strip() for l in open(a.registry) if l.strip())
        stale = collections.Counter()
        for r in rows:
            kn = r.get("kernel", "")
            if r["backend"] in ("direct", "im2col_gemm", "simt_tiled", "implicit_gemm", "tensor_core", "dispatcher") and kn:
                base = kn.split("+")[0]
                if base not in reg:
                    stale[base] += 1
        P(f"rows naming a kernel not in the CURRENT registry: {sum(stale.values())} ({len(stale)} distinct names)")
        for k, v in list(stale.items())[:10]:
            P("   ", k, v)
    # expected completeness: every (gpu,set,workload,precision) should have a row per requested backend/mode
    cells = collections.defaultdict(set)
    for r in rows:
        if r["fusion"] == "fused":
            cells[(r["_dir"], r["_set"], r["workload"], r["precision"])].add(variant(r))
    expect_core = {"direct", "im2col_gemm", "dispatcher:auto", "cudnn:library", "cutlass:library"}
    missing = [(k, expect_core - v) for k, v in cells.items() if k[1] in ("tuning_set", "heldout", "sweeps") and (expect_core - v)]
    P(f"(workload, precision) cells missing a core backend row: {len(missing)}")
    for k, m in missing[:5]:
        P("   ", k, sorted(m))
    meta_seen = {}
    for g in glob.glob(os.path.join(a.root, "raw", "*", "*", "meta.json")):
        m = json.load(open(g))
        meta_seen[g] = (m.get("gpu"), m.get("library_version"), m.get("cuda_runtime"), m.get("cudnn"), m.get("cutlass"), m.get("source_fingerprint", "(none: pre-audit run)"))
    P(f"meta.json files: {len(meta_seen)}; distinct (gpu, lib_version, cudart, cudnn, cutlass, fingerprint): {len(set(meta_seen.values()))}")
    for v in sorted(set(meta_seen.values()), key=str):
        P("   ", v)

    # ---------------------------------------------------------------- headline performance claims
    P("\n## 2. FP16 custom Tensor Core (tuned) vs cuDNN (>1 = custom faster)")
    for g in gpus:
        for s in ("tuning_set", "heldout", "sweeps"):
            sub = [r for r in rows if r["gpu"] == g]
            P(f"{g:16s} {s:11s} FP16 TC-tuned/cuDNN   ", summarize(ratio_table(sub, "tensor_core:tuned", "cudnn:library", [s], "fp16")))
    P("\n## 3. CUTLASS comparison (FP16+BF16), ratio = CUTLASS time / custom time (<1: CUTLASS faster)")
    for g in gpus:
        sub = [r for r in rows if r["gpu"] == g]
        for s in ("tuning_set", "heldout"):
            kernel_only, layout_incl = [], []
            idx = {}
            for r in sub:
                if r["status"] == "ok" and r["fusion"] == "fused" and r["_set"] == s and r["precision"] in ("fp16", "bf16"):
                    idx.setdefault((r["workload"], r["precision"]), {})[variant(r)] = r
            for k, d in idx.items():
                if "tensor_core:tuned" in d and "cutlass:library" in d:
                    mine = d["tensor_core:tuned"]["median_us"]
                    c = d["cutlass:library"]
                    kernel_only.append((k, c["median_us"] / mine))
                    if c.get("layout_inclusive_us"):
                        layout_incl.append((k, c["layout_inclusive_us"] / mine))
            P(f"{g:16s} {s:11s} kernel-only      ", summarize(kernel_only))
            P(f"{g:16s} {s:11s} layout-inclusive ", summarize(layout_incl))
    P("\n## 4. FP32: custom vs cuDNN FP32 (accuracy-gated)")
    for g in gpus:
        sub = [r for r in rows if r["gpu"] == g]
        for v in ("implicit_gemm:tuned", "simt_tiled:tuned", "im2col_gemm", "dispatcher:auto"):
            for s in ("tuning_set", "heldout"):
                P(f"{g:16s} {s:11s} {v:20s}/cuDNN FP32", summarize(ratio_table(sub, v, "cudnn:library", [s], "fp32")))
    P("\n## 5. autotuned vs fixed (same backend, fused, all sets)")
    for g in gpus:
        sub = [r for r in rows if r["gpu"] == g]
        for be in ("implicit_gemm", "simt_tiled", "tensor_core"):
            for prec in ("fp32", "fp16"):
                vals = ratio_table(sub, f"{be}:tuned", f"{be}:fixed", None, prec)
                if vals:
                    P(f"{g:16s} {be:14s} {prec}  tuned/fixed", summarize(vals))
    P("\n## 6. fusion: unfused time / fused time (same backend+mode, all sets)")
    fidx = {}
    for r in rows:
        if r["status"] == "ok":
            fidx.setdefault((r["gpu"], r["_set"], r["workload"], r["precision"], variant(r)), {})[r["fusion"]] = r["median_us"]
    byv = collections.defaultdict(list)
    for k, d in fidx.items():
        if "fused" in d and "unfused" in d:
            byv[(k[0], k[3], k[4])].append(d["unfused"] / d["fused"])
    for k in sorted(byv):
        P(f"{k[0]:16s} {k[1]:5s} {k[2]:20s} n={len(byv[k]):3d} geomean={geomean(byv[k]):.3f} median={np.median(byv[k]):.3f} max={max(byv[k]):.3f}")

    # ---------------------------------------------------------------- autotuning experiment from candidate lists
    P("\n## 7. Experiment B recomputed from the per-candidate records")
    for gdir in sorted(glob.glob(os.path.join(a.root, "raw", "*"))):
        f = os.path.join(gdir, "experiment_b", "experiment-b.jsonl")
        if not os.path.exists(f):
            continue
        recs = [json.loads(l) for l in open(f) if l.strip()]
        P(f"[{os.path.basename(gdir)}] {len(recs)} workload-precision records")
        for split in ("tuning", "heldout"):
            for prec in ("fp16", "fp32"):
                rs = [r for r in recs if r["split"] == split and r["precision"] == prec]
                if not rs:
                    continue
                sp, rank1, fixed = [], [], []
                topk = {k: [] for k in (1, 3, 8)}
                within5 = {k: 0 for k in topk}
                within10 = {k: 0 for k in topk}
                nmeas = []
                for r in rs:
                    meas = [(c["kernel"], c["median_us"], c["predicted_us"]) for c in r["candidates"] if "median_us" in c]
                    if len(meas) < 3:
                        continue
                    nmeas.append(len(meas))
                    best = min(m[1] for m in meas)
                    sp.append(spearman([m[2] for m in meas], [m[1] for m in meas]))
                    order = sorted(meas, key=lambda m: m[2])
                    rank1.append(order[0][1] == best)
                    for k in topk:
                        sl = min(m[1] for m in order[:k]) / best
                        topk[k].append(sl)
                        within5[k] += sl <= 1.05
                        within10[k] += sl <= 1.10
                    if r.get("fixed_us"):
                        fixed.append(r["fixed_us"] / best)
                n = len(nmeas)
                P(f"  {split:8s} {prec}: n={n} candidates/workload median={np.median(nmeas):.0f}  spearman(median)={np.nanmedian(sp):.3f}  true-best ranked #1: {np.mean(rank1)*100:.0f}%")
                if fixed:
                    P(f"     fixed config slowdown geomean={geomean(fixed):.2f} median={np.median(fixed):.2f} worst={max(fixed):.1f}")
                for k in topk:
                    v = np.array(topk[k])
                    P(f"     model top-{k}: geomean={geomean(v):.3f} median={np.median(v):.3f} worst={v.max():.3f} within5%={within5[k]/n*100:.0f}% within10%={within10[k]/n*100:.0f}%")
                ex = np.array([r["exhaustive_search_seconds"] for r in rs])
                P(f"     exhaustive search wall time: median={np.median(ex):.3f}s max={ex.max():.2f}s total={ex.sum():.1f}s")
                for k in (1, 3, 5, 8):
                    key = f"model_prior_k{k}_search_seconds"
                    vals = np.array([r[key] for r in rs if key in r])
                    if len(vals):
                        P(f"     top-{k} measured search: median={np.median(vals):.3f}s ({np.median(vals / ex[:len(vals)]) * 100:.0f}% of exhaustive); absolute saving per workload median={np.median(ex[:len(vals)] - vals):.3f}s")
                cal = [r for r in rs if "model_calibrated_k3_over_best" in r]
                if cal:
                    pr = [r["model_prior_k3_over_best"] for r in cal]
                    ca = [r["model_calibrated_k3_over_best"] for r in cal]
                    P(f"     calibration (top-3 slowdown geomean): prior={geomean(pr):.3f} calibrated={geomean(ca):.3f}")

    # ---------------------------------------------------------------- graphs
    P("\n## 8. CUDA Graphs (graphs.jsonl)")
    for gdir in sorted(glob.glob(os.path.join(a.root, "raw", "*"))):
        f = os.path.join(gdir, "graphs", "graphs.jsonl")
        if not os.path.exists(f):
            continue
        rs = [json.loads(l) for l in open(f) if l.strip()]
        for r in rs:
            if r.get("experiment") == "F_cuda_graphs" and r["precision"] == "fp16" and r["case"] == "b1_c32_14x14_k3":
                P(f"[{os.path.basename(gdir)}] layers={r['layers']:2d} device {r['eager_device_us']:.1f}->{r['graph_device_us']:.1f} us (x{r['device_speedup']:.2f}) host launch {r['eager_host_launch_us']:.1f}->{r['graph_host_launch_us']:.1f} us (x{r['eager_host_launch_us'] / r['graph_host_launch_us']:.1f}) identical={r['bitwise_identical_replays']}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out + ".txt", "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nwrote {a.out}.txt")


if __name__ == "__main__":
    main()
