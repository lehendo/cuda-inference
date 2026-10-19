#!/usr/bin/env python3
"""Final, independent recomputation of the headline comparisons from raw result rows (stdlib only).

usage: final_claims.py <results/raw> [--out results/verification/final_claims]

Inputs per GPU tag g in {a40, h100}:
  <raw>/audit_repr_<g>/{workloads,heldout_workloads}/results.jsonl            custom backends, CUTLASS, and the ORIGINAL
                                                                              (single-shot-selection) cuDNN / cuDNN-GEMM rows
  <raw>/audit_cudnn_retime_<g>/{rep1,rep2}/{workloads,heldout_workloads}/results.jsonl   cuDNN rows with the corrected
                                                                              (re-timed) algorithm selection, run twice
  <raw>/<g>/{tuning_set,heldout}/results.jsonl                                the first campaign (pre-audit), for context
Everything is read only; nothing under <raw> is modified.

Conventions: ratio = baseline_time / custom_time, so ratio > 1 means the custom backend is faster. A pair is used only if
both rows have status == ok. Duplicate keys are counted and the first occurrence is used. cuDNN comparisons are always
reported twice: with the corrected cuDNN rows (primary) and with the original rows from the earlier run (for the effect of
the correction). cuDNN-GEMM is a separately labelled diagnostic, never a replacement for unrestricted cuDNN.
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

GPUS = {"a40": "NVIDIA A40", "h100": "NVIDIA H100 NVL"}
SPLITS = {"workloads": "tuning", "heldout_workloads": "held-out"}
OLD_SPLITS = {"tuning_set": "tuning", "heldout": "held-out"}


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    rows = []
    with open(p) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def keyed(rows, stats, label):
    """(workload, precision, backend, mode, fusion) -> row; counts duplicates and non-ok rows."""
    out = {}
    for r in rows or []:
        k = (r["workload"], r["precision"], r["backend"], r.get("mode", ""), r.get("fusion", ""))
        if k in out:
            stats[label]["duplicate_keys"] += 1
            continue
        stats[label]["status_" + r["status"]] += 1
        if r["status"] != "ok" or not r.get("median_us"):
            continue
        out[k] = r
    return out


def geomean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def median(v):
    s = sorted(v)
    n = len(s)
    return float("nan") if n == 0 else (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def summarize(pairs):
    """pairs: list of (ratio, workload). Returns a formatted string."""
    if not pairs:
        return "n=0"
    r = [p[0] for p in pairs]
    worst = min(pairs)
    best = max(pairs)
    return (f"n={len(r):3d} geomean={geomean(r):6.3f} median={median(r):6.3f} wins={sum(x > 1 for x in r):3d} "
            f"losses={sum(x < 1 for x in r):3d} worst={worst[0]:5.2f} ({worst[1][:28]}) best={best[0]:5.2f}")


def algo_of(row):
    m = re.search(r"algo=([A-Z_0-9]+)", row.get("config", "") or "")
    return m.group(1) if m else "?"


def find(d, workload, prec, backend, mode, fusion="fused"):
    return d.get((workload, prec, backend, mode, fusion))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    raw = Path(sys.argv[1])
    out_prefix = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    stats = defaultdict(Counter)
    for g, gname in GPUS.items():
        emit(f"################ {gname} ################")
        data = {}      # split -> keyed rows of the audit re-run (custom, cutlass, original cudnn)
        corr = {}      # (rep, split) -> keyed rows of corrected cuDNN
        old = {}
        missing = []
        for ws, split in SPLITS.items():
            rows = load(raw / f"audit_repr_{g}" / ws / "results.jsonl")
            if rows is None:
                missing.append(f"audit_repr_{g}/{ws}")
            data[split] = keyed(rows, stats, f"{g}/{split}/audit_repr")
            for rep in ("rep1", "rep2"):
                rows = load(raw / f"audit_cudnn_retime_{g}" / rep / ws / "results.jsonl")
                if rows is None:
                    missing.append(f"audit_cudnn_retime_{g}/{rep}/{ws}")
                corr[(rep, split)] = keyed(rows, stats, f"{g}/{split}/retime_{rep}")
        for os_, split in OLD_SPLITS.items():
            old[split] = keyed(load(raw / g / os_ / "results.jsonl"), stats, f"{g}/{split}/first_campaign")
        if missing:
            emit("MISSING INPUTS (corrected-baseline results are PENDING for these): " + ", ".join(missing))

        emit("-- 1. input integrity")
        for label in sorted(k for k in stats if k.startswith(g + "/")):
            emit(f"   {label:40s} " + " ".join(f"{k}={v}" for k, v in sorted(stats[label].items())))

        emit("-- 2. cuDNN algorithm-selection stability (corrected rows, rep1 vs rep2; original single-shot rows for contrast)")
        for split in ("tuning", "held-out"):
            r1, r2, orig = corr[("rep1", split)], corr[("rep2", split)], data[split]
            for be in ("cudnn", "cudnn_gemm"):
                same = diff = 0
                lat = []
                changed = []
                for k, a in r1.items():
                    if k[2] != be or k not in r2:
                        continue
                    b = r2[k]
                    if algo_of(a) == algo_of(b):
                        same += 1
                    else:
                        diff += 1
                        changed.append(f"{k[0]}:{k[1]} {algo_of(a)}->{algo_of(b)} ({a['median_us']:.1f}/{b['median_us']:.1f}us)")
                    lat.append(max(a["median_us"], b["median_us"]) / min(a["median_us"], b["median_us"]))
                if same + diff == 0:
                    emit(f"   {split:8s} {be:10s} no paired corrected rows yet")
                    continue
                emit(f"   {split:8s} {be:10s} rep1==rep2 algorithm: {same}/{same + diff}; latency max/min ratio geomean "
                     f"{geomean(lat):.3f}, worst {max(lat):.2f}")
                for c in changed[:6]:
                    emit(f"       algorithm changed: {c}")
                # original single-shot rows vs corrected rep1
                o_same = o_diff = 0
                slow = []
                for k, a in r1.items():
                    if k[2] != be or k not in orig:
                        continue
                    if algo_of(a) == algo_of(orig[k]):
                        o_same += 1
                    else:
                        o_diff += 1
                    slow.append((orig[k]["median_us"] / a["median_us"], f"{k[0]}:{k[1]}"))
                if slow:
                    emit(f"       original(single-shot) vs corrected rep1: same algorithm {o_same}, different {o_diff}; "
                         f"original/corrected latency geomean {geomean([s[0] for s in slow]):.3f}, "
                         f"worst original slowdown {max(slow)[0]:.2f}x ({max(slow)[1]})")

        emit("-- 3. headline comparisons (ratio = baseline/custom; >1 custom faster)")
        comps = [
            ("FP16", "fp16", ("tensor_core", "tuned"), "cudnn", "median_us", "FP16 custom Tensor Core (tuned) vs unrestricted cuDNN"),
            ("FP16", "fp16", ("dispatcher", "auto"), "cudnn", "median_us", "FP16 dispatcher (auto) vs unrestricted cuDNN"),
            ("FP32", "fp32", ("implicit_gemm", "tuned"), "cudnn", "median_us", "FP32 custom implicit GEMM (tuned) vs unrestricted cuDNN"),
            ("FP32", "fp32", ("dispatcher", "auto"), "cudnn", "median_us", "FP32 dispatcher (auto) vs unrestricted cuDNN"),
            ("FP32", "fp32", ("implicit_gemm", "tuned"), "cudnn_gemm", "median_us", "[diagnostic] FP32 custom vs cuDNN GEMM-family only"),
        ]
        for tag, prec, (cb, cm), base, field, title in comps:
            emit(f"   {title}")
            for split in ("tuning", "held-out"):
                for src_label, src in (("corrected cuDNN (rep1)", corr[("rep1", split)]), ("corrected cuDNN (rep2)", corr[("rep2", split)]),
                                       ("original cuDNN rows", data[split])):
                    pairs = []
                    for k, row in data[split].items():
                        if k[1] != prec or k[2] != cb or k[3] != cm or k[4] != "fused":
                            continue
                        bk = (k[0], prec, base, "library", "fused")
                        if bk in src:
                            pairs.append((src[bk][field] / row["median_us"], k[0]))
                    emit(f"     {split:8s} {src_label:24s} {summarize(pairs)}")
        emit("   FP16 custom Tensor Core (tuned) vs CUTLASS (same-run rows)")
        for split in ("tuning", "held-out"):
            for label, field in (("kernel-only (NHWC)", "median_us"), ("layout-inclusive", "layout_inclusive_us")):
                pairs = []
                for k, row in data[split].items():
                    if k[1] != "fp16" or k[2] != "tensor_core" or k[3] != "tuned" or k[4] != "fused":
                        continue
                    ck = (k[0], "fp16", "cutlass", "library", "fused")
                    if ck in data[split] and data[split][ck].get(field):
                        pairs.append((data[split][ck][field] / row["median_us"], k[0]))
                emit(f"     {split:8s} {label:24s} {summarize(pairs)}")

        emit("-- 4. FP32 implicit GEMM: audit re-run vs first campaign, per workload and aggregate (old_us/new_us; >1 new faster)")
        for split in ("tuning", "held-out"):
            pairs = []
            for k, row in data[split].items():
                if k[1] != "fp32" or k[2] != "implicit_gemm" or k[3] != "tuned" or k[4] != "fused":
                    continue
                if k in old[split]:
                    pairs.append((old[split][k]["median_us"] / row["median_us"], k[0]))
            emit(f"     {split:8s} {summarize(pairs)}")
            top = sorted(pairs, reverse=True)[:3]
            emit("       largest per-workload gains: " + ", ".join(f"{w}={r:.2f}x" for r, w in top))
            emit("       tuned kernel unchanged in name for " +
                 str(sum(1 for k, row in data[split].items() if k[1] == "fp32" and k[2] == "implicit_gemm" and k[3] == "tuned"
                         and k in old[split] and row.get("kernel") == old[split][k].get("kernel"))) + " of " + str(len(pairs)) + " workloads")
        emit()

    if out_prefix:
        Path(out_prefix + ".txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
