#!/usr/bin/env python3
"""Aggregate raw gie_bench results into the tables the report and figures are built from.

Usage: aggregate_results.py --raw <dir-or-jsonl> [--raw ...] --out results/processed

Every table is derived from the raw rows; failed / unsupported / invalid rows are written to
failures.csv instead of being dropped. Speedups are computed only between rows measured with the same
GPU, workload, precision and fusion mode.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import geomean, key_cols, load_results, ok_rows, speedup_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="append", required=True)
    ap.add_argument("--out", default="results/processed")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load_results(a.raw)
    if df.empty:
        raise SystemExit("no results found")
    df.to_csv(f"{a.out}/all_results.csv", index=False)

    # ---- explicit failures / unsupported / validation failures -----------------------------------------
    bad = df[df["status"] != "ok"]
    cols = [c for c in key_cols() + ["backend", "mode", "kernel", "status", "reason"] if c in bad]
    bad[cols].to_csv(f"{a.out}/failures.csv", index=False)
    status_counts = df.groupby(["gpu", "variant", "status"]).size().unstack(fill_value=0)
    status_counts.to_csv(f"{a.out}/status_counts.csv")

    ok = ok_rows(df)

    # ---- per-workload wide table -----------------------------------------------------------------------
    wide = ok.pivot_table(index=key_cols() + ["category", "B", "C", "H", "W", "O", "Kh", "Kw", "Sh"], columns="variant",
                          values="median_us", aggfunc="first")
    wide.to_csv(f"{a.out}/per_workload_median_us.csv")

    # ---- speedups vs each baseline -------------------------------------------------------------------------
    summaries = []
    for baseline in ["direct", "im2col_gemm", "cudnn:library", "cutlass:library", "implicit_gemm:tuned"]:
        sp = speedup_table(df, baseline)
        if sp.empty:
            continue
        sp.to_csv(f"{a.out}/speedup_vs_{baseline.replace(':', '_')}.csv", index=False)
        for (gpu, prec, fusion, split, variant), g in sp.groupby(["gpu", "precision", "fusion", "split", "variant"]):
            if variant == baseline:
                continue
            s = g["speedup"].values
            summaries.append({
                "gpu": gpu, "precision": prec, "fusion": fusion, "split": split, "variant": variant, "baseline": baseline,
                "n_workloads": len(s), "geomean_speedup": geomean(s), "median_speedup": float(np.median(s)),
                "min_speedup": float(np.min(s)), "max_speedup": float(np.max(s)),
                "wins": int((s > 1.0).sum()), "losses": int((s < 1.0).sum()), "ties": int((s == 1.0).sum()),
            })
    if summaries:
        pd.DataFrame(summaries).to_csv(f"{a.out}/aggregate_speedups.csv", index=False)

    # ---- breakdowns (batch size, kernel size, dtype) against cuDNN if present else against direct -------------
    base = "cudnn:library" if (ok["variant"] == "cudnn:library").any() else "direct"
    sp = speedup_table(df, base)
    if not sp.empty:
        for name, col in (("by_batch", "B"), ("by_kernel_area", "kernel_area"), ("by_precision", "precision"), ("by_category", "category")):
            t = sp.groupby(["gpu", col, "variant"])["speedup"].agg(geomean).reset_index().rename(columns={"speedup": f"geomean_speedup_vs_{base.replace(':', '_')}"})
            t.to_csv(f"{a.out}/{name}.csv", index=False)

    # ---- temporary memory -----------------------------------------------------------------------------------------
    if "workspace_bytes" in ok:
        ok.groupby(["gpu", "variant"])["workspace_bytes"].agg(["median", "max"]).to_csv(f"{a.out}/workspace_memory.csv")

    # ---- autotuning: fixed manual configuration vs tuned, plus search cost ---------------------------------------------
    pieces = []
    for be in ("simt_tiled", "implicit_gemm", "tensor_core"):
        fx = ok[ok["variant"] == f"{be}:fixed"][key_cols() + ["median_us"]].rename(columns={"median_us": "fixed_us"})
        tu = ok[ok["variant"] == f"{be}:tuned"][key_cols() + ["median_us", "kernel", "tuning_seconds", "tuning_cache_hit"]].rename(
            columns={"median_us": "tuned_us", "kernel": "tuned_kernel"})
        m = fx.merge(tu, on=key_cols())
        if not m.empty:
            m["backend"] = be
            m["tuned_over_fixed_speedup"] = m["fixed_us"] / m["tuned_us"]
            pieces.append(m)
    if pieces:
        t = pd.concat(pieces)
        t.to_csv(f"{a.out}/autotune_vs_fixed.csv", index=False)

    # ---- fusion: fused vs unfused (same kernel family), total time including the elementwise passes ----------------------------
    fu = ok[ok["fusion"] == "fused"][key_cols()[:3] + ["variant", "median_us"]].rename(columns={"median_us": "fused_us"})
    un = ok[ok["fusion"] == "unfused"][key_cols()[:3] + ["variant", "median_us"]].rename(columns={"median_us": "unfused_us"})
    m = fu.merge(un, on=["gpu", "workload", "precision", "variant"])
    if not m.empty:
        m["fusion_speedup"] = m["unfused_us"] / m["fused_us"]
        m.to_csv(f"{a.out}/fusion.csv", index=False)

    # ---- Tensor Core vs SIMT (same precision, same fusion) --------------------------------------------------------------------------
    for tcv, simt in (("tensor_core:tuned", "implicit_gemm:tuned"), ("tensor_core:fixed", "implicit_gemm:fixed")):
        t1 = ok[ok["variant"] == tcv][key_cols() + ["median_us"]].rename(columns={"median_us": "tc_us"})
        t2 = ok[ok["variant"] == simt][key_cols() + ["median_us"]].rename(columns={"median_us": "simt_us"})
        m = t1.merge(t2, on=key_cols())
        if not m.empty:
            m["tc_over_simt_speedup"] = m["simt_us"] / m["tc_us"]
            m.to_csv(f"{a.out}/tensor_core_vs_simt_{tcv.split(':')[1]}.csv", index=False)

    # ---- im2col accounting: GEMM-only vs full pipeline -------------------------------------------------------------------------------------------------------
    im = ok[(ok["variant"] == "im2col_gemm") & ok.get("stage_gemm_us", pd.Series(dtype=float)).notna()] if "stage_gemm_us" in ok else pd.DataFrame()
    if not im.empty:
        im = im.assign(gemm_only_us=im["stage_gemm_us"], im2col_us=im["stage_im2col_us"], pipeline_us=im["median_us"])
        im[key_cols() + ["gemm_only_us", "im2col_us", "pipeline_us", "workspace_bytes"]].to_csv(f"{a.out}/im2col_accounting.csv", index=False)

    # ---- CUTLASS layout accounting -----------------------------------------------------------------------------------------------------------------------------------------------
    if "layout_inclusive_us" in ok:
        cu = ok[(ok["variant"] == "cutlass:library") & ok["layout_inclusive_us"].notna()]
        if not cu.empty:
            cu[key_cols() + ["median_us", "layout_inclusive_us"]].to_csv(f"{a.out}/cutlass_layout_accounting.csv", index=False)

    # ---- dispatcher selection quality: chosen vs the best *measured* custom backend on the same workload --------------------------------------------------
    custom = ok[ok["variant"].isin(["direct", "im2col_gemm", "simt_tiled:tuned", "implicit_gemm:tuned", "tensor_core:tuned",
                                    "simt_tiled:fixed", "implicit_gemm:fixed", "tensor_core:fixed"])]
    best = custom.groupby(key_cols())["median_us"].min().rename("best_custom_us").reset_index()
    disp = ok[ok["variant"] == "dispatcher:auto"][key_cols() + ["median_us", "kernel", "selection_source"]
                                                   if "selection_source" in ok else key_cols() + ["median_us", "kernel"]]
    d = disp.merge(best, on=key_cols())
    if not d.empty:
        d["regret"] = d["median_us"] / d["best_custom_us"]
        d["within_5pct_of_best"] = d["regret"] <= 1.05
        d.to_csv(f"{a.out}/dispatcher_selection.csv", index=False)
        with open(f"{a.out}/dispatcher_accuracy.json", "w") as fh:
            json.dump({"workloads": int(len(d)), "within_5pct": float(d["within_5pct_of_best"].mean()),
                       "geomean_regret": geomean(d["regret"]), "max_regret": float(d["regret"].max())}, fh, indent=1)

    # ---- winner / loser per workload ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    w = ok.loc[ok.groupby(key_cols())["median_us"].idxmin()][key_cols() + ["variant", "kernel", "median_us"]].rename(
        columns={"variant": "winner", "kernel": "winner_kernel", "median_us": "winner_us"})
    w.to_csv(f"{a.out}/winners.csv", index=False)
    w.groupby(["gpu", "precision", "fusion", "split", "winner"]).size().rename("workloads_won").reset_index().to_csv(f"{a.out}/win_counts.csv", index=False)

    # ---- environment manifest ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    env_cols = [c for c in ("gpu", "cc", "hostname", "slurm_job_id", "cuda_runtime", "cuda_driver", "cudnn", "cutlass", "git_rev") if c in df]
    df[env_cols].drop_duplicates().to_csv(f"{a.out}/environments.csv", index=False)
    print(f"wrote tables to {a.out}: {len(df)} rows, {len(ok)} ok, {len(bad)} non-ok")


if __name__ == "__main__":
    main()
