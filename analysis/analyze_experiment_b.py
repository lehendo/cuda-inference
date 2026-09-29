#!/usr/bin/env python3
"""Experiment B (autotuning): fixed config vs exhaustive search vs cost-model-guided search.

Reads gie_tune experiment-b output (experiment-b.jsonl, model_calibration.json) and writes a summary table and
a figure. Calibration was fitted on the *tuning* split only; the held-out split is the out-of-sample evaluation.
Usage: analyze_experiment_b.py --in <dir with experiment-b.jsonl> --out results/processed [--figures results/figures]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import geomean, save, setup_matplotlib, short_gpu, slug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, action="append")
    ap.add_argument("--out", default="results/processed")
    ap.add_argument("--figures", default="results/figures")
    a = ap.parse_args()
    rows = []
    for d in a.inp:
        p = os.path.join(d, "experiment-b.jsonl") if os.path.isdir(d) else d
        with open(p) as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    ks = sorted({int(c.split("_k")[1].split("_")[0]) for c in df.columns if c.startswith("model_prior_k") and c.endswith("_over_best")})
    recs = []
    for (gpu, split, prec), g in df.groupby(["gpu", "split", "precision"]):
        base = {"gpu": gpu, "split": split, "precision": prec, "workloads": len(g),
                "exhaustive_search_s_median": g["exhaustive_search_seconds"].median(), "candidates_measured_median": g["candidates_measured"].median()}
        if "fixed_over_best" in g:
            base["fixed_geomean_slowdown"] = geomean(g["fixed_over_best"].dropna())
            base["fixed_max_slowdown"] = g["fixed_over_best"].max()
            base["fixed_unsupported"] = int(g["fixed_over_best"].isna().sum())
        for tag in ("prior", "calibrated"):
            if f"spearman_{tag}" in g:
                base[f"spearman_{tag}_median"] = g[f"spearman_{tag}"].median()
                base[f"true_best_rank_{tag}_median"] = g[f"rank_of_true_best_{tag}"].median()
                base[f"true_best_top1_{tag}"] = float((g[f"rank_of_true_best_{tag}"] == 1).mean())
            for k in ks:
                c = f"model_{tag}_k{k}_over_best"
                if c in g:
                    base[f"{tag}_k{k}_geomean_slowdown"] = geomean(g[c].dropna())
                    base[f"{tag}_k{k}_max_slowdown"] = g[c].max()
                    base[f"{tag}_k{k}_search_cost_vs_exhaustive"] = (g[f"model_{tag}_k{k}_search_seconds"] / g["exhaustive_search_seconds"]).median()
        recs.append(base)
    out = pd.DataFrame(recs)
    out.to_csv(os.path.join(a.out, "experiment_b_summary.csv"), index=False)
    print(out.T.to_string())

    plt = setup_matplotlib()
    for (gpu, prec), g in out.groupby(["gpu", "precision"]):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for split, marker in (("tuning", "o"), ("heldout", "s")):
            r = g[g["split"] == split]
            if r.empty:
                continue
            r = r.iloc[0]
            for tag, ls in (("prior", "--"), ("calibrated", "-")):
                xs = [k for k in ks if f"{tag}_k{k}_geomean_slowdown" in r and pd.notna(r[f"{tag}_k{k}_geomean_slowdown"])]
                if xs:
                    ax[0].plot(xs, [r[f"{tag}_k{k}_geomean_slowdown"] for k in xs], marker=marker, ls=ls, label=f"{split}, {tag} model")
                    ax[1].plot(xs, [r[f"{tag}_k{k}_search_cost_vs_exhaustive"] for k in xs], marker=marker, ls=ls, label=f"{split}, {tag} model")
            if "fixed_geomean_slowdown" in r and pd.notna(r["fixed_geomean_slowdown"]):
                ax[0].axhline(r["fixed_geomean_slowdown"], color="gray", ls=":" if split == "tuning" else "-.", label=f"fixed config ({split})")
        ax[0].set_xlabel("candidates measured (top-K by model)")
        ax[0].set_ylabel("geomean slowdown vs exhaustive best (x)")
        ax[1].set_xlabel("candidates measured (top-K by model)")
        ax[1].set_ylabel("median search time / exhaustive search time")
        ax[0].legend(fontsize=7)
        fig.suptitle(f"Experiment B: model-guided search - {short_gpu(gpu)} - {prec}")
        save(fig, a.figures, f"expB_model_guided_{slug(short_gpu(gpu))}_{prec}")
        plt.close(fig)


if __name__ == "__main__":
    main()
