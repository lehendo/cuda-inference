"""Shared helpers for loading and labeling benchmark results.

All analysis reads machine-readable results (results.jsonl / meta.json written by gie_bench) and never
invents data. Rows with a non-"ok" status are kept and reported, never silently dropped.
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Iterable

import numpy as np
import pandas as pd

OPTIONAL_COLUMNS = ["tuning_seconds", "tuning_cache_hit", "selection_source", "stage_gemm_us", "stage_im2col_us", "layout_inclusive_us",
                    "conv_only_us", "setup_seconds", "workspace_bytes", "persistent_bytes", "reason", "kernel", "arithmetic_intensity",
                    "compulsory_bytes", "tflops", "images_per_s"]
MULTI_CONFIG_BACKENDS = {"simt_tiled", "implicit_gemm", "tensor_core"}

# Display order and a colour-blind-safe palette (Okabe-Ito) keyed by variant name.
VARIANT_ORDER = [
    "cpu_reference", "direct", "im2col_gemm", "simt_tiled:fixed", "simt_tiled:tuned", "implicit_gemm:fixed",
    "implicit_gemm:tuned", "tensor_core:fixed", "tensor_core:tuned", "dispatcher:auto", "cudnn:library", "cutlass:library",
]
VARIANT_COLORS = {
    "cpu_reference": "#999999", "direct": "#7f7f7f", "im2col_gemm": "#E69F00",
    "simt_tiled:fixed": "#56B4E9", "simt_tiled:tuned": "#0072B2",
    "implicit_gemm:fixed": "#a6d96a", "implicit_gemm:tuned": "#009E73",
    "tensor_core:fixed": "#f4a6c1", "tensor_core:tuned": "#CC79A7",
    "dispatcher:auto": "#000000", "cudnn:library": "#D55E00", "cutlass:library": "#8c564b",
}
PRETTY = {
    "direct": "naive direct CUDA", "im2col_gemm": "im2col + cuBLAS GEMM", "dispatcher:auto": "dispatcher (auto)",
    "cudnn:library": "cuDNN", "cutlass:library": "CUTLASS (NHWC, kernel-only)", "cpu_reference": "CPU reference",
}


def pretty(v: str) -> str:
    if v in PRETTY:
        return PRETTY[v]
    if ":" in v:
        b, m = v.split(":")
        return f"custom {b.replace('_', ' ')} ({m})"
    return v


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def load_results(paths: Iterable[str]) -> pd.DataFrame:
    """Load every results.jsonl found under the given files/directories."""
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "results.jsonl"), recursive=True)
        elif os.path.isfile(p):
            files.append(p)
    rows = []
    for f in sorted(files):
        meta = {}
        mp = os.path.join(os.path.dirname(f), "meta.json")
        if os.path.exists(mp):
            with open(mp) as fh:
                meta = json.load(fh)
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r["_source"] = f
                for k in ("hostname", "slurm_job_id", "cuda_runtime", "cuda_driver", "cudnn", "cutlass", "git_rev"):
                    if k in meta:
                        r.setdefault(k, meta[k])
                rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Early campaign versions wrote placeholder rows for baselines in the unfused-epilogue experiment, which
    # does not apply to them; those are not unsupported configurations, so they are excluded here.
    if "reason" in df:
        df = df[~df["reason"].fillna("").str.contains("measured with its own fused")]
    df["mode"] = df["mode"].fillna("n/a")
    df["variant"] = [
        f"{b}:{m}" if (b in MULTI_CONFIG_BACKENDS or b in ("cudnn", "cutlass", "dispatcher")) and m not in ("n/a", "host") else b
        for b, m in zip(df["backend"], df["mode"])
    ]
    for c in OPTIONAL_COLUMNS:  # measured only by some backends / modes
        if c not in df:
            df[c] = np.nan
    df["kernel_area"] = df["Kh"] * df["Kw"]
    for c in ("median_us", "images_per_s", "tflops", "workspace_bytes", "flops", "compulsory_bytes", "arithmetic_intensity"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["status"] == "ok") & df["median_us"].notna()]


def key_cols() -> list[str]:
    return ["gpu", "workload", "precision", "fusion", "split"]


def geomean(x) -> float:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v) and v > 0], dtype=float)
    return float(np.exp(np.mean(np.log(x)))) if x.size else float("nan")


def speedup_table(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """speedup = baseline_median_us / variant_median_us for matching (gpu, workload, precision, fusion)."""
    ok = ok_rows(df)
    base = ok[ok["variant"] == baseline][key_cols() + ["median_us"]].rename(columns={"median_us": "base_us"})
    base = base.drop_duplicates(key_cols())
    m = ok.merge(base, on=key_cols(), how="inner")
    m["speedup"] = m["base_us"] / m["median_us"]
    m["baseline"] = baseline
    return m


def short_gpu(g: str) -> str:
    return g.replace("NVIDIA ", "").replace("GeForce ", "")


def synthetic_banner(df: pd.DataFrame) -> str:
    if df["gpu"].astype(str).str.contains("SYNTHETIC").any():
        return "SYNTHETIC FIXTURE - NOT MEASURED DATA"
    return ""


def load_meta(paths: Iterable[str]) -> dict:
    """gpu name -> merged meta.json (measured ceilings etc.) from every meta.json next to a results.jsonl."""
    out: dict = {}
    for p in paths:
        for mp in glob.glob(os.path.join(p, "**", "meta.json"), recursive=True) if os.path.isdir(p) else []:
            with open(mp) as fh:
                m = json.load(fh)
            out[m.get("gpu", "unknown")] = m
    return out


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8,
    })
    return plt


def save(fig, out_dir: str, name: str, banner: str = ""):
    os.makedirs(out_dir, exist_ok=True)
    if banner:
        fig.text(0.5, 0.5, banner, fontsize=28, color="red", alpha=0.25, ha="center", va="center", rotation=20)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")


def precision_note(prec: str) -> str:
    return {
        "fp32": "FP32 storage, strict FP32 FMA (no TF32)",
        "fp16": "FP16 storage, FP32 accumulation",
        "bf16": "BF16 storage, FP32 accumulation",
        "tf32": "TF32 multiplies, FP32 accumulation",
    }.get(prec, prec)
