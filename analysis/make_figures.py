#!/usr/bin/env python3
"""Generate every figure from saved raw results. Usage: make_figures.py --raw <dir> [--raw ...] --figures results/figures"""
import argparse

import plot_latency
import plot_roofline
import plot_speedups
import plot_throughput
from common import load_meta, load_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="append", required=True)
    ap.add_argument("--figures", default="results/figures")
    a = ap.parse_args()
    df = load_results(a.raw)
    if df.empty:
        raise SystemExit("no results found")
    meta = load_meta(a.raw)
    plot_latency.plot(df, a.figures)
    plot_throughput.plot(df, a.figures)
    plot_speedups.plot(df, a.figures)
    plot_roofline.plot(df, a.figures, meta)
    print(f"figures written to {a.figures}")


if __name__ == "__main__":
    main()
