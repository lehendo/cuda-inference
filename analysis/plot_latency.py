"""Figure 1: steady-state kernel latency by backend, per workload (log scale)."""
import os

import numpy as np

from common import (VARIANT_COLORS, VARIANT_ORDER, ok_rows, precision_note, pretty, save, setup_matplotlib, short_gpu, slug,
                    synthetic_banner)


def plot(df, out_dir):
    plt = setup_matplotlib()
    ok = ok_rows(df)
    for (gpu, prec, split), g in ok[ok["fusion"] == "fused"].groupby(["gpu", "precision", "split"]):
        variants = [v for v in VARIANT_ORDER if v in set(g["variant"]) and v != "cpu_reference"]
        wl = list(dict.fromkeys(g.sort_values(["category", "workload"])["workload"]))
        if not variants or not wl:
            continue
        piv = g.pivot_table(index="workload", columns="variant", values="median_us", aggfunc="first").reindex(wl)
        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(wl) * max(1, len(variants) / 4)), 4.8))
        w = 0.8 / len(variants)
        for i, v in enumerate(variants):
            ax.bar(np.arange(len(wl)) + i * w, piv[v].values, w, label=pretty(v), color=VARIANT_COLORS.get(v))
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(wl)) + 0.4 - w / 2)
        ax.set_xticklabels(wl, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("median kernel latency (us, log)")
        ax.set_title(f"Steady-state latency by backend - {short_gpu(gpu)} - {precision_note(prec)} - {split} set")
        ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.55))
        save(fig, out_dir, f"fig01_latency_{slug(short_gpu(gpu))}_{prec}_{split}", synthetic_banner(g))
        plt.close(fig)
