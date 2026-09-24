"""Figure 2 (throughput by backend) and Figures 5/6 (sensitivity to batch size, kernel size, channels, spatial size)."""
import numpy as np

from common import (VARIANT_COLORS, VARIANT_ORDER, ok_rows, precision_note, pretty, save, setup_matplotlib, short_gpu, slug,
                    synthetic_banner)

SWEEPS = {  # category -> (x column, x label)
    "sweep_batch": ("B", "batch size"),
    "sweep_kernel": ("kernel_area", "kernel area Kh*Kw (1x1,3x3,5x5,7x7 -> 1,9,25,49)"),
    "sweep_channels": ("C", "input channels"),
    "sweep_spatial": ("H", "input height = width"),
    "sweep_irregular": ("H", "input height = width (irregular, not divisible by tile sizes)"),
}


def plot(df, out_dir):
    plt = setup_matplotlib()
    ok = ok_rows(df)
    ok = ok[ok["fusion"] == "fused"]
    # --- Figure 2: throughput (TFLOP/s and images/s) per workload -----------------------------------------------------
    for (gpu, prec, split), g in ok.groupby(["gpu", "precision", "split"]):
        if split == "sweep":
            continue
        variants = [v for v in VARIANT_ORDER if v in set(g["variant"]) and v != "cpu_reference"]
        wl = list(dict.fromkeys(g.sort_values(["category", "workload"])["workload"]))
        piv = g.pivot_table(index="workload", columns="variant", values="tflops", aggfunc="first").reindex(wl)
        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(wl) * max(1, len(variants) / 4)), 4.6))
        w = 0.8 / max(len(variants), 1)
        for i, v in enumerate(variants):
            ax.bar(np.arange(len(wl)) + i * w, piv[v].values, w, label=pretty(v), color=VARIANT_COLORS.get(v))
        ax.set_xticks(np.arange(len(wl)) + 0.4 - w / 2)
        ax.set_xticklabels(wl, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("effective TFLOP/s (2*MACs / median latency)")
        ax.set_title(f"Throughput by backend - {short_gpu(gpu)} - {precision_note(prec)} - {split} set")
        ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.55))
        save(fig, out_dir, f"fig02_throughput_{slug(short_gpu(gpu))}_{prec}_{split}", synthetic_banner(g))
        plt.close(fig)
    # --- Figures 5/6: one-factor sweeps -------------------------------------------------------------------------------
    for (gpu, prec), g in ok.groupby(["gpu", "precision"]):
        for cat, (xcol, xlabel) in SWEEPS.items():
            s = g[g["category"] == cat]
            if s.empty:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for v in [v for v in VARIANT_ORDER if v in set(s["variant"]) and v != "cpu_reference"]:
                t = s[s["variant"] == v].sort_values(xcol)
                axes[0].plot(t[xcol], t["tflops"], marker="o", ms=3, label=pretty(v), color=VARIANT_COLORS.get(v))
                axes[1].plot(t[xcol], t["median_us"], marker="o", ms=3, label=pretty(v), color=VARIANT_COLORS.get(v))
            axes[0].set_ylabel("effective TFLOP/s")
            axes[1].set_ylabel("median latency (us, log)")
            axes[1].set_yscale("log")
            if xcol in ("B", "C", "H"):
                axes[0].set_xscale("log", base=2) if cat in ("sweep_batch", "sweep_channels") else None
            for ax in axes:
                ax.set_xlabel(xlabel)
            axes[0].legend(ncol=2)
            fig.suptitle(f"Sensitivity: {cat.replace('sweep_', '')} - {short_gpu(gpu)} - {precision_note(prec)}")
            save(fig, out_dir, f"fig05_06_{cat}_{slug(short_gpu(gpu))}_{prec}", synthetic_banner(g))
            plt.close(fig)
