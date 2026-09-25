"""Figures 3, 4, 7, 8, 10: speedups vs naive CUDA / cuDNN, autotune gain, Tensor Core vs SIMT, winner matrix."""
import numpy as np

from common import (VARIANT_COLORS, VARIANT_ORDER, geomean, key_cols, ok_rows, precision_note, pretty, save, setup_matplotlib,
                    short_gpu, slug, speedup_table, synthetic_banner)


def _speedup_bars(plt, df, baseline, out_dir, fig_no, title_base):
    sp = speedup_table(df[df["fusion"] == "fused"], baseline)
    if sp.empty:
        return
    for (gpu, prec, split), g in sp.groupby(["gpu", "precision", "split"]):
        variants = [v for v in VARIANT_ORDER if v in set(g["variant"]) and v not in (baseline, "cpu_reference")]
        if not variants:
            continue
        wl = list(dict.fromkeys(g.sort_values(["category", "workload"])["workload"]))
        piv = g.pivot_table(index="workload", columns="variant", values="speedup", aggfunc="first").reindex(wl)
        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(wl) * max(1, len(variants) / 4)), 4.8))
        w = 0.8 / len(variants)
        for i, v in enumerate(variants):
            ax.bar(np.arange(len(wl)) + i * w, piv[v].values, w, label=f"{pretty(v)} (geomean {geomean(piv[v].values):.2f}x)", color=VARIANT_COLORS.get(v))
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(wl)) + 0.4 - w / 2)
        ax.set_xticklabels(wl, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel(f"speedup over {pretty(baseline)} (x, log)")
        ax.set_title(f"{title_base} - {short_gpu(gpu)} - {precision_note(prec)} - {split} set")
        ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0, -0.55))
        save(fig, out_dir, f"fig{fig_no}_speedup_vs_{baseline.split(':')[0]}_{slug(short_gpu(gpu))}_{prec}_{split}", synthetic_banner(g))
        plt.close(fig)


def _ratio_bars(plt, m, ylabel, title, name, out_dir, banner):
    """m: rows with key columns and a 'ratio' column (>1 means the numerator-side implementation is faster)."""
    for (gpu, prec, split), g in m.groupby(["gpu", "precision", "split"]):
        g = g.sort_values("workload")
        fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(g)), 4.2))
        ax.bar(range(len(g)), g["ratio"].values, color=["#009E73" if v >= 1 else "#D55E00" for v in g["ratio"].values])
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g["workload"], rotation=70, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} - {short_gpu(gpu)} - {precision_note(prec)} - {split} set (geomean {geomean(g['ratio'].values):.2f}x)")
        save(fig, out_dir, f"{name}_{slug(short_gpu(gpu))}_{prec}_{split}", banner)
        plt.close(fig)


def plot(df, out_dir):
    plt = setup_matplotlib()
    ok = ok_rows(df)
    fused = ok[ok["fusion"] == "fused"]
    _speedup_bars(plt, df, "direct", out_dir, "03", "Speedup over naive direct CUDA")
    _speedup_bars(plt, df, "cudnn:library", out_dir, "04", "Speedup over cuDNN (values < 1x mean slower than cuDNN)")

    # Fig 7: autotuned vs fixed manually-selected configuration (same backend)
    for be in ("implicit_gemm", "tensor_core", "simt_tiled"):
        fx = fused[fused["variant"] == f"{be}:fixed"][key_cols() + ["median_us"]].rename(columns={"median_us": "fixed_us"})
        tu = fused[fused["variant"] == f"{be}:tuned"][key_cols() + ["median_us"]].rename(columns={"median_us": "tuned_us"})
        m = fx.merge(tu, on=key_cols())
        if m.empty:
            continue
        m["ratio"] = m["fixed_us"] / m["tuned_us"]
        _ratio_bars(plt, m, "tuned speedup over fixed config (x)", f"Autotuning gain ({be.replace('_', ' ')})",
                    f"fig07_autotune_{be}", out_dir, synthetic_banner(df))

    # Fig 8: Tensor Core vs SIMT implicit GEMM (same precision, same fusion)
    for mode in ("tuned", "fixed"):
        tc = fused[fused["variant"] == f"tensor_core:{mode}"][key_cols() + ["median_us"]].rename(columns={"median_us": "tc_us"})
        si = fused[fused["variant"] == f"implicit_gemm:{mode}"][key_cols() + ["median_us"]].rename(columns={"median_us": "simt_us"})
        m = tc.merge(si, on=key_cols())
        if m.empty:
            continue
        m["ratio"] = m["simt_us"] / m["tc_us"]
        _ratio_bars(plt, m, "Tensor Core speedup over SIMT implicit GEMM (x)",
                    f"Tensor Core vs SIMT, same low-precision storage + FP32 accumulate ({mode} configs)",
                    f"fig08_tensor_core_vs_simt_{mode}", out_dir, synthetic_banner(df))

    # Fig 10: per-workload winner/loser matrix (time relative to the best implementation)
    for (gpu, prec, split), g in fused.groupby(["gpu", "precision", "split"]):
        variants = [v for v in VARIANT_ORDER if v in set(g["variant"]) and v != "cpu_reference"]
        wl = list(dict.fromkeys(g.sort_values(["category", "workload"])["workload"]))
        piv = g.pivot_table(index="workload", columns="variant", values="median_us", aggfunc="first").reindex(wl)[variants]
        rel = piv.div(piv.min(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(variants), 0.28 * len(wl) + 1.8))
        im = ax.imshow(np.log2(rel.values), cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=5)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([pretty(v) for v in variants], rotation=45, ha="right")
        ax.set_yticks(range(len(wl)))
        ax.set_yticklabels(wl, fontsize=7)
        for i in range(len(wl)):
            for j in range(len(variants)):
                v = rel.values[i, j]
                if np.isnan(v):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=6, color="gray")
                elif v == 1.0:
                    ax.text(j, i, "WIN", ha="center", va="center", fontsize=6, weight="bold")
                else:
                    ax.text(j, i, f"{v:.1f}x", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, label="log2(time / best time)")
        ax.set_title(f"Winner per workload (slower = redder) - {short_gpu(gpu)} - {precision_note(prec)} - {split}")
        save(fig, out_dir, f"fig10_winners_{slug(short_gpu(gpu))}_{prec}_{split}", synthetic_banner(g))
        plt.close(fig)
