"""Figure 9: roofline - estimated arithmetic intensity vs achieved performance, with measured and spec ceilings."""
import numpy as np

from common import (VARIANT_COLORS, VARIANT_ORDER, load_meta, ok_rows, precision_note, pretty, save, setup_matplotlib, short_gpu, slug,
                    synthetic_banner)


def plot(df, out_dir, meta=None):
    plt = setup_matplotlib()
    meta = meta or {}
    ok = ok_rows(df)
    ok = ok[(ok["fusion"] == "fused") & ok["arithmetic_intensity"].notna() & ok["tflops"].notna()]
    for (gpu, prec), g in ok.groupby(["gpu", "precision"]):
        m = meta.get(gpu, {})
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        for v in [v for v in VARIANT_ORDER if v in set(g["variant"]) and v != "cpu_reference"]:
            t = g[g["variant"] == v]
            ax.scatter(t["arithmetic_intensity"], t["tflops"], s=16, label=pretty(v), color=VARIANT_COLORS.get(v), alpha=0.85)
        xs = np.logspace(0, 3.2, 200)
        bw_meas, bw_spec = m.get("measured_bw_gbs"), m.get("est_peak_bw_gbs")
        if prec in ("fp16", "bf16"):
            ceil_m, ceil_s, cname = m.get("measured_fp16_tc_tflops"), m.get("est_peak_fp16_tc_tflops"), "Tensor Core (WMMA) FP16"
        else:
            ceil_m, ceil_s, cname = m.get("measured_fp32_tflops"), m.get("est_peak_fp32_tflops"), "FP32 FMA"
        # NOTE: GB/s * FLOP/B = GFLOP/s -> /1000 = TFLOP/s
        if bw_meas and ceil_m:
            ax.plot(xs, np.minimum(ceil_m, bw_meas * xs / 1000.0), color="k", lw=1.6, label=f"measured ceilings ({cname}: {ceil_m:.0f} TFLOP/s, copy BW {bw_meas:.0f} GB/s)")
        if bw_spec and ceil_s:
            ax.plot(xs, np.minimum(ceil_s, bw_spec * xs / 1000.0), color="k", lw=1.0, ls="--", label=f"spec-derived estimate ({ceil_s:.0f} TFLOP/s, {bw_spec:.0f} GB/s)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("arithmetic intensity (FLOP / compulsory byte; theoretical, not measured DRAM traffic)")
        ax.set_ylabel("achieved TFLOP/s (measured)")
        ax.set_title(f"Roofline - {short_gpu(gpu)} - {precision_note(prec)}")
        ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        save(fig, out_dir, f"fig09_roofline_{slug(short_gpu(gpu))}_{prec}", synthetic_banner(g))
        plt.close(fig)
