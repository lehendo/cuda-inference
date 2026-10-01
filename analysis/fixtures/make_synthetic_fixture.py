#!/usr/bin/env python3
"""Generates an explicitly SYNTHETIC results file used only to smoke-test the plotting/aggregation code.

The GPU name is "SYNTHETIC-FIXTURE", figures made from it carry a red watermark, and the output goes to
analysis/fixtures/out (never to results/). These numbers are invented and must never be reported.
"""
import json
import os
import random

random.seed(7)
out = os.path.join(os.path.dirname(__file__), "out", "raw")
os.makedirs(out, exist_ok=True)
workloads = [("lat_b1_c64_14_k3", "latency", 1, 64, 14, 64, 3), ("thr_b32_c64_56_k3", "throughput", 32, 64, 56, 64, 3),
             ("adv_c7", "adversarial", 4, 7, 33, 9, 5), ("sweep_batch_b8", "sweep_batch", 8, 64, 28, 64, 3),
             ("sweep_batch_b16", "sweep_batch", 16, 64, 28, 64, 3), ("sweep_batch_b32", "sweep_batch", 32, 64, 28, 64, 3)]
speed = {("direct", "library"): 1.0, ("im2col_gemm", "fixed"): 3.0, ("simt_tiled", "fixed"): 5.0, ("simt_tiled", "tuned"): 6.0,
         ("implicit_gemm", "fixed"): 8.0, ("implicit_gemm", "tuned"): 10.0, ("tensor_core", "fixed"): 25.0, ("tensor_core", "tuned"): 30.0,
         ("cudnn", "library"): 28.0}
with open(os.path.join(out, "results.jsonl"), "w") as f, open(os.path.join(out, "meta.json"), "w") as m:
    json.dump({"gpu": "SYNTHETIC-FIXTURE", "measured_fp32_tflops": 50, "measured_fp16_tc_tflops": 400, "measured_bw_gbs": 2500,
               "est_peak_fp32_tflops": 60, "est_peak_fp16_tc_tflops": 900, "est_peak_bw_gbs": 3300}, m)
    for prec in ("fp32", "fp16"):
        for name, cat, B, C, H, O, K in workloads:
            flops = 2.0 * B * O * H * H * C * K * K
            for (be, mode), sp in speed.items():
                if prec == "fp32" and be == "tensor_core":
                    continue
                us = (flops / 1e6 / 0.4) / sp * random.uniform(0.9, 1.1) + 5
                f.write(json.dumps({
                    "gpu": "SYNTHETIC-FIXTURE", "cc": 90, "workload": name, "category": cat, "split": "tune", "B": B, "C": C, "H": H, "W": H,
                    "O": O, "Kh": K, "Kw": K, "Sh": 1, "Sw": 1, "Ph": K // 2, "Pw": K // 2, "precision": prec, "bias": True, "relu": True,
                    "fusion": "fused", "backend": be, "mode": mode, "kernel": f"{be}_synthetic", "status": "ok", "median_us": us,
                    "flops": flops, "images_per_s": B / (us * 1e-6), "tflops": flops / (us * 1e-6) / 1e12,
                    "arithmetic_intensity": flops / (B * (C * H * H + O * H * H) * 2.0 + 1), "compulsory_bytes": 1.0, "workspace_bytes": 0}) + "\n")
print("wrote", out)
