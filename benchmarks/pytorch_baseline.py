#!/usr/bin/env python3
"""Application-level PyTorch comparison (optional): torch.nn.functional.conv2d on the same workload shapes.

Writes rows in the gie_bench results.jsonl schema (backend="pytorch", mode="library") so the standard
aggregation/plots pick them up. Notes on comparability:
  * Eager PyTorch has no fused conv+bias+ReLU: the ReLU is a separate kernel (unfused), bias is fused by cuDNN.
  * TF32 is explicitly disabled for fp32 (torch.backends.cudnn.allow_tf32 = False) to match strict FP32.
  * cudnn.benchmark=True: the algorithm search happens during warm-up, outside the timed region.
  * Tensors are random (torch generator, seeded) - this script reports timing only; correctness of the library
    kernels is established by the C++ test-suite and gie_bench validation, not here.
Usage: pytorch_baseline.py --workloads benchmarks/workloads.json --out results/raw/<dir>/pytorch --precisions fp32,fp16
"""
import argparse
import json
import os
import statistics

import torch
import torch.nn.functional as F


def load(path):
    with open(path) as fh:
        root = json.load(fh)
    for w in root["workloads"]:
        g = lambda k, d: w.get(k, d)
        K = g("K", 3)
        yield {
            "name": w["name"], "category": g("category", ""), "B": g("B", 1), "C": g("C", 1), "H": g("H", 1),
            "W": g("W", g("H", 1)), "O": g("O", g("C", 1)), "Kh": g("Kh", K), "Kw": g("Kw", K),
            "Sh": g("Sh", g("S", 1)), "Sw": g("Sw", g("S", 1)), "Ph": g("Ph", g("P", 0)), "Pw": g("Pw", g("P", 0)),
            "bias": g("bias", True), "relu": g("relu", True), "split": root.get("split", "unspecified"),
        }


def time_ms(fn, warmup, samples, min_ms):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); fn(); e.record(); torch.cuda.synchronize()
    probe = max(s.elapsed_time(e), 1e-4)
    inner = max(1, min(4096, int(min_ms / probe + 1)))
    out = []
    for _ in range(samples):
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / inner * 1000.0)  # us
    return out, inner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precisions", default="fp32,fp16")
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--min-sample-ms", type=float, default=1.0)
    ap.add_argument("--tag", default="pytorch")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.cuda.get_device_properties(0)
    gpu = dev.name
    meta = {"tag": a.tag, "gpu": gpu, "compute_capability": dev.major * 10 + dev.minor, "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    gen = torch.Generator(device="cuda").manual_seed(1234)
    with open(os.path.join(a.out, "results.jsonl"), "a") as out:
        for w in load(a.workloads):
            for prec in a.precisions.split(","):
                row = {"tag": a.tag, "gpu": gpu, "cc": meta["compute_capability"], "workload": w["name"], "category": w["category"],
                       "split": w["split"], "B": w["B"], "C": w["C"], "H": w["H"], "W": w["W"], "O": w["O"], "Kh": w["Kh"], "Kw": w["Kw"],
                       "Sh": w["Sh"], "Sw": w["Sw"], "Ph": w["Ph"], "Pw": w["Pw"], "precision": prec, "bias": w["bias"], "relu": w["relu"],
                       "fusion": "fused", "backend": "pytorch", "mode": "library", "kernel": "torch.nn.functional.conv2d (+F.relu, unfused)",
                       "status": "ok"}
                try:
                    x = torch.randn(w["B"], w["C"], w["H"], w["W"], device="cuda", dtype=dt[prec], generator=gen)
                    wt = torch.randn(w["O"], w["C"], w["Kh"], w["Kw"], device="cuda", dtype=dt[prec], generator=gen) * (3.0 / (w["C"] * w["Kh"] * w["Kw"])) ** 0.5
                    b = torch.randn(w["O"], device="cuda", dtype=dt[prec], generator=gen) if w["bias"] else None
                    stride, pad = (w["Sh"], w["Sw"]), (w["Ph"], w["Pw"])

                    def run():
                        y = F.conv2d(x, wt, b, stride=stride, padding=pad)
                        return F.relu(y) if w["relu"] else y

                    us, inner = time_ms(run, a.warmup, a.samples, a.min_sample_ms)
                    us_sorted = sorted(us)
                    med = statistics.median(us)
                    flops = 2.0 * w["B"] * w["O"] * ((w["H"] + 2 * w["Ph"] - w["Kh"]) // w["Sh"] + 1) * ((w["W"] + 2 * w["Pw"] - w["Kw"]) // w["Sw"] + 1) * w["C"] * w["Kh"] * w["Kw"]
                    row.update({"median_us": med, "mean_us": statistics.mean(us), "min_us": us_sorted[0], "p10_us": us_sorted[int(0.1 * (len(us) - 1))],
                                "p90_us": us_sorted[int(0.9 * (len(us) - 1))], "p99_us": us_sorted[-1], "stddev_us": statistics.pstdev(us),
                                "samples": len(us), "inner_iters": inner, "images_per_s": w["B"] / (med * 1e-6), "tflops": flops / (med * 1e-6) / 1e12,
                                "flops": flops, "validated": False})
                except Exception as e:  # noqa: BLE001 - recorded, not dropped
                    row.update({"status": "error", "reason": str(e)[:300]})
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(w["name"], prec, row.get("median_us", row.get("reason")))


if __name__ == "__main__":
    main()
