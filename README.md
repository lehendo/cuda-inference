# Hardware-Aware GPU Inference Engine

A from-scratch CUDA convolution inference library that implements several forward-convolution algorithms
(naive, explicit im2col, shared-memory tiled, implicit GEMM, Tensor Core), selects between them with a
capability-aware dispatcher and a measuring autotuner, and ships a reproducible benchmark / profiling / Slurm
workflow that compares everything against cuDNN and CUTLASS.

Everything reported below was measured on the hardware named next to the number; anything not measured is
marked as such. Detailed, regenerable results live in `results/` (raw JSON/CSV, processed tables, figures).

## What is implemented

| Component | Where | Notes |
|---|---|---|
| CPU reference | `src/cpu/cpu_reference.cpp` | nested loops, double accumulation, emits an error-scale array for the tolerance model |
| Naive direct CUDA | `src/kernels/direct_conv.cu` | one thread per output element |
| Explicit im2col + GEMM | `src/kernels/im2col.cu` | custom im2col kernel + cuBLAS strided-batched GEMM (library-assisted baseline); no output permutation |
| Tiled SIMT convolution | `src/kernels/simt_tiled_conv.cu` | shared-memory tiling, register blocking, 5 tile shapes x 2 channel chunks |
| Custom implicit GEMM (CUDA cores) | `src/kernels/implicit_gemm.cu` | never materializes im2col; lookup-table gather, 2-stage smem pipeline, 8 tile shapes |
| Custom Tensor Core convolution (WMMA) | `src/kernels/tensor_core_conv.cu` | WMMA 16x16x16, FP16/BF16 in, FP32 accumulate, 9 tile shapes, irregular M/N/K by predication |
| Custom Tensor Core convolution (mma.sync) | `src/kernels/tensor_core_mma.cu` | `mma.sync.m16n8k16` + `ldmatrix` + multi-stage 16-byte `cp.async`, NHWC repack in the timed pipeline, split-K; needs C_in % 8 == 0 |
| Fused epilogues | all custom kernels | bias and bias+ReLU applied to the FP32 accumulator before the single rounding |
| Dispatcher | `src/runtime/dispatcher.cpp` | capability check -> selection (explicit > tuning cache > autotune > cost model) -> plan -> execute |
| Autotuner + cache | `src/autotuner/` | validates every candidate against a double-precision reference, robust-median timing, JSON cache keyed by GPU/CUDA/problem |
| Roofline cost model | `src/runtime/perf_model.cpp`, `src/profiling/ceilings.cu` | spec-derived and *measured* ceilings; ranks / prunes candidates |
| CUDA Graphs | `src/runtime/cuda_graph.cpp` | capture/replay/update of layer chains |
| Benchmarks | `benchmarks/` | harness, cuDNN + CUTLASS (+ optional PyTorch) baselines, tuning experiments, graph/stream experiment |
| Analysis | `analysis/` | aggregation tables and figures generated only from saved results |
| Optional extra | `src/transformer/rmsnorm.cu` | fused (add +) RMSNorm, FP32/FP16/BF16 |

Supported: forward 2D convolution, NCHW / OIHW, stride, symmetric zero padding, optional bias and ReLU,
groups = 1, dilation = 1, FP32 (strict), FP16, BF16 (FP32 accumulation), TF32 (cuBLAS im2col and cuDNN only).
GPUs: compute capability 8.0 and newer (the build rejects older architectures).
See `docs/LIMITATIONS.md` for the full list of what is *not* supported.

## Build

Requirements: CUDA toolkit >= 12.x with cuBLAS (tested with 12.8), CMake >= 3.24, a C++17 compiler.
Optional: cuDNN 9 (`-DCUDNN_ROOT=...`) and a CUTLASS checkout (`-DCUTLASS_DIR=...`, tested with v3.9.2) for
the baselines; without them those baselines are compiled out and reported as "unsupported", never silently
replaced.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
      -DCUDNN_ROOT=/path/to/cudnn -DCUTLASS_DIR=/path/to/cutlass
cmake --build build -j
ctest --test-dir build --output-on-failure        # run on a GPU node
```
On the Illinois Campus Cluster use the helpers in `scripts/` (`env.sh`, `build.sh`, `submit.sh`,
`slurm/*.sbatch`); resources are configurable through `GPU_SLURM_*` environment variables.

## Usage

```cpp
#include "gpu_inference/conv2d.h"
using namespace gie;

ConvProblem prob;
prob.p.batch = 8; prob.p.in_channels = 64; prob.p.in_h = prob.p.in_w = 56;
prob.p.out_channels = 64; prob.p.kernel_h = prob.p.kernel_w = 3; prob.p.pad_h = prob.p.pad_w = 1;
prob.precision = Precision::kFp16;          // FP16 storage, FP32 accumulation
prob.has_bias = true; prob.activation = Activation::kReLU;

size_t ws = Conv2D::workspace_size(prob);   // 0 for the fused custom kernels
Conv2D::forward(x, w, &bias, prob, y, workspace, ws, stream);   // backend chosen by the dispatcher

Conv2DOptions opt; opt.backend = Backend::kTensorCore;          // or force a backend / a specific kernel
opt.autotune = true;                                            // measure + cache the best configuration
```
`examples/quickstart.cpp` is a complete program. Tools: `gie_info` (device + ceilings), `gie_bench`
(benchmarks), `gie_tune` (autotuning, Experiment B), `gie_graph_bench` (CUDA Graphs / streams),
`gie_rmsnorm_bench`.

## Correctness

Every kernel is validated against a double-precision reference (CPU for small shapes, GPU for large) on 26
public shapes (batch 1-32, 1x1 to 7x7 and non-square kernels, stride 1/2, odd and unequal sizes, tiny
outputs, boundary-dominated padding) with four input distributions, with and without bias/ReLU. The
tolerance is derived from the reduction length and the output type (`include/gpu_inference/reference.h`),
not tuned to pass. Tests also cover argument validation, aliasing, workspace size, the dispatcher,
the autotuner (including injected faulty kernels that must never be selected) and the tuning cache.
`compute-sanitizer` memcheck / racecheck / initcheck / synccheck are run by `scripts/run_sanitizers.sh` (reduced shapes; racecheck on
representative subsets for the concurrency and CUDA Graph tests; the cuBLAS-backed backend is excluded from synccheck because
of a divergent barrier inside a cuBLAS kernel on Hopper). Concurrency tests cover creating a plan while other streams are busy,
sharing one plan across streams and host threads, unaligned user pointers, grid limits and CUDA Graph replay.

## Results (measured; NVIDIA A40 sm_86 and H100 NVL sm_90, CUDA 12.8, cuDNN 9.5.1, CUTLASS 3.9.2)

All numbers are medians of 30 CUDA-event samples, validated against a double-precision reference, and regenerable from
`results/raw` (see Reproduction). Speedup = baseline time / our time on the same GPU and workload; > 1 is faster.
27 tuning workloads, 17 held-out workloads (never used to select tuning configurations or fit the cost model; their shapes
and results were visible during development, so they are not a blind test), 31 sweep workloads.

**cuDNN baseline.** An audit found that cuDNN's one-shot algorithm search can mis-rank small kernels (on two FP16 H100 shapes
one run picked a 48 us algorithm where another picked a 20 us one). The baseline now re-times every algorithm that fits the
workspace limit and passes the same accuracy gate as the custom kernels, and keeps the fastest; across two repetitions it
picks the same algorithm for 51-54 of 54 tuning rows (the few changes are ties within ~1-3%) with latencies repeating to
+-0.7% (geomean). All "vs cuDNN" numbers below use the corrected baseline (separate repetition files under
`results/raw/audit_cudnn_retime_*`); the correction lowered the H100 FP16 advantage from 1.49x / 1.66x to 1.32x / 1.34x.
Custom and CUTLASS rows come from a separate run of the same sources (`results/raw/audit_repr_*`), so each ratio pairs rows
from two runs on the same GPU model (run-to-run drift of the unchanged kernels was +-1%).

| FP16 storage / FP32 accumulate, fused conv+bias+ReLU | A40 tuning / held-out | H100 tuning / held-out |
|---|---|---|
| custom Tensor Core (tuned) vs unrestricted cuDNN NCHW (corrected) | 1.39x / 1.51x (faster on 23/27, 15/17; worst 0.54x) | 1.32x / 1.34x (20/27, 13/17; worst 0.48x) |
| custom Tensor Core (tuned) vs custom CUDA-core implicit GEMM | 2.5x geomean over 119 workloads | 2.8x |
| tuned vs fixed hand-picked configuration (autotuning gain) | 1.4-2.4x geomean by backend | 1.3-2.7x |
| CUTLASS NHWC kernel-only vs our path (we are slower) | CUTLASS 1.6-1.8x faster | CUTLASS 1.9-2.0x faster |
| same, if CUTLASS must also convert NCHW<->NHWC | parity (1.15x / 1.00x in our favour) | parity (1.08x / 0.98x) |

* Strict FP32: cuDNN is faster than every custom FP32 kernel on average; the best custom kernel (CUDA-core implicit GEMM,
  tuned) reaches 0.67x / 0.76x (A40) and 0.63x / 0.72x (H100) of unrestricted cuDNN (tuning / held-out; wins on 5-7 of 17-27).
  cuDNN's largest FP32 wins use FFT/FFT_TILING/Winograd algorithms that do fewer multiply-adds than direct convolution, so
  "effective TFLOP/s" computed from direct-convolution FLOPs can exceed the FMA ceiling. Against cuDNN restricted to its
  GEMM-family algorithms (a labelled diagnostic, not the primary comparison) the custom kernel reaches 0.82x / 0.84x (A40) and
  0.72x / 0.78x (H100). The dispatcher (0.90-0.97x of cuDNN in FP32; 1.50-1.60x in FP16) may also pick cuBLAS im2col, so it is
  not a custom-kernel result.
* FP32 kernel work in the audit: occupancy-capped and column-interleaved variants of the 8x8-tile kernels improve individual
  shapes (up to +29% on one H100 shape, +12% on A40) but change the aggregate over 44 workloads by only +0.6% (A40) / +1.4% (H100).
  The ablations did not support uncoalesced output stores as the bottleneck; removing the global-load path cut ~38% of the
  64x64 kernel's time on the H100 and its inner loop reached ~70% of the FMA ceiling even without loads.
* Cost-model-guided search (top-3 of ~20-44 candidates) finds a kernel within 0-9% (geomean) of the exhaustive optimum at 8-18% of the
  search cost; a fixed manual configuration is 2.5-3.5x (geomean) slower than the tuned one.
* Fusing bias+ReLU into the Tensor Core epilogue is 1.33x (A40) / 1.40x (H100) faster than separate elementwise kernels (geomean);
  CUDA Graph replay (un-profiled, 16 layer-chain cases per precision per GPU) cuts host launch cost by a geomean of 33-70x
  (1-3300x depending on chain length) but device time only by a geomean of 1.13-1.18x (range 1.00-1.37x); replay was bitwise
  identical to eager in every case. Independent-convolution streams (2/4/8 streams vs 1): 0.56-3.3x for the cuBLAS im2col cases and 1.5-4.8x for the tiny
  single-kernel cases, with 2 streams slower than 1 for one size; do not generalize beyond the cases listed in `docs/RESULTS.md`.
* On Hopper the `mma.sync`/WMMA path measures 566 TFLOP/s: ~59% of a clock-derived dense FP16 estimate (965) or ~68% of the
  datasheet-derived ~835 (NVIDIA's 1,671 figure for the H100 NVL carries a sparsity footnote; halved here). Neither number is a
  hard ceiling for `mma.sync`; `wgmma`/TMA are not implemented. The A40 microbenchmark reaches 99% of its estimate.
* Not available/verified: Nsight Compute hardware counters (`ERR_NVGPUCTRPERM`), A100/Blackwell GPUs. Counter-based profiling
  claims are therefore absent; Nsight Systems timelines (`results/raw/*/nsys`) are included.

Full tables, losses, accuracy and the methodology are in `docs/RESULTS.md` and `docs/BENCHMARK_METHODOLOGY.md` (kept out of
version control); figures are in `results/figures/` (`fig01`-`fig10`, `expB_*`) and tables in `results/processed/`.

## Reproduction

```bash
scripts/submit.sh scripts/slurm/inspect_env.sbatch      # record the environment
scripts/submit.sh scripts/slurm/build_test.sbatch       # build + all tests on one GPU
scripts/submit.sh scripts/slurm/bench_full.sbatch       # full benchmark campaign (or scripts/run_campaign.sh on a GPU node)
scripts/submit.sh scripts/slurm/profile_nsys.sbatch     # Nsight Systems timeline (GPU_NSYS_WORKLOAD / GPU_NSYS_BACKEND)
python analysis/aggregate_results.py --raw results/raw --out results/processed
python analysis/make_figures.py --raw results/raw --figures results/figures
```
