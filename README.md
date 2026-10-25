# CUDA Inference Engine

A hardware-aware GPU inference library built from scratch in CUDA C++, featuring custom convolution kernels, Tensor Core acceleration, and automated kernel tuning. The project is intended for experimentation with GPU kernel design, hardware-aware optimization, and reproducible inference performance measurements.

The engine implements multiple convolution algorithms, from shared-memory tiled CUDA kernels to implicit GEMM and Tensor Core kernels using `mma.sync`, `ldmatrix`, and asynchronous `cp.async`. A capability-aware dispatcher and autotuner select validated implementations based on workload characteristics and GPU architecture.

Benchmarked against NVIDIA cuDNN and CUTLASS on A40 and H100 GPUs.

## Performance

Custom FP16 Tensor Core convolution achieves **1.32–1.51x geometric mean speedup over accuracy-qualified cuDNN** across the evaluated GPU and workload sets.

| GPU | Tuning set (27 workloads) | Evaluation set (17 workloads) |
|---|---:|---:|
| NVIDIA A40 | **1.39x** | **1.51x** |
| NVIDIA H100 NVL | **1.32x** | **1.34x** |

Results measure fused FP16 convolution with FP32 accumulation, bias, and ReLU using an NCHW interface. Speedup is cuDNN execution time divided by custom execution time, so values above 1.0x favor the custom implementation.

The custom kernels outperform cuDNN on 23/27 tuning and 15/17 evaluation workloads on A40, and 20/27 and 13/17 on H100.

**Additional findings**

- **CUTLASS:** Its NHWC convolution kernel is approximately 1.6–2.0x faster than the custom Tensor Core pipeline when measured without CUTLASS layout conversions. With transformations required by the NCHW interface included, performance is approximately comparable.
- **FP32:** Custom implicit GEMM reaches 0.63–0.76x of unrestricted cuDNN performance on average across the evaluated sets.
- **Autotuning:** Cost-model-guided top-3 search comes within approximately 5–9% of exhaustive tuning on the measured tuning sets, using substantially fewer candidate measurements.
- **Kernel fusion:** Fused bias and ReLU improve Tensor Core execution by 1.33x on A40 and 1.40x on H100 over separate elementwise operations.

Measurements use CUDA 12.8, cuDNN 9.5.1, and CUTLASS 3.9.2. Results are geometric means over the specified workloads, not guarantees for arbitrary convolution shapes. The evaluation set was excluded from tuning and cost-model fitting, but its shapes were visible during development and it is not a blind test set.

## Features

### Custom CUDA kernels

| Backend | Implementation |
|---|---|
| Direct convolution | One CUDA thread per output element |
| Tiled SIMT | Shared-memory tiling and register blocking |
| Explicit im2col | Custom input unrolling with cuBLAS GEMM |
| Implicit GEMM | Custom CUDA-core GEMM without materializing the im2col matrix |
| Tensor Core WMMA | FP16/BF16 matrix operations with FP32 accumulation |
| Tensor Core MMA | `mma.sync`, `ldmatrix`, multi-stage `cp.async`, and split-K |

The im2col backend uses cuBLAS for matrix multiplication. The SIMT, implicit GEMM, WMMA, and MMA convolution kernels are custom implementations.

### Hardware-aware autotuning

A runtime dispatcher filters kernels by GPU capability, precision, tensor dimensions, shared-memory usage, and workspace requirements.

The autotuner validates candidate kernels against a double-precision reference, benchmarks supported configurations, and caches the fastest validated selection.

A roofline-inspired cost model estimates arithmetic intensity, occupancy, tile efficiency, and launch overhead to rank configurations and reduce tuning cost.

Cached selections are keyed by workload, backend, GPU characteristics, software versions, and source fingerprint.

### Additional optimizations

- **Fused epilogues:** Apply bias and ReLU directly to FP32 accumulators before output conversion.
- **CUDA Graphs:** Capture and replay convolution chains, reducing host launch overhead.
- **Split-K parallelism:** Improve GPU utilization for workloads with long reductions and relatively small output dimensions.
- **Reusable execution plans:** Cache kernel selection and setup state for repeated execution.
- **Fused RMSNorm:** Additional FP32/FP16/BF16 normalization operator with optional residual addition.

## Architecture

```text
                      Conv2D request
                            |
                            v
                    Problem validation
                            |
                            v
                    Execution plan cache
                            |
                            v
                    Capability filtering
                            |
                            v
                    Kernel selection
                  /         |          \
            Explicit      Cached      Autotune /
            override      winner      cost model
                  \         |          /
                            v
                      Execution plan
                            |
                            v
                       CUDA stream
                            |
                            v
                    Convolution output
```

The runtime separates kernel compatibility, configuration selection, plan creation, and execution.

Each kernel registers its supported configurations and hardware constraints. Execution plans own reusable setup state and launch kernels on caller-provided CUDA streams.

The optimized Tensor Core backend converts NCHW inputs into a channel-contiguous representation to enable vectorized `cp.async` transfers. Required layout transformations and weight repacking are included in its measured execution times.

## Getting started

### Requirements

- NVIDIA GPU with compute capability 8.0 or newer
- CUDA Toolkit 12.x or newer
- CMake 3.24 or newer
- C++17-compatible compiler
- cuBLAS, included with CUDA

Optional dependencies for library comparisons:

- cuDNN 9
- CUTLASS (tested with v3.9.2)

The core library builds without cuDNN or CUTLASS. Those benchmark backends are disabled when their dependencies are unavailable.

### Build

Configure CMake for your GPU architecture. For example, on an NVIDIA A40:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build -j
```

Use architecture `90` for H100. Builds have also been verified for `sm_80` and `sm_89`, although performance tests were conducted only on A40 and H100 NVL.

To enable cuDNN and CUTLASS comparisons, add:

```bash
-DCUDNN_ROOT=/path/to/cudnn \
-DCUTLASS_DIR=/path/to/cutlass
```

Run the correctness tests on a machine with an NVIDIA GPU:

```bash
ctest --test-dir build --output-on-failure
```

### Usage

The library exposes a C++ convolution API through [`conv2d.h`](include/gpu_inference/conv2d.h).

```cpp
#include "gpu_inference/conv2d.h"

using namespace gie;

ConvProblem prob;
prob.p.batch = 8;
prob.p.in_channels = 64;
prob.p.in_h = prob.p.in_w = 56;
prob.p.out_channels = 64;
prob.p.kernel_h = prob.p.kernel_w = 3;
prob.p.pad_h = prob.p.pad_w = 1;

prob.precision = Precision::kFp16;
prob.has_bias = true;
prob.activation = Activation::kReLU;

// x, w, bias, y, workspace, and stream are caller-provided.
size_t ws = Conv2D::workspace_size(prob);
Conv2D::forward(x, w, &bias, prob, y, workspace, ws, stream);
```

This is an API sketch. See [`examples/quickstart.cpp`](examples/quickstart.cpp) for the complete example, including device memory management and execution.

Available command-line tools include `gie_info`, `gie_bench`, `gie_tune`, `gie_graph_bench`, and `gie_rmsnorm_bench`.

## Benchmarking

Benchmarks use deterministic synthetic convolution workloads spanning batch size, channel count, spatial resolution, kernel size, padding, and stride.

GPU execution is measured with CUDA events after warmup. Short kernels are timed using repeated back-to-back launches to reduce event-resolution error. Reported execution measurements use the median of 30 samples.

The benchmark harness distinguishes:

- **Steady-state execution:** GPU kernel and required pipeline operations with data already resident on the device.
- **Layout-inclusive execution:** Required format transformations to match the public NCHW interface.
- **Setup and tuning:** Workspace preparation, algorithm search, and autotuning, measured separately.
- **Host overhead:** Launch and synchronization costs, reported separately from device execution.

For cuDNN, all available algorithms passing the workspace and numerical accuracy requirements are re-timed with a consistent CUDA-event measurement procedure. The fastest accepted algorithm is used as the baseline. Unrestricted cuDNN is the primary reference; GEMM-family-only comparisons are treated as diagnostic results.

Strict FP32 benchmarks disable implicit TF32 substitution.

### Reproducing benchmarks

The repository contains benchmark workloads, profiling tools, analysis scripts, and optional Slurm integration.

Run the benchmark harness on an allocated GPU:

```bash
scripts/run_benchmarks.sh --e2e --mode both \
  --backends implicit_gemm,tensor_core,dispatcher,cudnn,cutlass
```

Optional cuDNN and CUTLASS dependencies must be available to include those backends.

The repository also includes scripts for full benchmark campaigns and report generation:

```bash
# Full benchmark campaign on a Slurm cluster
scripts/submit.sh scripts/slurm/bench_full.sbatch

# Regenerate processed results and figures
scripts/make_report_data.sh
```

Slurm integration is optional and requires an appropriately configured cluster account.
