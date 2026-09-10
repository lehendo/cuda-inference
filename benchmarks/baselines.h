// Library baselines (NVIDIA cuDNN, CUTLASS) behind one small interface so the benchmark
// harness times every implementation with the identical protocol.
#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <memory>
#include <string>

#include "gpu_inference/conv2d.h"

namespace bench {

class Baseline {
 public:
  virtual ~Baseline() = default;
  virtual std::string name() const = 0;
  // One-time setup (descriptors, algorithm search, layout conversion of the static operands).
  // Returns false with *why filled when the problem is unsupported; never silently changes it.
  virtual bool setup(const gie::ConvProblem& prob, const void* x, const void* w, const void* bias,
                     void* y_scratch, std::string* why) = 0;
  virtual size_t workspace_bytes() const = 0;
  virtual std::string config() const = 0;           // selected algorithm / tile configuration
  virtual double setup_seconds() const = 0;         // algorithm search + descriptor creation
  // Steady-state kernel launch(es) on already-prepared operands. For baselines that need a
  // different layout this excludes layout conversion (see run_with_conversion).
  virtual void run(void* y, void* workspace, cudaStream_t stream) = 0;
  // Output in NCHW of the most recent run(), for validation (may convert into `y_nchw`).
  virtual void output_nchw(void* y_nchw, cudaStream_t stream) = 0;
  // Includes input/weight/output layout conversion when the baseline needs one; otherwise = run().
  virtual void run_with_conversion(const void* x, const void* w, void* y, void* workspace, cudaStream_t stream) {
    (void)x; (void)w;
    run(y, workspace, stream);
  }
  virtual bool needs_layout_conversion() const { return false; }
  // Optional accuracy gate. Given the NCHW output of a candidate algorithm it returns the worst
  // |error|/tolerance ratio (<= 1 passes). Baselines that choose among several algorithms (cuDNN) use it to
  // pick the fastest algorithm that meets the SAME accuracy bar as every other implementation, and record
  // the algorithms they rejected.
  using Acceptor = std::function<double(void* y_nchw)>;
  virtual void set_acceptor(Acceptor) {}
};

// Return nullptr when the dependency was not found at configure time.
// gemm_only: restrict the algorithm search to the GEMM-family algorithms (IMPLICIT_GEMM, IMPLICIT_PRECOMP_GEMM, GEMM), i.e. the
// algorithms that perform the same number of multiply-adds as a direct convolution (no FFT/Winograd). Reported as backend "cudnn_gemm".
std::unique_ptr<Baseline> make_cudnn_baseline(size_t workspace_limit_bytes, bool gemm_only = false);
std::unique_ptr<Baseline> make_cutlass_baseline();
std::string cudnn_version_string();   // "unavailable" when not built
std::string cutlass_version_string();

}  // namespace bench
