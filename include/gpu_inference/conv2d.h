// Convolution problem description and the public Conv2D entry point.
//
// Semantics (matches common deep-learning frameworks, i.e. cross-correlation):
//   Y[b,o,oh,ow] = bias[o] + sum_{c,r,s} X[b,c,oh*Sh-Ph+r*Dh, ow*Sw-Pw+s*Dw] * W[o,c,r,s]
// with zero padding outside X, followed by an optional ReLU.
//   H_out = floor((H + 2*Ph - Dh*(Kh-1) - 1) / Sh) + 1  (same for W)
// Layouts: X is NCHW, W is OIHW, Y is NCHW, all contiguous.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "gpu_inference/tensor.h"

namespace gie {

// Numerical policy. Accumulation is FP32 for every GPU precision; the policy
// names the storage/input type and any reduced-precision multiplier:
//   kFp32 : FP32 storage, strict FP32 arithmetic (no TF32).
//   kFp16 : FP16 storage (inputs/weights/bias/output), FP32 accumulation.
//   kBf16 : BF16 storage, FP32 accumulation.
//   kTf32 : FP32 storage, inputs rounded to TF32 (10-bit mantissa) for the
//           multiplies, FP32 accumulation. Never labeled equivalent to kFp32.
enum class Precision { kFp32, kFp16, kBf16, kTf32 };

DataType tensor_dtype(Precision p);
const char* to_string(Precision p);
Precision precision_from_string(const std::string& s);  // throws on unknown

enum class Activation { kNone, kReLU };
const char* to_string(Activation a);

enum class Backend {
  kAuto,          // dispatcher chooses
  kCpuReference,  // host reference (not dispatchable on the GPU path)
  kDirect,        // naive direct CUDA, one thread per output element
  kIm2colGemm,    // explicit im2col + cuBLAS GEMM (library-assisted baseline)
  kSimtTiled,     // shared-memory tiled, register-blocked direct convolution (CUDA cores)
  kImplicitGemm,  // custom implicit-GEMM on CUDA cores
  kTensorCore,    // custom implicit-GEMM on Tensor Cores (WMMA)
};
const char* to_string(Backend b);
Backend backend_from_string(const std::string& s);  // throws on unknown

struct Conv2DParams {
  int batch = 1;
  int in_channels = 1, in_h = 1, in_w = 1;
  int out_channels = 1, kernel_h = 1, kernel_w = 1;
  int stride_h = 1, stride_w = 1;
  int pad_h = 0, pad_w = 0;
  int dilation_h = 1, dilation_w = 1;  // only 1 is supported by the current backends
  int groups = 1;                      // only 1 is supported by the current backends

  int out_h() const { return (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1; }
  int out_w() const { return (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1; }

  // Implicit-GEMM view: C[M,N] = A[M,K] * B[K,N]
  int64_t gemm_m() const { return out_channels; }
  int64_t gemm_n() const { return int64_t(batch) * out_h() * out_w(); }
  int64_t gemm_k() const { return int64_t(in_channels) * kernel_h * kernel_w; }
  // FLOPs = 2 * B * C_out * H_out * W_out * C_in * Kh * Kw (one multiply-add = 2 FLOPs).
  int64_t flops() const { return 2 * gemm_m() * gemm_n() * gemm_k(); }

  TensorShape input_shape() const { return {batch, in_channels, in_h, in_w}; }
  TensorShape weight_shape() const { return {out_channels, in_channels, kernel_h, kernel_w}; }
  TensorShape output_shape() const { return {batch, out_channels, out_h(), out_w()}; }
  TensorShape bias_shape() const { return {1, out_channels, 1, 1}; }

  std::string to_string() const;
  bool operator==(const Conv2DParams& o) const;
};

// Complete operator description: geometry + numerics + fused epilogue.
struct ConvProblem {
  Conv2DParams p;
  Precision precision = Precision::kFp32;
  bool has_bias = false;
  Activation activation = Activation::kNone;

  DataType dtype() const { return tensor_dtype(precision); }
  std::string key() const;  // canonical string, used for plan/cache keys
  bool operator==(const ConvProblem& o) const { return key() == o.key(); }
};

// Throws gie::Error(kInvalidArgument / kUnsupported) with a specific message.
void validate_problem(const ConvProblem& prob);

struct Conv2DOptions {
  Backend backend = Backend::kAuto;
  std::string kernel;          // force one registered kernel configuration by name (optional)
  bool autotune = false;       // with kAuto: tune on a cache miss before executing
  bool force_retune = false;   // ignore a cached entry and tune again
  std::string cache_path;      // tuning cache file; empty -> default location
};

class Conv2D {
 public:
  // Executes Y = act(conv(X, W) [+ bias]) asynchronously on `stream`.
  // `workspace` must provide at least workspace_size(...) bytes when that is non-zero.
  // Plans are cached internally per (problem, options, device); the first call for a
  // new problem performs one-time setup (kernel selection, table upload), later calls
  // only launch. No device synchronization happens here.
  static void forward(const ConstTensorRef& input, const ConstTensorRef& weights,
                      const ConstTensorRef* bias, const ConvProblem& problem, const TensorRef& output,
                      void* workspace, size_t workspace_bytes, cudaStream_t stream,
                      const Conv2DOptions& options = {});

  static size_t workspace_size(const ConvProblem& problem, const Conv2DOptions& options = {});
  static void clear_plan_cache();
};

}  // namespace gie
