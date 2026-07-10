// Reference convolutions and error metrics.
//
// Reference precision policy (applies to every comparison in this repository):
//  * The reference consumes exactly the values the kernel consumes: tensors are
//    first quantized to the storage type (FP16/BF16) or, for TF32, rounded to the
//    TF32 multiplier precision with round-to-nearest-away (cvt.rna.tf32.f32).
//  * Products and sums are accumulated in double precision; bias is added, ReLU
//    applied, then the result is rounded once to float for comparison.
//  * Alongside y the reference emits scale[i] = sqrt(sum_k (x_k*w_k)^2 + (sum_k x_k*w_k)^2 + bias^2).
//    The first term is the random-walk magnitude of the partial sums (random-sign data), the second the
//    drift of partial sums when terms share a sign (e.g. all-positive inputs), which is the dominant
//    rounding-error source there. The tolerance model uses it (see TolerancePolicy) instead of an
//    arbitrary global epsilon.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_inference/conv2d.h"

namespace gie {

struct ReferenceOutput {
  std::vector<float> y;
  std::vector<float> scale;
};

// Straightforward nested-loop CPU implementation (double accumulation).
// `x`, `w`, `bias` hold already-quantized values; `bias` may be null.
void cpu_conv2d_reference(const ConvProblem& prob, const float* x, const float* w, const float* bias,
                          ReferenceOutput* out);

// GPU reference with double accumulation (direct, one thread per output). Reads the
// same device tensors the optimized kernels read. d_y/d_scale are float[numel(Y)].
void gpu_conv2d_reference(const ConvProblem& prob, const void* x, const void* w, const void* bias,
                          float* d_y, float* d_scale, cudaStream_t stream);

// Error tolerance: |err_i| <= c_acc * u_acc * sqrt(K) * scale_i      (accumulation)
//                          +  c_out * u_out * |ref_i|                 (output rounding)
//                          +  c_inter * u_out * scale_i               (only for two-rounding pipelines)
//                          +  c_in * u_in * scale_i                   (only TF32: multiplier-input rounding)
//                          +  abs_floor                               (subnormal outputs)
// u_acc = 2^-24 (FP32 accumulation), u_out is the unit roundoff of the output type.
// sqrt(K) models the random-walk growth of rounding error over K accumulations; the
// constants leave headroom over the observed errors of cuDNN / cuBLAS / the CPU reference.
struct TolerancePolicy {
  double c_acc = 8.0;
  double c_out = 1.0;
  // Pipelines that round the convolution result to the output type *before* bias/ReLU (cuBLAS im2col
  // epilogue, unfused bias/ReLU kernels) incur a second rounding of the pre-bias value |acc| <= scale.
  double c_inter = 0.0;
  // TF32 only: libraries round the multiplier inputs to 10 mantissa bits with their own rounding mode (round-
  // to-nearest-even, truncation, or emulation tricks) while the reference uses round-to-nearest-away, so
  // individual products differ by up to ~2*u_in. Independent per-term errors add as a random walk, giving a
  // term c_in*u_in*scale. This is an *accuracy* bound for TF32, deliberately far looser than the FP32 bound.
  double c_in = 0.0;
  double u_in = 0.00048828125;  // 2^-11, unit roundoff of the 10-bit TF32 mantissa
  double u_acc = 5.9604644775390625e-08;  // 2^-24
  double u_out = 5.9604644775390625e-08;
  double abs_floor = 1e-30;
  int64_t k = 1;
  static TolerancePolicy for_problem(const ConvProblem& prob, bool intermediate_rounding = false);
  double tol(double ref, double scale) const;
  std::string describe() const;
};

struct ErrorMetrics {
  int64_t n = 0;
  double max_abs = 0, mean_abs = 0, rms = 0;
  double max_rel = 0;            // |err| / max(|ref|, rel_floor), rel_floor = 1e-3 * rms(ref)
  double frac_over_tol = 0;
  int64_t over_tol = 0, nan_count = 0, inf_count = 0;
  double worst_tol_ratio = 0;    // max_i |err_i| / tol_i  (<= 1 means within tolerance)
  bool passed() const { return n > 0 && nan_count == 0 && inf_count == 0 && over_tol == 0; }
  std::string summary() const;
};

ErrorMetrics compare_outputs(const float* test, const float* ref, const float* scale, int64_t n,
                             const TolerancePolicy& tol);

// Host-side rounding helpers matching the device conversions.
float quantize_to(Precision p, float v);          // storage rounding for the kernel inputs
float round_to_tf32(float v);                      // round-to-nearest-away, 10-bit mantissa
std::vector<float> download_as_float(const void* d_ptr, DataType dt, int64_t n);

}  // namespace gie
