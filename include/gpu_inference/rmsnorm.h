// Optional transformer-inference extension: fused RMSNorm (and fused residual-add + RMSNorm).
//   y[r,:] = x[r,:] * rsqrt(mean(x[r,:]^2) + eps) * gamma[:]
// FP32 / FP16 / BF16 storage, FP32 reduction and arithmetic. One thread block per row, 16-byte vector loads
// when hidden is a multiple of the vector width and pointers are aligned (scalar fallback otherwise),
// warp-shuffle + shared-memory block reduction. The fused variant computes s = x + residual, writes s back
// (new residual stream) and y = rmsnorm(s) in a single kernel, removing one full read/write of the activations.
#pragma once

#include <cuda_runtime.h>

#include "gpu_inference/conv2d.h"  // DataType

namespace gie {

struct RmsNormParams {
  int rows = 1;
  int hidden = 1;
  float eps = 1e-6f;
  DataType dtype = DataType::kFloat32;
};

// y = rmsnorm(x). `residual_in` / `residual_out` both null: plain RMSNorm. Both non-null: fused add+norm
// (residual_out = x + residual_in is written; y = rmsnorm(residual_out)). residual_out may alias x.
void rmsnorm_forward(const RmsNormParams& p, const void* x, const void* gamma, void* y, const void* residual_in,
                     void* residual_out, cudaStream_t stream);

// Unfused baseline made of three simple kernels (square-mean per row, normalize, add) for comparison.
void rmsnorm_forward_unfused(const RmsNormParams& p, const void* x, const void* gamma, void* y, const void* residual_in,
                             void* residual_out, void* scratch_rows_f32, cudaStream_t stream);

}  // namespace gie
