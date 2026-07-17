// Fused output stage shared by all custom kernels: bias + optional ReLU applied to
// the FP32 accumulator before the single rounding to the output type, so the fused
// path writes the output tensor exactly once.
#pragma once

#include "conv_common.cuh"

namespace gie {
namespace kernels {

template <typename T>
__device__ __forceinline__ float apply_epilogue(float acc, const T* __restrict__ bias, int m, int relu) {
  if (bias != nullptr) acc += to_f32(bias[m]);
  if (relu) acc = fmaxf(acc, 0.0f);
  return acc;
}

}  // namespace kernels
}  // namespace gie
