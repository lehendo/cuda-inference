// Elementwise bias / ReLU passes. Each pass reads and writes the whole output
// tensor (2 * |Y| bytes of DRAM traffic), which is exactly the cost that fusing
// into a convolution epilogue removes.
#include "conv_common.cuh"
#include "gpu_inference/fused_ops.h"

namespace gie {
namespace {

using kernels::from_f32;
using kernels::to_f32;

template <typename T>
__global__ void bias_act_kernel(T* __restrict__ y, const T* __restrict__ bias, int64_t total, int O, int HWo,
                                int relu) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += int64_t(gridDim.x) * blockDim.x) {
    float v = to_f32(y[i]);
    if (bias != nullptr) v += to_f32(bias[(i / HWo) % O]);
    if (relu) v = fmaxf(v, 0.f);
    y[i] = from_f32<T>(v);
  }
}

template <typename T>
void run(const Conv2DParams& p, const void* bias, bool relu, void* y, cudaStream_t s) {
  const int64_t total = int64_t(p.batch) * p.out_channels * p.out_h() * p.out_w();
  int64_t blocks = (total + 255) / 256;
  if (blocks > 2147483647LL) blocks = 2147483647LL;
  bias_act_kernel<T><<<unsigned(blocks), 256, 0, s>>>(static_cast<T*>(y), static_cast<const T*>(bias), total,
                                                      p.out_channels, p.out_h() * p.out_w(), relu ? 1 : 0);
  GIE_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void launch_bias_act(const Conv2DParams& p, Precision prec, const void* bias, bool relu, void* y,
                     cudaStream_t s) {
  if (bias == nullptr && !relu) return;
  switch (prec) {
    case Precision::kFp32:
    case Precision::kTf32: run<float>(p, bias, relu, y, s); break;
    case Precision::kFp16: run<__half>(p, bias, relu, y, s); break;
    case Precision::kBf16: run<__nv_bfloat16>(p, bias, relu, y, s); break;
  }
}

void launch_bias_add(const Conv2DParams& p, Precision prec, const void* bias, void* y, cudaStream_t s) {
  launch_bias_act(p, prec, bias, false, y, s);
}

void launch_relu(const Conv2DParams& p, Precision prec, void* y, cudaStream_t s) {
  launch_bias_act(p, prec, nullptr, true, y, s);
}

}  // namespace gie
