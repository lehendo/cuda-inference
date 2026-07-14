#include <map>
#include <mutex>

#include "conv_common.cuh"

namespace gie {
namespace kernels {

void set_max_dynamic_smem(const void* func, size_t bytes) {
  if (bytes <= 48 * 1024) return;  // default limit already suffices
  static std::mutex mu;
  static std::map<std::pair<const void*, int>, size_t> configured;
  int dev = 0;
  GIE_CUDA_CHECK(cudaGetDevice(&dev));
  std::lock_guard<std::mutex> lock(mu);
  size_t& cur = configured[{func, dev}];
  if (bytes > cur) {
    GIE_CUDA_CHECK(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
    cur = bytes;
  }
}

}  // namespace kernels

std::vector<LutEntry> build_conv_lut(const Conv2DParams& p) {
  const int64_t K = p.gemm_k();
  std::vector<LutEntry> lut(static_cast<size_t>(K));
  const int khw = p.kernel_h * p.kernel_w;
  for (int64_t k = 0; k < K; ++k) {
    const int c = int(k / khw);
    const int rs = int(k % khw);
    const int r = rs / p.kernel_w;
    const int s = rs % p.kernel_w;
    lut[size_t(k)] = {c * p.in_h * p.in_w, r * p.dilation_h, s * p.dilation_w, 0};
  }
  return lut;
}

std::vector<LutEntry> build_conv_lut_chunked(const Conv2DParams& p) {
  const int cv = p.in_channels / 8;
  const int64_t chunks = p.gemm_k() / 8;
  std::vector<LutEntry> lut(static_cast<size_t>(chunks));
  for (int64_t q = 0; q < chunks; ++q) {
    const int rs = int(q / cv);
    const int r = rs / p.kernel_w;
    const int s = rs % p.kernel_w;
    lut[size_t(q)] = {int32_t(8 * (q % cv)), r * p.dilation_h, s * p.dilation_w, 0};
  }
  return lut;
}

}  // namespace gie
