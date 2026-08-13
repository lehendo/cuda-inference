#include "gpu_inference/rmsnorm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "gpu_inference/error.h"

namespace gie {
namespace {

__device__ __forceinline__ float cvt(float v) { return v; }
__device__ __forceinline__ float cvt(__half v) { return __half2float(v); }
__device__ __forceinline__ float cvt(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T back(float v);
template <> __device__ __forceinline__ float back<float>(float v) { return v; }
template <> __device__ __forceinline__ __half back<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 back<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

__device__ __forceinline__ float block_sum(float v, float* smem) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  v = (threadIdx.x < nw) ? smem[threadIdx.x] : 0.f;
  if (warp == 0) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) smem[0] = v;
  }
  __syncthreads();
  const float r = smem[0];
  __syncthreads();  // smem reusable
  return r;
}

// VEC = elements per 16-byte vector. `fast` => hidden % VEC == 0 and all pointers 16-byte aligned.
template <typename T, int VEC, bool FUSED>
__global__ void rmsnorm_kernel(const T* __restrict__ x, const T* __restrict__ gamma, T* __restrict__ y,
                               const T* __restrict__ res_in, T* __restrict__ res_out, int hidden, float eps, int fast) {
  __shared__ float smem[32];
  const size_t row = blockIdx.x;
  const T* xr = x + row * hidden;
  T* yr = y + row * hidden;
  const T* rin = FUSED ? res_in + row * hidden : nullptr;
  T* rout = FUSED ? res_out + row * hidden : nullptr;

  float ss = 0.f;
  if (fast) {
    const int nvec = hidden / VEC;
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
      uint4 raw = reinterpret_cast<const uint4*>(xr)[i];
      T v[VEC];
      memcpy(v, &raw, sizeof(raw));
      if (FUSED) {
        uint4 rr = reinterpret_cast<const uint4*>(rin)[i];
        T r[VEC];
        memcpy(r, &rr, sizeof(rr));
#pragma unroll
        for (int j = 0; j < VEC; ++j) v[j] = back<T>(cvt(v[j]) + cvt(r[j]));  // residual stream is stored in T
        memcpy(&raw, v, sizeof(raw));
        reinterpret_cast<uint4*>(rout)[i] = raw;
      }
#pragma unroll
      for (int j = 0; j < VEC; ++j) { const float f = cvt(v[j]); ss = fmaf(f, f, ss); }
    }
  } else {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      float f = cvt(xr[i]);
      if (FUSED) {
        const T s = back<T>(f + cvt(rin[i]));
        rout[i] = s;
        f = cvt(s);
      }
      ss = fmaf(f, f, ss);
    }
  }
  ss = block_sum(ss, smem);
  const float inv = rsqrtf(ss / float(hidden) + eps);

  const T* src = FUSED ? rout : xr;  // re-read (L1/L2 resident) after the reduction
  if (fast) {
    const int nvec = hidden / VEC;
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
      uint4 raw = reinterpret_cast<const uint4*>(src)[i], graw = reinterpret_cast<const uint4*>(gamma)[i];
      T v[VEC], g[VEC];
      memcpy(v, &raw, sizeof(raw));
      memcpy(g, &graw, sizeof(graw));
#pragma unroll
      for (int j = 0; j < VEC; ++j) v[j] = back<T>(cvt(v[j]) * inv * cvt(g[j]));
      memcpy(&raw, v, sizeof(raw));
      reinterpret_cast<uint4*>(yr)[i] = raw;
    }
  } else {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) yr[i] = back<T>(cvt(src[i]) * inv * cvt(gamma[i]));
  }
}

bool aligned16(const void* p) { return p == nullptr || reinterpret_cast<uintptr_t>(p) % 16 == 0; }

template <typename T, int VEC>
void run(const RmsNormParams& p, const void* x, const void* gamma, void* y, const void* rin, void* rout, cudaStream_t s) {
  const bool fused = rin != nullptr && rout != nullptr;
  const bool fast = p.hidden % VEC == 0 && aligned16(x) && aligned16(gamma) && aligned16(y) && aligned16(rin) && aligned16(rout);
  const int work = fast ? p.hidden / VEC : p.hidden;
  int threads = 32;
  while (threads < work && threads < 1024) threads <<= 1;
  threads = threads > 256 && work < 512 ? 128 : threads;  // small rows: fewer idle lanes
  if (fused)
    rmsnorm_kernel<T, VEC, true><<<p.rows, threads, 0, s>>>(static_cast<const T*>(x), static_cast<const T*>(gamma), static_cast<T*>(y),
                                                            static_cast<const T*>(rin), static_cast<T*>(rout), p.hidden, p.eps, fast);
  else
    rmsnorm_kernel<T, VEC, false><<<p.rows, threads, 0, s>>>(static_cast<const T*>(x), static_cast<const T*>(gamma), static_cast<T*>(y),
                                                             nullptr, nullptr, p.hidden, p.eps, fast);
  GIE_CUDA_CHECK(cudaGetLastError());
}

// ---- unfused baseline -----------------------------------------------------------------------------------------
template <typename T>
__global__ void ms_kernel(const T* __restrict__ x, float* __restrict__ out, int hidden) {  // one block per row
  __shared__ float smem[32];
  float ss = 0.f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) { const float f = cvt(x[size_t(blockIdx.x) * hidden + i]); ss += f * f; }
  ss = block_sum(ss, smem);
  if (threadIdx.x == 0) out[blockIdx.x] = ss / float(hidden);
}
template <typename T>
__global__ void scale_kernel(const T* __restrict__ x, const T* __restrict__ g, const float* __restrict__ ms, T* __restrict__ y, int hidden,
                             float eps, size_t total) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x)
    y[i] = back<T>(cvt(x[i]) * rsqrtf(ms[i / hidden] + eps) * cvt(g[i % hidden]));
}
template <typename T>
__global__ void add_kernel(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ o, size_t total) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x)
    o[i] = back<T>(cvt(a[i]) + cvt(b[i]));
}
template <typename T>
void run_unfused(const RmsNormParams& p, const void* x, const void* g, void* y, const void* rin, void* rout, void* scratch, cudaStream_t s) {
  const size_t total = size_t(p.rows) * p.hidden;
  const unsigned blocks = unsigned(std::min<size_t>((total + 255) / 256, 1u << 20));
  const T* src = static_cast<const T*>(x);
  if (rin != nullptr && rout != nullptr) {
    add_kernel<T><<<blocks, 256, 0, s>>>(src, static_cast<const T*>(rin), static_cast<T*>(rout), total);
    src = static_cast<const T*>(rout);
  }
  ms_kernel<T><<<p.rows, 256, 0, s>>>(src, static_cast<float*>(scratch), p.hidden);
  scale_kernel<T><<<blocks, 256, 0, s>>>(src, static_cast<const T*>(g), static_cast<const float*>(scratch), static_cast<T*>(y), p.hidden, p.eps, total);
  GIE_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void rmsnorm_forward(const RmsNormParams& p, const void* x, const void* gamma, void* y, const void* rin, void* rout, cudaStream_t s) {
  GIE_REQUIRE(p.rows > 0 && p.hidden > 0, StatusCode::kInvalidArgument, "rmsnorm: rows and hidden must be positive");
  GIE_REQUIRE(x && gamma && y, StatusCode::kInvalidArgument, "rmsnorm: null pointer");
  GIE_REQUIRE((rin == nullptr) == (rout == nullptr), StatusCode::kInvalidArgument, "rmsnorm: residual_in and residual_out must be given together");
  GIE_REQUIRE(y != x && y != gamma && y != rin && y != rout, StatusCode::kInvalidArgument,
              "rmsnorm: y must not alias x, gamma or the residual buffers");
  switch (p.dtype) {
    case DataType::kFloat32: run<float, 4>(p, x, gamma, y, rin, rout, s); break;
    case DataType::kFloat16: run<__half, 8>(p, x, gamma, y, rin, rout, s); break;
    case DataType::kBFloat16: run<__nv_bfloat16, 8>(p, x, gamma, y, rin, rout, s); break;
  }
}

void rmsnorm_forward_unfused(const RmsNormParams& p, const void* x, const void* g, void* y, const void* rin, void* rout, void* scratch,
                             cudaStream_t s) {
  GIE_REQUIRE(p.rows > 0 && p.hidden > 0 && x && g && y && scratch, StatusCode::kInvalidArgument, "rmsnorm_unfused: bad arguments");
  switch (p.dtype) {
    case DataType::kFloat32: run_unfused<float>(p, x, g, y, rin, rout, scratch, s); break;
    case DataType::kFloat16: run_unfused<__half>(p, x, g, y, rin, rout, scratch, s); break;
    case DataType::kBFloat16: run_unfused<__nv_bfloat16>(p, x, g, y, rin, rout, scratch, s); break;
  }
}

}  // namespace gie
