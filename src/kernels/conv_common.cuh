// Shared device/host helpers for all convolution kernels (private header).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_inference/conv2d.h"
#include "gpu_inference/kernel_registry.h"

namespace gie {
namespace kernels {

// Plain-old-data convolution geometry passed to kernels by value.
struct ConvArgs {
  int B, C, H, W;       // input
  int O, KH, KW;        // weights
  int SH, SW, PH, PW, DH, DW;
  int OH, OW;           // output
  int M, N, K;          // implicit-GEMM extents: M=O, N=B*OH*OW, K=C*KH*KW (all < 2^31, validated)
  int HWo;              // OH*OW
  int relu;
};

inline ConvArgs make_args(const ConvProblem& prob) {
  const Conv2DParams& p = prob.p;
  ConvArgs a;
  a.B = p.batch; a.C = p.in_channels; a.H = p.in_h; a.W = p.in_w;
  a.O = p.out_channels; a.KH = p.kernel_h; a.KW = p.kernel_w;
  a.SH = p.stride_h; a.SW = p.stride_w; a.PH = p.pad_h; a.PW = p.pad_w;
  a.DH = p.dilation_h; a.DW = p.dilation_w;
  a.OH = p.out_h(); a.OW = p.out_w();
  a.M = p.out_channels;
  a.N = int(p.gemm_n());
  a.K = int(p.gemm_k());
  a.HWo = a.OH * a.OW;
  a.relu = prob.activation == Activation::kReLU ? 1 : 0;
  return a;
}

// Storage-type traits per precision policy.
template <Precision P> struct PrecTraits;
template <> struct PrecTraits<Precision::kFp32> { using T = float; };
template <> struct PrecTraits<Precision::kTf32> { using T = float; };
template <> struct PrecTraits<Precision::kFp16> { using T = __half; };
template <> struct PrecTraits<Precision::kBf16> { using T = __nv_bfloat16; };

__device__ __forceinline__ float to_f32(float v) { return v; }
__device__ __forceinline__ float to_f32(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Per-kernel registration entry points (one per source file).
void register_direct(std::vector<KernelDesc>& out);
void register_im2col(std::vector<KernelDesc>& out);
void register_simt_tiled(std::vector<KernelDesc>& out);
void register_implicit_gemm(std::vector<KernelDesc>& out);
void register_tensor_core(std::vector<KernelDesc>& out);
void register_tensor_core_mma(std::vector<KernelDesc>& out);

// Opt a kernel in to > 48 KiB dynamic shared memory (idempotent, per function).
void set_max_dynamic_smem(const void* func, size_t bytes);

}  // namespace kernels
}  // namespace gie
