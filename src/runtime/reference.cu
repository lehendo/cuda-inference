// GPU reference convolution (double accumulation) + host error metrics + numeric helpers.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

#include "../kernels/conv_common.cuh"
#include "gpu_inference/reference.h"

namespace gie {
namespace {

using kernels::ConvArgs;
using kernels::make_args;
using kernels::PrecTraits;
using kernels::to_f32;

__device__ __forceinline__ float tf32_round(float v) {
  unsigned r;  // tf32 values live in a 32-bit untyped register
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v));
  return __uint_as_float(r);
}

template <typename T, bool TF32>
__device__ __forceinline__ double load_d(const T* p) {
  float v = to_f32(*p);
  if (TF32) v = tf32_round(v);
  return double(v);
}

template <typename T, bool TF32>
__global__ void gpu_ref_kernel(const T* __restrict__ x, const T* __restrict__ w, const T* __restrict__ bias,
                               float* __restrict__ y, float* __restrict__ scale, ConvArgs a) {
  const int64_t total = int64_t(a.B) * a.O * a.OH * a.OW;
  for (int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += int64_t(gridDim.x) * blockDim.x) {
    const int ow = int(idx % a.OW);
    const int oh = int((idx / a.OW) % a.OH);
    const int o = int((idx / a.HWo) % a.O);
    const int b = int(idx / (int64_t(a.HWo) * a.O));
    double acc = 0.0, sumsq = 0.0;
    for (int c = 0; c < a.C; ++c) {
      for (int r = 0; r < a.KH; ++r) {
        const int ih = oh * a.SH - a.PH + r * a.DH;
        if (ih < 0 || ih >= a.H) continue;
        for (int s = 0; s < a.KW; ++s) {
          const int iw = ow * a.SW - a.PW + s * a.DW;
          if (iw < 0 || iw >= a.W) continue;
          const double xv = load_d<T, TF32>(&x[((int64_t(b) * a.C + c) * a.H + ih) * a.W + iw]);
          const double wv = load_d<T, TF32>(&w[((int64_t(o) * a.C + c) * a.KH + r) * a.KW + s]);
          const double pr = xv * wv;
          acc += pr;
          sumsq += pr * pr;
        }
      }
    }
    sumsq += acc * acc;  // partial-sum drift term (see reference.h)
    if (bias != nullptr) {
      const double bv = double(to_f32(bias[o]));  // bias is not rounded to TF32 (added in FP32)
      acc += bv;
      sumsq += bv * bv;
    }
    if (a.relu && acc < 0.0) acc = 0.0;
    y[idx] = float(acc);
    scale[idx] = float(sqrt(sumsq));
  }
}

template <typename T, bool TF32>
void run_ref(const ConvProblem& prob, const void* x, const void* w, const void* bias, float* y, float* scale,
             cudaStream_t s) {
  const ConvArgs a = make_args(prob);
  const int64_t total = int64_t(a.B) * a.O * a.OH * a.OW;
  int64_t blocks = (total + 255) / 256;
  if (blocks > 2147483647LL) blocks = 2147483647LL;
  gpu_ref_kernel<T, TF32><<<unsigned(blocks), 256, 0, s>>>(static_cast<const T*>(x), static_cast<const T*>(w),
                                                          static_cast<const T*>(bias), y, scale, a);
  GIE_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void gpu_conv2d_reference(const ConvProblem& prob, const void* x, const void* w, const void* bias, float* d_y,
                          float* d_scale, cudaStream_t stream) {
  validate_problem(prob);
  const void* b = prob.has_bias ? bias : nullptr;
  switch (prob.precision) {
    case Precision::kFp32: run_ref<float, false>(prob, x, w, b, d_y, d_scale, stream); break;
    case Precision::kTf32: run_ref<float, true>(prob, x, w, b, d_y, d_scale, stream); break;
    case Precision::kFp16: run_ref<__half, false>(prob, x, w, b, d_y, d_scale, stream); break;
    case Precision::kBf16: run_ref<__nv_bfloat16, false>(prob, x, w, b, d_y, d_scale, stream); break;
  }
}

float round_to_tf32(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  if ((bits & 0x7f800000u) == 0x7f800000u) return v;  // inf/nan
  bits += 0x1000u;       // round to nearest, ties away from zero (cvt.rna)
  bits &= 0xffffe000u;   // keep 10 mantissa bits
  float r;
  std::memcpy(&r, &bits, 4);
  return r;
}

float quantize_to(Precision p, float v) {
  switch (p) {
    case Precision::kFp32: return v;
    case Precision::kTf32: return round_to_tf32(v);
    case Precision::kFp16: return __half2float(__float2half_rn(v));
    case Precision::kBf16: return __bfloat162float(__float2bfloat16_rn(v));
  }
  return v;
}

std::vector<float> download_as_float(const void* d_ptr, DataType dt, int64_t n) {
  std::vector<float> out(static_cast<size_t>(n));
  if (dt == DataType::kFloat32) {
    GIE_CUDA_CHECK(cudaMemcpy(out.data(), d_ptr, size_t(n) * 4, cudaMemcpyDeviceToHost));
  } else {
    std::vector<uint16_t> raw(static_cast<size_t>(n));
    GIE_CUDA_CHECK(cudaMemcpy(raw.data(), d_ptr, size_t(n) * 2, cudaMemcpyDeviceToHost));
    for (int64_t i = 0; i < n; ++i) {
      if (dt == DataType::kFloat16) {
        out[size_t(i)] = __half2float(__ushort_as_half(raw[size_t(i)]));
      } else {
        uint32_t bits = uint32_t(raw[size_t(i)]) << 16;  // bf16 is the high half of an fp32
        std::memcpy(&out[size_t(i)], &bits, 4);
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Error metrics and tolerance model (host)
// ---------------------------------------------------------------------------------------------

TolerancePolicy TolerancePolicy::for_problem(const ConvProblem& prob, bool intermediate_rounding) {
  TolerancePolicy t;
  t.k = prob.p.gemm_k();
  switch (prob.precision) {
    case Precision::kFp32:
      t.u_out = 5.9604644775390625e-08;  // 2^-24
      t.abs_floor = 1e-30;
      break;
    case Precision::kTf32:
      t.u_out = 5.9604644775390625e-08;
      t.abs_floor = 1e-30;
      t.c_in = 8.0;
      break;
    case Precision::kFp16:
      t.u_out = 4.8828125e-04;           // 2^-11
      t.abs_floor = 3.0e-8;              // half of the smallest fp16 subnormal (2^-24)
      break;
    case Precision::kBf16:
      t.u_out = 3.90625e-03;             // 2^-8
      t.abs_floor = 1e-30;
      break;
  }
  if (intermediate_rounding) t.c_inter = 1.0;  // unfused pipelines round the conv result before bias/ReLU
  return t;
}

double TolerancePolicy::tol(double ref, double scale) const {
  return c_acc * u_acc * std::sqrt(double(k)) * scale + c_out * u_out * std::fabs(ref) + c_inter * u_out * scale + c_in * u_in * scale + abs_floor;
}

std::string TolerancePolicy::describe() const {
  std::ostringstream os;
  os << "tol_i = " << c_acc << "*2^-24*sqrt(K=" << k << ")*scale_i + " << c_out << "*u_out(" << u_out
     << ")*|ref_i| + " << c_inter << "*u_out*scale_i + " << abs_floor;
  return os.str();
}

ErrorMetrics compare_outputs(const float* test, const float* ref, const float* scale, int64_t n,
                             const TolerancePolicy& tol) {
  ErrorMetrics m;
  m.n = n;
  double sum_abs = 0, sum_sq = 0, ref_sq = 0;
  for (int64_t i = 0; i < n; ++i) ref_sq += double(ref[i]) * double(ref[i]);
  const double rel_floor = std::max(1e-3 * std::sqrt(ref_sq / double(std::max<int64_t>(n, 1))), 1e-30);
  for (int64_t i = 0; i < n; ++i) {
    const double t = test[i];
    if (std::isnan(t)) { ++m.nan_count; continue; }
    if (std::isinf(t)) { ++m.inf_count; continue; }
    const double err = std::fabs(t - double(ref[i]));
    sum_abs += err;
    sum_sq += err * err;
    m.max_abs = std::max(m.max_abs, err);
    m.max_rel = std::max(m.max_rel, err / std::max(std::fabs(double(ref[i])), rel_floor));
    const double ti = tol.tol(ref[i], scale[i]);
    m.worst_tol_ratio = std::max(m.worst_tol_ratio, err / ti);
    if (err > ti) ++m.over_tol;
  }
  m.mean_abs = n ? sum_abs / double(n) : 0;
  m.rms = n ? std::sqrt(sum_sq / double(n)) : 0;
  m.frac_over_tol = n ? double(m.over_tol) / double(n) : 0;
  return m;
}

std::string ErrorMetrics::summary() const {
  std::ostringstream os;
  os << "max_abs=" << max_abs << " mean_abs=" << mean_abs << " rms=" << rms << " max_rel=" << max_rel
     << " over_tol=" << over_tol << "/" << n << " worst_tol_ratio=" << worst_tol_ratio << " nan=" << nan_count
     << " inf=" << inf_count;
  return os.str();
}

}  // namespace gie
