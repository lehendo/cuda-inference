#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstring>

#include "gpu_inference/testing.h"
#include "gpu_inference/reference.h"

namespace gie {

uint64_t Rng::next_u64() {
  uint64_t z = (s_ += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
double Rng::uniform() { return double(next_u64() >> 11) * (1.0 / 9007199254740992.0); }
float Rng::uniform(float lo, float hi) { return lo + float(uniform()) * (hi - lo); }
float Rng::normal() {
  double s = 0;
  for (int i = 0; i < 12; ++i) s += uniform();
  return float(s - 6.0);
}

const char* to_string(Distribution d) {
  switch (d) {
    case Distribution::kUniform: return "uniform";
    case Distribution::kNormal: return "normal";
    case Distribution::kSparseSigned: return "sparse_signed";
    case Distribution::kPositive: return "positive";
  }
  return "?";
}

ConstTensorRef ConvData::x_ref() const { return {dx.data(), prob.dtype(), prob.p.input_shape()}; }
ConstTensorRef ConvData::w_ref() const { return {dw.data(), prob.dtype(), prob.p.weight_shape()}; }
ConstTensorRef ConvData::b_ref() const { return {db.data(), prob.dtype(), prob.p.bias_shape()}; }
TensorRef ConvData::y_ref() { return {dy.data(), prob.dtype(), prob.p.output_shape()}; }

DeviceBuffer upload_as(const std::vector<float>& v, DataType dt) {
  DeviceBuffer buf(v.size() * dtype_size(dt));
  if (dt == DataType::kFloat32) {
    GIE_CUDA_CHECK(cudaMemcpy(buf.data(), v.data(), v.size() * 4, cudaMemcpyHostToDevice));
  } else {
    std::vector<uint16_t> raw(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
      if (dt == DataType::kFloat16) {
        const __half h = __float2half_rn(v[i]);
        std::memcpy(&raw[i], &h, 2);
      } else {
        const __nv_bfloat16 b = __float2bfloat16_rn(v[i]);
        std::memcpy(&raw[i], &b, 2);
      }
    }
    GIE_CUDA_CHECK(cudaMemcpy(buf.data(), raw.data(), raw.size() * 2, cudaMemcpyHostToDevice));
  }
  return buf;
}

namespace {

float draw(Rng& rng, Distribution d) {
  switch (d) {
    case Distribution::kUniform: return rng.uniform(-1.f, 1.f);
    case Distribution::kNormal: return rng.normal();
    case Distribution::kSparseSigned: return rng.uniform() < 0.7 ? 0.f : rng.uniform(-2.f, 2.f);
    case Distribution::kPositive: return rng.uniform(0.f, 1.f);
  }
  return 0.f;
}

}  // namespace

ConvData make_conv_data(const ConvProblem& prob, uint64_t seed, Distribution dist) {
  validate_problem(prob);
  ConvData d;
  d.prob = prob;
  const Conv2DParams& p = prob.p;
  Rng rng(seed);
  std::vector<float> rx(size_t(p.input_shape().numel())), rw(size_t(p.weight_shape().numel())),
      rb(size_t(p.out_channels));
  for (auto& v : rx) v = draw(rng, dist);
  const float wscale = std::sqrt(3.0f / float(p.gemm_k()));
  for (auto& v : rw) v = draw(rng, dist) * wscale;
  for (auto& v : rb) v = rng.uniform(-1.f, 1.f);

  // Device tensors hold the storage type; TF32 keeps raw FP32 (the kernel rounds the multiplier inputs).
  const DataType dt = prob.dtype();
  const bool half_storage = dt != DataType::kFloat32;
  d.hx.resize(rx.size());
  d.hw.resize(rw.size());
  d.hb.resize(rb.size());
  for (size_t i = 0; i < rx.size(); ++i) d.hx[i] = quantize_to(prob.precision, rx[i]);
  for (size_t i = 0; i < rw.size(); ++i) d.hw[i] = quantize_to(prob.precision, rw[i]);
  for (size_t i = 0; i < rb.size(); ++i) d.hb[i] = half_storage ? quantize_to(prob.precision, rb[i]) : rb[i];

  d.dx = upload_as(prob.precision == Precision::kTf32 ? rx : d.hx, dt);
  d.dw = upload_as(prob.precision == Precision::kTf32 ? rw : d.hw, dt);
  d.db = upload_as(d.hb, dt);
  d.dy = DeviceBuffer(size_t(p.output_shape().numel()) * dtype_size(dt));
  GIE_CUDA_CHECK(cudaMemset(d.dy.data(), 0xFF, d.dy.size()));  // NaN pattern: unwritten outputs are caught
  GIE_CUDA_CHECK(cudaDeviceSynchronize());  // uploads/memset must be complete before use on any stream
  return d;
}

}  // namespace gie
