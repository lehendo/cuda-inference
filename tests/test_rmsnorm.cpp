// RMSNorm extension: correctness vs a double-precision host reference for irregular hidden sizes.
#include <cmath>

#include "gpu_inference/reference.h"
#include "gpu_inference/rmsnorm.h"
#include "gpu_inference/testing.h"
#include "test_util.h"

using namespace gie;

static Precision prec_of(DataType d) { return d == DataType::kFloat32 ? Precision::kFp32 : (d == DataType::kFloat16 ? Precision::kFp16 : Precision::kBf16); }

static void run_case(int rows, int hidden, DataType dt, bool fused, bool unfused_impl) {
  Rng rng(uint64_t(rows) * 1315423911u + uint64_t(hidden));
  const Precision pr = prec_of(dt);
  const size_t n = size_t(rows) * hidden;
  std::vector<float> x(n), r(n), g(static_cast<size_t>(hidden));
  for (auto& v : x) v = quantize_to(pr, rng.normal() * 3.f);
  for (auto& v : r) v = quantize_to(pr, rng.normal());
  for (auto& v : g) v = quantize_to(pr, rng.uniform(0.5f, 1.5f));
  DeviceBuffer dx = upload_as(x, dt), dr = upload_as(r, dt), dg = upload_as(g, dt), dy(n * dtype_size(dt)), dres(n * dtype_size(dt)), scratch(size_t(rows) * 4);
  RmsNormParams p;
  p.rows = rows; p.hidden = hidden; p.dtype = dt;
  GIE_CUDA_CHECK(cudaMemset(dy.data(), 0xFF, dy.size()));
  if (unfused_impl) rmsnorm_forward_unfused(p, dx.data(), dg.data(), dy.data(), fused ? dr.data() : nullptr, fused ? dres.data() : nullptr, scratch.data(), nullptr);
  else rmsnorm_forward(p, dx.data(), dg.data(), dy.data(), fused ? dr.data() : nullptr, fused ? dres.data() : nullptr, nullptr);
  GIE_CUDA_CHECK(cudaDeviceSynchronize());
  const auto out = download_as_float(dy.data(), dt, int64_t(n));
  const auto res_out = fused ? download_as_float(dres.data(), dt, int64_t(n)) : std::vector<float>();
  const double u_out = dt == DataType::kFloat32 ? 6e-8 : (dt == DataType::kFloat16 ? 4.9e-4 : 3.9e-3);
  for (int row = 0; row < rows; ++row) {
    double ss = 0;
    std::vector<double> s(static_cast<size_t>(hidden));
    for (int i = 0; i < hidden; ++i) {
      double v = x[size_t(row) * hidden + i];
      if (fused) v = quantize_to(pr, float(v + r[size_t(row) * hidden + i]));  // residual stream stored in dtype
      s[size_t(i)] = v;
      ss += v * v;
    }
    const double inv = 1.0 / std::sqrt(ss / hidden + 1e-6);
    for (int i = 0; i < hidden; ++i) {
      const double ref = s[size_t(i)] * inv * g[size_t(i)];
      // FP32: rsqrtf (2 ulp) + two multiplies => a few ulp; 16 u is a bound with margin. 16-bit: output rounding dominates.
      const double tol = (dt == DataType::kFloat32 ? 16.0 : 3.0) * u_out * std::fabs(ref) + 1e-30 + (dt == DataType::kFloat16 ? 1e-7 : 0);
      CHECK_MSG(std::fabs(out[size_t(row) * hidden + i] - ref) <= tol, "rows=" << rows << " hidden=" << hidden << " dt=" << to_string(dt)
                                                                     << " fused=" << fused << " got " << out[size_t(row) * hidden + i] << " ref " << ref);
      if (fused) CHECK(res_out[size_t(row) * hidden + i] == float(s[size_t(i)]));
    }
  }
}

TEST(rmsnorm_all_dtypes_irregular_hidden) {
  for (DataType dt : {DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16})
    for (int hidden : {1, 7, 33, 64, 127, 1000, 1024, 4096, 5120, 8191})
      for (bool fused : {false, true}) run_case(5, hidden, dt, fused, false);
}

TEST(rmsnorm_unfused_baseline_is_correct) {
  for (DataType dt : {DataType::kFloat32, DataType::kFloat16})
    for (int hidden : {33, 4096}) for (bool fused : {false, true}) run_case(4, hidden, dt, fused, true);
}

TEST(rmsnorm_misaligned_pointers_use_scalar_path) {
  const int rows = 3, hidden = 64;
  std::vector<float> x(static_cast<size_t>(rows) * hidden + 4, 1.f), g(static_cast<size_t>(hidden), 1.f);
  DeviceBuffer dx = upload_as(x, DataType::kFloat32), dg = upload_as(g, DataType::kFloat32), dy(size_t(rows) * hidden * 4);
  RmsNormParams p;
  p.rows = rows; p.hidden = hidden;
  rmsnorm_forward(p, static_cast<const char*>(dx.data()) + 4, dg.data(), dy.data(), nullptr, nullptr, nullptr);  // 4-byte offset
  GIE_CUDA_CHECK(cudaDeviceSynchronize());
  const auto out = download_as_float(dy.data(), DataType::kFloat32, int64_t(rows) * hidden);
  for (float v : out) CHECK_NEAR(v, 1.0f, 1e-3);
}

TEST(rmsnorm_rejects_bad_arguments) {
  DeviceBuffer a(64), b(64), c(64);
  RmsNormParams p;
  p.rows = 1; p.hidden = 4;
  CHECK_THROWS_CODE(rmsnorm_forward(p, a.data(), b.data(), a.data(), nullptr, nullptr, nullptr), StatusCode::kInvalidArgument);
  CHECK_THROWS_CODE(rmsnorm_forward(p, a.data(), b.data(), c.data(), a.data(), nullptr, nullptr), StatusCode::kInvalidArgument);
  p.hidden = 0;
  CHECK_THROWS_CODE(rmsnorm_forward(p, a.data(), b.data(), c.data(), nullptr, nullptr, nullptr), StatusCode::kInvalidArgument);
}

GT_MAIN()
