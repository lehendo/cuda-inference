// Shared machinery: run one kernel against the double-precision GPU reference.
#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "gpu_inference/dispatcher.h"
#include "gpu_inference/execution_plan.h"
#include "gpu_inference/reference.h"
#include "gpu_inference/testing.h"
#include "shapes.h"
#include "test_util.h"

namespace gt {

struct RefBundle {
  gie::ConvData data;
  std::vector<float> ref, scale;
};

inline RefBundle make_bundle(const gie::ConvProblem& prob, uint64_t seed, gie::Distribution dist) {
  RefBundle b;
  b.data = gie::make_conv_data(prob, seed, dist);
  const int64_t n = prob.p.output_shape().numel();
  gie::DeviceBuffer dref(size_t(n) * 4), dsc(size_t(n) * 4);
  gie::gpu_conv2d_reference(prob, b.data.dx.data(), b.data.dw.data(), b.data.db.data(), static_cast<float*>(dref.data()),
                            static_cast<float*>(dsc.data()), nullptr);
  GIE_CUDA_CHECK(cudaDeviceSynchronize());
  b.ref.resize(size_t(n));
  b.scale.resize(size_t(n));
  GIE_CUDA_CHECK(cudaMemcpy(b.ref.data(), dref.data(), size_t(n) * 4, cudaMemcpyDeviceToHost));
  GIE_CUDA_CHECK(cudaMemcpy(b.scale.data(), dsc.data(), size_t(n) * 4, cudaMemcpyDeviceToHost));
  return b;
}

// Runs `k` once on the bundle's data and compares against the reference. Returns metrics;
// `ran` is false if the kernel was capability-rejected (reason in *why).
inline gie::ErrorMetrics run_kernel(const gie::KernelDesc& k, RefBundle& b, bool* ran, std::string* why,
                                    int repeats = 1) {
  const gie::ConvProblem& prob = b.data.prob;
  const gie::DeviceInfo& dev = gie::DeviceInfo::current();
  std::string reason;
  if (!k.supports(prob, dev, &reason)) {
    *ran = false;
    if (why) *why = reason;
    return {};
  }
  *ran = true;
  auto plan = gie::ExecutionPlan::create(prob, k);
  gie::DeviceBuffer ws(plan->workspace_bytes());
  // Poison: fill output with NaN pattern so unwritten elements fail the comparison.
  GIE_CUDA_CHECK(cudaMemset(b.data.dy.data(), 0xFF, b.data.dy.size()));
  const gie::ConstTensorRef bias = b.data.b_ref();
  for (int r = 0; r < repeats; ++r)
    plan->execute(b.data.x_ref(), b.data.w_ref(), prob.has_bias ? &bias : nullptr, b.data.y_ref(), ws.data(), ws.size(), nullptr);
  GIE_CUDA_CHECK(cudaDeviceSynchronize());
  const int64_t n = prob.p.output_shape().numel();
  const std::vector<float> out = gie::download_as_float(b.data.dy.data(), prob.dtype(), n);
  const bool double_round = k.backend == gie::Backend::kIm2colGemm && prob.precision != gie::Precision::kFp32 &&
                            (prob.has_bias || prob.activation != gie::Activation::kNone);
  return gie::compare_outputs(out.data(), b.ref.data(), b.scale.data(), n, gie::TolerancePolicy::for_problem(prob, double_round));
}

inline gie::ConvProblem make_problem(const gie::Conv2DParams& p, gie::Precision prec, bool bias, bool relu) {
  gie::ConvProblem prob;
  prob.p = p;
  prob.precision = prec;
  prob.has_bias = bias;
  prob.activation = relu ? gie::Activation::kReLU : gie::Activation::kNone;
  return prob;
}

inline const gie::Distribution kDists[] = {gie::Distribution::kUniform, gie::Distribution::kNormal,
                                           gie::Distribution::kSparseSigned, gie::Distribution::kPositive};

// Runs every registered kernel of `backends` over the correctness shape suite for each precision in
// `precisions`. Fails if any kernel produces a wrong result, and also if a (shape, backend,
// precision) cell had *no* kernel able to run it (so a suite can't pass by skipping everything).
inline void run_suite(const std::vector<gie::Backend>& backends, const std::vector<gie::Precision>& precisions,
                      int epilogue_mode /*-1: cycle none/bias/bias+relu per shape, else fixed 0..2*/) {
  const gie::DeviceInfo& dev = gie::DeviceInfo::current();
  int cell = 0;
  int total_runs = 0, skipped = 0;
  const bool reduced = std::getenv("GIE_TEST_REDUCED") != nullptr;  // compute-sanitizer runs: small shapes only
  for (const Shape& sh : correctness_shapes()) {
    if (reduced && sh.p.flops() > 30'000'000) continue;
    for (gie::Precision prec : precisions) {
      const int mode = epilogue_mode >= 0 ? epilogue_mode : (cell % 3);
      ++cell;
      const gie::ConvProblem prob = make_problem(sh.p, prec, mode >= 1, mode == 2);
      const gie::Distribution dist = kDists[cell % 4];
      RefBundle b = make_bundle(prob, 0xC0FFEEULL + uint64_t(cell), dist);
      for (gie::Backend be : backends) {
        // GIE_TEST_SKIP_CUBLAS=1: exclude the cuBLAS-backed im2col backend (used to run `compute-sanitizer
        // --tool synccheck` on Hopper, which aborts a cuBLAS nvJet kernel and would mask results for our kernels).
        if (be == gie::Backend::kIm2colGemm && std::getenv("GIE_TEST_SKIP_CUBLAS")) continue;
        int ran_in_backend = 0;
        for (const gie::KernelDesc* k : gie::kernels_for_backend(be)) {
          bool ran = false;
          std::string why;
          const gie::ErrorMetrics m = run_kernel(*k, b, &ran, &why);
          if (!ran) { ++skipped; continue; }
          ++ran_in_backend;
          ++total_runs;
          CHECK_MSG(m.passed(), k->name << " on " << sh.name << " " << prob.key() << " dist=" << gie::to_string(dist)
                                        << " -> " << m.summary());
        }
        // Only demand coverage for (backend, precision) pairs that exist in the registry at all.
        bool exists = false;
        for (const gie::KernelDesc* k : gie::kernels_for_backend(be)) {
          std::string w;
          // precision existence: a kernel for this precision exists if it supports a trivially valid problem
          gie::ConvProblem probe = make_problem(P(1, 4, 8, 8, 4, 1, 1, 1, 1, 0, 0), prec, false, false);
          if (k->supports(probe, dev, &w)) { exists = true; break; }
        }
        if (exists) CHECK_MSG(ran_in_backend > 0, "no " << gie::to_string(be) << " kernel could run " << sh.name << " " << prob.key());
      }
    }
  }
  std::cout << "  suite: " << total_runs << " kernel runs verified, " << skipped << " capability-rejected\n";
  // with GIE_TEST_SKIP_CUBLAS the only backend of the im2col suite is intentionally excluded, so zero runs is expected there
  CHECK(total_runs > 0 || std::getenv("GIE_TEST_SKIP_CUBLAS") != nullptr);
}

}  // namespace gt

namespace gt {

// Compares a device output (in the problem's dtype) with a bundle's double-precision reference.
inline gie::ErrorMetrics compare_device_output(const void* y_dev, const RefBundle& b, bool double_round = false) {
  const int64_t n = b.data.prob.p.output_shape().numel();
  const std::vector<float> out = gie::download_as_float(y_dev, b.data.prob.dtype(), n);
  return gie::compare_outputs(out.data(), b.ref.data(), b.scale.data(), n, gie::TolerancePolicy::for_problem(b.data.prob, double_round));
}

// Copies bundle `src`'s input tensors into bundle `dst`'s device buffers asynchronously on `stream`
// (same shapes), so the same plan/graph sees new data while `src`'s reference is the expected output.
inline void copy_inputs_async(RefBundle& dst, const RefBundle& src, cudaStream_t stream) {
  GIE_CUDA_CHECK(cudaMemcpyAsync(dst.data.dx.data(), src.data.dx.data(), src.data.dx.size(), cudaMemcpyDeviceToDevice, stream));
  GIE_CUDA_CHECK(cudaMemcpyAsync(dst.data.dw.data(), src.data.dw.data(), src.data.dw.size(), cudaMemcpyDeviceToDevice, stream));
  GIE_CUDA_CHECK(cudaMemcpyAsync(dst.data.db.data(), src.data.db.data(), src.data.db.size(), cudaMemcpyDeviceToDevice, stream));
}

}  // namespace gt
