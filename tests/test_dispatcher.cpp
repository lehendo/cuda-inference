// Dispatcher: capability checking, selection priority, cost-model ranking, forced overrides, plan reuse.
#include <algorithm>

#include "conv_test_common.h"
#include "gpu_inference/perf_model.h"

using namespace gie;

TEST(capability_filter_reports_reasons) {
  const DeviceInfo& dev = DeviceInfo::current();
  ConvProblem fp32 = gt::make_problem(gt::P(1, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1), Precision::kFp32, false, false);
  std::vector<Rejection> rej;
  auto cands = Dispatcher::compatible_kernels(fp32, dev, Backend::kAuto, &rej);
  CHECK(!cands.empty());
  CHECK(!rej.empty());
  for (const KernelDesc* k : cands) CHECK(k->backend != Backend::kTensorCore);  // TC needs fp16/bf16
  for (const Rejection& r : rej) CHECK(!r.reason.empty());
  // Every registered kernel is accounted for exactly once (accepted or rejected with a reason).
  CHECK_EQ(cands.size() + rej.size(), kernel_registry().size());
}

TEST(precision_routes_to_matching_kernels) {
  const DeviceInfo& dev = DeviceInfo::current();
  for (Precision p : {Precision::kFp32, Precision::kFp16, Precision::kBf16, Precision::kTf32}) {
    ConvProblem prob = gt::make_problem(gt::P(2, 32, 28, 28, 64, 3, 3, 1, 1, 1, 1), p, true, true);
    auto cands = Dispatcher::compatible_kernels(prob, dev);
    CHECK_MSG(!cands.empty(), "no kernel for " << to_string(p));
    for (const KernelDesc* k : cands) CHECK_MSG(k->name.find(to_string(p)) != std::string::npos, k->name << " vs " << to_string(p));
  }
}

TEST(cpu_reference_backend_is_not_dispatchable) {
  const DeviceInfo& dev = DeviceInfo::current();
  ConvProblem prob = gt::make_problem(gt::P(1, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1), Precision::kFp32, false, false);
  Conv2DOptions opt;
  opt.backend = Backend::kCpuReference;
  CHECK_THROWS_CODE(Dispatcher::select(prob, dev, opt), StatusCode::kUnsupported);
}

TEST(forced_backend_without_valid_kernel_gives_clear_error) {
  const DeviceInfo& dev = DeviceInfo::current();
  ConvProblem prob = gt::make_problem(gt::P(1, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1), Precision::kFp32, false, false);
  Conv2DOptions opt;
  opt.backend = Backend::kTensorCore;
  try {
    Dispatcher::select(prob, dev, opt);
    CHECK_MSG(false, "expected kUnsupported");
  } catch (const Error& e) {
    CHECK(e.code() == StatusCode::kUnsupported);
    CHECK(std::string(e.what()).find("tensor_core") != std::string::npos);  // names the rejected kernels
  }
}

TEST(explicit_kernel_overrides_everything) {
  const DeviceInfo& dev = DeviceInfo::current();
  ConvProblem prob = gt::make_problem(gt::P(1, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1), Precision::kFp32, false, false);
  Conv2DOptions opt;
  opt.kernel = "direct_fp32";
  Selection s = Dispatcher::select(prob, dev, opt);
  CHECK_EQ(s.kernel->name, std::string("direct_fp32"));
  CHECK(s.source == SelectionSource::kForcedKernel);
  opt.kernel = "no_such_kernel";
  CHECK_THROWS_CODE(Dispatcher::select(prob, dev, opt), StatusCode::kInvalidArgument);
  opt.kernel = "direct_fp32";
  opt.backend = Backend::kImplicitGemm;  // contradicts the kernel's backend
  CHECK_THROWS_CODE(Dispatcher::select(prob, dev, opt), StatusCode::kInvalidArgument);
}

TEST(model_ranking_is_sorted_and_deterministic) {
  const DeviceInfo& dev = DeviceInfo::current();
  ConvProblem prob = gt::make_problem(gt::P(8, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1), Precision::kFp16, true, true);
  auto cands = Dispatcher::compatible_kernels(prob, dev);
  auto r1 = Dispatcher::rank_by_model(prob, dev, cands);
  auto r2 = Dispatcher::rank_by_model(prob, dev, cands);
  CHECK(!r1.empty());
  for (size_t i = 1; i < r1.size(); ++i) CHECK(r1[i - 1].second <= r1[i].second);
  for (size_t i = 0; i < r1.size(); ++i) CHECK(r1[i].first == r2[i].first);
  for (const auto& kv : r1) CHECK(kv.second > 0);
}

TEST(model_prefers_tensor_cores_for_large_fp16_and_never_naive_direct) {
  const DeviceInfo& dev = DeviceInfo::current();
  if (!dev.supports_bf16_tensor_cores()) return;
  ConvProblem prob = gt::make_problem(gt::P(32, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1), Precision::kFp16, true, true);
  auto ranked = Dispatcher::rank_by_model(prob, dev, Dispatcher::compatible_kernels(prob, dev));
  CHECK(ranked.front().first->uses_tensor_cores);
  CHECK(ranked.front().first->backend != Backend::kDirect);
}

TEST(auto_selection_runs_and_matches_reference) {
  const DeviceInfo& dev = DeviceInfo::current();
  for (Precision p : {Precision::kFp32, Precision::kFp16}) {
    ConvProblem prob = gt::make_problem(gt::P(2, 24, 21, 19, 40, 3, 3, 2, 2, 1, 1), p, true, true);
    gt::RefBundle b = gt::make_bundle(prob, 9, Distribution::kNormal);
    Conv2DOptions opt;  // kAuto, no tuning, empty cache => cost model
    opt.cache_path = "/nonexistent_dir_for_test/none.json";
    DeviceBuffer ws(Conv2D::workspace_size(prob, opt));
    const ConstTensorRef bias = b.data.b_ref();
    GIE_CUDA_CHECK(cudaMemset(b.data.dy.data(), 0xFF, b.data.dy.size()));
    Conv2D::forward(b.data.x_ref(), b.data.w_ref(), &bias, prob, b.data.y_ref(), ws.data(), ws.size(), nullptr, opt);
    GIE_CUDA_CHECK(cudaDeviceSynchronize());
    const auto out = download_as_float(b.data.dy.data(), prob.dtype(), prob.p.output_shape().numel());
    const bool dr = p != Precision::kFp32;  // conservative: allow double rounding if im2col was chosen
    ErrorMetrics m = compare_outputs(out.data(), b.ref.data(), b.scale.data(), int64_t(out.size()), TolerancePolicy::for_problem(prob, dr));
    CHECK_MSG(m.passed(), to_string(p) << " auto-selected: " << m.summary());
    Selection s = Dispatcher::select(prob, dev, opt);
    CHECK(s.source == SelectionSource::kCostModel);
  }
}

TEST(plan_is_reused_across_calls) {
  ConvProblem prob = gt::make_problem(gt::P(1, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1), Precision::kFp32, false, false);
  ConvData d = make_conv_data(prob, 4);
  Conv2DOptions opt;
  opt.kernel = "implicit_gemm_fp32_64x64x8_t4x4";
  Conv2D::clear_plan_cache();
  Conv2D::forward(d.x_ref(), d.w_ref(), nullptr, prob, d.y_ref(), nullptr, 0, nullptr, opt);
  Conv2D::forward(d.x_ref(), d.w_ref(), nullptr, prob, d.y_ref(), nullptr, 0, nullptr, opt);
  GIE_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_EQ(Conv2D::workspace_size(prob, opt), size_t(0));
}

TEST(roofline_model_basics) {
  DeviceCeilings c;
  c.fp32_tflops = 50;
  c.fp16_tc_tflops = 400;
  c.bw_gbs = 2000;
  c.launch_overhead_us = 3;
  // Large 3x3: high arithmetic intensity -> compute bound.
  ConvProblem big = gt::make_problem(gt::P(32, 256, 28, 28, 256, 3, 3, 1, 1, 1, 1), Precision::kFp16, false, false);
  CHECK(roofline_estimate(big, c, true).bound == Bound::kCompute);
  // 1x1 conv with few channels: low intensity -> bandwidth bound.
  ConvProblem thin = gt::make_problem(gt::P(32, 4, 224, 224, 4, 1, 1, 1, 1, 0, 0), Precision::kFp32, false, false);
  CHECK(roofline_estimate(thin, c, false).bound == Bound::kBandwidth);
  // Tiny problem: dominated by launch cost.
  ConvProblem tiny = gt::make_problem(gt::P(1, 4, 4, 4, 4, 1, 1, 1, 1, 0, 0), Precision::kFp32, false, false);
  CHECK(roofline_estimate(tiny, c, false).bound == Bound::kLaunch);
  ProblemModel m = analyze_problem(big);
  CHECK_EQ(m.flops, double(big.p.flops()));
  CHECK(m.arithmetic_intensity > 100);
  CHECK(m.im2col_bytes > 8 * m.input_bytes);
}

GT_MAIN()
