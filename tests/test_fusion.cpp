// Fused epilogues: conv, conv+bias, conv+bias+ReLU across every backend; fused vs unfused equality.
#include "conv_test_common.h"
#include "gpu_inference/fused_ops.h"
#include "gpu_inference/perf_model.h"

using namespace gie;

static const std::vector<Backend> kAll = {Backend::kDirect, Backend::kIm2colGemm, Backend::kSimtTiled,
                                          Backend::kImplicitGemm, Backend::kTensorCore};

TEST(conv_only) { gt::run_suite(kAll, {Precision::kFp32, Precision::kFp16}, 0); }
TEST(conv_bias) { gt::run_suite(kAll, {Precision::kFp32, Precision::kFp16}, 1); }
TEST(conv_bias_relu) { gt::run_suite(kAll, {Precision::kFp32, Precision::kFp16}, 2); }

TEST(relu_zeroes_negatives_and_bias_matters) {
  // Strongly negative bias => many pre-activations below zero: ReLU must clamp them to exactly 0.
  ConvProblem prob = gt::make_problem(gt::P(2, 16, 12, 12, 16, 3, 3, 1, 1, 1, 1), Precision::kFp32, true, true);
  gt::RefBundle b = gt::make_bundle(prob, 77, Distribution::kNormal);
  size_t zeros = 0, positives = 0;
  for (float v : b.ref) { (v == 0.f ? zeros : positives)++; CHECK(v >= 0.f); }
  CHECK(zeros > 0 && positives > 0);  // the test data exercises both sides of the ReLU
  for (Backend be : kAll) {
    const auto cands = Dispatcher::compatible_kernels(prob, DeviceInfo::current(), be);
    if (cands.empty()) continue;  // e.g. Tensor Core has no FP32 kernel
    const KernelDesc* k = cands.front();
    bool ran; std::string why;
    ErrorMetrics m = gt::run_kernel(*k, b, &ran, &why);
    CHECK_MSG(ran && m.passed(), k->name << " " << m.summary());
  }
}

// FP32: a fused epilogue and "conv, then separate bias and ReLU kernels" perform the same
// arithmetic in the same order, so the results must be bitwise identical.
TEST(fused_equals_unfused_fp32_bitwise) {
  const Conv2DParams p = gt::P(4, 32, 17, 19, 40, 3, 3, 1, 1, 1, 1);
  ConvProblem fused = gt::make_problem(p, Precision::kFp32, true, true);
  ConvProblem plain = gt::make_problem(p, Precision::kFp32, false, false);
  ConvData d = make_conv_data(fused, 31, Distribution::kNormal);
  const int64_t n = p.output_shape().numel();
  for (const char* name : {"implicit_gemm_fp32_128x128x8_t8x8", "implicit_gemm_fp32_64x64x8_t4x4",
                           "simt_tiled_fp32_tc32_t8x8_r4x2_cc8", "direct_fp32"}) {
    const KernelDesc* k = find_kernel(name);
    CHECK_MSG(k != nullptr, name);
    if (!k) continue;
    auto pf = ExecutionPlan::create(fused, *k);
    auto pp = ExecutionPlan::create(plain, *k);
    DeviceBuffer y_unfused(size_t(n) * 4);
    GIE_CUDA_CHECK(cudaMemset(d.dy.data(), 0xFF, d.dy.size()));
    pf->execute_unchecked(d.dx.data(), d.dw.data(), d.db.data(), d.dy.data(), nullptr, 0, nullptr);
    pp->execute_unchecked(d.dx.data(), d.dw.data(), nullptr, y_unfused.data(), nullptr, 0, nullptr);
    launch_bias_add(p, Precision::kFp32, d.db.data(), y_unfused.data(), nullptr);
    launch_relu(p, Precision::kFp32, y_unfused.data(), nullptr);
    GIE_CUDA_CHECK(cudaDeviceSynchronize());
    const auto a = download_as_float(d.dy.data(), DataType::kFloat32, n);
    const auto bb = download_as_float(y_unfused.data(), DataType::kFloat32, n);
    CHECK_MSG(a == bb, name << ": fused and unfused FP32 outputs differ");
  }
}

TEST(fused_vs_unfused_fp16_within_tolerance) {
  if (!DeviceInfo::current().supports_bf16_tensor_cores()) return;
  const Conv2DParams p = gt::P(2, 32, 14, 14, 48, 3, 3, 1, 1, 1, 1);
  ConvProblem fused = gt::make_problem(p, Precision::kFp16, true, true);
  gt::RefBundle b = gt::make_bundle(fused, 5, Distribution::kUniform);
  const KernelDesc* k = find_kernel("tensor_core_fp16_128x128x32_w2x4");
  CHECK(k != nullptr);
  bool ran; std::string why;
  ErrorMetrics m = gt::run_kernel(*k, b, &ran, &why);
  CHECK_MSG(ran && m.passed(), m.summary());
}

TEST(traffic_model_accounts_for_fusion) {
  ConvProblem prob = gt::make_problem(gt::P(8, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1), Precision::kFp16, true, true);
  ProblemModel m = analyze_problem(prob);
  // Each separately-launched elementwise op re-reads and re-writes Y once.
  CHECK_EQ(m.unfused_extra_traffic, 2.0 * m.output_bytes);
  CHECK_EQ(m.output_bytes, 8.0 * 64 * 56 * 56 * 2);
  CHECK(m.unrolled_blowup > 8.9 && m.unrolled_blowup < 9.1);  // 3x3, stride 1: ~9x the input
}

GT_MAIN()
