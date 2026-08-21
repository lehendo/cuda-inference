// Custom implicit-GEMM (CUDA cores): all tile configurations, FP32 and FP16.
#include "conv_test_common.h"

using namespace gie;

TEST(implicit_gemm_fp32_fp16_all_configs) {
  gt::run_suite({Backend::kImplicitGemm}, {Precision::kFp32, Precision::kFp16}, -1);
}

TEST(implicit_gemm_needs_no_workspace) {
  // The defining property: no materialized im2col matrix.
  ConvProblem prob = gt::make_problem(gt::P(8, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1), Precision::kFp32, false, false);
  for (const KernelDesc* k : kernels_for_backend(Backend::kImplicitGemm)) CHECK_EQ(k->workspace_bytes(prob), size_t(0));
  const KernelDesc* col = find_kernel("im2col_gemm_fp32");
  CHECK(col != nullptr);
  CHECK(col->workspace_bytes(prob) >= size_t(8) * 64 * 9 * 56 * 56 * 4);  // B*K*Ho*Wo floats
}

TEST(implicit_gemm_all_distributions_one_shape) {
  ConvProblem prob = gt::make_problem(gt::P(2, 24, 19, 21, 40, 3, 3, 2, 2, 1, 1), Precision::kFp32, true, true);
  for (Distribution d : gt::kDists) {
    gt::RefBundle b = gt::make_bundle(prob, 17, d);
    for (const KernelDesc* k : kernels_for_backend(Backend::kImplicitGemm)) {
      bool ran;
      std::string why;
      ErrorMetrics m = gt::run_kernel(*k, b, &ran, &why);
      if (ran) CHECK_MSG(m.passed(), k->name << " " << to_string(d) << " " << m.summary());
    }
  }
}

GT_MAIN()
