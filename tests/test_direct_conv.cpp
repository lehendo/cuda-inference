// Direct (naive), explicit im2col+cuBLAS, and shared-memory tiled SIMT backends, FP32 and FP16.
#include "conv_test_common.h"

using namespace gie;

TEST(gpu_reference_matches_cpu_reference) {
  // The double-precision GPU reference is the oracle for large shapes, so it must agree with the
  // independent nested-loop CPU reference on every small shape.
  for (const gt::Shape& sh : gt::correctness_shapes()) {
    if (sh.p.flops() > 50'000'000) continue;
    for (Precision prec : {Precision::kFp32, Precision::kFp16}) {
      ConvProblem prob = gt::make_problem(sh.p, prec, true, true);
      gt::RefBundle b = gt::make_bundle(prob, 99, Distribution::kNormal);
      ReferenceOutput cpu;
      cpu_conv2d_reference(prob, b.data.hx.data(), b.data.hw.data(), b.data.hb.data(), &cpu);
      ErrorMetrics m = compare_outputs(b.ref.data(), cpu.y.data(), cpu.scale.data(), int64_t(cpu.y.size()),
                                       TolerancePolicy::for_problem(prob));
      CHECK_MSG(m.passed(), sh.name << " " << to_string(prec) << ": " << m.summary());
    }
  }
}

TEST(direct_fp32_fp16_bf16) {
  gt::run_suite({Backend::kDirect}, {Precision::kFp32, Precision::kFp16, Precision::kBf16}, -1);
}

TEST(im2col_gemm_all_precisions) {
  gt::run_suite({Backend::kIm2colGemm}, {Precision::kFp32, Precision::kFp16, Precision::kBf16}, -1);
}

TEST(simt_tiled_fp32_fp16) {
  gt::run_suite({Backend::kSimtTiled}, {Precision::kFp32, Precision::kFp16}, -1);
}

TEST(repeated_execution_is_stable) {
  // Same kernel launched repeatedly must give identical results (no stale shared/global state).
  ConvProblem prob = gt::make_problem(gt::P(2, 16, 14, 14, 32, 3, 3, 1, 1, 1, 1), Precision::kFp32, true, true);
  gt::RefBundle b = gt::make_bundle(prob, 5, Distribution::kUniform);
  for (Backend be : {Backend::kDirect, Backend::kIm2colGemm, Backend::kSimtTiled}) {
    if (be == Backend::kIm2colGemm && std::getenv("GIE_TEST_SKIP_CUBLAS")) continue;
    const KernelDesc* k = kernels_for_backend(be).front();
    bool ran;
    std::string why;
    ErrorMetrics m = gt::run_kernel(*k, b, &ran, &why, /*repeats=*/5);
    CHECK_MSG(ran && m.passed(), k->name << " " << m.summary());
  }
}

GT_MAIN()
