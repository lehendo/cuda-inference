// Custom Tensor Core (WMMA) convolution: FP16 and BF16, irregular dimensions, alignment paths.
#include "conv_test_common.h"

using namespace gie;

static bool device_has_tc() { return DeviceInfo::current().supports_bf16_tensor_cores(); }

TEST(tensor_core_fp16_bf16_all_configs) {
  if (!device_has_tc()) { std::cout << "  SKIP: needs sm_80+\n"; return; }
  gt::run_suite({Backend::kTensorCore}, {Precision::kFp16, Precision::kBf16}, -1);
}

TEST(tensor_core_unsupported_for_fp32) {
  ConvProblem prob = gt::make_problem(gt::P(1, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1), Precision::kFp32, false, false);
  const DeviceInfo& dev = DeviceInfo::current();
  CHECK(Dispatcher::compatible_kernels(prob, dev, Backend::kTensorCore).empty());
  std::vector<Rejection> rej;
  Dispatcher::compatible_kernels(prob, dev, Backend::kTensorCore, &rej);
  CHECK(!rej.empty());
  CHECK(rej.front().reason.find("fp16") != std::string::npos || rej.front().reason.find("bf16") != std::string::npos);
}

TEST(tensor_core_vector_and_scalar_weight_paths) {
  // K % 8 == 0 takes the 16-byte weight-load path; K % 8 != 0 takes the predicated scalar path.
  if (!device_has_tc()) return;
  for (int cin : {8, 16, 3, 5, 7}) {  // K = 72, 144 (vector path) vs 27, 45, 63 (scalar path)
    ConvProblem prob = gt::make_problem(gt::P(2, cin, 17, 19, 21, 3, 3, 1, 1, 1, 1), Precision::kFp16, true, false);
    gt::RefBundle b = gt::make_bundle(prob, 41 + uint64_t(cin), Distribution::kUniform);
    for (const KernelDesc* k : kernels_for_backend(Backend::kTensorCore)) {
      bool ran;
      std::string why;
      ErrorMetrics m = gt::run_kernel(*k, b, &ran, &why);
      if (ran) CHECK_MSG(m.passed(), k->name << " cin=" << cin << " " << m.summary());
    }
  }
}

TEST(tensor_core_misaligned_weight_pointer) {
  // A weight pointer offset by 2 bytes is not 16-byte aligned: the kernel must fall back to the
  // scalar path instead of issuing misaligned vector loads.
  if (!device_has_tc()) return;
  ConvProblem prob = gt::make_problem(gt::P(1, 16, 9, 9, 16, 3, 3, 1, 1, 1, 1), Precision::kFp16, false, false);
  gt::RefBundle b = gt::make_bundle(prob, 8, Distribution::kUniform);
  const int64_t wn = prob.p.weight_shape().numel();
  DeviceBuffer padded(size_t(wn + 8) * 2);
  GIE_CUDA_CHECK(cudaMemset(padded.data(), 0, padded.size()));
  void* shifted = static_cast<char*>(padded.data()) + 2;  // 2-byte offset
  GIE_CUDA_CHECK(cudaMemcpy(shifted, b.data.dw.data(), size_t(wn) * 2, cudaMemcpyDeviceToDevice));
  const KernelDesc* k = kernels_for_backend(Backend::kTensorCore).front();
  auto plan = ExecutionPlan::create(prob, *k);
  GIE_CUDA_CHECK(cudaMemset(b.data.dy.data(), 0xFF, b.data.dy.size()));
  plan->execute_unchecked(b.data.dx.data(), shifted, nullptr, b.data.dy.data(), nullptr, 0, nullptr);
  GIE_CUDA_CHECK(cudaDeviceSynchronize());
  const auto out = download_as_float(b.data.dy.data(), prob.dtype(), prob.p.output_shape().numel());
  ErrorMetrics m = compare_outputs(out.data(), b.ref.data(), b.scale.data(), int64_t(out.size()), TolerancePolicy::for_problem(prob));
  CHECK_MSG(m.passed(), m.summary());
}

TEST(tensor_core_large_reduction_k4608) {
  // Longest reduction in the suite (512 channels, 3x3): checks accumulation-error behaviour at scale.
  if (!device_has_tc()) return;
  ConvProblem prob = gt::make_problem(gt::P(2, 512, 9, 9, 64, 3, 3, 1, 1, 1, 1), Precision::kFp16, true, true);
  gt::RefBundle b = gt::make_bundle(prob, 123, Distribution::kPositive);
  for (const KernelDesc* k : kernels_for_backend(Backend::kTensorCore)) {
    bool ran;
    std::string why;
    ErrorMetrics m = gt::run_kernel(*k, b, &ran, &why);
    if (ran) CHECK_MSG(m.passed(), k->name << " " << m.summary());
  }
}

TEST(tcmma_family_runs_on_every_eligible_shape) {
  // mma.sync/ldmatrix/cp.async family: needs C_in % 8 == 0. Every eligible shape must be exercised by at least one
  // tcmma kernel per precision (all tile shapes and split-K variants that fit the device are verified).
  if (!device_has_tc()) return;
  int verified = 0, shapes = 0;
  for (const gt::Shape& sh : gt::correctness_shapes()) {
    if (sh.p.in_channels % 8 != 0) continue;
    for (Precision prec : {Precision::kFp16, Precision::kBf16}) {
      ++shapes;
      ConvProblem prob = gt::make_problem(sh.p, prec, shapes % 2 == 0, shapes % 3 == 0);
      gt::RefBundle b = gt::make_bundle(prob, 500 + uint64_t(shapes), gt::kDists[shapes % 4]);
      int ran_here = 0;
      for (const KernelDesc* k : kernels_for_backend(Backend::kTensorCore)) {
        if (k->name.rfind("tcmma_", 0) != 0) continue;
        bool ran;
        std::string why;
        ErrorMetrics m = gt::run_kernel(*k, b, &ran, &why);
        if (!ran) continue;
        ++ran_here;
        ++verified;
        CHECK_MSG(m.passed(), k->name << " on " << sh.name << " " << prob.key() << " -> " << m.summary());
      }
      CHECK_MSG(ran_here > 0, "no tcmma kernel could run " << sh.name << " " << to_string(prec));
    }
  }
  std::cout << "  tcmma: " << verified << " kernel runs verified over " << shapes << " (shape, precision) cells\n";
}

GT_MAIN()
