// Argument validation: shapes, dtypes, null pointers, aliasing, workspace size, bias consistency.
#include "conv_test_common.h"

using namespace gie;

namespace {
struct Fixture {
  ConvProblem prob = gt::make_problem(gt::P(2, 8, 10, 10, 8, 3, 3, 1, 1, 1, 1), Precision::kFp32, true, false);
  ConvData d = make_conv_data(prob, 1);
  const KernelDesc* k = find_kernel("im2col_gemm_fp32");
  std::shared_ptr<ExecutionPlan> plan = ExecutionPlan::create(prob, *k);
  DeviceBuffer ws{plan->workspace_bytes()};
};
}  // namespace

TEST(valid_call_succeeds) {
  Fixture f;
  const ConstTensorRef bias = f.d.b_ref();
  f.plan->execute(f.d.x_ref(), f.d.w_ref(), &bias, f.d.y_ref(), f.ws.data(), f.ws.size(), nullptr);
  GIE_CUDA_CHECK(cudaDeviceSynchronize());
}

TEST(rejects_null_pointers_and_bad_shapes) {
  Fixture f;
  const ConstTensorRef bias = f.d.b_ref();
  ConstTensorRef x = f.d.x_ref();
  x.data = nullptr;
  CHECK_THROWS_CODE(f.plan->execute(x, f.d.w_ref(), &bias, f.d.y_ref(), f.ws.data(), f.ws.size(), nullptr), StatusCode::kInvalidArgument);
  x = f.d.x_ref();
  x.shape.h += 1;
  CHECK_THROWS_CODE(f.plan->execute(x, f.d.w_ref(), &bias, f.d.y_ref(), f.ws.data(), f.ws.size(), nullptr), StatusCode::kInvalidArgument);
  ConstTensorRef w = f.d.w_ref();
  w.dtype = DataType::kFloat16;
  CHECK_THROWS_CODE(f.plan->execute(f.d.x_ref(), w, &bias, f.d.y_ref(), f.ws.data(), f.ws.size(), nullptr), StatusCode::kInvalidArgument);
  TensorRef y = f.d.y_ref();
  y.shape.c += 1;
  CHECK_THROWS_CODE(f.plan->execute(f.d.x_ref(), f.d.w_ref(), &bias, y, f.ws.data(), f.ws.size(), nullptr), StatusCode::kInvalidArgument);
}

TEST(rejects_bias_mismatch) {
  Fixture f;
  CHECK_THROWS_CODE(f.plan->execute(f.d.x_ref(), f.d.w_ref(), nullptr, f.d.y_ref(), f.ws.data(), f.ws.size(), nullptr), StatusCode::kInvalidArgument);
  ConvProblem nb = f.prob;
  nb.has_bias = false;
  auto plan = ExecutionPlan::create(nb, *f.k);
  const ConstTensorRef bias = f.d.b_ref();
  CHECK_THROWS_CODE(plan->execute(f.d.x_ref(), f.d.w_ref(), &bias, f.d.y_ref(), f.ws.data(), f.ws.size(), nullptr), StatusCode::kInvalidArgument);
  ConstTensorRef bad = f.d.b_ref();
  bad.shape.c = 3;
  CHECK_THROWS_CODE(f.plan->execute(f.d.x_ref(), f.d.w_ref(), &bad, f.d.y_ref(), f.ws.data(), f.ws.size(), nullptr), StatusCode::kInvalidArgument);
}

TEST(rejects_output_aliasing_inputs) {
  Fixture f;
  const ConstTensorRef bias = f.d.b_ref();
  // Output overlapping the input tensor (in-place) must be refused, including partial overlap.
  TensorRef y = f.d.y_ref();
  y.data = f.d.dx.data();
  CHECK_THROWS_CODE(f.plan->execute(f.d.x_ref(), f.d.w_ref(), &bias, y, f.ws.data(), f.ws.size(), nullptr), StatusCode::kInvalidArgument);
  DeviceBuffer big(size_t(f.prob.p.output_shape().numel()) * 4 + 4096);
  TensorRef y2 = f.d.y_ref();
  y2.data = static_cast<char*>(big.data());
  ConstTensorRef x2 = f.d.x_ref();
  x2.data = static_cast<char*>(big.data()) + 1024;  // input starts inside the output range
  CHECK_THROWS_CODE(f.plan->execute(x2, f.d.w_ref(), &bias, y2, f.ws.data(), f.ws.size(), nullptr), StatusCode::kInvalidArgument);
}

TEST(rejects_small_or_missing_workspace) {
  Fixture f;
  CHECK(f.plan->workspace_bytes() > 0);
  const ConstTensorRef bias = f.d.b_ref();
  CHECK_THROWS_CODE(f.plan->execute(f.d.x_ref(), f.d.w_ref(), &bias, f.d.y_ref(), nullptr, 0, nullptr), StatusCode::kWorkspaceTooSmall);
  CHECK_THROWS_CODE(f.plan->execute(f.d.x_ref(), f.d.w_ref(), &bias, f.d.y_ref(), f.ws.data(), f.ws.size() - 1, nullptr), StatusCode::kWorkspaceTooSmall);
}

TEST(plan_rejects_unsupported_kernel_precision_combo) {
  ConvProblem prob = gt::make_problem(gt::P(1, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1), Precision::kFp32, false, false);
  CHECK_THROWS_CODE(ExecutionPlan::create(prob, *find_kernel("tensor_core_fp16_128x128x32_w2x4")), StatusCode::kUnsupported);
}

TEST(forward_api_end_to_end_with_validation) {
  ConvProblem prob = gt::make_problem(gt::P(2, 8, 10, 10, 8, 3, 3, 1, 1, 1, 1), Precision::kFp32, true, true);
  gt::RefBundle b = gt::make_bundle(prob, 3, Distribution::kNormal);
  Conv2DOptions opt;
  opt.backend = Backend::kIm2colGemm;
  DeviceBuffer ws(Conv2D::workspace_size(prob, opt));
  const ConstTensorRef bias = b.data.b_ref();
  CHECK_THROWS_CODE(Conv2D::forward(b.data.x_ref(), b.data.w_ref(), &bias, prob, b.data.y_ref(), nullptr, 0, nullptr, opt),
                    StatusCode::kWorkspaceTooSmall);
  GIE_CUDA_CHECK(cudaMemset(b.data.dy.data(), 0xFF, b.data.dy.size()));
  Conv2D::forward(b.data.x_ref(), b.data.w_ref(), &bias, prob, b.data.y_ref(), ws.data(), ws.size(), nullptr, opt);
  GIE_CUDA_CHECK(cudaDeviceSynchronize());
  const auto out = download_as_float(b.data.dy.data(), prob.dtype(), prob.p.output_shape().numel());
  CHECK(compare_outputs(out.data(), b.ref.data(), b.scale.data(), int64_t(out.size()), TolerancePolicy::for_problem(prob)).passed());
}

GT_MAIN()
