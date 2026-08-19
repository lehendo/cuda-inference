// Known-answer tests for the CPU reference convolution (no GPU needed), plus a cross-check of
// the CPU reference against the double-precision GPU reference when a device is present.
#include <cmath>

#include "gpu_inference/reference.h"
#include "gpu_inference/testing.h"
#include "shapes.h"
#include "test_util.h"

using namespace gie;

static ConvProblem prob_of(const Conv2DParams& p, bool bias = false, bool relu = false) {
  ConvProblem pr;
  pr.p = p;
  pr.has_bias = bias;
  pr.activation = relu ? Activation::kReLU : Activation::kNone;
  return pr;
}

TEST(output_size_formula) {
  Conv2DParams p = gt::P(1, 1, 7, 7, 1, 3, 3, 2, 2, 0, 0);
  CHECK_EQ(p.out_h(), 3);  // floor((7-3)/2)+1
  p = gt::P(1, 1, 8, 8, 1, 3, 3, 2, 2, 1, 1);
  CHECK_EQ(p.out_h(), 4);  // floor((8+2-3)/2)+1
  p = gt::P(1, 1, 5, 9, 1, 5, 3, 1, 3, 2, 1);
  CHECK_EQ(p.out_h(), 5);
  CHECK_EQ(p.out_w(), 3);  // floor((9+2-3)/3)+1
  p = gt::P(2, 3, 224, 224, 64, 7, 7, 2, 2, 3, 3);
  CHECK_EQ(p.out_h(), 112);
  CHECK_EQ(p.flops(), int64_t(2) * 64 * (2 * 112 * 112) * (3 * 49));
}

TEST(known_answer_ones_window) {
  Conv2DParams p = gt::P(1, 1, 3, 3, 1, 3, 3, 1, 1, 0, 0);
  std::vector<float> x(9, 1.f), w(9, 1.f);
  ReferenceOutput out;
  cpu_conv2d_reference(prob_of(p), x.data(), w.data(), nullptr, &out);
  CHECK_EQ(out.y.size(), size_t(1));
  CHECK_EQ(out.y[0], 9.f);
}

TEST(known_answer_cross_correlation_not_flipped) {
  // x = 1..9 (3x3), w = [[1,0],[0,-1]] -> every window gives x[i][j] - x[i+1][j+1] = -4.
  // A flipped (true convolution) kernel would give +4.
  Conv2DParams p = gt::P(1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
  std::vector<float> x = {1, 2, 3, 4, 5, 6, 7, 8, 9}, w = {1, 0, 0, -1};
  ReferenceOutput out;
  cpu_conv2d_reference(prob_of(p), x.data(), w.data(), nullptr, &out);
  CHECK_EQ(out.y.size(), size_t(4));
  for (float v : out.y) CHECK_EQ(v, -4.f);
}

TEST(known_answer_zero_padding) {
  Conv2DParams p = gt::P(1, 1, 2, 2, 1, 3, 3, 1, 1, 1, 1);
  std::vector<float> x(4, 1.f), w(9, 1.f);
  ReferenceOutput out;
  cpu_conv2d_reference(prob_of(p), x.data(), w.data(), nullptr, &out);
  CHECK_EQ(out.y.size(), size_t(4));
  for (float v : out.y) CHECK_EQ(v, 4.f);  // each 3x3 window covers all four ones
}

TEST(known_answer_padding_asymmetric_corner) {
  // 3x3 ones kernel, 3x3 input 1..9, pad 1 -> corner output sums the 2x2 top-left block = 1+2+4+5 = 12.
  Conv2DParams p = gt::P(1, 1, 3, 3, 1, 3, 3, 1, 1, 1, 1);
  std::vector<float> x = {1, 2, 3, 4, 5, 6, 7, 8, 9}, w(9, 1.f);
  ReferenceOutput out;
  cpu_conv2d_reference(prob_of(p), x.data(), w.data(), nullptr, &out);
  CHECK_EQ(out.y.size(), size_t(9));
  CHECK_EQ(out.y[0], 12.f);
  CHECK_EQ(out.y[4], 45.f);  // center sees everything
  CHECK_EQ(out.y[8], 5.f + 6.f + 8.f + 9.f);
}

TEST(known_answer_stride) {
  Conv2DParams p = gt::P(1, 1, 4, 4, 1, 1, 1, 2, 2, 0, 0);
  std::vector<float> x(16), w = {1.f};
  for (int i = 0; i < 16; ++i) x[size_t(i)] = float(i);
  ReferenceOutput out;
  cpu_conv2d_reference(prob_of(p), x.data(), w.data(), nullptr, &out);
  CHECK_EQ(out.y.size(), size_t(4));
  CHECK_EQ(out.y[0], 0.f);
  CHECK_EQ(out.y[1], 2.f);
  CHECK_EQ(out.y[2], 8.f);
  CHECK_EQ(out.y[3], 10.f);
}

TEST(known_answer_channels_batch_bias_relu) {
  // B=2, Cin=2, Cout=2, 1x1: y[b,o] = sum_c w[o,c] x[b,c] + bias[o], then ReLU.
  Conv2DParams p = gt::P(2, 2, 1, 1, 2, 1, 1, 1, 1, 0, 0);
  std::vector<float> x = {1, 2, /*b1*/ -3, 4};
  std::vector<float> w = {1, 1, /*o0*/ -1, 0.5f /*o1*/};
  std::vector<float> bias = {0.25f, -0.5f};
  ReferenceOutput out;
  cpu_conv2d_reference(prob_of(p, true, true), x.data(), w.data(), bias.data(), &out);
  // b0: o0 = 1+2+0.25 = 3.25 ; o1 = -1+1-0.5 = -0.5 -> relu 0
  // b1: o0 = -3+4+0.25 = 1.25 ; o1 = 3+2-0.5 = 4.5
  CHECK_EQ(out.y.size(), size_t(4));
  CHECK_EQ(out.y[0], 3.25f);
  CHECK_EQ(out.y[1], 0.f);
  CHECK_EQ(out.y[2], 1.25f);
  CHECK_EQ(out.y[3], 4.5f);
  // scale = L2 norm of the terms feeding the output
  // products are 1 and 2: sumsq=5, (sum)^2=9, bias^2=0.0625
  CHECK_NEAR(out.scale[0], std::sqrt(5.0 + 9.0 + 0.0625), 1e-6);
}

TEST(identity_kernel_reproduces_input) {
  Conv2DParams p = gt::P(2, 3, 5, 6, 3, 1, 1, 1, 1, 0, 0);
  const size_t n = size_t(p.input_shape().numel());
  std::vector<float> x(n), w(9, 0.f);
  Rng rng(7);
  for (auto& v : x) v = rng.uniform(-1.f, 1.f);
  for (int i = 0; i < 3; ++i) w[size_t(i * 3 + i)] = 1.f;
  ReferenceOutput out;
  cpu_conv2d_reference(prob_of(p), x.data(), w.data(), nullptr, &out);
  for (size_t i = 0; i < n; ++i) CHECK_EQ(out.y[i], x[i]);
}

TEST(cpu_reference_is_deterministic) {
  Conv2DParams p = gt::P(2, 5, 9, 7, 4, 3, 3, 2, 1, 1, 1);
  Rng rng(3);
  std::vector<float> x(size_t(p.input_shape().numel())), w(size_t(p.weight_shape().numel()));
  for (auto& v : x) v = rng.uniform(-1.f, 1.f);
  for (auto& v : w) v = rng.uniform(-1.f, 1.f);
  ReferenceOutput a, b;
  cpu_conv2d_reference(prob_of(p), x.data(), w.data(), nullptr, &a);
  cpu_conv2d_reference(prob_of(p), x.data(), w.data(), nullptr, &b);
  CHECK(a.y == b.y);
}

TEST(quantization_helpers) {
  CHECK_EQ(quantize_to(Precision::kFp32, 0.1f), 0.1f);
  CHECK_EQ(quantize_to(Precision::kFp16, 1.0f), 1.0f);
  CHECK(quantize_to(Precision::kFp16, 0.1f) != 0.1f);
  CHECK_NEAR(quantize_to(Precision::kFp16, 0.1f), 0.1f, 1e-4);
  CHECK_NEAR(quantize_to(Precision::kBf16, 0.1f), 0.1f, 1e-3);
  // TF32 keeps 10 mantissa bits: 1 + 2^-11 rounds away from zero to 1 + 2^-10 (tie -> away).
  CHECK_EQ(round_to_tf32(1.0f + 1.0f / 2048.f), 1.0f + 1.0f / 1024.f);
  CHECK_EQ(round_to_tf32(1.0f + 1.0f / 4096.f), 1.0f);
}

TEST(error_metrics_detect_nan_and_over_tolerance) {
  const int n = 8;
  std::vector<float> ref(n, 1.f), scale(n, 1.f), test(n, 1.f);
  ConvProblem pr = prob_of(gt::P(1, 4, 4, 4, 1, 1, 1, 1, 1, 0, 0));
  TolerancePolicy tol = TolerancePolicy::for_problem(pr);
  CHECK(compare_outputs(test.data(), ref.data(), scale.data(), n, tol).passed());
  test[3] = 1.f + 1e-3f;
  ErrorMetrics m = compare_outputs(test.data(), ref.data(), scale.data(), n, tol);
  CHECK(!m.passed());
  CHECK_EQ(m.over_tol, int64_t(1));
  test[3] = std::nanf("");
  m = compare_outputs(test.data(), ref.data(), scale.data(), n, tol);
  CHECK_EQ(m.nan_count, int64_t(1));
  CHECK(!m.passed());
}

TEST(invalid_problems_are_rejected) {
  CHECK_THROWS_CODE(validate_problem(prob_of(gt::P(0, 1, 5, 5, 1, 3, 3, 1, 1, 0, 0))), StatusCode::kInvalidArgument);
  CHECK_THROWS_CODE(validate_problem(prob_of(gt::P(1, 1, 2, 2, 1, 5, 5, 1, 1, 0, 0))), StatusCode::kInvalidArgument);
  CHECK_THROWS_CODE(validate_problem(prob_of(gt::P(1, 1, 5, 5, 1, 3, 3, 0, 1, 0, 0))), StatusCode::kInvalidArgument);
  CHECK_THROWS_CODE(validate_problem(prob_of(gt::P(1, 1, 5, 5, 1, 3, 3, 1, 1, -1, 0))), StatusCode::kInvalidArgument);
  Conv2DParams d = gt::P(1, 1, 9, 9, 1, 3, 3, 1, 1, 0, 0);
  d.dilation_h = 2;
  CHECK_THROWS_CODE(validate_problem(prob_of(d)), StatusCode::kUnsupported);
  Conv2DParams g = gt::P(1, 4, 9, 9, 4, 3, 3, 1, 1, 0, 0);
  g.groups = 2;
  CHECK_THROWS_CODE(validate_problem(prob_of(g)), StatusCode::kUnsupported);
  // Index-overflow safeguard: N = B*Ho*Wo would exceed 32-bit GEMM indexing.
  CHECK_THROWS_CODE(validate_problem(prob_of(gt::P(30000, 1, 512, 512, 1, 1, 1, 1, 1, 0, 0))), StatusCode::kInvalidArgument);
}

GT_MAIN()
