// Deterministic synthetic data and device-buffer helpers used by tests, the
// autotuner and the benchmark harness. No datasets or model weights are needed.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_inference/conv2d.h"
#include "gpu_inference/reference.h"

namespace gie {

// Platform-independent PRNG (SplitMix64) so seeds reproduce identical data everywhere.
class Rng {
 public:
  explicit Rng(uint64_t seed) : s_(seed) {}
  uint64_t next_u64();
  double uniform();                  // [0, 1)
  float uniform(float lo, float hi);
  float normal();                    // Irwin-Hall approximation (sum of 12 uniforms - 6)

 private:
  uint64_t s_;
};

enum class Distribution {
  kUniform,       // x ~ U(-1,1)
  kNormal,        // x ~ approx N(0,1)
  kSparseSigned,  // 70% zeros, rest U(-2,2): exercises ReLU and zero handling
  kPositive,      // x ~ U(0,1): all-positive accumulation (worst case for rounding growth)
};
const char* to_string(Distribution d);

struct ConvData {
  ConvProblem prob;
  // Reference-view values: what the reference consumes (quantized per the policy).
  std::vector<float> hx, hw, hb;
  // Device tensors in the problem's storage dtype (raw fp32 for TF32).
  DeviceBuffer dx, dw, db, dy;

  ConstTensorRef x_ref() const;
  ConstTensorRef w_ref() const;
  ConstTensorRef b_ref() const;
  TensorRef y_ref();
};

// Weights are scaled by sqrt(3/K) (He-style) so |y| stays O(1); bias ~ U(-1,1).
ConvData make_conv_data(const ConvProblem& prob, uint64_t seed, Distribution dist = Distribution::kUniform);

// Converts host floats to the device storage dtype and uploads.
DeviceBuffer upload_as(const std::vector<float>& v, DataType dt);

}  // namespace gie
