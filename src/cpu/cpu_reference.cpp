// Backend A: CPU reference convolution (nested loops, double accumulation).
// Correctness oracle only -- no attempt is made to make it fast.
#include <cmath>

#include "gpu_inference/reference.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gie {

void cpu_conv2d_reference(const ConvProblem& prob, const float* x, const float* w, const float* bias,
                          ReferenceOutput* out) {
  validate_problem(prob);
  const Conv2DParams& p = prob.p;
  const int OH = p.out_h(), OW = p.out_w();
  const int64_t total = int64_t(p.batch) * p.out_channels * OH * OW;
  out->y.assign(size_t(total), 0.f);
  out->scale.assign(size_t(total), 0.f);
  const bool relu = prob.activation == Activation::kReLU;

#pragma omp parallel for schedule(static)
  for (int64_t idx = 0; idx < total; ++idx) {  // each output is computed independently => deterministic
    const int ow = int(idx % OW);
    const int oh = int((idx / OW) % OH);
    const int o = int((idx / (int64_t(OW) * OH)) % p.out_channels);
    const int b = int(idx / (int64_t(OW) * OH * p.out_channels));
    double acc = 0.0, sumsq = 0.0;
    for (int c = 0; c < p.in_channels; ++c) {
      for (int r = 0; r < p.kernel_h; ++r) {
        const int ih = oh * p.stride_h - p.pad_h + r * p.dilation_h;
        if (ih < 0 || ih >= p.in_h) continue;
        for (int s = 0; s < p.kernel_w; ++s) {
          const int iw = ow * p.stride_w - p.pad_w + s * p.dilation_w;
          if (iw < 0 || iw >= p.in_w) continue;
          const double xv = x[((int64_t(b) * p.in_channels + c) * p.in_h + ih) * p.in_w + iw];
          const double wv = w[((int64_t(o) * p.in_channels + c) * p.kernel_h + r) * p.kernel_w + s];
          acc += xv * wv;
          sumsq += xv * xv * wv * wv;
        }
      }
    }
    // scale: magnitude that governs accumulation rounding error = sqrt(sum of squared products +
    // (sum of products)^2 + bias^2): the first term models random-sign terms, the second the drift of
    // partial sums when terms share a sign.
    sumsq += acc * acc;
    if (prob.has_bias && bias != nullptr) {
      acc += double(bias[o]);
      sumsq += double(bias[o]) * double(bias[o]);
    }
    if (relu && acc < 0.0) acc = 0.0;
    out->y[size_t(idx)] = float(acc);
    out->scale[size_t(idx)] = float(std::sqrt(sumsq));
  }
}

}  // namespace gie
