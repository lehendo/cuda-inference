#include "gpu_inference/timing.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "gpu_inference/error.h"

namespace gie {
namespace {

double percentile(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return 0;
  const double pos = q * double(sorted.size() - 1);
  const size_t lo = size_t(std::floor(pos)), hi = size_t(std::ceil(pos));
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - double(lo));
}

}  // namespace

TimingStats summarize(std::vector<double> v, int inner) {
  TimingStats t;
  t.inner_iters = inner;
  t.samples = int(v.size());
  t.raw_us = v;
  if (v.empty()) return t;
  std::sort(v.begin(), v.end());
  t.min_us = v.front();
  t.max_us = v.back();
  t.median_us = percentile(v, 0.5);
  t.p10_us = percentile(v, 0.10);
  t.p90_us = percentile(v, 0.90);
  t.p99_us = percentile(v, 0.99);
  t.iqr_us = percentile(v, 0.75) - percentile(v, 0.25);
  t.mean_us = std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
  double ss = 0;
  for (double x : v) ss += (x - t.mean_us) * (x - t.mean_us);
  t.stddev_us = v.size() > 1 ? std::sqrt(ss / double(v.size() - 1)) : 0;
  return t;
}

TimingStats time_stream_work(const std::function<void(cudaStream_t)>& launch, cudaStream_t stream,
                             const TimingOptions& opt) {
  cudaEvent_t start, stop;
  GIE_CUDA_CHECK(cudaEventCreate(&start));
  GIE_CUDA_CHECK(cudaEventCreate(&stop));
  struct Guard {
    cudaEvent_t a, b;
    ~Guard() { cudaEventDestroy(a); cudaEventDestroy(b); }
  } guard{start, stop};

  auto time_batch = [&](int n) {
    GIE_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < n; ++i) launch(stream);
    GIE_CUDA_CHECK(cudaEventRecord(stop, stream));
    GIE_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    GIE_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return double(ms);
  };

  for (int i = 0; i < opt.warmup_iters; ++i) launch(stream);
  GIE_CUDA_CHECK(cudaStreamSynchronize(stream));

  int inner = opt.inner;
  if (opt.flush_bytes > 0) inner = 1;
  if (inner <= 0) {
    const double probe_ms = std::max(time_batch(4) / 4.0, 1e-4);  // coarse per-launch estimate
    inner = int(std::clamp(std::ceil(opt.min_sample_ms / probe_ms), 1.0, double(opt.max_inner)));
  }

  std::vector<double> us;
  us.reserve(size_t(opt.samples));
  if (opt.flush_bytes > 0 && opt.flush_buffer != nullptr) {
    for (int s = 0; s < opt.samples; ++s) {
      GIE_CUDA_CHECK(cudaMemsetAsync(opt.flush_buffer, s & 0xFF, opt.flush_bytes, stream));  // evicts L2
      us.push_back(time_batch(1) * 1000.0);
    }
    return summarize(std::move(us), 1);
  }
  for (int s = 0; s < opt.samples; ++s) us.push_back(time_batch(inner) * 1000.0 / inner);
  return summarize(std::move(us), inner);
}

}  // namespace gie
