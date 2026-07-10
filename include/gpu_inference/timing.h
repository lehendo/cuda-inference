// CUDA-event based timing with robust summary statistics.
#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <vector>

namespace gie {

struct TimingStats {
  int samples = 0;
  int inner_iters = 1;     // launches averaged inside each sample
  double median_us = 0, mean_us = 0, min_us = 0, max_us = 0;
  double p10_us = 0, p90_us = 0, p99_us = 0;
  double stddev_us = 0;
  double iqr_us = 0;
  std::vector<double> raw_us;  // per-sample (per-launch average) latency
};

struct TimingOptions {
  int warmup_iters = 5;
  int samples = 20;
  // Each sample times `inner` back-to-back launches between one event pair. If
  // inner <= 0 it is chosen so a sample lasts at least min_sample_ms (short kernels
  // need batching: event resolution is ~0.5 us and launch gaps are amortized).
  int inner = 0;
  double min_sample_ms = 1.0;
  int max_inner = 4096;
  // Cold-cache protocol: if flush_bytes > 0, each sample is a single launch preceded by an
  // untimed memset of `flush_buffer` (size >= 2x L2) so inputs/weights are not L2-resident.
  void* flush_buffer = nullptr;
  size_t flush_bytes = 0;
};

// `launch` must enqueue the work on `stream` without synchronizing.
TimingStats time_stream_work(const std::function<void(cudaStream_t)>& launch, cudaStream_t stream,
                             const TimingOptions& opt = {});

TimingStats summarize(std::vector<double> per_launch_us, int inner);

}  // namespace gie
