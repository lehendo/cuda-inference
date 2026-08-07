#include "gpu_inference/perf_model.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>

namespace gie {

namespace {
std::mutex g_cal_mu;
std::map<std::string, double>& cal_map() {
  static std::map<std::string, double> m;
  return m;
}
}  // namespace

void set_model_calibration(const std::map<std::string, double>& m) {
  std::lock_guard<std::mutex> lock(g_cal_mu);
  cal_map() = m;
}
std::map<std::string, double> get_model_calibration() {
  std::lock_guard<std::mutex> lock(g_cal_mu);
  return cal_map();
}

const char* to_string(Bound b) {
  switch (b) {
    case Bound::kCompute: return "compute";
    case Bound::kBandwidth: return "bandwidth";
    case Bound::kLaunch: return "launch";
  }
  return "?";
}

DeviceCeilings theoretical_ceilings(const DeviceInfo& dev) {
  DeviceCeilings c;
  c.fp32_tflops = dev.est_peak_fp32_tflops();
  c.fp16_tc_tflops = dev.est_peak_fp16_tc_tflops();
  c.bw_gbs = dev.est_peak_bw_gbs();
  c.launch_overhead_us = 4.0;
  c.measured = false;
  c.source = "theoretical estimate from SM count, max clock and architecture constants";
  return c;
}

ProblemModel analyze_problem(const ConvProblem& prob) {
  const Conv2DParams& p = prob.p;
  const double esz = double(dtype_size(prob.dtype()));
  ProblemModel m;
  m.flops = double(p.flops());
  m.input_bytes = double(p.input_shape().numel()) * esz;
  m.weight_bytes = double(p.weight_shape().numel()) * esz;
  m.output_bytes = double(p.output_shape().numel()) * esz;
  m.bias_bytes = prob.has_bias ? double(p.out_channels) * esz : 0.0;
  m.compulsory_bytes = m.input_bytes + m.weight_bytes + m.output_bytes + m.bias_bytes;
  m.im2col_bytes = double(p.batch) * double(p.gemm_k()) * double(p.out_h()) * double(p.out_w()) * esz;
  m.im2col_extra_traffic = 2.0 * m.im2col_bytes;
  m.unfused_extra_traffic = 2.0 * m.output_bytes;
  m.arithmetic_intensity = m.flops / m.compulsory_bytes;
  m.unrolled_blowup = m.im2col_bytes / m.input_bytes;
  return m;
}

namespace {

double bytes_per_us(const DeviceCeilings& c) { return c.bw_gbs * 1e3; }       // GB/s -> B/us
double flops_per_us(double tflops) { return tflops * 1e6; }                    // TFLOP/s -> FLOP/us

Bound classify(double compute_us, double memory_us, double launch_us) {
  if (launch_us >= compute_us && launch_us >= memory_us) return Bound::kLaunch;
  return compute_us >= memory_us ? Bound::kCompute : Bound::kBandwidth;
}

int blocks_per_sm(const KernelDesc& k, const ConvProblem& prob, const DeviceInfo& dev) {
  if (!k.function || k.threads_per_block <= 0) return 1;
  static std::mutex mu;
  static std::map<std::tuple<const void*, size_t, int>, int> cache;
  const size_t smem = k.dynamic_smem_bytes ? k.dynamic_smem_bytes(prob) : 0;
  const auto key = std::make_tuple(k.function, smem, dev.device);
  {
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
  }
  int cur = 0;
  cudaGetDevice(&cur);
  if (cur != dev.device) cudaSetDevice(dev.device);
  if (smem > 48 * 1024)  // occupancy calculation needs the opt-in attribute to be set first
    cudaFuncSetAttribute(k.function, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
  int n = 0;
  cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k.function, k.threads_per_block, smem);
  if (cur != dev.device) cudaSetDevice(cur);
  if (e != cudaSuccess) {
    (void)cudaGetLastError();
    n = 1;
  }
  n = std::max(n, 1);
  std::lock_guard<std::mutex> lock(mu);
  cache[key] = n;
  return n;
}

}  // namespace

Estimate roofline_estimate(const ConvProblem& prob, const DeviceCeilings& c, bool tensor_cores) {
  const ProblemModel m = analyze_problem(prob);
  Estimate e;
  e.compute_us = m.flops / flops_per_us(tensor_cores ? c.fp16_tc_tflops : c.fp32_tflops);
  e.memory_us = m.compulsory_bytes / bytes_per_us(c);
  e.launch_us = c.launch_overhead_us;
  e.time_us = std::max(e.compute_us, e.memory_us);
  e.bound = classify(e.compute_us, e.memory_us, e.launch_us);
  if (e.bound == Bound::kLaunch) e.time_us = e.launch_us;
  return e;
}

double backend_efficiency_prior(const KernelDesc& k) {
  if (k.efficiency_prior > 0) return k.efficiency_prior;
  switch (k.backend) {
    case Backend::kDirect: return 0.05;
    case Backend::kSimtTiled: return 0.40;
    case Backend::kImplicitGemm: return 0.55;
    case Backend::kTensorCore: return 0.30;
    case Backend::kIm2colGemm: return 0.60;
    default: return 0.1;
  }
}

Estimate estimate_kernel(const ConvProblem& prob, const KernelDesc& k, const DeviceInfo& dev, const DeviceCeilings& c) {
  const ProblemModel m = analyze_problem(prob);
  Estimate e;

  if (k.tile_efficiency) {
    e.tile_efficiency = k.tile_efficiency(prob);
  } else {
    const auto pad = [](int64_t v, int64_t t) { return double(v) / double(((v + t - 1) / t) * t); };
    e.tile_efficiency = pad(prob.p.gemm_m(), k.tile_m) * pad(prob.p.gemm_n(), k.tile_n) * pad(prob.p.gemm_k(), k.tile_k);
  }
  const int64_t blocks = k.num_blocks ? std::max<int64_t>(k.num_blocks(prob), 1) : 1;
  e.blocks_per_sm = blocks_per_sm(k, prob, dev);
  const double slots = double(dev.sm_count) * e.blocks_per_sm;
  e.wave_efficiency = double(blocks) / (std::ceil(double(blocks) / slots) * slots);

  const double peak = k.uses_tensor_cores ? c.fp16_tc_tflops : c.fp32_tflops;
  const double eff = backend_efficiency_prior(k) * std::max(e.tile_efficiency, 1e-3) * std::max(e.wave_efficiency, 1e-3);
  e.compute_us = m.flops / (flops_per_us(peak) * eff);

  double traffic = m.compulsory_bytes;
  int launches = k.launches;
  if (k.backend == Backend::kIm2colGemm) {
    traffic += m.im2col_extra_traffic;
    launches = 2;  // im2col + GEMM
    if (prob.has_bias || prob.activation != Activation::kNone) {
      traffic += m.unfused_extra_traffic;
      launches = 3;
    }
  }
  if (k.extra_traffic_bytes) traffic += k.extra_traffic_bytes(prob);
  e.memory_us = traffic / bytes_per_us(c);
  e.launch_us = c.launch_overhead_us * launches;
  e.bound = classify(e.compute_us, e.memory_us, e.launch_us);
  e.time_us = std::max(e.compute_us, e.memory_us) + e.launch_us;
  {
    std::lock_guard<std::mutex> lock(g_cal_mu);
    auto it = cal_map().find(to_string(k.backend));
    if (it != cal_map().end()) {
      e.compute_us *= it->second;
      e.memory_us *= it->second;
      e.time_us = std::max(e.compute_us, e.memory_us) + e.launch_us;
    }
  }
  return e;
}

}  // namespace gie
