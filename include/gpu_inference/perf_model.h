// Roofline-inspired analytical performance model.
//
// Everything here is a *static estimate*. FLOP counts are exact for the
// operator; byte counts are compulsory-traffic lower bounds, not measured DRAM
// traffic; time estimates use ceilings that are either theoretical (from the
// spec-derived DeviceInfo) or measured by microbenchmarks (device_ceilings).
// The model is used to rank/prune autotuning candidates; its agreement with
// measurements is evaluated by bench/gie_tune, never assumed.
#pragma once

#include <map>
#include <string>

#include "gpu_inference/kernel_registry.h"

namespace gie {

struct DeviceCeilings {
  double fp32_tflops = 0;       // CUDA-core FMA ceiling
  double fp16_tc_tflops = 0;    // Tensor Core (WMMA) FP16->FP32 ceiling
  double bw_gbs = 0;            // achievable global-memory bandwidth
  double launch_overhead_us = 4.0;
  bool measured = false;        // false: derived from DeviceInfo estimates
  std::string source;           // human-readable provenance
};

DeviceCeilings theoretical_ceilings(const DeviceInfo& dev);
// Runs (or loads cached) microbenchmarks on `device`. Result cached in
// $GIE_CEILING_DIR (default ~/.cache/gpu_inference). Allocates ~1 GiB briefly.
DeviceCeilings measure_ceilings(int device, bool use_cache = true);
// Loads previously measured ceilings if present (never runs microbenchmarks).
bool load_cached_ceilings(int device, DeviceCeilings* out);

enum class Bound { kCompute, kBandwidth, kLaunch };
const char* to_string(Bound b);

struct ProblemModel {
  double flops = 0;
  double input_bytes = 0, weight_bytes = 0, output_bytes = 0, bias_bytes = 0;
  double compulsory_bytes = 0;      // read X, W, bias once + write Y once
  double im2col_bytes = 0;          // size of the explicit unrolled matrix
  double im2col_extra_traffic = 0;  // write + read of the unrolled matrix
  double unfused_extra_traffic = 0; // extra read+write of Y per separately-launched elementwise op
  double arithmetic_intensity = 0;  // flops / compulsory_bytes (FLOP/byte)
  double unrolled_blowup = 0;       // im2col_bytes / input_bytes
};
ProblemModel analyze_problem(const ConvProblem& prob);

struct Estimate {
  double time_us = 0;
  double compute_us = 0, memory_us = 0, launch_us = 0;
  Bound bound = Bound::kCompute;
  double tile_efficiency = 1.0;   // useful / padded work
  double wave_efficiency = 1.0;   // block-level load balance over SMs
  int blocks_per_sm = 0;
};

// Idealized roofline bound: no tiling waste, perfect overlap.
Estimate roofline_estimate(const ConvProblem& prob, const DeviceCeilings& c, bool tensor_cores);
// Per-kernel estimate including padding waste, wave quantization and a backend
// efficiency factor (fraction of the ceiling a well-tuned kernel of that family
// is expected to reach; these defaults are priors, not measurements).
Estimate estimate_kernel(const ConvProblem& prob, const KernelDesc& k, const DeviceInfo& dev,
                         const DeviceCeilings& c);

double backend_efficiency_prior(const KernelDesc& k);

// Optional per-backend multiplicative corrections to estimate_kernel().time_us, fitted from
// *tuning-set* measurements (gie_tune calibrate) and then evaluated on held-out workloads.
// Keys are backend names (to_string(Backend)). Empty map => uncalibrated priors.
void set_model_calibration(const std::map<std::string, double>& scale_by_backend);
std::map<std::string, double> get_model_calibration();

}  // namespace gie
