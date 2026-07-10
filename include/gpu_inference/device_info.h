// GPU capability and (estimated) peak-rate queries.
//
// Peaks reported here are *theoretical estimates* derived from SM count, the
// maximum SM clock reported by the driver, and per-architecture throughput
// constants. They are not measurements; see perf_model.h / device_ceilings for
// measured ceilings.
#pragma once

#include <cuda_runtime.h>

#include <string>

namespace gie {

struct DeviceInfo {
  int device = 0;
  std::string name;
  int cc_major = 0, cc_minor = 0;
  int sm_count = 0;
  size_t global_mem_bytes = 0;
  size_t l2_bytes = 0;
  size_t smem_per_block_default = 0;  // static limit without opt-in
  size_t smem_per_block_optin = 0;    // maximum with cudaFuncAttributeMaxDynamicSharedMemorySize
  size_t smem_per_sm = 0;
  int regs_per_sm = 0;
  int max_threads_per_sm = 0;
  int max_threads_per_block = 0;
  int max_blocks_per_sm = 0;
  int warp_size = 32;
  int max_sm_clock_khz = 0;
  int mem_clock_khz = 0;
  int mem_bus_bits = 0;
  int driver_version = 0;   // e.g. 13000 for 13.0
  int runtime_version = 0;  // e.g. 12080 for 12.8

  int cc() const { return cc_major * 10 + cc_minor; }
  bool supports_fp16_tensor_cores() const { return cc() >= 70; }
  bool supports_bf16_tensor_cores() const { return cc() >= 80; }
  bool supports_tf32_tensor_cores() const { return cc() >= 80; }
  bool supports_cp_async() const { return cc() >= 80; }

  // Theoretical estimates (TFLOP/s, GB/s) -- NOT measurements.
  double est_peak_fp32_tflops() const;
  double est_peak_fp16_tc_tflops() const;  // dense, FP32 accumulate
  double est_peak_bw_gbs() const;

  std::string describe() const;

  // Queried once per device and cached (thread-safe); call freely.
  static const DeviceInfo& get(int device);
  static const DeviceInfo& current();
};

}  // namespace gie
