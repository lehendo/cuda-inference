#include "gpu_inference/device_info.h"

#include <map>
#include <mutex>
#include <sstream>

#include "gpu_inference/error.h"

namespace gie {
namespace {

int attr(cudaDeviceAttr a, int dev) {
  int v = 0;
  GIE_CUDA_CHECK(cudaDeviceGetAttribute(&v, a, dev));
  return v;
}

// FP32 FMA lanes per SM by compute capability (NVIDIA architecture whitepapers).
int fp32_lanes_per_sm(int cc) {
  if (cc >= 86) return 128;  // GA10x, Ada, Hopper, Blackwell
  return 64;                 // Volta, Turing, GA100
}

// Dense FP16 Tensor Core FLOP/clk/SM with FP32 accumulation (whitepapers / spec sheets).
int fp16_tc_flops_per_clk_sm(int cc) {
  if (cc >= 90) return 4096;
  if (cc >= 89) return 1024;
  if (cc >= 86) return 1024;
  if (cc >= 80) return 2048;
  if (cc >= 70) return 1024;
  return 0;
}

}  // namespace

double DeviceInfo::est_peak_fp32_tflops() const {
  return double(sm_count) * fp32_lanes_per_sm(cc()) * 2.0 * double(max_sm_clock_khz) * 1e3 / 1e12;
}

double DeviceInfo::est_peak_fp16_tc_tflops() const {
  return double(sm_count) * fp16_tc_flops_per_clk_sm(cc()) * double(max_sm_clock_khz) * 1e3 / 1e12;
}

double DeviceInfo::est_peak_bw_gbs() const {
  return 2.0 * double(mem_clock_khz) * 1e3 * (double(mem_bus_bits) / 8.0) / 1e9;
}

std::string DeviceInfo::describe() const {
  std::ostringstream os;
  os << name << " (sm_" << cc() << ", " << sm_count << " SMs, " << global_mem_bytes / (1024 * 1024)
     << " MiB, L2 " << l2_bytes / 1024 << " KiB, smem/block " << smem_per_block_default / 1024 << "/"
     << smem_per_block_optin / 1024 << " KiB (default/opt-in), driver " << driver_version / 1000 << "."
     << (driver_version % 1000) / 10 << ", runtime " << runtime_version / 1000 << "."
     << (runtime_version % 1000) / 10 << ")";
  return os.str();
}

const DeviceInfo& DeviceInfo::get(int device) {
  static std::mutex mu;
  static std::map<int, DeviceInfo> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(device);
  if (it != cache.end()) return it->second;

  DeviceInfo d;
  d.device = device;
  cudaDeviceProp prop;
  GIE_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  d.name = prop.name;
  d.cc_major = prop.major;
  d.cc_minor = prop.minor;
  d.global_mem_bytes = prop.totalGlobalMem;
  d.smem_per_block_default = prop.sharedMemPerBlock;
  d.warp_size = prop.warpSize;
  d.sm_count = attr(cudaDevAttrMultiProcessorCount, device);
  d.l2_bytes = size_t(attr(cudaDevAttrL2CacheSize, device));
  d.smem_per_block_optin = size_t(attr(cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  d.smem_per_sm = size_t(attr(cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
  d.regs_per_sm = attr(cudaDevAttrMaxRegistersPerMultiprocessor, device);
  d.max_threads_per_sm = attr(cudaDevAttrMaxThreadsPerMultiProcessor, device);
  d.max_threads_per_block = attr(cudaDevAttrMaxThreadsPerBlock, device);
  d.max_blocks_per_sm = attr(cudaDevAttrMaxBlocksPerMultiprocessor, device);
  d.max_sm_clock_khz = attr(cudaDevAttrClockRate, device);
  d.mem_clock_khz = attr(cudaDevAttrMemoryClockRate, device);
  d.mem_bus_bits = attr(cudaDevAttrGlobalMemoryBusWidth, device);
  GIE_CUDA_CHECK(cudaDriverGetVersion(&d.driver_version));
  GIE_CUDA_CHECK(cudaRuntimeGetVersion(&d.runtime_version));
  return cache.emplace(device, std::move(d)).first->second;
}

const DeviceInfo& DeviceInfo::current() {
  int dev = 0;
  GIE_CUDA_CHECK(cudaGetDevice(&dev));
  return get(dev);
}

}  // namespace gie
