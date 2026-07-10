// Registry of concrete kernel implementations ("candidates").
//
// Every launchable kernel configuration -- a backend plus a specific tile/launch
// configuration -- is one KernelDesc. The dispatcher filters the registry by
// capability, the cost model ranks it, and the autotuner measures it.
// The tile parameters recorded in `params` are exactly the template parameters
// of the instantiated kernel; nothing is listed that the kernel ignores.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gpu_inference/conv2d.h"
#include "gpu_inference/device_info.h"

namespace gie {

// Bump when kernel behavior changes in a way that invalidates tuning results.
constexpr int kLibraryVersion = 2;  // 2: added the mma.sync Tensor Core family; invalidates tuning results from version 1

struct LaunchArgs {
  const ConvProblem* problem = nullptr;
  const DeviceInfo* dev = nullptr;
  const void* x = nullptr;
  const void* w = nullptr;
  const void* bias = nullptr;  // nullptr when !problem->has_bias
  void* y = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  const void* plan_data = nullptr;   // device-resident per-problem tables (see needs_lut)
  void* plan_state = nullptr;        // opaque per-plan state created by KernelDesc::create_state
  cudaStream_t stream = nullptr;
  // Optional, backend-specific profiling hooks. Backends with distinct stages
  // (im2col + GEMM) record into events[0..2]: start, after im2col, after GEMM.
  cudaEvent_t* stage_events = nullptr;
};

struct KernelDesc {
  std::string name;  // unique, stable; used in caches and result files
  Backend backend = Backend::kDirect;
  int version = 1;   // bump when this kernel's code changes behavior/performance
  std::vector<std::pair<std::string, int>> params;  // tile/launch parameters
  int threads_per_block = 0;

  // Tile extents in the implicit-GEMM view (M = C_out, N = B*Ho*Wo, K = C_in*Kh*Kw).
  // Used by the cost model for padding-waste and wave-quantization estimates.
  int tile_m = 1, tile_n = 1, tile_k = 1;
  bool uses_tensor_cores = false;
  bool needs_lut = false;           // requires the (c,r,s) offset table in plan_data
  bool lut_chunked = false;         // table has one entry per 8 consecutive k (r,s,c-chunk order, C % 8 == 0)
  int launches = 1;                 // kernel launches per forward call (cost model)
  double efficiency_prior = 0;      // cost-model prior: fraction of the compute ceiling this family reaches (0 = backend default)
  // Extra DRAM traffic (bytes) beyond the compulsory X/W/Y traffic, e.g. layout conversions, split-K partials.
  double (*extra_traffic_bytes)(const ConvProblem&) = nullptr;
  const void* function = nullptr;   // primary __global__ function (occupancy queries); may be null

  // Capability check. On failure, writes the reason to *why.
  bool (*supports)(const ConvProblem&, const DeviceInfo&, std::string* why) = nullptr;
  size_t (*workspace_bytes)(const ConvProblem&) = nullptr;       // per-call temporary memory
  size_t (*dynamic_smem_bytes)(const ConvProblem&) = nullptr;    // dynamic shared memory per block
  int64_t (*num_blocks)(const ConvProblem&) = nullptr;           // launched thread blocks (first kernel)
  // Fraction of tile slots doing useful work (<= 1); null means derive it from tile_m/n/k
  // over the implicit-GEMM extents. Used only by the cost model.
  double (*tile_efficiency)(const ConvProblem&) = nullptr;
  std::shared_ptr<void> (*create_state)(const ConvProblem&) = nullptr;  // per-plan state (may be null)
  size_t (*persistent_bytes)(const ConvProblem&) = nullptr;      // plan-owned memory, e.g. cuBLAS workspace
  void (*launch)(const LaunchArgs&) = nullptr;

  std::string describe() const;
};

const std::vector<KernelDesc>& kernel_registry();
// TESTING ONLY: mutable access so tests can inject deliberately faulty kernels (to prove the
// autotuner never selects an incorrect or failing candidate). Not for production use.
std::vector<KernelDesc>& kernel_registry_for_testing();
const KernelDesc* find_kernel(const std::string& name);
std::vector<const KernelDesc*> kernels_for_backend(Backend b);

// Per-(c,r,s) lookup entry for implicit-GEMM style gathers: element offset of the
// (c, 0, 0) tap inside one image, and the (r, s) tap displacement including dilation.
struct alignas(16) LutEntry {
  int32_t c_off;  // c * H * W
  int32_t r_off;  // r * dilation_h
  int32_t s_off;  // s * dilation_w
  int32_t pad_;
};
std::vector<LutEntry> build_conv_lut(const Conv2DParams& p);
// One entry per 8 consecutive k for kernels that gather 16-byte channel vectors from an NHWC copy of the
// input: chunk q -> tap (r,s) = q / (C/8), channel offset 8*(q % (C/8)).  c_off is the channel offset in
// elements, r_off/s_off the dilated tap displacement. Requires C % 8 == 0.
std::vector<LutEntry> build_conv_lut_chunked(const Conv2DParams& p);

}  // namespace gie
