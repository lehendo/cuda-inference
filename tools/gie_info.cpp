// Prints the detected GPU environment and the kernels each device can run.
#include <iostream>

#include "gpu_inference/conv2d.h"
#include "gpu_inference/dispatcher.h"
#include "gpu_inference/perf_model.h"

// Per-kernel static resource report (no performance counters needed): registers, local-memory spills,
// shared memory and theoretical occupancy from the CUDA occupancy API for a sample problem.
static void kernel_report(const gie::DeviceInfo& dev) {
  using namespace gie;
  for (Precision prec : {Precision::kFp32, Precision::kFp16}) {
    ConvProblem prob;
    prob.p.batch = 32; prob.p.in_channels = 64; prob.p.in_h = prob.p.in_w = 56; prob.p.out_channels = 64;
    prob.p.kernel_h = prob.p.kernel_w = 3; prob.p.pad_h = prob.p.pad_w = 1;
    prob.precision = prec;
    std::cout << "\n# kernel resources, sample problem " << prob.key() << " on " << dev.name << "\n";
    std::cout << "kernel,threads,regs,spill_local_bytes,static_smem,dynamic_smem,blocks_per_sm,occupancy_pct,workspace_bytes\n";
    for (const KernelDesc& k : kernel_registry()) {
      std::string why;
      if (!k.function || !k.supports(prob, dev, &why)) continue;
      cudaFuncAttributes at;
      if (cudaFuncGetAttributes(&at, k.function) != cudaSuccess) { (void)cudaGetLastError(); continue; }
      const size_t dyn = k.dynamic_smem_bytes ? k.dynamic_smem_bytes(prob) : 0;
      if (dyn > 48 * 1024) cudaFuncSetAttribute(k.function, cudaFuncAttributeMaxDynamicSharedMemorySize, int(dyn));
      int blocks = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k.function, k.threads_per_block, dyn);
      const double occ = 100.0 * blocks * k.threads_per_block / dev.max_threads_per_sm;
      std::cout << k.name << "," << k.threads_per_block << "," << at.numRegs << "," << at.localSizeBytes << "," << at.sharedSizeBytes << ","
                << dyn << "," << blocks << "," << occ << "," << k.workspace_bytes(prob) << "\n";
    }
  }
}

int main(int argc, char** argv) {
  using namespace gie;
  const bool measure = argc > 1 && std::string(argv[1]) == "--measure-ceilings";
  const bool report = argc > 1 && std::string(argv[1]) == "--kernel-report";
  if (argc > 1 && std::string(argv[1]) == "--list-kernels") {  // machine-readable registry listing (one name per line)
    for (const auto& k : kernel_registry()) std::cout << k.name << "\n";
    return 0;
  }
  try {
    int n = 0;
    GIE_CUDA_CHECK(cudaGetDeviceCount(&n));
    std::cout << "CUDA devices: " << n << "\n";
    for (int i = 0; i < n; ++i) {
      const DeviceInfo& d = DeviceInfo::get(i);
      std::cout << "[" << i << "] " << d.describe() << "\n";
      std::cout << "    theoretical (spec-derived, NOT measured): fp32 " << d.est_peak_fp32_tflops()
                << " TFLOP/s, fp16-TC " << d.est_peak_fp16_tc_tflops() << " TFLOP/s, bandwidth "
                << d.est_peak_bw_gbs() << " GB/s\n";
      if (measure) {
        const DeviceCeilings c = measure_ceilings(i, false);
        std::cout << "    measured ceilings: fp32 " << c.fp32_tflops << " TFLOP/s, fp16-TC " << c.fp16_tc_tflops
                  << " TFLOP/s, copy bandwidth " << c.bw_gbs << " GB/s, launch " << c.launch_overhead_us << " us\n";
      }
      if (report) kernel_report(d);
      int per_backend[7] = {0};
      for (const auto& k : kernel_registry()) ++per_backend[int(k.backend)];
      std::cout << "    registered kernels: " << kernel_registry().size() << " (direct " << per_backend[int(Backend::kDirect)]
                << ", im2col_gemm " << per_backend[int(Backend::kIm2colGemm)] << ", simt_tiled "
                << per_backend[int(Backend::kSimtTiled)] << ", implicit_gemm " << per_backend[int(Backend::kImplicitGemm)]
                << ", tensor_core " << per_backend[int(Backend::kTensorCore)] << ")\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
