// Measured device ceilings: achievable FP32 FMA throughput, FP16 Tensor Core (WMMA)
// throughput, device-memory copy bandwidth and back-to-back launch cost. These are
// micro-benchmark results, distinct from the spec-derived estimates in DeviceInfo.
#include <mma.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>

#include "gpu_inference/json_lite.h"
#include "gpu_inference/perf_model.h"
#include "gpu_inference/timing.h"

namespace gie {
namespace {

using namespace nvcuda;

__global__ void fma_kernel(float* out, int iters) {
  float a0 = threadIdx.x * 1e-3f, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
  const float b = 1.0001f, c = 0.9999f;
  for (int i = 0; i < iters; ++i) {
    a0 = fmaf(a0, b, c); a1 = fmaf(a1, b, c); a2 = fmaf(a2, b, c); a3 = fmaf(a3, b, c);
    a4 = fmaf(a4, b, c); a5 = fmaf(a5, b, c); a6 = fmaf(a6, b, c); a7 = fmaf(a7, b, c);
  }
  const float s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  if (s == 123456.789f) out[0] = s;  // defeats dead-code elimination
}

__global__ void wmma_kernel(float* out, int iters) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
  wmma::fill_fragment(fa, __float2half(0.5f));
  wmma::fill_fragment(fb, __float2half(0.25f));
  // Distinct accumulator seeds: with identical inputs the compiler could merge the 8 chains into one
  // (common-subexpression elimination) and the measured rate would be inflated.
#pragma unroll
  for (int j = 0; j < 8; ++j) wmma::fill_fragment(acc[j], float(j + 1));
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) wmma::mma_sync(acc[j], fa, fb, acc[j]);
  }
  float s = 0;  // consume EVERY accumulator element, otherwise the compiler drops the unused HMMA halves
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int e = 0; e < acc[j].num_elements; ++e) s += acc[j].x[e];
  if (s == 123456.789f) out[0] = s;
}

__global__ void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) dst[i] = src[i];
}

__global__ void empty_kernel() {}

double best_ms(const std::function<void(cudaStream_t)>& f, cudaStream_t s, int reps = 8) {
  TimingOptions o;
  o.warmup_iters = 2;
  o.samples = reps;
  o.inner = 1;
  TimingStats t = time_stream_work(f, s, o);
  return t.min_us / 1000.0;
}

std::string cache_dir() {
  if (const char* d = std::getenv("GIE_CEILING_DIR")) return d;
  const char* home = std::getenv("HOME");
  return std::string(home ? home : ".") + "/.cache/gpu_inference";
}

std::string cache_file(const DeviceInfo& d) {
  std::string n = d.name;
  for (char& c : n)
    if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
  return cache_dir() + "/ceilings_" + n + "_sm" + std::to_string(d.cc()) + ".json";
}

void mkdirs(const std::string& dir) {
  const std::string cmd = "mkdir -p '" + dir + "'";
  if (std::system(cmd.c_str()) != 0) throw Error(StatusCode::kIo, "cannot create " + dir);
}

}  // namespace

bool load_cached_ceilings(int device, DeviceCeilings* out) {
  try {
    const DeviceInfo& d = DeviceInfo::get(device);
    const json::Value v = json::parse_file(cache_file(d));
    if (!v.contains("fp32_tflops") || !v.contains("bw_gbs") || !v.contains("fp16_tc_tflops")) return false;
    out->fp32_tflops = v.find("fp32_tflops")->as_double();
    out->fp16_tc_tflops = v.find("fp16_tc_tflops")->as_double();
    out->bw_gbs = v.find("bw_gbs")->as_double();
    out->launch_overhead_us = v.find("launch_overhead_us") ? v.find("launch_overhead_us")->as_double() : 4.0;
    out->measured = true;
    out->source = "microbenchmark (" + cache_file(d) + ")";
    return true;
  } catch (const Error&) {
    return false;
  }
}

DeviceCeilings measure_ceilings(int device, bool use_cache) {
  DeviceCeilings c;
  if (use_cache && load_cached_ceilings(device, &c)) return c;

  int prev = 0;
  GIE_CUDA_CHECK(cudaGetDevice(&prev));
  GIE_CUDA_CHECK(cudaSetDevice(device));
  const DeviceInfo& d = DeviceInfo::get(device);
  cudaStream_t s;
  GIE_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  float* scratch = nullptr;
  GIE_CUDA_CHECK(cudaMalloc(&scratch, 16));

  // FP32 FMA throughput
  {
    const int iters = 1 << 15, blocks = d.sm_count * 8, threads = 256;
    const double ms = best_ms([&](cudaStream_t st) { fma_kernel<<<blocks, threads, 0, st>>>(scratch, iters); }, s);
    c.fp32_tflops = double(blocks) * threads * iters * 8 * 2 / (ms * 1e-3) / 1e12;
  }
  // Tensor Core throughput (FP16 inputs, FP32 accumulate)
  if (d.supports_fp16_tensor_cores()) {
    const int iters = 1 << 12, blocks = d.sm_count * 4, threads = 256;
    const double ms = best_ms([&](cudaStream_t st) { wmma_kernel<<<blocks, threads, 0, st>>>(scratch, iters); }, s);
    c.fp16_tc_tflops = double(blocks) * (threads / 32) * iters * 8 * (2.0 * 16 * 16 * 16) / (ms * 1e-3) / 1e12;
  }
  // Global-memory copy bandwidth (read + write), working set far above L2
  {
    const size_t bytes = std::max<size_t>(size_t(256) << 20, 8 * d.l2_bytes);
    void *src = nullptr, *dst = nullptr;
    GIE_CUDA_CHECK(cudaMalloc(&src, bytes));
    GIE_CUDA_CHECK(cudaMalloc(&dst, bytes));
    GIE_CUDA_CHECK(cudaMemset(src, 1, bytes));
    const size_t n4 = bytes / 16;
    const double ms = best_ms([&](cudaStream_t st) {
      copy_kernel<<<d.sm_count * 16, 256, 0, st>>>(static_cast<const float4*>(src), static_cast<float4*>(dst), n4);
    }, s, 10);
    c.bw_gbs = 2.0 * double(bytes) / (ms * 1e-3) / 1e9;
    cudaFree(src);
    cudaFree(dst);
  }
  // Back-to-back empty-kernel launch cost
  {
    TimingOptions o;
    o.warmup_iters = 100;
    o.samples = 15;
    o.inner = 500;
    c.launch_overhead_us = time_stream_work([&](cudaStream_t st) { empty_kernel<<<1, 32, 0, st>>>(); }, s, o).median_us;
  }
  cudaFree(scratch);
  cudaStreamDestroy(s);
  GIE_CUDA_CHECK(cudaSetDevice(prev));

  c.measured = true;
  c.source = "microbenchmark: FMA loop, WMMA m16n16k16 loop, float4 copy, empty-kernel stream";
  json::Value v = json::Value::object();
  v["gpu"] = d.name;
  v["compute_capability"] = d.cc();
  v["fp32_tflops"] = c.fp32_tflops;
  v["fp16_tc_tflops"] = c.fp16_tc_tflops;
  v["bw_gbs"] = c.bw_gbs;
  v["launch_overhead_us"] = c.launch_overhead_us;
  v["est_peak_fp32_tflops"] = d.est_peak_fp32_tflops();
  v["est_peak_fp16_tc_tflops"] = d.est_peak_fp16_tc_tflops();
  v["est_peak_bw_gbs"] = d.est_peak_bw_gbs();
  v["method"] = c.source;
  try {
    mkdirs(cache_dir());
    json::write_file(cache_file(d), v);
  } catch (const Error&) {
    // Cache is best-effort; a read-only home must not break measurement.
  }
  return c;
}

}  // namespace gie
