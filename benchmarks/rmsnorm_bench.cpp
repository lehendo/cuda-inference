// gie_rmsnorm_bench: fused RMSNorm / fused add+RMSNorm vs the unfused three-kernel baseline.
#include <fstream>
#include <iostream>

#include "gpu_inference/device_info.h"
#include "gpu_inference/json_lite.h"
#include "gpu_inference/rmsnorm.h"
#include "gpu_inference/testing.h"
#include "gpu_inference/timing.h"

using namespace gie;

int main(int argc, char** argv) {
  try {
    std::string out_dir = argc > 1 ? argv[1] : "results/raw/rmsnorm";
    if (std::system(("mkdir -p '" + out_dir + "'").c_str()) != 0) throw Error(StatusCode::kIo, "mkdir failed");
    std::ofstream out(out_dir + "/rmsnorm.jsonl", std::ios::app);
    const DeviceInfo& dev = DeviceInfo::get(0);
    cudaStream_t s;
    GIE_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    for (DataType dt : {DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16}) {
      for (int rows : {1, 32, 512, 4096}) {
        for (int hidden : {768, 2048, 4096, 5120, 8192, 4097}) {
          const size_t n = size_t(rows) * hidden;
          std::vector<float> h(n, 0.5f), g(static_cast<size_t>(hidden), 1.f);
          DeviceBuffer x = upload_as(h, dt), r = upload_as(h, dt), gm = upload_as(g, dt), y(n * dtype_size(dt)), res(n * dtype_size(dt)), sc(size_t(rows) * 4);
          RmsNormParams p;
          p.rows = rows; p.hidden = hidden; p.dtype = dt;
          for (int fused = 0; fused < 2; ++fused) {
            TimingOptions to;
            to.warmup_iters = 10; to.samples = 30; to.min_sample_ms = 1.0;
            const void* ri = fused ? r.data() : nullptr;
            void* ro = fused ? res.data() : nullptr;
            const TimingStats a = time_stream_work([&](cudaStream_t st) { rmsnorm_forward(p, x.data(), gm.data(), y.data(), ri, ro, st); }, s, to);
            const TimingStats b = time_stream_work([&](cudaStream_t st) { rmsnorm_forward_unfused(p, x.data(), gm.data(), y.data(), ri, ro, sc.data(), st); }, s, to);
            // minimum traffic: read x (+res), write y (+res_out), read gamma
            const double bytes = double(dtype_size(dt)) * (double(n) * (fused ? 4.0 : 2.0) + hidden);
            json::Value j = json::Value::object();
            j["gpu"] = dev.name; j["dtype"] = to_string(dt); j["rows"] = rows; j["hidden"] = hidden; j["fused_add"] = bool(fused);
            j["fused_kernel_us"] = a.median_us; j["unfused_us"] = b.median_us; j["speedup"] = b.median_us / a.median_us;
            j["fused_gbs"] = bytes / (a.median_us * 1e-6) / 1e9;
            j["compulsory_bytes"] = bytes;
            out << j.dump() << "\n";
            std::cout << to_string(dt) << " rows=" << rows << " hidden=" << hidden << " add=" << fused << ": fused " << a.median_us << " us ("
                      << j.find("fused_gbs")->as_double() << " GB/s), unfused " << b.median_us << " us\n";
          }
        }
      }
    }
    cudaStreamDestroy(s);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
