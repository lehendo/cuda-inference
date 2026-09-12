// gie_graph_bench: Experiment F (CUDA Graphs) and a stream-concurrency measurement.
//
// A "network" of L identical small layers (ping-pong between two activation buffers) is executed
//   eager : L ExecutionPlan launches per iteration issued from the host
//   graph : one cudaGraphLaunch of the captured sequence
// Reported separately: graph capture and instantiate cost, per-iteration device time (CUDA events),
// per-iteration host launch cost (steady_clock, no sync inside the loop), and a bitwise check that
// graph replay reproduces the eager result across repeated replays. Also: independent convolutions
// run sequentially on one stream vs spread over several streams.
#include <chrono>
#include <fstream>
#include <iostream>

#include "gpu_inference/cuda_graph.h"
#include "gpu_inference/dispatcher.h"
#include "gpu_inference/execution_plan.h"
#include "gpu_inference/json_lite.h"
#include "gpu_inference/perf_model.h"
#include "gpu_inference/testing.h"
#include "gpu_inference/timing.h"

using namespace gie;

namespace {
double host_us_per_iter(const std::function<void()>& f, cudaStream_t s, int iters = 200) {
  for (int i = 0; i < 20; ++i) f();
  GIE_CUDA_CHECK(cudaStreamSynchronize(s));
  std::vector<double> v;
  for (int rep = 0; rep < 7; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) f();
    v.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters);
    GIE_CUDA_CHECK(cudaStreamSynchronize(s));
  }
  return summarize(v, 1).median_us;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::string out_dir = "results/raw/graphs", kernel_name;
    Precision prec = Precision::kFp16;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
      else if (a == "--precision" && i + 1 < argc) prec = precision_from_string(argv[++i]);
      else if (a == "--kernel" && i + 1 < argc) kernel_name = argv[++i];
    }
    GIE_CUDA_CHECK(cudaSetDevice(0));
    const DeviceInfo& dev = DeviceInfo::get(0);
    const std::string mk = "mkdir -p '" + out_dir + "'";
    if (std::system(mk.c_str()) != 0) throw Error(StatusCode::kIo, "cannot create output dir");
    std::ofstream out(out_dir + "/graphs.jsonl", std::ios::app);

    struct Case { const char* name; int batch, ch, hw, k; };
    const std::vector<Case> cases = {{"b1_c32_14x14_k3", 1, 32, 14, 3}, {"b1_c64_7x7_k1", 1, 64, 7, 1},
                                     {"b1_c64_28x28_k3", 1, 64, 28, 3}, {"b4_c128_14x14_k3", 4, 128, 14, 3}};
    const std::vector<int> depths = {1, 4, 16, 48};
    cudaStream_t stream;
    GIE_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    for (const Case& c : cases) {
      ConvProblem prob;
      prob.p.batch = c.batch; prob.p.in_channels = c.ch; prob.p.out_channels = c.ch;
      prob.p.in_h = prob.p.in_w = c.hw; prob.p.kernel_h = prob.p.kernel_w = c.k; prob.p.pad_h = prob.p.pad_w = c.k / 2;
      prob.precision = prec;
      prob.has_bias = true;
      prob.activation = Activation::kReLU;
      // choose the kernel: explicit, else the cost-model top candidate (documented, no tuning here)
      const KernelDesc* k = nullptr;
      if (!kernel_name.empty()) k = find_kernel(kernel_name);
      if (!k) {
        auto cands = Dispatcher::compatible_kernels(prob, dev);
        k = Dispatcher::rank_by_model(prob, dev, cands).front().first;
      }
      auto plan = ExecutionPlan::create(prob, *k, 0);
      ConvData d = make_conv_data(prob, 11);
      DeviceBuffer a_buf(d.dy.size()), b_buf(d.dy.size());
      GIE_CUDA_CHECK(cudaMemcpy(a_buf.data(), d.dx.data(), d.dx.size(), cudaMemcpyDeviceToDevice));
      DeviceBuffer ws(plan->workspace_bytes());

      for (int depth : depths) {
        // layer i reads buf[i%2] and writes buf[(i+1)%2] (channels/spatial preserved by the chosen shapes)
        auto enqueue = [&](cudaStream_t s) {
          void* src = a_buf.data();
          void* dst = b_buf.data();
          for (int l = 0; l < depth; ++l) {
            plan->execute_unchecked(src, d.dw.data(), d.db.data(), dst, ws.data(), ws.size(), s);
            std::swap(src, dst);
          }
        };
        CudaGraphExecutor graph;
        graph.capture(enqueue);

        // correctness: eager vs graph, repeated replays must give bit-identical results
        auto snapshot = [&]() {
          GIE_CUDA_CHECK(cudaStreamSynchronize(stream));
          return download_as_float((depth % 2 == 0) ? a_buf.data() : b_buf.data(), prob.dtype(), prob.p.input_shape().numel());
        };
        // Keep activations bounded: restore the input before each run.
        auto reset_input = [&]() { GIE_CUDA_CHECK(cudaMemcpyAsync(a_buf.data(), d.dx.data(), d.dx.size(), cudaMemcpyDeviceToDevice, stream)); };
        reset_input();
        enqueue(stream);
        const auto eager_out = snapshot();
        bool identical = true;
        for (int rep = 0; rep < 5; ++rep) {
          reset_input();
          graph.launch(stream);
          identical = identical && (snapshot() == eager_out);
        }

        TimingOptions to;
        to.warmup_iters = 10;
        to.samples = 30;
        to.min_sample_ms = 2.0;
        const TimingStats t_eager = time_stream_work([&](cudaStream_t s) { enqueue(s); }, stream, to);
        const TimingStats t_graph = time_stream_work([&](cudaStream_t s) { graph.launch(s); }, stream, to);
        const double h_eager = host_us_per_iter([&] { enqueue(stream); }, stream);
        const double h_graph = host_us_per_iter([&] { graph.launch(stream); }, stream);

        json::Value r = json::Value::object();
        r["experiment"] = "F_cuda_graphs";
        r["gpu"] = dev.name;
        r["case"] = c.name;
        r["precision"] = to_string(prec);
        r["kernel"] = k->name;
        r["layers"] = depth;
        r["graph_nodes"] = double(graph.node_count());
        r["capture_us"] = graph.capture_us();
        r["instantiate_us"] = graph.instantiate_us();
        r["eager_device_us"] = t_eager.median_us;
        r["graph_device_us"] = t_graph.median_us;
        r["eager_host_launch_us"] = h_eager;
        r["graph_host_launch_us"] = h_graph;
        r["device_speedup"] = t_eager.median_us / t_graph.median_us;
        r["bitwise_identical_replays"] = identical;
        out << r.dump() << "\n";
        out.flush();
        std::cout << c.name << " L=" << depth << " eager " << t_eager.median_us << " us, graph " << t_graph.median_us
                  << " us (x" << t_eager.median_us / t_graph.median_us << "), host launch " << h_eager << " -> " << h_graph
                  << " us, identical=" << identical << ", capture " << graph.capture_us() << " + instantiate " << graph.instantiate_us() << " us\n";
      }

      // Stream concurrency: S independent copies of the layer, one stream vs S streams.
      for (int S : {2, 4, 8}) {
        std::vector<ConvData> ds;
        std::vector<DeviceBuffer> wss;
        for (int s = 0; s < S; ++s) { ds.push_back(make_conv_data(prob, 100 + uint64_t(s))); wss.emplace_back(plan->workspace_bytes()); }
        std::vector<cudaStream_t> streams(static_cast<size_t>(S));
        std::vector<cudaEvent_t> done(static_cast<size_t>(S));
        for (int s = 0; s < S; ++s) {
          GIE_CUDA_CHECK(cudaStreamCreateWithFlags(&streams[size_t(s)], cudaStreamNonBlocking));
          GIE_CUDA_CHECK(cudaEventCreateWithFlags(&done[size_t(s)], cudaEventDisableTiming));
        }
        cudaEvent_t fork;
        GIE_CUDA_CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
        auto run_one = [&](int s, cudaStream_t st) {
          plan->execute_unchecked(ds[size_t(s)].dx.data(), ds[size_t(s)].dw.data(), ds[size_t(s)].db.data(), ds[size_t(s)].dy.data(), wss[size_t(s)].data(), wss[size_t(s)].size(), st);
        };
        auto seq = [&](cudaStream_t s) { for (int i = 0; i < S; ++i) run_one(i, s); };
        auto par = [&](cudaStream_t s) {
          GIE_CUDA_CHECK(cudaEventRecord(fork, s));
          for (int i = 0; i < S; ++i) {
            GIE_CUDA_CHECK(cudaStreamWaitEvent(streams[size_t(i)], fork, 0));
            run_one(i, streams[size_t(i)]);
            GIE_CUDA_CHECK(cudaEventRecord(done[size_t(i)], streams[size_t(i)]));
            GIE_CUDA_CHECK(cudaStreamWaitEvent(s, done[size_t(i)], 0));
          }
        };
        TimingOptions to2;
        to2.warmup_iters = 10; to2.samples = 30; to2.min_sample_ms = 2.0;
        const TimingStats ts = time_stream_work(seq, stream, to2), tp = time_stream_work(par, stream, to2);
        json::Value r = json::Value::object();
        r["experiment"] = "F_stream_concurrency";
        r["gpu"] = dev.name;
        r["case"] = c.name;
        r["kernel"] = k->name;
        r["independent_convs"] = S;
        r["one_stream_us"] = ts.median_us;
        r["multi_stream_us"] = tp.median_us;
        r["speedup"] = ts.median_us / tp.median_us;
        out << r.dump() << "\n";
        out.flush();
        std::cout << c.name << " " << S << " independent convs: 1 stream " << ts.median_us << " us, " << S << " streams "
                  << tp.median_us << " us (x" << ts.median_us / tp.median_us << ")\n";
        GIE_CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int s = 0; s < S; ++s) { cudaStreamDestroy(streams[size_t(s)]); cudaEventDestroy(done[size_t(s)]); }
        cudaEventDestroy(fork);
      }
    }
    cudaStreamDestroy(stream);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
