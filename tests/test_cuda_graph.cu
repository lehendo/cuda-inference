// CUDA Graph capture/replay correctness for every kernel family: replay with changing input data, replay on
// different streams, repeated replays, in-place graph update for new buffers, chained layers.
#include "conv_test_common.h"
#include "gpu_inference/cuda_graph.h"

using namespace gie;

namespace {

struct Stream {
  cudaStream_t s = nullptr;
  Stream() { GIE_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(s); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
};

ConvProblem prob_of(Precision prec) {
  return gt::make_problem(gt::P(2, 32, 15, 14, 32, 3, 3, 1, 1, 1, 1), prec, true, true);
}

bool dr(const KernelDesc& k, const ConvProblem& p) {
  return k.backend == Backend::kIm2colGemm && p.precision != Precision::kFp32;
}

const char* kFp16[] = {"direct_fp16", "im2col_gemm_fp16", "simt_tiled_fp16_tc32_t8x8_r4x2_cc8", "implicit_gemm_fp16_64x64x8_t4x4",
                       "tensor_core_fp16_64x64x32_w2x2", "tcmma_fp16_128x128x32_s3_w2x4_sk1", "tcmma_fp16_64x64x32_s4_w2x2_sk4"};
const char* kFp32[] = {"direct_fp32", "im2col_gemm_fp32", "simt_tiled_fp32_tc32_t8x8_r4x2_cc8", "implicit_gemm_fp32_64x64x8_t4x4"};

template <typename F>
void for_each_kernel(F&& f) {
  const bool tc = DeviceInfo::current().supports_bf16_tensor_cores();
  for (const char* n : kFp16) {
    if (!tc && (std::string(n).rfind("tensor_core", 0) == 0 || std::string(n).rfind("tcmma", 0) == 0)) continue;
    f(n, Precision::kFp16);
  }
  for (const char* n : kFp32) f(n, Precision::kFp32);
}

}  // namespace

TEST(graph_replay_with_changing_inputs_matches_reference) {
  for_each_kernel([&](const char* name, Precision prec) {
    const ConvProblem prob = prob_of(prec);
    const KernelDesc* k = find_kernel(name);
    CHECK_MSG(k && k->supports(prob, DeviceInfo::current(), nullptr), name);
    if (!k) return;
    auto plan = ExecutionPlan::create(prob, *k);  // plans must exist BEFORE capture (creation allocates/synchronizes)
    DeviceBuffer ws(plan->workspace_bytes());
    gt::RefBundle bufs = gt::make_bundle(prob, 1, Distribution::kUniform);
    std::vector<gt::RefBundle> data;
    for (int i = 0; i < 3; ++i) data.push_back(gt::make_bundle(prob, 50 + uint64_t(i), i == 1 ? Distribution::kSparseSigned : Distribution::kNormal));
    CudaGraphExecutor graph;
    graph.capture([&](cudaStream_t s) {
      plan->execute_unchecked(bufs.data.dx.data(), bufs.data.dw.data(), bufs.data.db.data(), bufs.data.dy.data(), ws.data(), ws.size(), s);
    });
    CHECK(graph.valid() && graph.node_count() >= 1);
    Stream replay_stream, other_stream;
    for (int rep = 0; rep < 6; ++rep) {
      Stream& st = (rep % 2) ? other_stream : replay_stream;  // the graph may be launched on any stream
      const gt::RefBundle& src = data[size_t(rep) % data.size()];
      gt::copy_inputs_async(bufs, src, st.s);
      GIE_CUDA_CHECK(cudaMemsetAsync(bufs.data.dy.data(), 0xFF, bufs.data.dy.size(), st.s));
      graph.launch(st.s);
      GIE_CUDA_CHECK(cudaStreamSynchronize(st.s));
      const ErrorMetrics m = gt::compare_device_output(bufs.data.dy.data(), src, dr(*k, prob));
      CHECK_MSG(m.passed(), name << " replay " << rep << ": " << m.summary());
    }
  });
}

TEST(graph_node_counts_reflect_pipeline_structure) {
  const ConvProblem prob = prob_of(Precision::kFp16);
  struct Expect { const char* kernel; size_t min_nodes; };
  // im2col: im2col kernel + >=1 GEMM kernel + bias/ReLU pass; tcmma: 2 transposes + main kernel (+ reduce for split-K)
  for (Expect e : {Expect{"direct_fp16", 1}, Expect{"im2col_gemm_fp16", 3}, Expect{"tcmma_fp16_128x128x32_s3_w2x4_sk1", 3}, Expect{"tcmma_fp16_64x64x32_s4_w2x2_sk4", 4}}) {
    if (!DeviceInfo::current().supports_bf16_tensor_cores() && std::string(e.kernel).rfind("tcmma", 0) == 0) continue;
    const KernelDesc* k = find_kernel(e.kernel);
    auto plan = ExecutionPlan::create(prob, *k);
    DeviceBuffer ws(plan->workspace_bytes());
    gt::RefBundle b = gt::make_bundle(prob, 2, Distribution::kUniform);
    CudaGraphExecutor g;
    g.capture([&](cudaStream_t s) { plan->execute_unchecked(b.data.dx.data(), b.data.dw.data(), b.data.db.data(), b.data.dy.data(), ws.data(), ws.size(), s); });
    CHECK_MSG(g.node_count() >= e.min_nodes, e.kernel << ": " << g.node_count() << " nodes < " << e.min_nodes);
  }
}

TEST(graph_recapture_update_switches_buffers_in_place) {
  const ConvProblem prob = prob_of(Precision::kFp32);
  const KernelDesc* k = find_kernel("implicit_gemm_fp32_64x64x8_t4x4");
  auto plan = ExecutionPlan::create(prob, *k);
  gt::RefBundle b1 = gt::make_bundle(prob, 5, Distribution::kNormal), b2 = gt::make_bundle(prob, 6, Distribution::kUniform);
  auto enqueue_for = [&](gt::RefBundle& b) {
    return [&b, &plan](cudaStream_t s) { plan->execute_unchecked(b.data.dx.data(), b.data.dw.data(), b.data.db.data(), b.data.dy.data(), nullptr, 0, s); };
  };
  CudaGraphExecutor g;
  g.capture(enqueue_for(b1));
  Stream st;
  g.launch(st.s);
  GIE_CUDA_CHECK(cudaStreamSynchronize(st.s));
  CHECK(gt::compare_device_output(b1.data.dy.data(), b1).passed());
  const bool in_place = g.recapture_update(enqueue_for(b2));  // same topology, different pointers
  CHECK_MSG(in_place, "same-topology update should succeed in place");
  GIE_CUDA_CHECK(cudaMemset(b2.data.dy.data(), 0xFF, b2.data.dy.size()));
  g.launch(st.s);
  GIE_CUDA_CHECK(cudaStreamSynchronize(st.s));
  CHECK_MSG(gt::compare_device_output(b2.data.dy.data(), b2).passed(), "graph must write b2's output after the update");
}

TEST(graph_chain_of_layers_equals_eager_chain) {
  const ConvProblem prob = gt::make_problem(gt::P(1, 32, 12, 12, 32, 3, 3, 1, 1, 1, 1), Precision::kFp32, true, true);
  const KernelDesc* k = find_kernel("im2col_gemm_fp32");  // cuBLAS inside capture: workspace comes from the caller
  auto plan = ExecutionPlan::create(prob, *k);
  gt::RefBundle b = gt::make_bundle(prob, 9, Distribution::kNormal);
  DeviceBuffer a_buf(b.data.dy.size()), b_buf(b.data.dy.size()), ws(plan->workspace_bytes());
  auto chain = [&](cudaStream_t s) {
    void* src = a_buf.data();
    void* dst = b_buf.data();
    for (int l = 0; l < 4; ++l) {
      plan->execute_unchecked(src, b.data.dw.data(), b.data.db.data(), dst, ws.data(), ws.size(), s);
      std::swap(src, dst);
    }
  };
  Stream st;
  GIE_CUDA_CHECK(cudaMemcpy(a_buf.data(), b.data.dx.data(), b.data.dx.size(), cudaMemcpyDeviceToDevice));
  chain(st.s);
  GIE_CUDA_CHECK(cudaStreamSynchronize(st.s));
  const auto eager = download_as_float(a_buf.data(), DataType::kFloat32, prob.p.input_shape().numel());  // 4 layers: result back in a_buf
  CudaGraphExecutor g;
  g.capture(chain);
  for (int rep = 0; rep < 3; ++rep) {
    GIE_CUDA_CHECK(cudaMemcpy(a_buf.data(), b.data.dx.data(), b.data.dx.size(), cudaMemcpyDeviceToDevice));
    g.launch(st.s);
    GIE_CUDA_CHECK(cudaStreamSynchronize(st.s));
    CHECK(download_as_float(a_buf.data(), DataType::kFloat32, prob.p.input_shape().numel()) == eager);
  }
}

GT_MAIN()
