#include "gpu_inference/cuda_graph.h"

#include "gpu_inference/nvtx.h"

#include <chrono>

namespace gie {
namespace {
using clock = std::chrono::steady_clock;
double us_since(clock::time_point t0) { return std::chrono::duration<double, std::micro>(clock::now() - t0).count(); }
}  // namespace

CudaGraphExecutor::~CudaGraphExecutor() { reset(); }

void CudaGraphExecutor::reset() {
  if (exec_) cudaGraphExecDestroy(exec_);
  if (graph_) cudaGraphDestroy(graph_);
  exec_ = nullptr;
  graph_ = nullptr;
  nodes_ = 0;
}

void CudaGraphExecutor::capture_graph(const Enqueue& enqueue, cudaGraph_t* g, double* us) {
  cudaStream_t s;
  GIE_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  struct Guard {
    cudaStream_t s;
    ~Guard() { cudaStreamDestroy(s); }
  } guard{s};
  const auto t0 = clock::now();
  GIE_CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  try {
    enqueue(s);
  } catch (...) {
    cudaGraph_t partial = nullptr;
    cudaStreamEndCapture(s, &partial);  // leave the stream usable, drop the partial graph
    if (partial) cudaGraphDestroy(partial);
    throw;
  }
  GIE_CUDA_CHECK(cudaStreamEndCapture(s, g));
  *us = us_since(t0);
}

void CudaGraphExecutor::capture(const Enqueue& enqueue) {
  NvtxRange range("gie::CudaGraphExecutor::capture");
  reset();
  cudaStream_t warm;
  GIE_CUDA_CHECK(cudaStreamCreateWithFlags(&warm, cudaStreamNonBlocking));
  try {
    enqueue(warm);
    GIE_CUDA_CHECK(cudaStreamSynchronize(warm));
  } catch (...) {
    cudaStreamDestroy(warm);
    throw;
  }
  cudaStreamDestroy(warm);
  capture_graph(enqueue, &graph_, &capture_us_);
  const auto t0 = clock::now();
  GIE_CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, 0));
  instantiate_us_ = us_since(t0);
  size_t n = 0;
  GIE_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &n));
  nodes_ = n;
}

bool CudaGraphExecutor::recapture_update(const Enqueue& enqueue) {
  if (!exec_) {
    capture(enqueue);
    return false;
  }
  cudaGraph_t g2 = nullptr;
  double us = 0;
  capture_graph(enqueue, &g2, &us);
  capture_us_ = us;
  cudaGraphExecUpdateResultInfo info;
  const cudaError_t e = cudaGraphExecUpdate(exec_, g2, &info);
  if (e == cudaSuccess && info.result == cudaGraphExecUpdateSuccess) {
    cudaGraphDestroy(graph_);
    graph_ = g2;
    return true;
  }
  (void)cudaGetLastError();
  cudaGraphExecDestroy(exec_);
  exec_ = nullptr;
  cudaGraphDestroy(graph_);
  graph_ = g2;
  const auto t0 = clock::now();
  GIE_CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, 0));
  instantiate_us_ = us_since(t0);
  return false;
}

void CudaGraphExecutor::launch(cudaStream_t stream) const {
  NvtxRange range("gie::CudaGraphExecutor::launch");
  GIE_REQUIRE(exec_ != nullptr, StatusCode::kInvalidArgument, "CudaGraphExecutor::launch before capture()");
  GIE_CUDA_CHECK(cudaGraphLaunch(exec_, stream));
}

}  // namespace gie
