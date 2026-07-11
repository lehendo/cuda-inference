// CUDA Graph capture/replay of an arbitrary sequence of asynchronous launches (typically a chain of
// ExecutionPlan::execute_unchecked calls). Graphs remove per-launch CPU cost and inter-kernel launch gaps,
// which matters for small, launch-bound layers. The core library never requires them.
#pragma once

#include <cuda_runtime.h>

#include <functional>

#include "gpu_inference/error.h"

namespace gie {

class CudaGraphExecutor {
 public:
  using Enqueue = std::function<void(cudaStream_t)>;  // must not allocate, synchronize or call cudaMalloc/cudaFree

  CudaGraphExecutor() = default;
  ~CudaGraphExecutor();
  CudaGraphExecutor(const CudaGraphExecutor&) = delete;
  CudaGraphExecutor& operator=(const CudaGraphExecutor&) = delete;

  // Runs `enqueue` once eagerly (warms lazy library initialization, which is illegal during capture),
  // then captures it into a graph and instantiates an executable graph. Throws on failure.
  void capture(const Enqueue& enqueue);

  // Re-captures `enqueue` (e.g. after the caller switched to different input/output buffers of the same
  // shapes) and updates the existing executable graph in place when the topology is unchanged
  // (cudaGraphExecUpdate); falls back to a full re-instantiation. Returns true if updated in place.
  bool recapture_update(const Enqueue& enqueue);

  void launch(cudaStream_t stream) const;

  bool valid() const { return exec_ != nullptr; }
  size_t node_count() const { return nodes_; }
  double capture_us() const { return capture_us_; }
  double instantiate_us() const { return instantiate_us_; }

 private:
  void capture_graph(const Enqueue& enqueue, cudaGraph_t* g, double* us);
  void reset();
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  size_t nodes_ = 0;
  double capture_us_ = 0, instantiate_us_ = 0;
};

}  // namespace gie
