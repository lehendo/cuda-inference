// An ExecutionPlan binds one ConvProblem to one concrete kernel and owns all
// one-time setup (capability check, device tables, cuBLAS handle). Creating a
// plan may allocate and synchronize; execute() only launches work.
#pragma once

#include <cuda_runtime.h>

#include <memory>

#include "gpu_inference/kernel_registry.h"

namespace gie {

class ExecutionPlan {
 public:
  // Throws gie::Error(kUnsupported) if `kernel` cannot run `problem` on the device.
  static std::shared_ptr<ExecutionPlan> create(const ConvProblem& problem, const KernelDesc& kernel,
                                               int device = -1);
  ~ExecutionPlan();
  ExecutionPlan(const ExecutionPlan&) = delete;
  ExecutionPlan& operator=(const ExecutionPlan&) = delete;

  // Validates arguments (shapes, dtypes, null pointers, aliasing, workspace size) and
  // launches asynchronously on `stream`. Never synchronizes the device.
  void execute(const ConstTensorRef& x, const ConstTensorRef& w, const ConstTensorRef* bias,
               const TensorRef& y, void* workspace, size_t workspace_bytes, cudaStream_t stream,
               cudaEvent_t* stage_events = nullptr) const;

  // Raw-pointer fast path: assumes arguments were validated once by the caller
  // (used by benchmark/autotuner timing loops and CUDA Graph capture).
  void execute_unchecked(const void* x, const void* w, const void* bias, void* y, void* workspace,
                         size_t workspace_bytes, cudaStream_t stream,
                         cudaEvent_t* stage_events = nullptr) const;

  const ConvProblem& problem() const { return problem_; }
  const KernelDesc& kernel() const { return *kernel_; }
  size_t workspace_bytes() const { return workspace_bytes_; }       // per-call, caller-provided
  size_t persistent_bytes() const { return persistent_bytes_ + lut_bytes_; }  // plan-owned
  int device() const { return device_; }

 private:
  ExecutionPlan() = default;
  ConvProblem problem_;
  const KernelDesc* kernel_ = nullptr;
  int device_ = 0;
  size_t workspace_bytes_ = 0;
  size_t persistent_bytes_ = 0;
  size_t lut_bytes_ = 0;
  DeviceBuffer lut_;
  std::shared_ptr<void> state_;
};

}  // namespace gie
