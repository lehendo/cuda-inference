#include "gpu_inference/execution_plan.h"

#include "gpu_inference/nvtx.h"

#include <algorithm>
#include <sstream>

namespace gie {
namespace {

struct DeviceGuard {
  int prev = 0;
  explicit DeviceGuard(int dev) {
    GIE_CUDA_CHECK(cudaGetDevice(&prev));
    if (prev != dev) GIE_CUDA_CHECK(cudaSetDevice(dev));
  }
  ~DeviceGuard() { cudaSetDevice(prev); }
};

bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
  if (!a || !b || an == 0 || bn == 0) return false;
  const auto pa = reinterpret_cast<uintptr_t>(a), pb = reinterpret_cast<uintptr_t>(b);
  return pa < pb + bn && pb < pa + an;
}

void check_tensor(const char* what, const void* data, DataType dtype, const TensorShape& shape,
                  DataType want_dtype, const TensorShape& want_shape) {
  if (data == nullptr) throw Error(StatusCode::kInvalidArgument, std::string(what) + ": null device pointer");
  if (dtype != want_dtype)
    throw Error(StatusCode::kInvalidArgument, std::string(what) + ": dtype " + to_string(dtype) + " but the problem's precision policy needs " + to_string(want_dtype));
  if (shape != want_shape)
    throw Error(StatusCode::kInvalidArgument, std::string(what) + ": shape " + shape.to_string() + " but expected " + want_shape.to_string());
}

// Copies host data to the device and returns only when the copy has COMPLETED, ordering against nothing else.
// A private non-blocking stream is used so that (a) completion is guaranteed (a pageable cudaMemcpy may return
// before the DMA has landed, while callers later read the data from their own non-blocking streams, which do
// not order against the legacy default stream) and (b) unrelated work on any other stream is neither waited
// for nor delayed (unlike cudaDeviceSynchronize).
void upload_and_wait(void* dst, const void* src, size_t bytes) {
  cudaStream_t s = nullptr;
  GIE_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s);
  if (e == cudaSuccess) e = cudaStreamSynchronize(s);
  cudaStreamDestroy(s);
  GIE_CUDA_CHECK(e);
}

}  // namespace

std::shared_ptr<ExecutionPlan> ExecutionPlan::create(const ConvProblem& problem, const KernelDesc& kernel, int device) {
  NvtxRange range("gie::ExecutionPlan::create");
  validate_problem(problem);
  if (device < 0) GIE_CUDA_CHECK(cudaGetDevice(&device));
  DeviceGuard guard(device);
  const DeviceInfo& dev = DeviceInfo::get(device);

  std::string why;
  if (!kernel.supports(problem, dev, &why))
    throw Error(StatusCode::kUnsupported, kernel.name + " cannot run " + problem.key() + " on " + dev.name + ": " + why);

  std::shared_ptr<ExecutionPlan> plan(new ExecutionPlan());
  plan->problem_ = problem;
  plan->kernel_ = &kernel;
  plan->device_ = device;
  plan->workspace_bytes_ = kernel.workspace_bytes(problem);
  if (kernel.persistent_bytes) plan->persistent_bytes_ = kernel.persistent_bytes(problem);
  if (kernel.create_state) plan->state_ = kernel.create_state(problem);
  if (kernel.needs_lut) {
    const std::vector<LutEntry> lut = kernel.lut_chunked ? build_conv_lut_chunked(problem.p) : build_conv_lut(problem.p);
    plan->lut_ = DeviceBuffer(lut.size() * sizeof(LutEntry));
    upload_and_wait(plan->lut_.data(), lut.data(), lut.size() * sizeof(LutEntry));
    plan->lut_bytes_ = lut.size() * sizeof(LutEntry);
  }
  return plan;
}

ExecutionPlan::~ExecutionPlan() = default;

void ExecutionPlan::execute(const ConstTensorRef& x, const ConstTensorRef& w, const ConstTensorRef* bias,
                            const TensorRef& y, void* workspace, size_t workspace_bytes, cudaStream_t stream,
                            cudaEvent_t* stage_events) const {
  const Conv2DParams& p = problem_.p;
  const DataType dt = problem_.dtype();
  check_tensor("input", x.data, x.dtype, x.shape, dt, p.input_shape());
  check_tensor("weights", w.data, w.dtype, w.shape, dt, p.weight_shape());
  check_tensor("output", y.data, y.dtype, y.shape, dt, p.output_shape());
  const void* bias_ptr = nullptr;
  if (problem_.has_bias) {
    if (bias == nullptr) throw Error(StatusCode::kInvalidArgument, "problem has_bias=true but no bias tensor was given");
    check_tensor("bias", bias->data, bias->dtype, bias->shape, dt, p.bias_shape());
    bias_ptr = bias->data;
  } else if (bias != nullptr && bias->data != nullptr) {
    throw Error(StatusCode::kInvalidArgument, "a bias tensor was given but the problem has has_bias=false");
  }

  const size_t xb = size_t(p.input_shape().numel()) * dtype_size(dt);
  const size_t wb = size_t(p.weight_shape().numel()) * dtype_size(dt);
  const size_t yb = size_t(p.output_shape().numel()) * dtype_size(dt);
  const size_t bb = size_t(p.out_channels) * dtype_size(dt);
  if (overlaps(y.data, yb, x.data, xb) || overlaps(y.data, yb, w.data, wb) || overlaps(y.data, yb, bias_ptr, bb))
    throw Error(StatusCode::kInvalidArgument, "output aliases an input tensor (in-place convolution is not supported)");
  if (workspace_bytes_ > 0) {
    if (reinterpret_cast<uintptr_t>(workspace) % 256 != 0)
      throw Error(StatusCode::kInvalidArgument, "workspace must be 256-byte aligned (kernels place 16-byte vector buffers inside it)");
    if (workspace == nullptr || workspace_bytes < workspace_bytes_)
      throw Error(StatusCode::kWorkspaceTooSmall, kernel_->name + " needs " + std::to_string(workspace_bytes_) +
                                                      " bytes of workspace, got " + std::to_string(workspace_bytes));
    if (overlaps(workspace, workspace_bytes_, x.data, xb) || overlaps(workspace, workspace_bytes_, w.data, wb) ||
        overlaps(workspace, workspace_bytes_, y.data, yb))
      throw Error(StatusCode::kInvalidArgument, "workspace aliases a tensor");
  }
  int cur = 0;
  GIE_CUDA_CHECK(cudaGetDevice(&cur));
  if (cur != device_)
    throw Error(StatusCode::kInvalidArgument, "plan was created for device " + std::to_string(device_) + " but device " +
                                                  std::to_string(cur) + " is current");
  execute_unchecked(x.data, w.data, bias_ptr, y.data, workspace, workspace_bytes, stream, stage_events);
}

void ExecutionPlan::execute_unchecked(const void* x, const void* w, const void* bias, void* y, void* workspace,
                                      size_t workspace_bytes, cudaStream_t stream, cudaEvent_t* stage_events) const {
  NvtxRange range(kernel_->name.c_str());  // visible in Nsight Systems timelines
  LaunchArgs la;
  la.problem = &problem_;
  la.dev = &DeviceInfo::get(device_);
  la.x = x;
  la.w = w;
  la.bias = problem_.has_bias ? bias : nullptr;
  la.y = y;
  la.workspace = workspace;
  la.workspace_bytes = workspace_bytes;
  la.plan_data = lut_.data();
  la.plan_state = state_.get();
  la.stream = stream;
  la.stage_events = stage_events;
  kernel_->launch(la);
}

}  // namespace gie
