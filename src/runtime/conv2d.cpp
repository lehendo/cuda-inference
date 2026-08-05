#include "gpu_inference/conv2d.h"

#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <tuple>

#include "gpu_inference/autotuner.h"
#include "gpu_inference/dispatcher.h"
#include "gpu_inference/execution_plan.h"

namespace gie {

DataType tensor_dtype(Precision p) {
  switch (p) {
    case Precision::kFp32:
    case Precision::kTf32: return DataType::kFloat32;
    case Precision::kFp16: return DataType::kFloat16;
    case Precision::kBf16: return DataType::kBFloat16;
  }
  return DataType::kFloat32;
}

const char* to_string(Precision p) {
  switch (p) {
    case Precision::kFp32: return "fp32";
    case Precision::kFp16: return "fp16";
    case Precision::kBf16: return "bf16";
    case Precision::kTf32: return "tf32";
  }
  return "?";
}

Precision precision_from_string(const std::string& s) {
  if (s == "fp32") return Precision::kFp32;
  if (s == "fp16") return Precision::kFp16;
  if (s == "bf16") return Precision::kBf16;
  if (s == "tf32") return Precision::kTf32;
  throw Error(StatusCode::kInvalidArgument, "unknown precision '" + s + "' (fp32|fp16|bf16|tf32)");
}

const char* to_string(Activation a) { return a == Activation::kReLU ? "relu" : "none"; }

const char* to_string(Backend b) {
  switch (b) {
    case Backend::kAuto: return "auto";
    case Backend::kCpuReference: return "cpu_reference";
    case Backend::kDirect: return "direct";
    case Backend::kIm2colGemm: return "im2col_gemm";
    case Backend::kSimtTiled: return "simt_tiled";
    case Backend::kImplicitGemm: return "implicit_gemm";
    case Backend::kTensorCore: return "tensor_core";
  }
  return "?";
}

Backend backend_from_string(const std::string& s) {
  for (Backend b : {Backend::kAuto, Backend::kCpuReference, Backend::kDirect, Backend::kIm2colGemm,
                    Backend::kSimtTiled, Backend::kImplicitGemm, Backend::kTensorCore}) {
    if (s == to_string(b)) return b;
  }
  throw Error(StatusCode::kInvalidArgument, "unknown backend '" + s + "'");
}

std::string Conv2DParams::to_string() const {
  std::ostringstream os;
  os << "B" << batch << "_C" << in_channels << "_" << in_h << "x" << in_w << "_O" << out_channels << "_K"
     << kernel_h << "x" << kernel_w << "_s" << stride_h << "x" << stride_w << "_p" << pad_h << "x" << pad_w
     << "_d" << dilation_h << "x" << dilation_w << "_g" << groups;
  return os.str();
}

bool Conv2DParams::operator==(const Conv2DParams& o) const { return to_string() == o.to_string(); }

std::string ConvProblem::key() const {
  std::ostringstream os;
  os << p.to_string() << "_" << gie::to_string(precision) << "_b" << int(has_bias) << "_" << gie::to_string(activation);
  return os.str();
}

void validate_problem(const ConvProblem& prob) {
  const Conv2DParams& p = prob.p;
  auto bad = [](const std::string& m) { throw Error(StatusCode::kInvalidArgument, m); };
  if (p.batch <= 0 || p.in_channels <= 0 || p.in_h <= 0 || p.in_w <= 0 || p.out_channels <= 0 ||
      p.kernel_h <= 0 || p.kernel_w <= 0)
    bad("all extents must be positive: " + p.to_string());
  if (p.stride_h <= 0 || p.stride_w <= 0) bad("strides must be positive");
  if (p.pad_h < 0 || p.pad_w < 0) bad("padding must be non-negative");
  if (p.dilation_h != 1 || p.dilation_w != 1)
    throw Error(StatusCode::kUnsupported, "dilation != 1 is not supported yet");
  if (p.groups != 1) throw Error(StatusCode::kUnsupported, "grouped convolution is not supported yet");
  // The effective (dilated) kernel must produce at least one output position.
  const int64_t eff_h = int64_t(p.dilation_h) * (p.kernel_h - 1) + 1;
  const int64_t eff_w = int64_t(p.dilation_w) * (p.kernel_w - 1) + 1;
  if (int64_t(p.in_h) + 2 * p.pad_h < eff_h || int64_t(p.in_w) + 2 * p.pad_w < eff_w)
    bad("kernel larger than padded input: " + p.to_string());
  if (p.out_h() <= 0 || p.out_w() <= 0) bad("empty output: " + p.to_string());
  // Index-overflow safeguards. Kernels use 32-bit indices for GEMM coordinates
  // (m, n, k) and per-image offsets, and 64-bit arithmetic for global tensor offsets.
  const int64_t lim = std::numeric_limits<int32_t>::max();
  if (p.gemm_n() > lim || p.gemm_k() > lim || p.gemm_m() > lim)
    bad("GEMM extents exceed 32-bit indexing: " + p.to_string());
  if (int64_t(p.in_channels) * p.in_h * p.in_w > lim)
    bad("per-image input size exceeds 32-bit indexing: " + p.to_string());
  if (int64_t(p.out_channels) * p.out_h() * p.out_w() > lim)
    bad("per-image output size exceeds 32-bit indexing: " + p.to_string());
}

namespace {

struct PlanKey {
  std::string problem, backend, kernel;
  int device;
  bool autotune, retune;
  std::string cache;
  bool operator<(const PlanKey& o) const {
    return std::tie(problem, backend, kernel, device, autotune, retune, cache) <
           std::tie(o.problem, o.backend, o.kernel, o.device, o.autotune, o.retune, o.cache);
  }
};

std::mutex g_plan_mu;
std::map<PlanKey, std::shared_ptr<ExecutionPlan>>& plan_cache() {
  static std::map<PlanKey, std::shared_ptr<ExecutionPlan>> c;
  return c;
}

std::shared_ptr<ExecutionPlan> get_plan(const ConvProblem& prob, const Conv2DOptions& opt) {
  validate_problem(prob);
  int dev = 0;
  GIE_CUDA_CHECK(cudaGetDevice(&dev));
  PlanKey key{prob.key(), to_string(opt.backend), opt.kernel, dev, opt.autotune, opt.force_retune, opt.cache_path};
  {
    std::lock_guard<std::mutex> lock(g_plan_mu);
    auto it = plan_cache().find(key);
    if (it != plan_cache().end()) return it->second;
  }
  // Slow path: selection (may autotune) and setup happen outside the lock.
  const DeviceInfo& di = DeviceInfo::get(dev);
  Selection sel = Dispatcher::select(prob, di, opt);
  std::shared_ptr<ExecutionPlan> plan = ExecutionPlan::create(prob, *sel.kernel, dev);
  std::lock_guard<std::mutex> lock(g_plan_mu);
  return plan_cache().emplace(key, plan).first->second;
}

}  // namespace

void Conv2D::forward(const ConstTensorRef& input, const ConstTensorRef& weights, const ConstTensorRef* bias,
                     const ConvProblem& problem, const TensorRef& output, void* workspace,
                     size_t workspace_bytes, cudaStream_t stream, const Conv2DOptions& options) {
  auto plan = get_plan(problem, options);
  plan->execute(input, weights, bias, output, workspace, workspace_bytes, stream);
}

size_t Conv2D::workspace_size(const ConvProblem& problem, const Conv2DOptions& options) {
  return get_plan(problem, options)->workspace_bytes();
}

void Conv2D::clear_plan_cache() {
  std::lock_guard<std::mutex> lock(g_plan_mu);
  plan_cache().clear();
}

}  // namespace gie
