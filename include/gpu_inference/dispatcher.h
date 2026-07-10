// Kernel dispatcher: capability checking -> configuration selection.
// (Workspace planning and execution live in ExecutionPlan.)
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "gpu_inference/kernel_registry.h"

namespace gie {

enum class SelectionSource {
  kForcedKernel,   // Conv2DOptions::kernel
  kForcedBackend,  // Conv2DOptions::backend with the best-ranked config of that backend
  kTuningCache,    // measured winner from the persisted autotuning cache
  kTuned,          // autotuned just now
  kCostModel,      // ranked by the analytical model (no measurement available)
};
const char* to_string(SelectionSource s);

struct Selection {
  const KernelDesc* kernel = nullptr;
  SelectionSource source = SelectionSource::kCostModel;
  double predicted_us = 0;  // model estimate for the chosen kernel (0 if n/a)
  std::string reason;
};

struct Rejection {
  std::string kernel;
  std::string reason;
};

class Dispatcher {
 public:
  // Step 1: capability checking. Returns every registered kernel (optionally
  // restricted to one backend) that can run `prob` on `dev`; rejected candidates
  // are reported with the reason instead of being silently dropped.
  static std::vector<const KernelDesc*> compatible_kernels(const ConvProblem& prob,
                                                           const DeviceInfo& dev,
                                                           Backend only = Backend::kAuto,
                                                           std::vector<Rejection>* rejected = nullptr);

  // Step 2: configuration selection. Priority: explicit kernel > tuning cache >
  // (autotune if requested) > cost-model ranking. Throws kUnsupported with the
  // collected rejection reasons when nothing can run the problem.
  static Selection select(const ConvProblem& prob, const DeviceInfo& dev, const Conv2DOptions& opt);

  // Cost-model ranking of compatible kernels, best first. Ceilings are the
  // measured ones when cached for this device, otherwise theoretical.
  static std::vector<std::pair<const KernelDesc*, double>> rank_by_model(
      const ConvProblem& prob, const DeviceInfo& dev, const std::vector<const KernelDesc*>& cands);
};

}  // namespace gie
