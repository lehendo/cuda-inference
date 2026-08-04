#include "gpu_inference/dispatcher.h"

#include <algorithm>
#include <sstream>

#include "gpu_inference/autotuner.h"
#include "gpu_inference/nvtx.h"
#include "gpu_inference/perf_model.h"

namespace gie {

const char* to_string(SelectionSource s) {
  switch (s) {
    case SelectionSource::kForcedKernel: return "forced_kernel";
    case SelectionSource::kForcedBackend: return "forced_backend";
    case SelectionSource::kTuningCache: return "tuning_cache";
    case SelectionSource::kTuned: return "tuned";
    case SelectionSource::kCostModel: return "cost_model";
  }
  return "?";
}

std::vector<const KernelDesc*> Dispatcher::compatible_kernels(const ConvProblem& prob, const DeviceInfo& dev,
                                                              Backend only, std::vector<Rejection>* rejected) {
  std::vector<const KernelDesc*> out;
  for (const KernelDesc& k : kernel_registry()) {
    if (only != Backend::kAuto && k.backend != only) continue;
    std::string why;
    if (k.supports(prob, dev, &why)) out.push_back(&k);
    else if (rejected) rejected->push_back({k.name, why});
  }
  return out;
}

std::vector<std::pair<const KernelDesc*, double>> Dispatcher::rank_by_model(
    const ConvProblem& prob, const DeviceInfo& dev, const std::vector<const KernelDesc*>& cands) {
  DeviceCeilings c;
  if (!load_cached_ceilings(dev.device, &c)) c = theoretical_ceilings(dev);
  std::vector<std::pair<const KernelDesc*, double>> ranked;
  for (const KernelDesc* k : cands) ranked.emplace_back(k, estimate_kernel(prob, *k, dev, c).time_us);
  std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
  return ranked;
}

Selection Dispatcher::select(const ConvProblem& prob, const DeviceInfo& dev, const Conv2DOptions& opt) {
  NvtxRange range("gie::Dispatcher::select");
  validate_problem(prob);
  Selection sel;

  if (!opt.kernel.empty()) {
    const KernelDesc* k = find_kernel(opt.kernel);
    if (!k) throw Error(StatusCode::kInvalidArgument, "unknown kernel '" + opt.kernel + "'");
    if (opt.backend != Backend::kAuto && k->backend != opt.backend)
      throw Error(StatusCode::kInvalidArgument, "kernel '" + opt.kernel + "' belongs to backend " + to_string(k->backend) + ", not " + to_string(opt.backend));
    std::string why;
    if (!k->supports(prob, dev, &why)) throw Error(StatusCode::kUnsupported, k->name + ": " + why);
    sel.kernel = k;
    sel.source = SelectionSource::kForcedKernel;
    sel.reason = "explicit kernel";
    return sel;
  }
  if (opt.backend == Backend::kCpuReference)
    throw Error(StatusCode::kUnsupported, "the CPU reference runs on host buffers via cpu_conv2d_reference(); it is not a GPU backend");

  std::vector<Rejection> rejected;
  std::vector<const KernelDesc*> cands = compatible_kernels(prob, dev, opt.backend, &rejected);
  if (cands.empty()) {
    std::ostringstream os;
    os << "no kernel can run " << prob.key() << " on " << dev.name << " (backend=" << to_string(opt.backend) << "):";
    for (const auto& r : rejected) os << "\n  - " << r.kernel << ": " << r.reason;
    throw Error(StatusCode::kUnsupported, os.str());
  }

  TuneCache cache(opt.cache_path);
  std::string cached_name;
  double cached_us = 0;
  if (!opt.force_retune && cache.lookup(prob, dev, &cached_name, &cached_us, nullptr, opt.backend)) {
    const KernelDesc* k = find_kernel(cached_name);
    if (k && std::find(cands.begin(), cands.end(), k) != cands.end()) {
      sel.kernel = k;
      sel.source = SelectionSource::kTuningCache;
      sel.predicted_us = cached_us;
      sel.reason = "measured winner from tuning cache";
      return sel;
    }
  }
  if (opt.autotune || opt.force_retune) {
    TuneOptions to;
    to.only_backend = opt.backend;
    to.cache_path = opt.cache_path;
    to.force_retune = opt.force_retune;
    TuneResult r = Autotuner::tune(prob, dev, to);
    sel.kernel = find_kernel(r.best_kernel);
    sel.source = SelectionSource::kTuned;
    sel.predicted_us = r.best_us;
    sel.reason = "autotuned";
    return sel;
  }
  const auto ranked = rank_by_model(prob, dev, cands);
  sel.kernel = ranked.front().first;
  sel.predicted_us = ranked.front().second;
  sel.source = opt.backend == Backend::kAuto ? SelectionSource::kCostModel : SelectionSource::kForcedBackend;
  sel.reason = "best-ranked by the analytical cost model (no measurement available)";
  return sel;
}

}  // namespace gie
