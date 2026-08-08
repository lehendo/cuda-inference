#include "gpu_inference/autotuner.h"

#include <algorithm>
#include <chrono>
#include <iostream>

#include "gpu_inference/execution_plan.h"
#include "gpu_inference/nvtx.h"
#include "gpu_inference/perf_model.h"
#include "gpu_inference/reference.h"
#include "gpu_inference/testing.h"

namespace gie {

TuneResult Autotuner::tune(const ConvProblem& prob, const DeviceInfo& dev, const TuneOptions& opt) {
  NvtxRange range("gie::Autotuner::tune");
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  validate_problem(prob);

  TuneResult res;
  res.problem = prob;

  TuneCache cache(opt.cache_path);
  if (opt.use_cache && !opt.force_retune) {
    std::string name;
    double us = 0, cold_s = 0;
    if (cache.lookup(prob, dev, &name, &us, &cold_s, opt.only_backend)) {
      const KernelDesc* k = find_kernel(name);
      if (k) {
        res.best_kernel = name;
        res.best_us = us;
        res.from_cache = true;
        res.search_seconds = cold_s;  // cold cost recorded when the entry was created
        return res;
      }
    }
  }

  int prev_dev = 0;
  GIE_CUDA_CHECK(cudaGetDevice(&prev_dev));
  if (prev_dev != dev.device) GIE_CUDA_CHECK(cudaSetDevice(dev.device));
  struct Restore {
    int d;
    ~Restore() { cudaSetDevice(d); }
  } restore{prev_dev};

  // 1. Capability filter (hardware limits, precision, grid limits ...).
  std::vector<Rejection> rejected;
  std::vector<const KernelDesc*> cands = Dispatcher::compatible_kernels(prob, dev, opt.only_backend, &rejected);
  for (const Rejection& r : rejected) {
    CandidateResult cr;
    cr.kernel = r.kernel;
    cr.status = "rejected: " + r.reason;
    res.candidates.push_back(cr);
  }
  res.candidates_total = int(cands.size());
  if (cands.empty()) throw Error(StatusCode::kUnsupported, "autotune: no compatible kernel for " + prob.key());

  // 2. Cost-model ranking and optional pruning.
  auto ranked = Dispatcher::rank_by_model(prob, dev, cands);
  if (opt.model_keep > 0 && int(ranked.size()) > opt.model_keep) ranked.resize(size_t(opt.model_keep));
  if (int(ranked.size()) > opt.max_candidates) ranked.resize(size_t(opt.max_candidates));

  // 3. Shared synthetic data and the double-precision GPU reference (computed once).
  ConvData data = make_conv_data(prob, opt.seed, Distribution::kUniform);
  const int64_t out_n = prob.p.output_shape().numel();
  DeviceBuffer d_ref(size_t(out_n) * 4), d_scale(size_t(out_n) * 4);
  std::vector<float> h_ref(static_cast<size_t>(out_n)), h_scale(static_cast<size_t>(out_n));
  cudaStream_t stream;
  GIE_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  struct StreamGuard {
    cudaStream_t s;
    ~StreamGuard() { cudaStreamDestroy(s); }
  } sg{stream};
  if (opt.validate) {
    gpu_conv2d_reference(prob, data.dx.data(), data.dw.data(), data.db.data(), static_cast<float*>(d_ref.data()),
                         static_cast<float*>(d_scale.data()), stream);
    GIE_CUDA_CHECK(cudaStreamSynchronize(stream));
    GIE_CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref.data(), size_t(out_n) * 4, cudaMemcpyDeviceToHost));
    GIE_CUDA_CHECK(cudaMemcpy(h_scale.data(), d_scale.data(), size_t(out_n) * 4, cudaMemcpyDeviceToHost));
  }

  // 4. Validate, then time, each candidate.
  const void* bias = prob.has_bias ? data.db.data() : nullptr;
  double best_us = 0;
  for (const auto& rk : ranked) {
    const KernelDesc& k = *rk.first;
    CandidateResult cr;
    cr.kernel = k.name;
    cr.backend = k.backend;
    cr.predicted_us = rk.second;
    try {
      auto plan = ExecutionPlan::create(prob, k, dev.device);
      DeviceBuffer ws(plan->workspace_bytes());
      GIE_CUDA_CHECK(cudaMemsetAsync(data.dy.data(), 0xFF, data.dy.size(), stream));
      plan->execute_unchecked(data.dx.data(), data.dw.data(), bias, data.dy.data(), ws.data(), ws.size(), stream);
      GIE_CUDA_CHECK(cudaStreamSynchronize(stream));
      GIE_CUDA_CHECK(cudaGetLastError());
      bool ok = true;
      if (opt.validate) {
        const std::vector<float> out = download_as_float(data.dy.data(), prob.dtype(), out_n);
        const bool double_round = k.backend == Backend::kIm2colGemm && prob.precision != Precision::kFp32 &&
                                  (prob.has_bias || prob.activation != Activation::kNone);
        const ErrorMetrics m = compare_outputs(out.data(), h_ref.data(), h_scale.data(), out_n,
                                               TolerancePolicy::for_problem(prob, double_round));
        cr.max_abs_err = m.max_abs;
        cr.frac_over_tol = m.frac_over_tol;
        if (!m.passed()) {
          ok = false;
          cr.status = "failed validation: " + m.summary();
        }
      }
      if (ok) {
        cr.timing = time_stream_work(
            [&](cudaStream_t s) { plan->execute_unchecked(data.dx.data(), data.dw.data(), bias, data.dy.data(), ws.data(), ws.size(), s); },
            stream, opt.timing);
        GIE_CUDA_CHECK(cudaGetLastError());
        cr.measured = true;
        cr.status = "ok";
        if (res.best_kernel.empty() || cr.timing.median_us < best_us) {
          best_us = cr.timing.median_us;
          res.best_kernel = k.name;
        }
      }
    } catch (const Error& e) {
      cr.status = std::string("error: ") + e.what();
    }
    if (opt.verbose)
      std::cerr << "[tune] " << prob.key() << " " << k.name << ": " << cr.status << (cr.measured ? " " + std::to_string(cr.timing.median_us) + " us" : "") << "\n";
    res.candidates.push_back(std::move(cr));
  }
  res.candidates_measured = int(ranked.size());

  if (res.best_kernel.empty()) throw Error(StatusCode::kUnsupported, "autotune: every candidate failed for " + prob.key());
  res.best_us = best_us;
  res.search_seconds = std::chrono::duration<double>(clock::now() - t0).count();
  if (opt.use_cache) cache.store(res, dev, opt.only_backend);
  return res;
}

}  // namespace gie
