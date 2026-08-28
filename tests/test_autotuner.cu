// Autotuner + persistent cache: validation gating, cache hits, staleness, corruption, retuning.
#include <unistd.h>

#include <cstdio>
#include <fstream>

#include "conv_test_common.h"
#include "gpu_inference/autotuner.h"
#include "gpu_inference/json_lite.h"

using namespace gie;

namespace {

__global__ void garbage_kernel(float* y, int64_t n) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) y[i] = 0.f;
}

// A "kernel" that is extremely fast but wrong: it must never win the autotune.
void launch_garbage(const LaunchArgs& la) {
  const int64_t n = la.problem->p.output_shape().numel();
  garbage_kernel<<<64, 256, 0, la.stream>>>(static_cast<float*>(la.y), n);
}

void launch_throwing(const LaunchArgs&) { throw Error(StatusCode::kCudaError, "injected launch failure"); }

bool always(const ConvProblem& p, const DeviceInfo&, std::string*) { return p.precision == Precision::kFp32; }
size_t no_ws(const ConvProblem&) { return 0; }

void inject_faulty_kernels() {
  static bool done = false;
  if (done) return;
  done = true;
  KernelDesc bad;
  bad.name = "test_fast_but_wrong_fp32";
  bad.backend = Backend::kDirect;
  bad.supports = &always;
  bad.workspace_bytes = &no_ws;
  bad.launch = &launch_garbage;
  kernel_registry_for_testing().push_back(bad);
  KernelDesc thr = bad;
  thr.name = "test_throwing_fp32";
  thr.launch = &launch_throwing;
  kernel_registry_for_testing().push_back(thr);
}

std::string tmp_cache(const char* tag) {
  std::string p = std::string("/tmp/gie_test_cache_") + tag + "_" + std::to_string(::getpid()) + ".json";
  std::remove(p.c_str());
  return p;
}

TuneOptions fast_opts(const std::string& cache) {
  TuneOptions o;
  o.timing.warmup_iters = 2;
  o.timing.samples = 5;
  o.timing.min_sample_ms = 0.2;
  o.cache_path = cache;
  return o;
}

}  // namespace

TEST(incorrect_or_failing_candidates_are_never_selected) {
  inject_faulty_kernels();
  const DeviceInfo& dev = DeviceInfo::current();
  ConvProblem prob = gt::make_problem(gt::P(2, 16, 14, 14, 32, 3, 3, 1, 1, 1, 1), Precision::kFp32, true, true);
  TuneOptions o = fast_opts(tmp_cache("faulty"));
  TuneResult r = Autotuner::tune(prob, dev, o);
  CHECK(r.best_kernel != "test_fast_but_wrong_fp32");
  CHECK(r.best_kernel != "test_throwing_fp32");
  bool saw_wrong = false, saw_throw = false;
  for (const auto& c : r.candidates) {
    if (c.kernel == "test_fast_but_wrong_fp32") { saw_wrong = true; CHECK(!c.measured); CHECK(c.status.find("failed validation") == 0); }
    if (c.kernel == "test_throwing_fp32") { saw_throw = true; CHECK(!c.measured); CHECK(c.status.find("error:") == 0); }
  }
  CHECK(saw_wrong && saw_throw);
  // The reported winner must itself be correct.
  gt::RefBundle b = gt::make_bundle(prob, 12, Distribution::kNormal);
  bool ran; std::string why;
  ErrorMetrics m = gt::run_kernel(*find_kernel(r.best_kernel), b, &ran, &why);
  CHECK_MSG(ran && m.passed(), r.best_kernel << " " << m.summary());
  CHECK(r.best_us > 0);
}

TEST(search_space_and_cost_are_reported) {
  const DeviceInfo& dev = DeviceInfo::current();
  ConvProblem prob = gt::make_problem(gt::P(2, 32, 14, 14, 32, 3, 3, 1, 1, 1, 1), Precision::kFp16, true, true);
  TuneOptions o = fast_opts(tmp_cache("space"));
  TuneResult r = Autotuner::tune(prob, dev, o);
  CHECK(r.candidates_total >= 10);
  CHECK(r.candidates_measured == r.candidates_total);
  CHECK(r.search_seconds > 0);
  int measured = 0;
  for (const auto& c : r.candidates) if (c.measured) { ++measured; CHECK(c.timing.median_us > 0); CHECK(c.timing.samples == 5); }
  CHECK(measured >= 5);
}

TEST(cache_hit_skips_search_and_persists_across_instances) {
  const DeviceInfo& dev = DeviceInfo::current();
  ConvProblem prob = gt::make_problem(gt::P(1, 16, 12, 12, 16, 3, 3, 1, 1, 1, 1), Precision::kFp32, false, false);
  const std::string path = tmp_cache("hit");
  TuneOptions o = fast_opts(path);
  TuneResult cold = Autotuner::tune(prob, dev, o);
  CHECK(!cold.from_cache);
  TuneResult warm = Autotuner::tune(prob, dev, o);
  CHECK(warm.from_cache);
  CHECK_EQ(warm.best_kernel, cold.best_kernel);
  CHECK(warm.candidates.empty());
  TuneCache fresh(path);  // a new instance reads the file
  std::string name; double us = 0;
  CHECK(fresh.lookup(prob, dev, &name, &us));
  CHECK_EQ(name, cold.best_kernel);
  // force_retune ignores the cached entry
  o.force_retune = true;
  CHECK(!Autotuner::tune(prob, dev, o).from_cache);
  // clear() empties the cache
  fresh.clear();
  CHECK_EQ(fresh.size(), size_t(0));
  CHECK(!fresh.lookup(prob, dev, &name, &us));
  std::remove(path.c_str());
}

TEST(cache_key_separates_problems_precisions_and_fusion) {
  const DeviceInfo& dev = DeviceInfo::current();
  ConvProblem a = gt::make_problem(gt::P(1, 16, 12, 12, 16, 3, 3, 1, 1, 1, 1), Precision::kFp32, false, false);
  ConvProblem b = a; b.precision = Precision::kFp16;
  ConvProblem c = a; c.has_bias = true;
  ConvProblem d = a; d.p.batch = 2;
  const std::string ka = TuneCache::make_key(a, dev);
  CHECK(ka != TuneCache::make_key(b, dev));
  CHECK(ka != TuneCache::make_key(c, dev));
  CHECK(ka != TuneCache::make_key(d, dev));
  CHECK(ka.find(dev.name) != std::string::npos);
  CHECK(ka.find("cudart") != std::string::npos);
}

TEST(stale_and_corrupt_entries_are_rejected) {
  const DeviceInfo& dev = DeviceInfo::current();
  ConvProblem prob = gt::make_problem(gt::P(1, 16, 12, 12, 16, 3, 3, 1, 1, 1, 1), Precision::kFp32, false, false);
  const std::string path = tmp_cache("stale");
  TuneOptions o = fast_opts(path);
  TuneResult cold = Autotuner::tune(prob, dev, o);

  // 1. kernel version mismatch => entry ignored
  json::Value root = json::parse_file(path);
  json::Value& ents = root["entries"];
  for (const auto& kv : ents.items()) const_cast<json::Value&>(kv.second)["kernel_version"] = 999;
  json::write_file(path, root);
  std::string name; double us;
  CHECK(!TuneCache(path).lookup(prob, dev, &name, &us));

  // 2. unknown kernel name => ignored
  for (const auto& kv : ents.items()) { const_cast<json::Value&>(kv.second)["kernel_version"] = 1; const_cast<json::Value&>(kv.second)["kernel"] = "deleted_kernel"; }
  json::write_file(path, root);
  CHECK(!TuneCache(path).lookup(prob, dev, &name, &us));

  // 3. corrupt file => treated as empty (no crash), and a fresh tune repairs it
  { std::ofstream f(path); f << "{ this is not json"; }
  CHECK(!TuneCache(path).lookup(prob, dev, &name, &us));
  TuneResult again = Autotuner::tune(prob, dev, o);
  CHECK(!again.from_cache);
  CHECK(TuneCache(path).lookup(prob, dev, &name, &us));
  std::remove(path.c_str());
  (void)cold;
}

TEST(model_pruning_limits_search) {
  const DeviceInfo& dev = DeviceInfo::current();
  ConvProblem prob = gt::make_problem(gt::P(4, 64, 28, 28, 64, 3, 3, 1, 1, 1, 1), Precision::kFp32, true, true);
  TuneOptions o = fast_opts(tmp_cache("prune"));
  o.use_cache = false;
  o.model_keep = 4;
  TuneResult r = Autotuner::tune(prob, dev, o);
  CHECK_EQ(r.candidates_measured, 4);
  int attempted = 0;
  for (const auto& c : r.candidates) if (c.status.find("rejected") != 0) ++attempted;
  CHECK_EQ(attempted, 4);
}

TEST(backend_restricted_tuning) {
  const DeviceInfo& dev = DeviceInfo::current();
  ConvProblem prob = gt::make_problem(gt::P(2, 32, 14, 14, 32, 3, 3, 1, 1, 1, 1), Precision::kFp32, false, false);
  TuneOptions o = fast_opts(tmp_cache("only"));
  o.only_backend = Backend::kSimtTiled;
  TuneResult r = Autotuner::tune(prob, dev, o);
  CHECK(find_kernel(r.best_kernel)->backend == Backend::kSimtTiled);
}

TEST(tuning_is_reproducible_in_validation_and_seed) {
  // Same seed => same synthetic data => same correctness verdict for every candidate.
  const DeviceInfo& dev = DeviceInfo::current();
  ConvProblem prob = gt::make_problem(gt::P(2, 8, 10, 10, 8, 3, 3, 1, 1, 1, 1), Precision::kFp32, true, true);
  TuneOptions o = fast_opts("");
  o.use_cache = false;
  TuneResult a = Autotuner::tune(prob, dev, o), b = Autotuner::tune(prob, dev, o);
  CHECK_EQ(a.candidates.size(), b.candidates.size());
  for (size_t i = 0; i < a.candidates.size(); ++i) {
    CHECK_EQ(a.candidates[i].kernel, b.candidates[i].kernel);
    CHECK_EQ(a.candidates[i].measured, b.candidates[i].measured);
    CHECK_NEAR(a.candidates[i].max_abs_err, b.candidates[i].max_abs_err, 1e-12);
  }
}

GT_MAIN()
