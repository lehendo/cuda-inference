// Kernel autotuner and persistent tuning cache.
//
// Procedure per workload: enumerate registered kernels -> capability filter ->
// (optional) cost-model pruning -> correctness validation against a GPU
// double-precision reference -> warmup -> repeated event timing -> robust
// median -> persist winner. A candidate that fails validation or errors out is
// never selected.
#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "gpu_inference/dispatcher.h"
#include "gpu_inference/timing.h"

namespace gie {

struct TuneOptions {
  TimingOptions timing;                 // timing protocol for each candidate
  uint64_t seed = 0x5eed1234abcdULL;    // synthetic data seed (reproducible tuning)
  bool validate = true;                 // reject candidates that fail the reference check
  Backend only_backend = Backend::kAuto;// restrict the search to one backend
  int model_keep = 0;                   // >0: keep only the N best candidates per the cost model
  int max_candidates = 128;             // hard cap on the search space
  bool use_cache = true;
  bool force_retune = false;
  std::string cache_path;               // empty -> TuneCache::default_path()
  bool verbose = false;
};

struct CandidateResult {
  std::string kernel;
  Backend backend = Backend::kDirect;
  bool measured = false;     // timed (passed validation)
  std::string status;        // "ok", "rejected: ...", "failed validation: ...", "error: ..."
  double predicted_us = 0;   // cost-model estimate
  TimingStats timing;
  double max_abs_err = 0, frac_over_tol = 0;
};

struct TuneResult {
  ConvProblem problem;
  std::string best_kernel;
  double best_us = 0;
  bool from_cache = false;
  double search_seconds = 0;   // wall-clock cost of the search (cold tuning cost)
  int candidates_total = 0;    // after capability filtering
  int candidates_measured = 0; // after model pruning
  std::vector<CandidateResult> candidates;
};

// JSON cache. Keys include GPU model/cc, CUDA runtime+driver versions, the full
// problem description, precision and library version; entries also record the
// winning kernel's version. Stale or mismatching entries are ignored.
class TuneCache {
 public:
  static std::string default_path();  // $GIE_TUNE_CACHE or ~/.cache/gpu_inference/tune_cache.json
  explicit TuneCache(std::string path = "");
  // `scope` restricts the search that produced the entry (Backend::kAuto = all backends); entries of
  // different scopes never collide.
  static std::string make_key(const ConvProblem& prob, const DeviceInfo& dev, Backend scope = Backend::kAuto);

  // Returns true and fills `kernel`/`us` when a valid entry exists.
  bool lookup(const ConvProblem& prob, const DeviceInfo& dev, std::string* kernel, double* us,
              double* tuned_seconds = nullptr, Backend scope = Backend::kAuto) const;
  void store(const TuneResult& r, const DeviceInfo& dev, Backend scope = Backend::kAuto);
  void clear();   // removes all entries and the file contents
  size_t size() const;
  const std::string& path() const { return path_; }

 private:
  void load_locked() const;
  void save_locked() const;
  std::string path_;
  mutable std::mutex mu_;
  struct Entry {
    std::string kernel;
    int kernel_version = 0;
    double us = 0;
    double tuned_seconds = 0;
    std::string timestamp;
  };
  mutable std::map<std::string, Entry> entries_;
  mutable bool loaded_ = false;
};

class Autotuner {
 public:
  static TuneResult tune(const ConvProblem& prob, const DeviceInfo& dev, const TuneOptions& opt = {});
};

}  // namespace gie
