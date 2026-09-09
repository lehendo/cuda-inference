// gie_tune: autotuning driver and Experiment B (search strategies).
//
//   gie_tune --clear-cache [--cache FILE]
//   gie_tune tune          --workloads F --precisions fp32,fp16 --out DIR
//   gie_tune experiment-b  --tuning-set F --heldout-set F --precisions ... --out DIR [--keep 1,3,5]
//
// Experiment B compares, per workload, three ways of choosing a kernel:
//   fixed       : one manually chosen configuration per precision (what a hard-coded library does)
//   exhaustive  : measure every capability-valid candidate (the oracle, with its search cost)
//   model-guided: measure only the top-K candidates ranked by the analytical cost model
// The model's per-backend scale factors are fitted ONLY on the tuning set (median measured/predicted
// ratio) and then evaluated on the held-out set, so held-out numbers are an out-of-sample test.
// Everything measured here is appended to DIR/*.jsonl; nothing is printed that is not in those files.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>

#include "gpu_inference/autotuner.h"
#include "gpu_inference/json_lite.h"
#include "gpu_inference/perf_model.h"

using namespace gie;

namespace {

struct W { std::string name, category; Conv2DParams p; bool bias = true, relu = true; };

std::vector<W> load(const std::string& path) {
  const json::Value root = json::parse_file(path);
  std::vector<W> out;
  for (const json::Value& w : root.find("workloads")->elements()) {
    auto geti = [&](const char* k, int def) { const json::Value* v = w.find(k); return v ? int(v->as_int()) : def; };
    W x;
    x.name = w.find("name")->as_string();
    x.category = w.find("category") ? w.find("category")->as_string() : "";
    Conv2DParams& p = x.p;
    p.batch = geti("B", 1); p.in_channels = geti("C", 1); p.in_h = geti("H", 1); p.in_w = geti("W", p.in_h);
    p.out_channels = geti("O", p.in_channels);
    p.kernel_h = geti("Kh", geti("K", 3)); p.kernel_w = geti("Kw", geti("K", 3));
    p.stride_h = geti("Sh", geti("S", 1)); p.stride_w = geti("Sw", geti("S", 1));
    p.pad_h = geti("Ph", geti("P", 0)); p.pad_w = geti("Pw", geti("P", 0));
    x.bias = w.find("bias") ? w.find("bias")->as_bool(true) : true;
    x.relu = w.find("relu") ? w.find("relu")->as_bool(true) : true;
    out.push_back(x);
  }
  return out;
}

std::vector<std::string> csv(const std::string& s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(t);
  return v;
}

double spearman(const std::vector<double>& a, const std::vector<double>& b) {
  const size_t n = a.size();
  if (n < 3) return std::nan("");
  auto ranks = [](const std::vector<double>& v) {
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t i, size_t j) { return v[i] < v[j]; });
    std::vector<double> r(v.size());
    for (size_t k = 0; k < idx.size();) {  // average ranks for ties
      size_t e = k;
      while (e + 1 < idx.size() && v[idx[e + 1]] == v[idx[k]]) ++e;
      for (size_t m = k; m <= e; ++m) r[idx[m]] = 0.5 * double(k + e);
      k = e + 1;
    }
    return r;
  };
  const auto ra = ranks(a), rb = ranks(b);
  const double ma = std::accumulate(ra.begin(), ra.end(), 0.0) / double(n), mb = std::accumulate(rb.begin(), rb.end(), 0.0) / double(n);
  double sab = 0, saa = 0, sbb = 0;
  for (size_t i = 0; i < n; ++i) { sab += (ra[i] - ma) * (rb[i] - mb); saa += (ra[i] - ma) * (ra[i] - ma); sbb += (rb[i] - mb) * (rb[i] - mb); }
  return (saa > 0 && sbb > 0) ? sab / std::sqrt(saa * sbb) : std::nan("");
}

std::string fixed_name(Precision p) {
  const std::string pn = to_string(p);
  if (p == Precision::kFp16 || p == Precision::kBf16) return "tensor_core_" + pn + "_128x128x32_w2x4";
  return "implicit_gemm_" + pn + "_128x128x8_t8x8";
}

struct Args {
  std::string cmd, workloads, tuning_set, heldout_set, out = "results/raw/tune", cache;
  std::vector<Precision> precisions{Precision::kFp32, Precision::kFp16};
  std::vector<int> keeps{1, 3, 5, 8};
  int samples = 7;
  bool clear = false;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto need = [&]() -> std::string { if (i + 1 >= argc) throw Error(StatusCode::kInvalidArgument, "missing value for " + s); return argv[++i]; };
    if (s == "tune" || s == "experiment-b") a.cmd = s;
    else if (s == "--clear-cache") a.clear = true;
    else if (s == "--workloads") a.workloads = need();
    else if (s == "--tuning-set") a.tuning_set = need();
    else if (s == "--heldout-set") a.heldout_set = need();
    else if (s == "--out") a.out = need();
    else if (s == "--cache") a.cache = need();
    else if (s == "--samples") a.samples = std::stoi(need());
    else if (s == "--precisions") { a.precisions.clear(); for (auto& t : csv(need())) a.precisions.push_back(precision_from_string(t)); }
    else if (s == "--keep") { a.keeps.clear(); for (auto& t : csv(need())) a.keeps.push_back(std::stoi(t)); }
    else throw Error(StatusCode::kInvalidArgument, "unknown argument " + s);
  }
  return a;
}

ConvProblem to_problem(const W& w, Precision p) {
  ConvProblem pr;
  pr.p = w.p;
  pr.precision = p;
  pr.has_bias = w.bias;
  pr.activation = w.relu ? Activation::kReLU : Activation::kNone;
  return pr;
}

TuneOptions base_opts(const Args& a) {
  TuneOptions o;
  o.timing.samples = a.samples;
  o.timing.warmup_iters = 3;
  o.timing.min_sample_ms = 0.5;
  o.use_cache = false;  // experiments measure the search itself; the cache must not short-circuit it
  o.cache_path = a.cache;
  return o;
}

// One exhaustive search over all valid candidates (the oracle) with every candidate's timing kept.
struct Oracle {
  TuneResult tr;
  std::map<std::string, double> us;      // kernel -> measured median
  std::map<std::string, double> pred;    // kernel -> model prediction (current calibration)
};

Oracle exhaustive(const ConvProblem& prob, const DeviceInfo& dev, const Args& a) {
  Oracle o;
  o.tr = Autotuner::tune(prob, dev, base_opts(a));
  for (const auto& c : o.tr.candidates)
    if (c.measured) { o.us[c.kernel] = c.timing.median_us; o.pred[c.kernel] = c.predicted_us; }
  return o;
}

json::Value experiment_workload(const W& w, Precision prec, const DeviceInfo& dev, const Args& a,
                                const std::string& split, const std::map<std::string, double>& cal, Oracle* keep_oracle) {
  const ConvProblem prob = to_problem(w, prec);
  json::Value r = json::Value::object();
  r["split"] = split;
  r["workload"] = w.name;
  r["category"] = w.category;
  r["precision"] = to_string(prec);
  r["gpu"] = dev.name;
  Oracle orc = exhaustive(prob, dev, a);
  r["exhaustive_best_kernel"] = orc.tr.best_kernel;
  r["exhaustive_best_us"] = orc.tr.best_us;
  r["exhaustive_search_seconds"] = orc.tr.search_seconds;
  r["candidates_valid"] = orc.tr.candidates_total;
  r["candidates_measured"] = int(orc.us.size());
  json::Value cands = json::Value::array();
  for (const auto& c : orc.tr.candidates) {
    json::Value cj = json::Value::object();
    cj["kernel"] = c.kernel;
    cj["status"] = c.status;
    if (c.measured) { cj["median_us"] = c.timing.median_us; cj["predicted_us"] = c.predicted_us; }
    cands.push_back(cj);
  }
  r["candidates"] = cands;

  // fixed manual configuration
  const KernelDesc* fk = find_kernel(fixed_name(prec));
  if (fk && orc.us.count(fk->name)) { r["fixed_kernel"] = fk->name; r["fixed_us"] = orc.us[fk->name]; r["fixed_over_best"] = orc.us[fk->name] / orc.tr.best_us; }
  else r["fixed_kernel"] = fk ? fk->name + " (unsupported or failed for this workload)" : "none";

  // model-guided top-K (uncalibrated and calibrated)
  for (int pass = 0; pass < 2; ++pass) {
    set_model_calibration(pass == 0 ? std::map<std::string, double>{} : cal);
    if (pass == 1 && cal.empty()) break;
    const std::string tag = pass == 0 ? "prior" : "calibrated";
    std::vector<const KernelDesc*> valid = Dispatcher::compatible_kernels(prob, dev);
    auto ranked = Dispatcher::rank_by_model(prob, dev, valid);
    // rank correlation between model predictions and measurements over the measured candidates
    std::vector<double> p_us, m_us;
    for (const auto& kv : ranked) {
      auto it = orc.us.find(kv.first->name);
      if (it != orc.us.end()) { p_us.push_back(kv.second); m_us.push_back(it->second); }
    }
    r["spearman_" + tag] = spearman(p_us, m_us);
    int top1_rank_of_best = 0;
    for (size_t i = 0; i < ranked.size(); ++i)
      if (ranked[i].first->name == orc.tr.best_kernel) { top1_rank_of_best = int(i) + 1; break; }
    r["rank_of_true_best_" + tag] = top1_rank_of_best;
    for (int k : a.keeps) {
      // Search cost = actually run the pruned search so the saving is measured, not extrapolated.
      TuneOptions to = base_opts(a);
      to.model_keep = k;
      const TuneResult tr = Autotuner::tune(prob, dev, to);
      const std::string key = "model_" + tag + "_k" + std::to_string(k);
      r[key + "_kernel"] = tr.best_kernel;
      r[key + "_search_seconds"] = tr.search_seconds;
      auto it = orc.us.find(tr.best_kernel);
      // re-expressed against the oracle's timing of the same kernel (same measurement conditions)
      const double us_oracle_conditions = it != orc.us.end() ? it->second : tr.best_us;
      r[key + "_us"] = us_oracle_conditions;
      r[key + "_over_best"] = us_oracle_conditions / orc.tr.best_us;
    }
  }
  set_model_calibration({});
  if (keep_oracle) *keep_oracle = std::move(orc);
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args a = parse(argc, argv);
    int dev_id = 0;
    GIE_CUDA_CHECK(cudaGetDevice(&dev_id));
    const DeviceInfo& dev = DeviceInfo::get(dev_id);
    if (a.clear) {
      TuneCache c(a.cache);
      c.clear();
      std::cout << "cleared tuning cache " << c.path() << "\n";
      if (a.cmd.empty()) return 0;
    }
    const std::string mk = "mkdir -p '" + a.out + "'";
    if (std::system(mk.c_str()) != 0) throw Error(StatusCode::kIo, "cannot create " + a.out);
    std::ofstream out(a.out + "/" + (a.cmd.empty() ? "tune" : a.cmd) + ".jsonl", std::ios::app);

    if (a.cmd == "tune") {
      for (const W& w : load(a.workloads)) {
        for (Precision prec : a.precisions) {
          const ConvProblem prob = to_problem(w, prec);
          TuneOptions to = base_opts(a);
          to.use_cache = true;
          json::Value r = json::Value::object();
          r["workload"] = w.name;
          r["precision"] = to_string(prec);
          r["gpu"] = dev.name;
          try {
            const TuneResult tr = Autotuner::tune(prob, dev, to);
            r["best_kernel"] = tr.best_kernel;
            r["best_us"] = tr.best_us;
            r["from_cache"] = tr.from_cache;
            r["search_seconds"] = tr.search_seconds;
            r["candidates_measured"] = tr.candidates_measured;
            std::cout << w.name << " " << to_string(prec) << " -> " << tr.best_kernel << " " << tr.best_us << " us"
                      << (tr.from_cache ? " (cache)" : " (search " + std::to_string(tr.search_seconds) + " s)") << "\n";
          } catch (const Error& e) {
            r["error"] = e.what();
            std::cout << w.name << " " << to_string(prec) << " FAILED: " << e.what() << "\n";
          }
          out << r.dump() << "\n";
          out.flush();
        }
      }
      return 0;
    }

    if (a.cmd == "experiment-b") {
      // Phase 1: tuning set -> record oracle data, fit per-backend scale factors from measured/predicted.
      std::map<std::string, std::vector<double>> ratios;
      std::vector<json::Value> tuning_records;
      for (const W& w : load(a.tuning_set)) {
        for (Precision prec : a.precisions) {
          try {
            Oracle orc;
            json::Value r = experiment_workload(w, prec, dev, a, "tuning", {}, &orc);
            for (const auto& kv : orc.us) {
              const KernelDesc* k = find_kernel(kv.first);
              const double pred = orc.pred[kv.first];
              if (k && pred > 0 && kv.second > 0) ratios[to_string(k->backend)].push_back(kv.second / pred);
            }
            out << r.dump() << "\n";
            out.flush();
            std::cerr << "[tuning] " << w.name << " " << to_string(prec) << " best=" << orc.tr.best_kernel << "\n";
          } catch (const Error& e) { std::cerr << "[tuning] " << w.name << " failed: " << e.what() << "\n"; }
        }
      }
      std::map<std::string, double> cal;
      json::Value calj = json::Value::object();
      for (auto& kv : ratios) {
        std::sort(kv.second.begin(), kv.second.end());
        cal[kv.first] = kv.second[kv.second.size() / 2];  // median ratio: robust to outliers
        calj[kv.first] = cal[kv.first];
      }
      json::write_file(a.out + "/model_calibration.json", calj);
      std::cerr << "fitted calibration (tuning set only): " << calj.dump() << "\n";

      // Phase 2: held-out set -> evaluate prior and calibrated model-guided search (out of sample).
      for (const W& w : load(a.heldout_set)) {
        for (Precision prec : a.precisions) {
          try {
            json::Value r = experiment_workload(w, prec, dev, a, "heldout", cal, nullptr);
            out << r.dump() << "\n";
            out.flush();
            std::cerr << "[heldout] " << w.name << " " << to_string(prec) << "\n";
          } catch (const Error& e) { std::cerr << "[heldout] " << w.name << " failed: " << e.what() << "\n"; }
        }
      }
      return 0;
    }
    std::cerr << "usage: gie_tune --clear-cache | tune --workloads F | experiment-b --tuning-set F --heldout-set F\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
