// gie_bench: reproducible benchmark harness.
//
// For every (workload, precision, fusion mode, backend) it
//   1. builds deterministic synthetic tensors (fixed seed) and a double-precision GPU reference,
//   2. selects the kernel (custom backends: autotuned and/or a fixed manual configuration),
//   3. validates the output against the reference with the documented tolerance policy,
//   4. times the steady-state launch with CUDA events (warmup, repeated samples, robust stats),
//   5. appends one JSON line (and one CSV row) per measurement -- including failures,
//      unsupported combinations and validation failures, which are never dropped.
// Tuning cost, library setup cost, workspace and layout-conversion cost are reported in separate
// fields so they never leak into steady-state latency.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <unistd.h>

#include "baselines.h"
#include "gpu_inference/autotuner.h"
#include "gpu_inference/build_info.h"
#include "gpu_inference/conv2d.h"
#include "gpu_inference/dispatcher.h"
#include "gpu_inference/execution_plan.h"
#include "gpu_inference/fused_ops.h"
#include "gpu_inference/json_lite.h"
#include "gpu_inference/nvtx.h"
#include "gpu_inference/perf_model.h"
#include "gpu_inference/reference.h"
#include "gpu_inference/testing.h"
#include "gpu_inference/timing.h"

using namespace gie;

namespace {

struct Workload {
  std::string name, category, split;
  Conv2DParams p;
  bool bias = true, relu = true;
};

struct Opts {
  std::string workloads = "benchmarks/workloads.json";
  std::string split_label;  // overrides the split recorded in the file (optional)
  std::vector<std::string> kernels;  // explicit kernel names to time as extra rows (diagnostics / ablations)
  std::vector<std::string> backends = {"direct", "im2col_gemm", "simt_tiled", "implicit_gemm", "tensor_core", "dispatcher", "cudnn", "cutlass"};
  std::vector<Precision> precisions = {Precision::kFp32, Precision::kFp16};
  std::string mode = "tuned";     // tuned | fixed | both  (custom multi-config backends)
  std::string fusion = "fused";   // fused | unfused | both
  int warmup = 5, samples = 30;
  double min_sample_ms = 1.0;
  bool flush_l2 = false, validate = true, e2e = false, skip_transfers = false, retune = false;
  int tune_samples = 7;
  int model_keep = 0;
  std::string out_dir = "results/raw/run";
  std::string tag = "run";
  int device = 0;
  uint64_t seed = 0x1234ABCDULL;
  std::string filter, category;
  std::string cache_path;
  double cpu_max_gflop = 0.3;
  size_t cudnn_ws_mb = 1024;
  bool list = false;
  int max_workloads = 0;
};

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(tok);
  return out;
}

void usage() {
  std::cout << "gie_bench [options]\n"
               "  --workloads FILE        workload JSON (default benchmarks/workloads.json)\n"
               "  --backends LIST         comma list of: cpu_reference,direct,im2col_gemm,simt_tiled,implicit_gemm,\n"
               "                          tensor_core,dispatcher,cudnn,cutlass\n"
               "  --kernels LIST          also time these registered kernels by name (backend column = 'kernel')\n"
               "  --precisions LIST       fp32,fp16,bf16,tf32 (default fp32,fp16)\n"
               "  --mode tuned|fixed|both configuration policy for simt_tiled/implicit_gemm/tensor_core\n"
               "  --fusion fused|unfused|both\n"
               "  --warmup N --samples N --min-sample-ms X   timing protocol (defaults 5, 30, 1.0)\n"
               "  --flush-l2              evict L2 before every timed launch (cold-cache protocol)\n"
               "  --no-validate           skip correctness checking (not recommended)\n"
               "  --e2e                   also measure host-synchronous latency and H2D/D2H transfers\n"
               "  --skip-transfers        with --e2e: omit the H2D/D2H memcpy rows (use when re-running one backend so they are not duplicated)\n"
               "  --retune                ignore cached tuning results\n"
               "  --tune-samples N        timing samples per candidate during tuning (default 7)\n"
               "  --model-keep N          prune tuning to the N best candidates per the cost model\n"
               "  --cache FILE            tuning cache path (default $GIE_TUNE_CACHE)\n"
               "  --filter SUBSTR --category NAME --max-workloads N\n"
               "  --out DIR --tag NAME --device N --seed N\n"
               "  --cpu-max-gflop X       run the CPU reference only below X GFLOP (default 0.3)\n"
               "  --cudnn-ws-mb N         cuDNN workspace limit (default 1024)\n"
               "  --list                  print the selected workloads and exit\n";
}

Opts parse_args(int argc, char** argv) {
  Opts o;
  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) throw Error(StatusCode::kInvalidArgument, std::string("missing value for ") + argv[i]);
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--workloads") o.workloads = need(i);
    else if (a == "--backends") o.backends = split_csv(need(i));
    else if (a == "--kernels") o.kernels = split_csv(need(i));
    else if (a == "--precisions") { o.precisions.clear(); for (auto& s : split_csv(need(i))) o.precisions.push_back(precision_from_string(s)); }
    else if (a == "--mode") o.mode = need(i);
    else if (a == "--fusion") o.fusion = need(i);
    else if (a == "--warmup") o.warmup = std::stoi(need(i));
    else if (a == "--samples") o.samples = std::stoi(need(i));
    else if (a == "--min-sample-ms") o.min_sample_ms = std::stod(need(i));
    else if (a == "--flush-l2") o.flush_l2 = true;
    else if (a == "--no-validate") o.validate = false;
    else if (a == "--e2e") o.e2e = true;
    else if (a == "--skip-transfers") o.skip_transfers = true;
    else if (a == "--retune") o.retune = true;
    else if (a == "--tune-samples") o.tune_samples = std::stoi(need(i));
    else if (a == "--model-keep") o.model_keep = std::stoi(need(i));
    else if (a == "--cache") o.cache_path = need(i);
    else if (a == "--filter") o.filter = need(i);
    else if (a == "--category") o.category = need(i);
    else if (a == "--max-workloads") o.max_workloads = std::stoi(need(i));
    else if (a == "--out") o.out_dir = need(i);
    else if (a == "--tag") o.tag = need(i);
    else if (a == "--device") o.device = std::stoi(need(i));
    else if (a == "--seed") o.seed = std::stoull(need(i));
    else if (a == "--cpu-max-gflop") o.cpu_max_gflop = std::stod(need(i));
    else if (a == "--cudnn-ws-mb") o.cudnn_ws_mb = size_t(std::stoul(need(i)));
    else if (a == "--list") o.list = true;
    else if (a == "--help" || a == "-h") { usage(); std::exit(0); }
    else throw Error(StatusCode::kInvalidArgument, "unknown option " + a);
  }
  if (o.mode != "tuned" && o.mode != "fixed" && o.mode != "both") throw Error(StatusCode::kInvalidArgument, "--mode must be tuned|fixed|both");
  if (o.fusion != "fused" && o.fusion != "unfused" && o.fusion != "both") throw Error(StatusCode::kInvalidArgument, "--fusion must be fused|unfused|both");
  return o;
}

std::vector<Workload> load_workloads(const Opts& o) {
  const json::Value root = json::parse_file(o.workloads);
  const json::Value* list = root.find("workloads");
  if (!list || !list->is_array()) throw Error(StatusCode::kIo, o.workloads + ": missing 'workloads' array");
  const std::string file_split = root.find("split") ? root.find("split")->as_string() : "unspecified";
  std::vector<Workload> out;
  for (const json::Value& w : list->elements()) {
    auto geti = [&](const char* k, int def) { const json::Value* v = w.find(k); return v ? int(v->as_int()) : def; };
    Workload wl;
    wl.name = w.find("name") ? w.find("name")->as_string() : "unnamed";
    wl.category = w.find("category") ? w.find("category")->as_string() : "uncategorized";
    wl.split = !o.split_label.empty() ? o.split_label : file_split;
    Conv2DParams& p = wl.p;
    p.batch = geti("B", 1);
    p.in_channels = geti("C", 1);
    p.in_h = geti("H", 1);
    p.in_w = geti("W", p.in_h);
    p.out_channels = geti("O", p.in_channels);
    p.kernel_h = geti("Kh", geti("K", 3));
    p.kernel_w = geti("Kw", geti("K", 3));
    p.stride_h = geti("Sh", geti("S", 1));
    p.stride_w = geti("Sw", geti("S", 1));
    p.pad_h = geti("Ph", geti("P", 0));
    p.pad_w = geti("Pw", geti("P", 0));
    wl.bias = w.find("bias") ? w.find("bias")->as_bool(true) : true;
    wl.relu = w.find("relu") ? w.find("relu")->as_bool(true) : true;
    if (!o.filter.empty() && wl.name.find(o.filter) == std::string::npos) continue;
    if (!o.category.empty() && wl.category != o.category) continue;
    out.push_back(wl);
    if (o.max_workloads > 0 && int(out.size()) >= o.max_workloads) break;
  }
  return out;
}

// ---- result sink -----------------------------------------------------------------------------------
const std::vector<std::string>& csv_columns() {
  static const std::vector<std::string> c = {
      "tag", "gpu", "cc", "workload", "category", "split", "B", "C", "H", "W", "O", "Kh", "Kw", "Sh", "Sw", "Ph", "Pw",
      "precision", "bias", "relu", "fusion", "backend", "mode", "kernel", "status", "reason", "median_us", "mean_us", "min_us",
      "p10_us", "p90_us", "p99_us", "stddev_us", "samples", "inner_iters", "images_per_s", "tflops", "workspace_bytes",
      "persistent_bytes", "tuning_seconds", "tuning_cache_hit", "setup_seconds", "max_abs_err", "frac_over_tol",
      "worst_tol_ratio", "flops", "compulsory_bytes", "arithmetic_intensity", "model_pred_us", "roofline_bound",
      "conv_only_us", "layout_inclusive_us", "src_fp", "warmup", "layout", "stage_im2col_us", "stage_gemm_us", "host_sync_latency_us", "flush_l2", "config"};
  return c;
}

class Sink {
 public:
  Sink(const std::string& dir) {
    const std::string cmd = "mkdir -p '" + dir + "'";
    if (std::system(cmd.c_str()) != 0) throw Error(StatusCode::kIo, "cannot create " + dir);
    jsonl_.open(dir + "/results.jsonl", std::ios::app);
    const bool fresh = !std::ifstream(dir + "/results.csv").good();
    csv_.open(dir + "/results.csv", std::ios::app);
    if (!jsonl_ || !csv_) throw Error(StatusCode::kIo, "cannot open result files in " + dir);
    if (fresh) {
      for (size_t i = 0; i < csv_columns().size(); ++i) csv_ << (i ? "," : "") << csv_columns()[i];
      csv_ << "\n";
    }
  }
  void write(const json::Value& r) {
    jsonl_ << r.dump() << "\n";
    jsonl_.flush();
    for (size_t i = 0; i < csv_columns().size(); ++i) {
      csv_ << (i ? "," : "");
      const json::Value* v = r.find(csv_columns()[i]);
      if (!v || v->is_null()) continue;
      std::string s = v->is_string() ? v->as_string() : v->dump();
      if (v->type() == json::Value::Type::kBool) s = v->as_bool() ? "1" : "0";
      if (s.find_first_of(",\"\n") != std::string::npos) {
        std::string q = "\"";
        for (char c : s) { if (c == '"') q += '"'; q += c; }
        s = q + "\"";
      }
      csv_ << s;
    }
    csv_ << "\n";
    csv_.flush();
  }

 private:
  std::ofstream jsonl_, csv_;
};

// ---- measurement context ----------------------------------------------------------------------------
struct Ctx {
  Opts opt;
  const DeviceInfo* dev = nullptr;
  DeviceCeilings ceilings;
  cudaStream_t stream = nullptr;
  DeviceBuffer flush;
  Sink* sink = nullptr;
  std::string cudnn_ver, cutlass_ver;
};

TimingOptions timing_opts(const Ctx& c) {
  TimingOptions t;
  t.warmup_iters = c.opt.warmup;
  t.samples = c.opt.samples;
  t.min_sample_ms = c.opt.min_sample_ms;
  if (c.opt.flush_l2) {
    t.flush_buffer = c.flush.data();
    t.flush_bytes = c.flush.size();
  }
  return t;
}

json::Value base_record(const Ctx& c, const Workload& wl, const ConvProblem& prob, const std::string& fusion,
                        const std::string& backend, const std::string& mode) {
  json::Value r = json::Value::object();
  r["tag"] = c.opt.tag;
  r["gpu"] = c.dev->name;
  r["cc"] = c.dev->cc();
  r["workload"] = wl.name;
  r["category"] = wl.category;
  r["split"] = wl.split;
  const Conv2DParams& p = prob.p;
  r["B"] = p.batch; r["C"] = p.in_channels; r["H"] = p.in_h; r["W"] = p.in_w; r["O"] = p.out_channels;
  r["Kh"] = p.kernel_h; r["Kw"] = p.kernel_w; r["Sh"] = p.stride_h; r["Sw"] = p.stride_w; r["Ph"] = p.pad_h; r["Pw"] = p.pad_w;
  r["precision"] = to_string(prob.precision);
  r["bias"] = prob.has_bias;
  r["relu"] = prob.activation == Activation::kReLU;
  r["fusion"] = fusion;
  r["backend"] = backend;
  r["mode"] = mode;
  r["status"] = "ok";
  r["flops"] = double(p.flops());
  const ProblemModel m = analyze_problem(prob);
  r["compulsory_bytes"] = m.compulsory_bytes;
  r["arithmetic_intensity"] = m.arithmetic_intensity;
  r["roofline_bound"] = to_string(roofline_estimate(prob, c.ceilings, prob.precision == Precision::kFp16 || prob.precision == Precision::kBf16).bound);
  r["flush_l2"] = c.opt.flush_l2;
  r["seed"] = double(c.opt.seed);
  r["src_fp"] = build_info().source_fingerprint;
  r["warmup"] = c.opt.warmup;
  r["layout"] = backend == "cutlass" ? "NHWC/KRSC" : "NCHW/OIHW";
  return r;
}

void fill_timing(json::Value& r, const TimingStats& t, const ConvProblem& prob) {
  r["median_us"] = t.median_us;
  r["mean_us"] = t.mean_us;
  r["min_us"] = t.min_us;
  r["p10_us"] = t.p10_us;
  r["p90_us"] = t.p90_us;
  r["p99_us"] = t.p99_us;
  r["stddev_us"] = t.stddev_us;
  r["samples"] = t.samples;
  r["inner_iters"] = t.inner_iters;
  if (t.median_us > 0) {
    r["images_per_s"] = double(prob.p.batch) / (t.median_us * 1e-6);
    r["tflops"] = double(prob.p.flops()) / (t.median_us * 1e-6) / 1e12;
  }
}

void fail_record(json::Value& r, const std::string& status, const std::string& reason) {
  r["status"] = status;
  r["reason"] = reason;
}

std::string classify_error(const Error& e) {
  if (e.code() == StatusCode::kUnsupported) return "unsupported";
  const std::string w = e.what();
  if (w.find("out of memory") != std::string::npos) return "oom";
  return "error";
}

// ---- reference bundle per (workload, precision) -------------------------------------------------------
struct Bundle {
  ConvData data;                 // device tensors for the fused problem
  std::vector<float> ref, scale; // reference output (host)
  bool have_ref = false;
};

Bundle make_bundle(const Ctx& c, const ConvProblem& prob) {
  Bundle b;
  b.data = make_conv_data(prob, c.opt.seed, Distribution::kUniform);
  if (c.opt.validate) {
    const int64_t n = prob.p.output_shape().numel();
    DeviceBuffer dref(size_t(n) * 4), dsc(size_t(n) * 4);
    gpu_conv2d_reference(prob, b.data.dx.data(), b.data.dw.data(), b.data.db.data(), static_cast<float*>(dref.data()),
                         static_cast<float*>(dsc.data()), c.stream);
    GIE_CUDA_CHECK(cudaStreamSynchronize(c.stream));
    b.ref.resize(size_t(n));
    b.scale.resize(size_t(n));
    GIE_CUDA_CHECK(cudaMemcpy(b.ref.data(), dref.data(), size_t(n) * 4, cudaMemcpyDeviceToHost));
    GIE_CUDA_CHECK(cudaMemcpy(b.scale.data(), dsc.data(), size_t(n) * 4, cudaMemcpyDeviceToHost));
    b.have_ref = true;
  }
  return b;
}

void validate_into(json::Value& r, const Bundle& b, const ConvProblem& prob, const void* y, bool two_roundings) {
  if (!b.have_ref) { r["validated"] = false; return; }
  const int64_t n = prob.p.output_shape().numel();
  const std::vector<float> out = download_as_float(y, prob.dtype(), n);
  const ErrorMetrics m = compare_outputs(out.data(), b.ref.data(), b.scale.data(), n, TolerancePolicy::for_problem(prob, two_roundings));
  r["validated"] = true;
  r["max_abs_err"] = m.max_abs;
  r["mean_abs_err"] = m.mean_abs;
  r["rms_err"] = m.rms;
  r["max_rel_err"] = m.max_rel;
  r["frac_over_tol"] = m.frac_over_tol;
  r["worst_tol_ratio"] = m.worst_tol_ratio;
  r["nan_count"] = double(m.nan_count);
  r["inf_count"] = double(m.inf_count);
  if (!m.passed()) fail_record(r, "failed_validation", m.summary());
}

// ---- custom-kernel measurement ----------------------------------------------------------------------------
// fused: the kernel's own epilogue. unfused: conv-only kernel + separate bias pass + separate ReLU pass.
void measure_kernel(Ctx& c, const Workload& wl, const ConvProblem& prob, Bundle& b, const KernelDesc& k,
                    const std::string& fusion, json::Value& r) {
  const bool unfused = fusion == "unfused";
  ConvProblem conv_prob = prob;
  if (unfused) { conv_prob.has_bias = false; conv_prob.activation = Activation::kNone; }
  auto plan = ExecutionPlan::create(conv_prob, k, c.dev->device);
  DeviceBuffer ws(plan->workspace_bytes());
  r["kernel"] = unfused ? k.name + "+bias+relu(unfused)" : k.name;
  r["workspace_bytes"] = double(plan->workspace_bytes());
  r["persistent_bytes"] = double(plan->persistent_bytes());
  r["model_pred_us"] = estimate_kernel(prob, k, *c.dev, c.ceilings).time_us;

  const void* bias = prob.has_bias ? b.data.db.data() : nullptr;
  cudaEvent_t stage[3] = {nullptr, nullptr, nullptr};
  const bool staged = k.backend == Backend::kIm2colGemm && !unfused;
  if (staged) for (auto& e : stage) GIE_CUDA_CHECK(cudaEventCreate(&e));

  auto launch = [&](cudaStream_t s, cudaEvent_t* ev = nullptr) {
    plan->execute_unchecked(b.data.dx.data(), b.data.dw.data(), unfused ? nullptr : bias, b.data.dy.data(), ws.data(), ws.size(), s, ev);
    if (unfused) {
      if (prob.has_bias) launch_bias_add(prob.p, prob.precision, bias, b.data.dy.data(), s);
      if (prob.activation == Activation::kReLU) launch_relu(prob.p, prob.precision, b.data.dy.data(), s);
    }
  };

  // correctness first: no timing for a kernel that produces wrong output
  GIE_CUDA_CHECK(cudaMemsetAsync(b.data.dy.data(), 0xFF, b.data.dy.size(), c.stream));
  launch(c.stream);
  GIE_CUDA_CHECK(cudaStreamSynchronize(c.stream));
  const bool two_roundings = prob.precision != Precision::kFp32 && (unfused || k.backend == Backend::kIm2colGemm) &&
                             (prob.has_bias || prob.activation != Activation::kNone);
  validate_into(r, b, prob, b.data.dy.data(), two_roundings);
  if (r.find("status")->as_string() != "ok") {
    if (staged) for (auto e : stage) cudaEventDestroy(e);
    return;
  }

  const TimingStats t = time_stream_work([&](cudaStream_t s) { launch(s); }, c.stream, timing_opts(c));
  GIE_CUDA_CHECK(cudaGetLastError());
  fill_timing(r, t, prob);

  if (staged) {  // GEMM-only vs whole-pipeline time (Experiment A accounting)
    std::vector<double> t_col, t_gemm;
    for (int i = 0; i < 20; ++i) {
      launch(c.stream, stage);
      GIE_CUDA_CHECK(cudaEventSynchronize(stage[2]));
      float a = 0, g = 0;
      GIE_CUDA_CHECK(cudaEventElapsedTime(&a, stage[0], stage[1]));
      GIE_CUDA_CHECK(cudaEventElapsedTime(&g, stage[1], stage[2]));
      if (i >= 5) { t_col.push_back(a * 1000.0); t_gemm.push_back(g * 1000.0); }
    }
    r["stage_im2col_us"] = summarize(t_col, 1).median_us;
    r["stage_gemm_us"] = summarize(t_gemm, 1).median_us;
    for (auto e : stage) cudaEventDestroy(e);
  }
  if (c.opt.e2e) {  // host-synchronous latency: launch + wait, what a naive caller observes
    std::vector<double> us;
    for (int i = 0; i < 50; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      launch(c.stream);
      GIE_CUDA_CHECK(cudaStreamSynchronize(c.stream));
      if (i >= 10) us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    r["host_sync_latency_us"] = summarize(us, 1).median_us;
  }
  (void)wl;
}

std::string fixed_kernel_name(const std::string& backend, Precision p) {
  const std::string pn = to_string(p);
  if (backend == "simt_tiled") return "simt_tiled_" + pn + "_tc32_t8x8_r4x2_cc8";
  if (backend == "implicit_gemm") return "implicit_gemm_" + pn + "_128x128x8_t8x8";
  if (backend == "tensor_core") return "tensor_core_" + pn + "_128x128x32_w2x4";
  if (backend == "direct") return "direct_" + pn;
  if (backend == "im2col_gemm") return "im2col_gemm_" + pn;
  return "";
}

void run_custom_backend(Ctx& c, const Workload& wl, const ConvProblem& prob, Bundle& b, const std::string& backend,
                        const std::string& fusion) {
  const Backend be = backend_from_string(backend);
  std::vector<std::string> modes;
  if (be == Backend::kDirect || be == Backend::kIm2colGemm) modes = {"fixed"};
  else if (c.opt.mode == "both") modes = {"fixed", "tuned"};
  else modes = {c.opt.mode};

  ConvProblem conv_prob = prob;  // the problem actually handed to the conv kernel
  if (fusion == "unfused") { conv_prob.has_bias = false; conv_prob.activation = Activation::kNone; }

  for (const std::string& mode : modes) {
    json::Value r = base_record(c, wl, prob, fusion, backend, mode);
    try {
      const KernelDesc* k = nullptr;
      if (mode == "fixed") {
        k = find_kernel(fixed_kernel_name(backend, prob.precision));
        if (!k) throw Error(StatusCode::kUnsupported, "no fixed " + backend + " kernel for " + to_string(prob.precision));
        std::string why;
        if (!k->supports(conv_prob, *c.dev, &why)) throw Error(StatusCode::kUnsupported, k->name + ": " + why);
      } else {
        TuneOptions to;
        to.only_backend = be;
        to.cache_path = c.opt.cache_path;
        to.force_retune = c.opt.retune;
        to.model_keep = c.opt.model_keep;
        to.timing.samples = c.opt.tune_samples;
        to.timing.warmup_iters = 3;
        to.timing.min_sample_ms = 0.5;
        const TuneResult tr = Autotuner::tune(conv_prob, *c.dev, to);
        k = find_kernel(tr.best_kernel);
        r["tuning_seconds"] = tr.search_seconds;
        r["tuning_cache_hit"] = tr.from_cache;
        if (!tr.from_cache) {
          r["tune_candidates"] = tr.candidates_measured;
          r["tune_best_us"] = tr.best_us;
        }
      }
      measure_kernel(c, wl, prob, b, *k, fusion, r);
    } catch (const Error& e) {
      fail_record(r, classify_error(e), e.what());
    }
    c.sink->write(r);
  }
}

void run_named_kernel(Ctx& c, const Workload& wl, const ConvProblem& prob, Bundle& b, const std::string& name, const std::string& fusion) {
  json::Value r = base_record(c, wl, prob, fusion, "kernel", "named");
  try {
    const KernelDesc* k = find_kernel(name);
    if (!k) throw Error(StatusCode::kUnsupported, "unknown kernel " + name);
    ConvProblem conv_prob = prob;
    if (fusion == "unfused") { conv_prob.has_bias = false; conv_prob.activation = Activation::kNone; }
    std::string why;
    if (!k->supports(conv_prob, *c.dev, &why)) throw Error(StatusCode::kUnsupported, name + ": " + why);
    measure_kernel(c, wl, prob, b, *k, fusion, r);
  } catch (const Error& e) {
    fail_record(r, classify_error(e), e.what());
  }
  c.sink->write(r);
}

void run_dispatcher(Ctx& c, const Workload& wl, const ConvProblem& prob, Bundle& b, const std::string& fusion) {
  json::Value r = base_record(c, wl, prob, fusion, "dispatcher", "auto");
  try {
    if (fusion == "unfused") throw Error(StatusCode::kUnsupported, "dispatcher is measured with its own fused selection only");
    Conv2DOptions opt;
    opt.autotune = true;
    opt.force_retune = c.opt.retune;
    opt.cache_path = c.opt.cache_path;
    const auto t0 = std::chrono::steady_clock::now();
    const Selection sel = Dispatcher::select(prob, *c.dev, opt);
    const double select_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    r["tuning_seconds"] = select_s;
    r["selection_source"] = to_string(sel.source);
    measure_kernel(c, wl, prob, b, *sel.kernel, "fused", r);
    r["config"] = std::string("selected ") + sel.kernel->name + " via " + to_string(sel.source);

    // warm dispatch overhead: host time of Conv2D::forward (plan lookup + validation + launch) vs raw launch
    if (r.find("status")->as_string() == "ok") {
      Conv2DOptions fo;
      fo.kernel = sel.kernel->name;
      DeviceBuffer ws(Conv2D::workspace_size(prob, fo));
      const ConstTensorRef bias = b.data.b_ref();
      auto plan = ExecutionPlan::create(prob, *sel.kernel, c.dev->device);
      auto host_us = [&](const std::function<void()>& f) {
        for (int i = 0; i < 50; ++i) f();
        GIE_CUDA_CHECK(cudaStreamSynchronize(c.stream));
        std::vector<double> v;
        for (int rep = 0; rep < 9; ++rep) {
          const auto rep_start = std::chrono::steady_clock::now();
          for (int i = 0; i < 200; ++i) f();
          v.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - rep_start).count() / 200.0);
          GIE_CUDA_CHECK(cudaStreamSynchronize(c.stream));
        }
        return summarize(v, 1).median_us;
      };
      const double via_api = host_us([&] {
        Conv2D::forward(b.data.x_ref(), b.data.w_ref(), prob.has_bias ? &bias : nullptr, prob, b.data.y_ref(), ws.data(), ws.size(), c.stream, fo);
      });
      const double raw = host_us([&] {
        plan->execute_unchecked(b.data.dx.data(), b.data.dw.data(), prob.has_bias ? b.data.db.data() : nullptr, b.data.dy.data(), ws.data(), ws.size(), c.stream);
      });
      r["dispatch_api_host_us"] = via_api;
      r["raw_launch_host_us"] = raw;
      r["warm_dispatch_overhead_us"] = via_api - raw;
    }
  } catch (const Error& e) {
    fail_record(r, classify_error(e), e.what());
  }
  c.sink->write(r);
}

void run_baseline(Ctx& c, const Workload& wl, const ConvProblem& prob, Bundle& b, const std::string& backend,
                  const std::string& fusion) {
  json::Value r = base_record(c, wl, prob, fusion, backend, "library");
  try {
    if (fusion == "unfused") throw Error(StatusCode::kUnsupported, backend + " is measured with its own fused pipeline only");
    std::unique_ptr<bench::Baseline> bl = backend.rfind("cudnn", 0) == 0 ? bench::make_cudnn_baseline(c.opt.cudnn_ws_mb << 20, backend == "cudnn_gemm") : bench::make_cutlass_baseline();
    if (!bl) throw Error(StatusCode::kUnsupported, backend + " was not found at configure time (see build log)");
    DeviceBuffer y_scratch(b.data.dy.size());
    std::string why;
    if (b.have_ref) {
      const bool two_acc = prob.precision != Precision::kFp32 && prob.has_bias && prob.activation != Activation::kNone && backend.rfind("cudnn", 0) == 0;
      bl->set_acceptor([&](void* y_nchw) -> double {
        bl->output_nchw(y_nchw, nullptr);
        cudaDeviceSynchronize();
        const int64_t n = prob.p.output_shape().numel();
        const std::vector<float> out = download_as_float(y_nchw, prob.dtype(), n);
        const ErrorMetrics m = compare_outputs(out.data(), b.ref.data(), b.scale.data(), n, TolerancePolicy::for_problem(prob, two_acc));
        return (m.nan_count || m.inf_count) ? 1e30 : m.worst_tol_ratio;
      });
    }
    if (!bl->setup(prob, b.data.dx.data(), b.data.dw.data(), prob.has_bias ? b.data.db.data() : nullptr, y_scratch.data(), &why))
      throw Error(StatusCode::kUnsupported, why);
    r["kernel"] = bl->name();
    r["config"] = bl->config();
    r["setup_seconds"] = bl->setup_seconds();
    r["workspace_bytes"] = double(bl->workspace_bytes());
    DeviceBuffer ws(bl->workspace_bytes());

    // y_nchw receives results (cuDNN writes it directly; CUTLASS converts from its NHWC buffer)
    void* y = b.data.dy.data();
    GIE_CUDA_CHECK(cudaMemsetAsync(y, 0xFF, b.data.dy.size(), c.stream));
    bl->run(y, ws.data(), c.stream);
    bl->output_nchw(y, c.stream);
    GIE_CUDA_CHECK(cudaStreamSynchronize(c.stream));
    const bool two = prob.precision != Precision::kFp32 && prob.has_bias && prob.activation != Activation::kNone && backend.rfind("cudnn", 0) == 0;
    validate_into(r, b, prob, y, two);
    if (r.find("status")->as_string() == "ok") {
      const TimingStats t = time_stream_work([&](cudaStream_t s) { bl->run(y, ws.data(), s); }, c.stream, timing_opts(c));
      GIE_CUDA_CHECK(cudaGetLastError());
      fill_timing(r, t, prob);
      if (bl->needs_layout_conversion()) {
        const TimingStats tc = time_stream_work([&](cudaStream_t s) { bl->run_with_conversion(b.data.dx.data(), b.data.dw.data(), y, ws.data(), s); },
                                                c.stream, timing_opts(c));
        r["layout_inclusive_us"] = tc.median_us;
        r["conv_only_us"] = t.median_us;
      }
    }
  } catch (const Error& e) {
    fail_record(r, classify_error(e), e.what());
  }
  c.sink->write(r);
}

void run_cpu_reference(Ctx& c, const Workload& wl, const ConvProblem& prob, Bundle& b, const std::string& fusion) {
  json::Value r = base_record(c, wl, prob, fusion, "cpu_reference", "host");
  try {
    if (double(prob.p.flops()) / 1e9 > c.opt.cpu_max_gflop)
      throw Error(StatusCode::kUnsupported, "skipped: " + std::to_string(prob.p.flops() / 1e9) + " GFLOP exceeds --cpu-max-gflop");
    ReferenceOutput out;
    const auto t0 = std::chrono::steady_clock::now();
    cpu_conv2d_reference(prob, b.data.hx.data(), b.data.hw.data(), prob.has_bias ? b.data.hb.data() : nullptr, &out);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    r["kernel"] = "cpu_reference(double accum, OpenMP)";
    r["median_us"] = us;
    r["samples"] = 1;
    r["images_per_s"] = double(prob.p.batch) / (us * 1e-6);
    r["tflops"] = double(prob.p.flops()) / (us * 1e-6) / 1e12;
    if (b.have_ref) {
      const ErrorMetrics m = compare_outputs(out.y.data(), b.ref.data(), b.scale.data(), int64_t(out.y.size()), TolerancePolicy::for_problem(prob));
      r["max_abs_err"] = m.max_abs;
      if (!m.passed()) fail_record(r, "failed_validation", m.summary());
    }
  } catch (const Error& e) {
    fail_record(r, classify_error(e), e.what());
  }
  c.sink->write(r);
}

void run_transfers(Ctx& c, const Workload& wl, const ConvProblem& prob) {
  // Host<->device copy cost, measured separately from kernel time (pinned host memory).
  json::Value r = base_record(c, wl, prob, "n/a", "memcpy", "pinned");
  try {
    const size_t in_bytes = (size_t(prob.p.input_shape().numel()) + size_t(prob.p.weight_shape().numel())) * dtype_size(prob.dtype());
    const size_t out_bytes = size_t(prob.p.output_shape().numel()) * dtype_size(prob.dtype());
    void *h_in = nullptr, *h_out = nullptr;
    GIE_CUDA_CHECK(cudaMallocHost(&h_in, in_bytes));
    GIE_CUDA_CHECK(cudaMallocHost(&h_out, out_bytes));
    DeviceBuffer d_in(in_bytes), d_out(out_bytes);
    TimingOptions to = timing_opts(c);
    to.flush_bytes = 0;
    to.inner = 1;
    const TimingStats h2d = time_stream_work([&](cudaStream_t s) { cudaMemcpyAsync(d_in.data(), h_in, in_bytes, cudaMemcpyHostToDevice, s); }, c.stream, to);
    const TimingStats d2h = time_stream_work([&](cudaStream_t s) { cudaMemcpyAsync(h_out, d_out.data(), out_bytes, cudaMemcpyDeviceToHost, s); }, c.stream, to);
    r["kernel"] = "cudaMemcpyAsync";
    r["h2d_us"] = h2d.median_us;
    r["d2h_us"] = d2h.median_us;
    r["h2d_bytes"] = double(in_bytes);
    r["d2h_bytes"] = double(out_bytes);
    r["median_us"] = h2d.median_us + d2h.median_us;
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
  } catch (const Error& e) {
    fail_record(r, classify_error(e), e.what());
  }
  c.sink->write(r);
}

json::Value make_meta(const Ctx& c, int argc, char** argv) {
  json::Value m = json::Value::object();
  m["tag"] = c.opt.tag;
  const DeviceInfo& d = *c.dev;
  m["gpu"] = d.name;
  m["compute_capability"] = d.cc();
  m["sm_count"] = d.sm_count;
  m["gpu_memory_bytes"] = double(d.global_mem_bytes);
  m["l2_bytes"] = double(d.l2_bytes);
  m["cuda_runtime"] = d.runtime_version;
  m["cuda_driver"] = d.driver_version;
  m["est_peak_fp32_tflops"] = d.est_peak_fp32_tflops();
  m["est_peak_fp16_tc_tflops"] = d.est_peak_fp16_tc_tflops();
  m["est_peak_bw_gbs"] = d.est_peak_bw_gbs();
  m["measured_fp32_tflops"] = c.ceilings.fp32_tflops;
  m["measured_fp16_tc_tflops"] = c.ceilings.fp16_tc_tflops;
  m["measured_bw_gbs"] = c.ceilings.bw_gbs;
  m["measured_launch_overhead_us"] = c.ceilings.launch_overhead_us;
  m["ceilings_source"] = c.ceilings.source;
  m["cudnn"] = c.cudnn_ver;
  m["cutlass"] = c.cutlass_ver;
  m["library_version"] = kLibraryVersion;
  char host[256] = {0};
  gethostname(host, sizeof(host) - 1);
  m["hostname"] = host;
  if (const char* j = std::getenv("SLURM_JOB_ID")) m["slurm_job_id"] = j;
  if (const char* p = std::getenv("SLURM_JOB_PARTITION")) m["slurm_partition"] = p;
#ifdef GIE_GIT_REV
  m["git_rev"] = GIE_GIT_REV;
#endif
  {  // non-mutating build provenance (no git commit required)
    const BuildInfo& bi = build_info();
    m["source_fingerprint"] = bi.source_fingerprint;
    m["build_type"] = bi.build_type;
    m["cuda_archs"] = bi.cuda_archs;
    m["nvcc_version"] = bi.nvcc_version;
    m["cxx_compiler"] = bi.cxx_compiler;
    m["cuda_flags"] = bi.cuda_flags;
    m["cxx_flags"] = bi.cxx_flags;
    m["nvtx"] = bi.has_nvtx;
  }
  m["input_layout"] = "NCHW activations / OIHW weights for custom kernels, cuBLAS and cuDNN; CUTLASS runs NHWC/KRSC (conversions reported separately)";
  m["timing_mode"] = c.opt.flush_l2 ? "cold_l2_single_launch_events" : "warm_back_to_back_batched_events";
  m["seed"] = double(c.opt.seed);
  m["warmup"] = c.opt.warmup;
  m["samples"] = c.opt.samples;
  m["min_sample_ms"] = c.opt.min_sample_ms;
  m["flush_l2"] = c.opt.flush_l2;
  m["precisions"] = [&] { json::Value a = json::Value::array(); for (auto p : c.opt.precisions) a.push_back(to_string(p)); return a; }();
  m["backends"] = [&] { json::Value a = json::Value::array(); for (auto& s : c.opt.backends) a.push_back(s); return a; }();
  std::string cmd;
  for (int i = 0; i < argc; ++i) cmd += std::string(i ? " " : "") + argv[i];
  m["command"] = cmd;
  m["tolerance_policy"] = "tol_i = 8*2^-24*sqrt(K)*scale_i + c_out*u_out*|ref_i| + floor; see reference.h";
  return m;
}

}  // namespace

// A sticky device error (illegal/misaligned address, ...) poisons every later CUDA call in this process;
// continuing would turn all remaining rows into misleading "error" rows, so stop loudly instead.
void abort_if_context_corrupted(const char* where) {
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    std::cerr << "[gie_bench] FATAL: CUDA context is corrupted after " << where << ": " << cudaGetErrorString(e)
              << ". Rows written so far are valid; rerun the remaining backends in a fresh process.\n";
    std::exit(4);
  }
}

int main(int argc, char** argv) {
  try {
    Ctx c;
    c.opt = parse_args(argc, argv);
    const std::vector<Workload> workloads = load_workloads(c.opt);
    if (c.opt.list) {
      for (const Workload& w : workloads) std::cout << w.name << "\t" << w.category << "\t" << w.p.to_string() << "\n";
      return 0;
    }
    GIE_CUDA_CHECK(cudaSetDevice(c.opt.device));
    c.dev = &DeviceInfo::get(c.opt.device);
    c.ceilings = measure_ceilings(c.opt.device, true);
    GIE_CUDA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
    c.flush = DeviceBuffer(std::max<size_t>(2 * c.dev->l2_bytes, 128u << 20));
    c.cudnn_ver = bench::cudnn_version_string();
    c.cutlass_ver = bench::cutlass_version_string();
    Sink sink(c.opt.out_dir);
    c.sink = &sink;
    json::write_file(c.opt.out_dir + "/meta.json", make_meta(c, argc, argv));
    std::cerr << "[gie_bench] " << c.dev->describe() << "\n[gie_bench] " << workloads.size() << " workloads -> " << c.opt.out_dir << "\n";

    int idx = 0;
    for (const Workload& wl : workloads) {
      ++idx;
      for (Precision prec : c.opt.precisions) {
        ConvProblem prob;
        prob.p = wl.p;
        prob.precision = prec;
        prob.has_bias = wl.bias;
        prob.activation = wl.relu ? Activation::kReLU : Activation::kNone;
        std::cerr << "[" << idx << "/" << workloads.size() << "] " << wl.name << " " << to_string(prec) << " " << prob.p.to_string() << "\n";
        NvtxRange workload_range((wl.name + "_" + to_string(prec)).c_str());
        try {
          validate_problem(prob);
          // memory feasibility: x, w, y, reference y+scale (8 B/elem) and an equal-size im2col-ish margin
          const double need = double(prob.p.output_shape().numel()) * 8.0 + double(analyze_problem(prob).compulsory_bytes) * 2.0 +
                              double(analyze_problem(prob).im2col_bytes);
          if (need > 0.85 * double(c.dev->global_mem_bytes)) throw Error(StatusCode::kUnsupported, "skipped_memory: estimated working set exceeds device memory budget");
          Bundle b = make_bundle(c, prob);
          if (c.opt.e2e && !c.opt.skip_transfers) run_transfers(c, wl, prob);
          std::vector<std::string> fusions;
          if (c.opt.fusion == "both") fusions = {"fused", "unfused"};
          else fusions = {c.opt.fusion};
          const bool has_epilogue = prob.has_bias || prob.activation != Activation::kNone;
          for (const std::string& fusion : fusions) {
            if (fusion == "unfused" && !has_epilogue) continue;  // identical to fused; no duplicate rows
            for (const std::string& be : c.opt.backends) {
              // The unfused-epilogue experiment compares the custom/cuBLAS pipelines; the dispatcher and the
              // library baselines have no separately launched bias/ReLU path, so they are simply not part of it.
              if (fusion == "unfused" && (be == "dispatcher" || be == "cudnn" || be == "cudnn_gemm" || be == "cutlass" || be == "cpu_reference")) continue;
              if (be == "cpu_reference") run_cpu_reference(c, wl, prob, b, fusion);
              else if (be == "dispatcher") run_dispatcher(c, wl, prob, b, fusion);
              else if (be == "cudnn" || be == "cudnn_gemm" || be == "cutlass") run_baseline(c, wl, prob, b, be, fusion);
              else run_custom_backend(c, wl, prob, b, be, fusion);
              abort_if_context_corrupted(be.c_str());
            }
            if (fusion == "fused")
              for (const std::string& kn : c.opt.kernels) {
                run_named_kernel(c, wl, prob, b, kn, fusion);
                abort_if_context_corrupted(kn.c_str());
              }
          }
        } catch (const Error& e) {
          // Record the failure for every requested backend: nothing is silently dropped.
          for (const std::string& be : c.opt.backends) {
            json::Value r = base_record(c, wl, prob, "fused", be, "n/a");
            fail_record(r, classify_error(e), e.what());
            sink.write(r);
          }
          std::cerr << "  workload failed: " << e.what() << "\n";
          (void)cudaGetLastError();
        }
      }
    }
    std::cerr << "[gie_bench] done\n";
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
