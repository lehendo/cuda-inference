// cuDNN baseline (legacy cudnnConvolutionForward API, still shipped in cuDNN 9).
//
// Documented behaviour:
//  * Algorithm selection: cudnnFindConvolutionForwardAlgorithmEx benchmarks every algorithm on
//    the real operands once during setup (excluded from steady-state timing; reported as
//    setup_seconds). The Find pass times each algorithm once, which mis-ranks ~20 us kernels run to run
//    (measured: IMPLICIT_GEMM 20 us vs IMPLICIT_PRECOMP_GEMM 48 us chosen in different runs of the same problem), so the
//    candidates that fit the workspace limit, pass the fused-path probe and meet the accuracy gate are all
//    re-timed with the steady-state method (CUDA events, median of repeated batches) and the fastest is used.
//  * Math mode: FP32 uses CUDNN_FMA_MATH (strict FP32, no TF32 tensor ops) so it is comparable to
//    the strict-FP32 custom kernels; TF32 uses CUDNN_DEFAULT_MATH (TF32 allowed); FP16/BF16 use
//    CUDNN_TENSOR_OP_MATH. Accumulation/compute type is always FP32.
//  * Layout: NCHW/OIHW like the custom kernels. (cuDNN's tensor-core paths are NHWC-native; for
//    FP16/BF16 NCHW it may insert internal layout transforms, which are part of its measured time.)
//  * Fusion: bias(+ReLU) uses cudnnConvolutionBiasActivationForward; ReLU without bias is
//    cudnnConvolutionForward + cudnnActivationForward (two kernels); both are reported as such.
#include "baselines.h"

#ifdef GIE_HAS_CUDNN
#include <cudnn.h>

#include <algorithm>
#include <chrono>
#include <sstream>
#include <vector>

#include "gpu_inference/tensor.h"

namespace bench {
namespace {

#define CUDNN_OK(expr)                                                                              \
  do {                                                                                              \
    cudnnStatus_t s_ = (expr);                                                                      \
    if (s_ != CUDNN_STATUS_SUCCESS)                                                                 \
      throw gie::Error(gie::StatusCode::kCudnnError, std::string(#expr) + ": " + cudnnGetErrorString(s_)); \
  } while (0)

const char* algo_name(cudnnConvolutionFwdAlgo_t a) {
  switch (a) {
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: return "IMPLICIT_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: return "IMPLICIT_PRECOMP_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_GEMM: return "GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: return "DIRECT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT: return "FFT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: return "FFT_TILING";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: return "WINOGRAD";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: return "WINOGRAD_NONFUSED";
    default: return "UNKNOWN";
  }
}

class CudnnBaseline : public Baseline {
 public:
  explicit CudnnBaseline(size_t ws_limit, bool gemm_only) : ws_limit_(ws_limit), gemm_only_(gemm_only) { CUDNN_OK(cudnnCreate(&h_)); }
  ~CudnnBaseline() override {
    if (xd_) cudnnDestroyTensorDescriptor(xd_);
    if (yd_) cudnnDestroyTensorDescriptor(yd_);
    if (bd_) cudnnDestroyTensorDescriptor(bd_);
    if (wd_) cudnnDestroyFilterDescriptor(wd_);
    if (cd_) cudnnDestroyConvolutionDescriptor(cd_);
    if (ad_) cudnnDestroyActivationDescriptor(ad_);
    cudnnDestroy(h_);
  }
  std::string name() const override { return gemm_only_ ? "cudnn_gemm" : "cudnn"; }
  size_t workspace_bytes() const override { return ws_bytes_; }
  std::string config() const override { return config_; }
  double setup_seconds() const override { return setup_s_; }
  void set_acceptor(Acceptor a) override { acceptor_ = std::move(a); }

  bool setup(const gie::ConvProblem& prob, const void* x, const void* w, const void* bias, void* y,
             std::string* why) override {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    prob_ = prob;
    x_ = x; w_ = w; bias_ = bias;
    const gie::Conv2DParams& p = prob.p;
    cudnnDataType_t dt = CUDNN_DATA_FLOAT;
    if (prob.precision == gie::Precision::kFp16) dt = CUDNN_DATA_HALF;
    if (prob.precision == gie::Precision::kBf16) dt = CUDNN_DATA_BFLOAT16;
    try {
      CUDNN_OK(cudnnCreateTensorDescriptor(&xd_));
      CUDNN_OK(cudnnCreateTensorDescriptor(&yd_));
      CUDNN_OK(cudnnCreateTensorDescriptor(&bd_));
      CUDNN_OK(cudnnCreateFilterDescriptor(&wd_));
      CUDNN_OK(cudnnCreateConvolutionDescriptor(&cd_));
      CUDNN_OK(cudnnCreateActivationDescriptor(&ad_));
      CUDNN_OK(cudnnSetTensor4dDescriptor(xd_, CUDNN_TENSOR_NCHW, dt, p.batch, p.in_channels, p.in_h, p.in_w));
      CUDNN_OK(cudnnSetTensor4dDescriptor(yd_, CUDNN_TENSOR_NCHW, dt, p.batch, p.out_channels, p.out_h(), p.out_w()));
      CUDNN_OK(cudnnSetTensor4dDescriptor(bd_, CUDNN_TENSOR_NCHW, dt, 1, p.out_channels, 1, 1));
      CUDNN_OK(cudnnSetFilter4dDescriptor(wd_, dt, CUDNN_TENSOR_NCHW, p.out_channels, p.in_channels, p.kernel_h, p.kernel_w));
      CUDNN_OK(cudnnSetConvolution2dDescriptor(cd_, p.pad_h, p.pad_w, p.stride_h, p.stride_w, p.dilation_h, p.dilation_w,
                                               CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
      cudnnMathType_t mt = CUDNN_TENSOR_OP_MATH;
      if (prob.precision == gie::Precision::kFp32) mt = CUDNN_FMA_MATH;
      if (prob.precision == gie::Precision::kTf32) mt = CUDNN_DEFAULT_MATH;
      CUDNN_OK(cudnnSetConvolutionMathType(cd_, mt));
      CUDNN_OK(cudnnSetActivationDescriptor(ad_, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

      // Algorithm search on the real operands. The search needs a scratch workspace.
      gie::DeviceBuffer scratch(ws_limit_);
      cudnnConvolutionFwdAlgoPerf_t perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
      int n = 0;
      CUDNN_OK(cudnnFindConvolutionForwardAlgorithmEx(h_, xd_, x, wd_, w, cd_, yd_, y, CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &n,
                                                      perf, scratch.data(), scratch.size()));
      const bool relu = prob.activation == gie::Activation::kReLU;
      fused_bias_act_ = prob.has_bias;  // BiasActivation kernel handles bias (+ReLU or identity)
      // Candidates arrive sorted by their single Find measurement. Keep those that (a) fit the workspace limit,
      // (b) support the fused path if needed and (c) pass the accuracy gate (if one was installed); the final choice below is
      // made on re-timed steady-state measurements, not on the Find order.
      std::string rejected;
      struct Cand { cudnnConvolutionFwdAlgo_t algo; size_t ws; std::string note; float find_ms; };
      std::vector<Cand> accepted;
      // Every algorithm that fits the workspace limit, supports the fused path and meets the accuracy gate is re-timed (at most
      // CUDNN_CONVOLUTION_FWD_ALGO_COUNT, so the cost is bounded): a single noisy Find measurement must not be able to hide the
      // fastest valid algorithm by ranking it outside a short list.
      for (int i = 0; i < n; ++i) {
        if (perf[i].status != CUDNN_STATUS_SUCCESS || perf[i].memory > ws_limit_) continue;
        if (gemm_only_ && perf[i].algo != CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM &&
            perf[i].algo != CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM && perf[i].algo != CUDNN_CONVOLUTION_FWD_ALGO_GEMM)
          continue;
        algo_ = perf[i].algo;
        ws_bytes_ = perf[i].memory;
        algo_note_.clear();
        if (fused_bias_act_ && !probe_bias_act(relu, y)) {
          // The documented fused path is only guaranteed for IMPLICIT_PRECOMP_GEMM; fall back to it.
          algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
          size_t need = 0;
          CUDNN_OK(cudnnGetConvolutionForwardWorkspaceSize(h_, xd_, wd_, cd_, yd_, algo_, &need));
          ws_bytes_ = need;
          if (!probe_bias_act(relu, y)) continue;
          algo_note_ = " (fused path forced to IMPLICIT_PRECOMP_GEMM)";
        }
        if (acceptor_) {
          scratch_.reset();
          scratch_ = gie::DeviceBuffer(ws_bytes_);
          run(y, nullptr, nullptr);
          GIE_CUDA_CHECK(cudaDeviceSynchronize());
          const double ratio = acceptor_(y);
          if (!(ratio <= 1.0)) {
            rejected += std::string(rejected.empty() ? "" : "; ") + algo_name(algo_) + algo_note_ + " (error/tolerance " +
                        std::to_string(ratio) + ")";
            continue;
          }
        }
        accepted.push_back({algo_, ws_bytes_, algo_note_, perf[i].time});
      }
      std::string retime_note;
      if (accepted.size() > 1) {
        // Re-time the accepted candidates with the same events-based steady-state protocol as the benchmark harness.
        double best_us = 1e300;
        size_t best = 0;
        for (size_t c = 0; c < accepted.size(); ++c) {
          algo_ = accepted[c].algo; ws_bytes_ = accepted[c].ws; algo_note_ = accepted[c].note;
          scratch_.reset();
          scratch_ = gie::DeviceBuffer(std::max<size_t>(ws_bytes_, 1));
          const double us = steady_us(y);
          retime_note += std::string(retime_note.empty() ? "" : ", ") + algo_name(algo_) + " find=" +
                         std::to_string(double(accepted[c].find_ms) * 1e3).substr(0, 6) + "us retimed=" + std::to_string(us).substr(0, 6) + "us";
          if (us < best_us) { best_us = us; best = c; }
        }
        algo_ = accepted[best].algo; ws_bytes_ = accepted[best].ws; algo_note_ = accepted[best].note;
      } else if (accepted.size() == 1) {
        algo_ = accepted[0].algo; ws_bytes_ = accepted[0].ws; algo_note_ = accepted[0].note;
      }
      const bool chosen = !accepted.empty();
      if (!chosen) {
        *why = rejected.empty() ? "cudnnFind returned no usable algorithm within the workspace limit"
                                : "no cuDNN algorithm met the accuracy tolerance; rejected: " + rejected;
        return false;
      }
      std::ostringstream os;
      os << "algo=" << algo_name(algo_) << algo_note_ << " math="
         << (mt == CUDNN_FMA_MATH ? "FMA" : (mt == CUDNN_DEFAULT_MATH ? "DEFAULT(tf32 allowed)" : "TENSOR_OP"))
         << " pipeline=" << (fused_bias_act_ ? "conv+bias+act fused" : (relu ? "conv, then relu (2 kernels)" : "conv"));
      if (!retime_note.empty()) os << " | re-timed candidates: " << retime_note;
      if (!rejected.empty()) os << " | faster algorithms rejected for accuracy: " << rejected;
      config_ = os.str();
      scratch_.reset();
      scratch_ = gie::DeviceBuffer(ws_bytes_);
    } catch (const std::exception& e) {
      *why = e.what();
      return false;
    }
    setup_s_ = std::chrono::duration<double>(clock::now() - t0).count();
    return true;
  }

  void run(void* y, void* workspace, cudaStream_t stream) override {
    CUDNN_OK(cudnnSetStream(h_, stream));
    const float alpha = 1.f, beta = 0.f;
    const bool relu = prob_.activation == gie::Activation::kReLU;
    void* ws = workspace ? workspace : scratch_.data();
    if (fused_bias_act_) {
      CUDNN_OK(cudnnConvolutionBiasActivationForward(h_, &alpha, xd_, x_, wd_, w_, cd_, algo_, ws, ws_bytes_, &beta, yd_, y,
                                                     bd_, bias_, relu ? ad_ : identity_desc(), yd_, y));
    } else {
      CUDNN_OK(cudnnConvolutionForward(h_, &alpha, xd_, x_, wd_, w_, cd_, algo_, ws, ws_bytes_, &beta, yd_, y));
      if (relu) CUDNN_OK(cudnnActivationForward(h_, ad_, &alpha, yd_, y, &beta, yd_, y));
    }
  }
  void output_nchw(void*, cudaStream_t) override {}  // already NCHW

 private:
  cudnnActivationDescriptor_t identity_desc() {
    if (!id_) {
      CUDNN_OK(cudnnCreateActivationDescriptor(&id_));
      CUDNN_OK(cudnnSetActivationDescriptor(id_, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0));
    }
    return id_;
  }
  // Median over 5 batches of the per-launch time of run() on a private stream (events; no host timing).
  double steady_us(void* y) {
    cudaStream_t st = nullptr;
    GIE_CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    cudaEvent_t e0, e1;
    GIE_CUDA_CHECK(cudaEventCreate(&e0));
    GIE_CUDA_CHECK(cudaEventCreate(&e1));
    for (int i = 0; i < 5; ++i) run(y, nullptr, st);
    GIE_CUDA_CHECK(cudaStreamSynchronize(st));
    std::vector<double> v;
    for (int b = 0; b < 5; ++b) {
      constexpr int kIters = 20;
      GIE_CUDA_CHECK(cudaEventRecord(e0, st));
      for (int i = 0; i < kIters; ++i) run(y, nullptr, st);
      GIE_CUDA_CHECK(cudaEventRecord(e1, st));
      GIE_CUDA_CHECK(cudaEventSynchronize(e1));
      float ms = 0.f;
      GIE_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      v.push_back(double(ms) * 1e3 / kIters);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    CUDNN_OK(cudnnSetStream(h_, nullptr));  // do not leave the handle bound to the stream destroyed below
    cudaStreamDestroy(st);
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  }
  bool probe_bias_act(bool relu, void* y) {
    gie::DeviceBuffer ws(std::max<size_t>(ws_bytes_, 1));
    const float alpha = 1.f, beta = 0.f;
    cudnnStatus_t s = cudnnConvolutionBiasActivationForward(h_, &alpha, xd_, x_, wd_, w_, cd_, algo_, ws.data(), ws_bytes_, &beta, yd_, y,
                                                            bd_, bias_, relu ? ad_ : identity_desc(), yd_, y);
    cudaDeviceSynchronize();
    (void)cudaGetLastError();
    return s == CUDNN_STATUS_SUCCESS;
  }

  size_t ws_limit_;
  bool gemm_only_ = false;
  cudnnHandle_t h_ = nullptr;
  cudnnTensorDescriptor_t xd_ = nullptr, yd_ = nullptr, bd_ = nullptr;
  cudnnFilterDescriptor_t wd_ = nullptr;
  cudnnConvolutionDescriptor_t cd_ = nullptr;
  cudnnActivationDescriptor_t ad_ = nullptr, id_ = nullptr;
  cudnnConvolutionFwdAlgo_t algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  gie::ConvProblem prob_;
  const void *x_ = nullptr, *w_ = nullptr, *bias_ = nullptr;
  size_t ws_bytes_ = 0;
  gie::DeviceBuffer scratch_;
  bool fused_bias_act_ = false;
  std::string config_, algo_note_;
  Acceptor acceptor_;
  double setup_s_ = 0;
};

}  // namespace

std::unique_ptr<Baseline> make_cudnn_baseline(size_t ws_limit, bool gemm_only) { return std::make_unique<CudnnBaseline>(ws_limit, gemm_only); }
std::string cudnn_version_string() {
  return std::to_string(CUDNN_MAJOR) + "." + std::to_string(CUDNN_MINOR) + "." + std::to_string(CUDNN_PATCHLEVEL) +
         " (runtime " + std::to_string(cudnnGetVersion()) + ")";
}

}  // namespace bench
#else
namespace bench {
std::unique_ptr<Baseline> make_cudnn_baseline(size_t, bool) { return nullptr; }
std::string cudnn_version_string() { return "unavailable"; }
}  // namespace bench
#endif
