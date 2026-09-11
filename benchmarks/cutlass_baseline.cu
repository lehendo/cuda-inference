// CUTLASS baseline: NVIDIA CUTLASS 3.x "2.x-style" implicit-GEMM Conv2d fprop (public device API).
//
// Documented configuration:
//  * FP16/BF16-class path: OpClassTensorOp, Sm80 kernels (mma.sync 16x8x16, FP32 accumulate),
//    IteratorAlgorithm::kOptimized, several threadblock shapes; the fastest one that can_implement()
//    the problem is chosen once in setup (search cost reported as setup_seconds).
//  * FP32 path: OpClassSimt (strict FP32 FMA), 128x128x8 threadblock.
//  * CUTLASS convolution is NHWC/KRSC; operands are converted once during setup and the
//    steady-state `run()` is kernel-only. `run_with_conversion()` additionally times the
//    NCHW->NHWC / OIHW->KRSC / NHWC->NCHW layout transforms (tiled, coalesced transpose kernels), so reports can
//    show both kernel-only and conversion-inclusive numbers.
//  * Epilogue fusion: bias is supplied as the epilogue source operand through a zero-stride NHWC
//    TensorRef (every pixel reads bias[c]); LinearCombinationRelu supplies the ReLU.
//  * Requires C_in and C_out multiples of 8 for the 16-bit Tensor Core kernels (16-byte alignment).
#include "baselines.h"

#ifdef GIE_HAS_CUTLASS
#include <cuda_fp16.h>

#include <chrono>
#include <functional>
#include <type_traits>
#include <sstream>
#include <vector>

#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_types.h"
#include "cutlass/version.h"
#include "gpu_inference/tensor.h"

namespace bench {
namespace {

// ---- layout conversion kernels -----------------------------------------------------------------
// Tiled (32x32 shared-memory) batched transpose dst[b][c][r] = src[b][r][c]: both global accesses are coalesced.
// NCHW->NHWC is rows=C, cols=H*W; OIHW->KRSC is batch=O, rows=C, cols=Kh*Kw; NHWC->NCHW is rows=H*W, cols=C.
// The same transpose structure is used by the library's own NHWC Tensor Core path, so the layout-inclusive
// comparison is not skewed by a naive conversion.
template <typename T>
__global__ void transpose_batched(const T* __restrict__ src, T* __restrict__ dst, int rows, int cols) {
  __shared__ T tile[32][33];
  const int64_t base = int64_t(blockIdx.z) * rows * cols;
  const int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
  for (int j = threadIdx.y; j < 32; j += 8) {
    const int r = r0 + j, c = c0 + threadIdx.x;
    if (r < rows && c < cols) tile[j][threadIdx.x] = src[base + int64_t(r) * cols + c];
  }
  __syncthreads();
  for (int j = threadIdx.y; j < 32; j += 8) {
    const int c = c0 + j, r = r0 + threadIdx.x;
    if (c < cols && r < rows) dst[base + int64_t(c) * rows + r] = tile[threadIdx.x][j];
  }
}
template <typename T>
void transpose(const void* s, void* d, int batch, int rows, int cols, cudaStream_t st) {
  dim3 grid(unsigned((cols + 31) / 32), unsigned((rows + 31) / 32), unsigned(batch));
  transpose_batched<T><<<grid, dim3(32, 8), 0, st>>>(static_cast<const T*>(s), static_cast<T*>(d), rows, cols);
}
template <typename T>
void to_nhwc(const void* s, void* d, int N, int C, int H, int W, cudaStream_t st) { transpose<T>(s, d, N, C, H * W, st); }
template <typename T>
void to_nchw(const void* s, void* d, int N, int C, int H, int W, cudaStream_t st) { transpose<T>(s, d, N, H * W, C, st); }

// ---- CUTLASS kernel definitions -----------------------------------------------------------------
using NHWC = cutlass::layout::TensorNHWC;

enum class Epi { kPlain, kBias, kRelu, kBiasRelu };

template <typename Elem, Epi E, int Count>
struct EpilogueFor;
template <typename Elem, int Count>
struct EpilogueFor<Elem, Epi::kPlain, Count> {
  using type = cutlass::epilogue::thread::LinearCombination<Elem, Count, float, float>;
};
template <typename Elem, int Count>
struct EpilogueFor<Elem, Epi::kBias, Count> {
  using type = cutlass::epilogue::thread::LinearCombination<Elem, Count, float, float,
                                                            cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};
template <typename Elem, int Count>
struct EpilogueFor<Elem, Epi::kRelu, Count> {
  using type = cutlass::epilogue::thread::LinearCombinationRelu<Elem, Count, float, float>;
};
template <typename Elem, int Count>
struct EpilogueFor<Elem, Epi::kBiasRelu, Count> {
  using type = cutlass::epilogue::thread::LinearCombinationRelu<Elem, Count, float, float,
                                                                cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

// Count = elements per epilogue access: 128-bit vectors for Tensor Core kernels, 1 for the SIMT kernels
// (the SIMT epilogue shared-memory layout is built for scalar accesses; a wider Count faults with a
// misaligned shared read, found with compute-sanitizer).
template <typename Elem, typename OpClass, typename TB, typename Warp, typename Inst, int Stages, Epi E>
struct ConvDef {
  static constexpr int kCount = std::is_same<OpClass, cutlass::arch::OpClassSimt>::value ? 1 : 128 / cutlass::sizeof_bits<Elem>::value;
  using Epilogue = typename EpilogueFor<Elem, E, kCount>::type;
  using Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
      Elem, NHWC, Elem, NHWC, Elem, NHWC, float, OpClass, cutlass::arch::Sm80, TB, Warp, Inst, Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
  using Op = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
};

struct Runner {
  virtual ~Runner() = default;
  virtual std::string name() const = 0;
  // Returns "" on success, else the reason.
  virtual std::string init(const cutlass::conv::Conv2dProblemSize& ps, void* x, void* w, void* bias, void* y, bool has_source) = 0;
  virtual size_t workspace_bytes() const = 0;
  virtual bool run(cudaStream_t s, void* ws) = 0;
};

template <typename Def, typename Elem>
class RunnerImpl : public Runner {
 public:
  explicit RunnerImpl(std::string n) : name_(std::move(n)) {}
  std::string name() const override { return name_; }
  std::string init(const cutlass::conv::Conv2dProblemSize& ps, void* x, void* w, void* bias, void* y, bool has_source) override {
    using Op = typename Def::Op;
    auto* ex = reinterpret_cast<Elem*>(x);
    auto* ew = reinterpret_cast<Elem*>(w);
    auto* ey = reinterpret_cast<Elem*>(y);
    auto* eb = reinterpret_cast<Elem*>(bias);
    NHWC la = NHWC::packed(ps.activation_extent());
    NHWC lb = NHWC::packed(ps.filter_extent());
    NHWC ld = NHWC::packed(ps.output_extent());
    // zero-stride layout: element (n,h,w,c) of the "source" tensor reads bias[c]
    NHWC lc_bias(NHWC::Stride(0));
    typename Op::Arguments args(
        ps, {ex, la}, {ew, lb},
        has_source ? cutlass::TensorRef<Elem, NHWC>(eb, lc_bias) : cutlass::TensorRef<Elem, NHWC>(ey, ld),
        {ey, ld}, {1.0f, 0.0f});
    cutlass::Status st = op_.can_implement(args);
    if (st != cutlass::Status::kSuccess) return std::string("can_implement: ") + cutlass::cutlassGetStatusString(st);
    ws_bytes_ = op_.get_workspace_size(args);
    ws_.reset();
    ws_ = gie::DeviceBuffer(ws_bytes_);
    st = op_.initialize(args, ws_.data());
    if (st != cutlass::Status::kSuccess) return std::string("initialize: ") + cutlass::cutlassGetStatusString(st);
    return "";
  }
  size_t workspace_bytes() const override { return ws_bytes_; }
  bool run(cudaStream_t s, void*) override { return op_.run(s) == cutlass::Status::kSuccess; }

 private:
  std::string name_;
  typename Def::Op op_;
  gie::DeviceBuffer ws_;
  size_t ws_bytes_ = 0;
};

using Half = cutlass::half_t;
using BF16 = cutlass::bfloat16_t;

template <typename Elem, typename OpClass, typename TB, typename Warp, typename Inst, int Stages, Epi E>
std::unique_ptr<Runner> make(const char* label) {
  return std::make_unique<RunnerImpl<ConvDef<Elem, OpClass, TB, Warp, Inst, Stages, E>, Elem>>(label);
}

using TensorOp = cutlass::arch::OpClassTensorOp;
using Simt = cutlass::arch::OpClassSimt;
template <int M, int N, int K> using GS = cutlass::gemm::GemmShape<M, N, K>;

template <typename Elem, Epi E>
void add_tensorop(std::vector<std::unique_ptr<Runner>>& v, const char* tag) {
  using Inst = GS<16, 8, 16>;
  v.push_back(make<Elem, TensorOp, GS<128, 128, 64>, GS<64, 64, 64>, Inst, 3, E>((std::string("tensorop_128x128x64_w64x64x64_s3_") + tag).c_str()));
  v.push_back(make<Elem, TensorOp, GS<128, 64, 64>, GS<64, 32, 64>, Inst, 3, E>((std::string("tensorop_128x64x64_w64x32x64_s3_") + tag).c_str()));
  v.push_back(make<Elem, TensorOp, GS<64, 64, 64>, GS<32, 32, 64>, Inst, 4, E>((std::string("tensorop_64x64x64_w32x32x64_s4_") + tag).c_str()));
}

template <typename Elem, Epi E>
void add_simt(std::vector<std::unique_ptr<Runner>>& v, const char* tag) {
  v.push_back(make<Elem, Simt, GS<128, 128, 8>, GS<32, 64, 8>, GS<1, 1, 1>, 4, E>((std::string("simt_128x128x8_w32x64x8_s4_") + tag).c_str()));
}

template <typename Elem, bool Tensor, Epi E>
void add_for(std::vector<std::unique_ptr<Runner>>& v, const char* tag) {
  if constexpr (Tensor) add_tensorop<Elem, E>(v, tag);
  else add_simt<Elem, E>(v, tag);
}

template <typename Elem, bool Tensor>
std::vector<std::unique_ptr<Runner>> candidates(Epi e) {
  std::vector<std::unique_ptr<Runner>> v;
  switch (e) {
    case Epi::kPlain: add_for<Elem, Tensor, Epi::kPlain>(v, "plain"); break;
    case Epi::kBias: add_for<Elem, Tensor, Epi::kBias>(v, "bias"); break;
    case Epi::kRelu: add_for<Elem, Tensor, Epi::kRelu>(v, "relu"); break;
    case Epi::kBiasRelu: add_for<Elem, Tensor, Epi::kBiasRelu>(v, "bias_relu"); break;
  }
  return v;
}

class CutlassBaseline : public Baseline {
 public:
  std::string name() const override { return "cutlass"; }
  size_t workspace_bytes() const override { return ws_bytes_; }
  std::string config() const override { return config_; }
  double setup_seconds() const override { return setup_s_; }
  bool needs_layout_conversion() const override { return true; }

  bool setup(const gie::ConvProblem& prob, const void* x, const void* w, const void* bias, void*, std::string* why) override {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    prob_ = prob;
    const gie::Conv2DParams& p = prob.p;
    if (prob.precision == gie::Precision::kTf32) { *why = "CUTLASS TF32 convolution not instantiated in this baseline"; return false; }
    const bool tensor = prob.precision != gie::Precision::kFp32;
    if (tensor && (p.in_channels % 8 != 0 || p.out_channels % 8 != 0)) {
      *why = "CUTLASS 16-bit Tensor Core kernels need C_in and C_out multiples of 8 (alignment)";
      return false;
    }
    esz_ = int(gie::dtype_size(prob.dtype()));
    const size_t xe = size_t(p.input_shape().numel()), we = size_t(p.weight_shape().numel()), ye = size_t(p.output_shape().numel());
    x_nhwc_ = gie::DeviceBuffer(xe * esz_);
    w_krsc_ = gie::DeviceBuffer(we * esz_);
    y_nhwc_ = gie::DeviceBuffer(ye * esz_);
    bias_ = bias;
    x_nchw_ = x;
    w_oihw_ = w;
    convert_inputs(x, w, nullptr);
    GIE_CUDA_CHECK(cudaDeviceSynchronize());

    cutlass::conv::Conv2dProblemSize ps({p.batch, p.in_h, p.in_w, p.in_channels}, {p.out_channels, p.kernel_h, p.kernel_w, p.in_channels},
                                        {p.pad_h, p.pad_h, p.pad_w, p.pad_w}, {p.stride_h, p.stride_w}, {p.dilation_h, p.dilation_w},
                                        cutlass::conv::Mode::kCrossCorrelation, 1, 1);
    const bool relu = prob.activation == gie::Activation::kReLU;
    const Epi epi = prob.has_bias ? (relu ? Epi::kBiasRelu : Epi::kBias) : (relu ? Epi::kRelu : Epi::kPlain);
    std::vector<std::unique_ptr<Runner>> cands;
    switch (prob.precision) {
      case gie::Precision::kFp32: cands = candidates<float, false>(epi); break;
      case gie::Precision::kFp16: cands = candidates<Half, true>(epi); break;
      case gie::Precision::kBf16: cands = candidates<BF16, true>(epi); break;
      default: break;
    }
    std::string last_reason = "no candidate";
    double best_us = 1e30;
    cudaStream_t s = nullptr;
    cudaEvent_t e0, e1;
    GIE_CUDA_CHECK(cudaEventCreate(&e0));
    GIE_CUDA_CHECK(cudaEventCreate(&e1));
    for (auto& c : cands) {
      const std::string r = c->init(ps, x_nhwc_.data(), w_krsc_.data(), const_cast<void*>(bias), y_nhwc_.data(), prob.has_bias);
      if (!r.empty()) { last_reason = c->name() + ": " + r; continue; }
      bool ok = true;
      for (int i = 0; i < 3 && ok; ++i) ok = c->run(s, nullptr);
      cudaEventRecord(e0, s);
      for (int i = 0; i < 10 && ok; ++i) ok = c->run(s, nullptr);
      cudaEventRecord(e1, s);
      cudaEventSynchronize(e1);
      float ms = 0;
      cudaEventElapsedTime(&ms, e0, e1);
      if (!ok || cudaGetLastError() != cudaSuccess) { last_reason = c->name() + ": run failed"; continue; }
      if (ms * 100.0 < best_us) {
        best_us = ms * 100.0;
        best_ = std::move(c);
      }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (!best_) { *why = last_reason; return false; }
    // Re-bind the winner (a later candidate may have re-initialized shared buffers).
    const std::string r = best_->init(ps, x_nhwc_.data(), w_krsc_.data(), const_cast<void*>(bias), y_nhwc_.data(), prob.has_bias);
    if (!r.empty()) { *why = r; return false; }
    ws_bytes_ = best_->workspace_bytes();
    config_ = best_->name() + " (CUTLASS " + std::to_string(CUTLASS_MAJOR) + "." + std::to_string(CUTLASS_MINOR) + "." +
              std::to_string(CUTLASS_PATCH) + ", kOptimized, NHWC)";
    setup_s_ = std::chrono::duration<double>(clock::now() - t0).count();
    return true;
  }

  void run(void* y, void* ws, cudaStream_t s) override {
    (void)y;
    if (!best_->run(s, ws)) throw gie::Error(gie::StatusCode::kInternal, "CUTLASS run failed");
  }
  void output_nchw(void* y_nchw, cudaStream_t s) override { convert_output(y_nchw, s); }
  void run_with_conversion(const void* x, const void* w, void* y, void* ws, cudaStream_t s) override {
    convert_inputs(x, w, s);
    run(nullptr, ws, s);
    convert_output(y, s);
  }

 private:
  void convert_inputs(const void* x, const void* w, cudaStream_t s) {
    const gie::Conv2DParams& p = prob_.p;
    auto go = [&](auto tag) {
      using T = decltype(tag);
      to_nhwc<T>(x, x_nhwc_.data(), p.batch, p.in_channels, p.in_h, p.in_w, s);
      // OIHW -> KRSC is NCHW->NHWC with N=O, C=C_in, H=Kh, W=Kw
      transpose<T>(w, w_krsc_.data(), p.out_channels, p.in_channels, p.kernel_h * p.kernel_w, s);
    };
    if (esz_ == 4) go(float{}); else go(__half{});  // 16-bit conversion is a bit copy, type-agnostic
  }
  void convert_output(void* y, cudaStream_t s) {
    const gie::Conv2DParams& p = prob_.p;
    if (esz_ == 4) to_nchw<float>(y_nhwc_.data(), y, p.batch, p.out_channels, p.out_h(), p.out_w(), s);
    else to_nchw<__half>(y_nhwc_.data(), y, p.batch, p.out_channels, p.out_h(), p.out_w(), s);
  }

  gie::ConvProblem prob_;
  gie::DeviceBuffer x_nhwc_, w_krsc_, y_nhwc_;
  const void *bias_ = nullptr, *x_nchw_ = nullptr, *w_oihw_ = nullptr;
  std::unique_ptr<Runner> best_;
  size_t ws_bytes_ = 0;
  int esz_ = 4;
  std::string config_;
  double setup_s_ = 0;
};

}  // namespace

std::unique_ptr<Baseline> make_cutlass_baseline() { return std::make_unique<CutlassBaseline>(); }
std::string cutlass_version_string() {
  return std::to_string(CUTLASS_MAJOR) + "." + std::to_string(CUTLASS_MINOR) + "." + std::to_string(CUTLASS_PATCH);
}

}  // namespace bench
#else
namespace bench {
std::unique_ptr<Baseline> make_cutlass_baseline() { return nullptr; }
std::string cutlass_version_string() { return "unavailable"; }
}  // namespace bench
#endif
