// Backend C: explicit im2col + GEMM (library-assisted baseline).
//
// 1. A custom kernel unrolls the input into col[B][K][Ho*Wo], K = C*Kh*Kw, with zero padding
//    written explicitly. The layout keeps the pixel index contiguous, so reads are coalesced
//    for stride 1 and writes always are.
// 2. cuBLAS computes Y_b[O, Ho*Wo] = W[O, K] * col_b[K, Ho*Wo] as one strided-batched GEMM
//    (batch = B, weight stride 0). Because the batch dimension is kept separate, the GEMM
//    writes NCHW directly: no output permutation is materialized.
// 3. Bias/ReLU run as a separate elementwise pass (this backend is intentionally unfused).
//
// Memory cost: the col matrix is B*K*Ho*Wo elements (Kh*Kw times the input for stride 1) which
// is written once and read once -- reported as the workspace size.
// The GEMM is cuBLAS: this backend is a library-assisted baseline, NOT a custom GEMM.
#include <cublas_v2.h>

#include <algorithm>
#include <mutex>

#include "conv_common.cuh"
#include "gpu_inference/fused_ops.h"

namespace gie {
namespace kernels {
namespace {

#define GIE_CUBLAS_CHECK(expr)                                                                      \
  do {                                                                                              \
    cublasStatus_t st_ = (expr);                                                                    \
    if (st_ != CUBLAS_STATUS_SUCCESS)                                                               \
      throw Error(StatusCode::kCublasError, std::string(#expr) + " failed, status " + std::to_string(int(st_))); \
  } while (0)

constexpr int kThreads = 256;
// cuBLAS scratch lives in the CALLER-provided per-call workspace (after the unrolled matrix), never in the plan:
// two streams that execute the same plan concurrently with distinct workspaces therefore never share scratch
// memory, and CUDA Graph capture never allocates. The cuBLAS handle itself is not thread-safe, so its use
// (stream binding + workspace binding + GEMM submission) is serialized with a mutex.
constexpr size_t kCublasWorkspaceBytes = 32u << 20;

size_t align256(size_t v) { return (v + 255) & ~size_t(255); }

template <Precision P>
size_t col_bytes(const ConvProblem& prob) {
  return size_t(prob.p.batch) * prob.p.gemm_k() * prob.p.out_h() * prob.p.out_w() * dtype_size(tensor_dtype(P));
}

struct Im2colState {
  cublasHandle_t handle = nullptr;
  std::mutex mu;
  ~Im2colState() {
    if (handle) cublasDestroy(handle);
  }
};

template <typename T>
__global__ void __launch_bounds__(kThreads)
    im2col_kernel(const T* __restrict__ x, T* __restrict__ col, ConvArgs a) {
  for (int b = blockIdx.z; b < a.B; b += gridDim.z) {
    const T* xb = x + int64_t(b) * a.C * a.H * a.W;
    T* colb = col + int64_t(b) * a.K * a.HWo;
    for (int k = blockIdx.y; k < a.K; k += gridDim.y) {
      const int c = k / (a.KH * a.KW);
      const int rs = k - c * a.KH * a.KW;
      const int r = rs / a.KW;
      const int s = rs - r * a.KW;
      const T* xc = xb + int64_t(c) * a.H * a.W;
      for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < a.HWo; p += gridDim.x * blockDim.x) {
        const int oh = p / a.OW;
        const int ow = p - oh * a.OW;
        const int ih = oh * a.SH - a.PH + r * a.DH;
        const int iw = ow * a.SW - a.PW + s * a.DW;
        T v = from_f32<T>(0.f);
        if (ih >= 0 && ih < a.H && iw >= 0 && iw < a.W) v = xc[int64_t(ih) * a.W + iw];
        colb[int64_t(k) * a.HWo + p] = v;
      }
    }
  }
}

template <Precision P>
void launch_im2col_gemm(const LaunchArgs& la) {
  using T = typename PrecTraits<P>::T;
  const ConvProblem& prob = *la.problem;
  const ConvArgs a = make_args(prob);
  auto* st = static_cast<Im2colState*>(la.plan_state);
  T* col = static_cast<T*>(la.workspace);

  if (la.stage_events) GIE_CUDA_CHECK(cudaEventRecord(la.stage_events[0], la.stream));
  dim3 grid(unsigned(std::min<int64_t>(ceil_div(a.HWo, kThreads), 65535)), unsigned(std::min(a.K, 65535)),
            unsigned(std::min(a.B, 65535)));
  im2col_kernel<T><<<grid, kThreads, 0, la.stream>>>(static_cast<const T*>(la.x), col, a);
  GIE_CUDA_CHECK(cudaGetLastError());
  if (la.stage_events) GIE_CUDA_CHECK(cudaEventRecord(la.stage_events[1], la.stream));

  std::lock_guard<std::mutex> lock(st->mu);
  GIE_CUBLAS_CHECK(cublasSetStream(st->handle, la.stream));
  GIE_CUBLAS_CHECK(cublasSetWorkspace(st->handle, static_cast<char*>(la.workspace) + align256(col_bytes<P>(prob)), kCublasWorkspaceBytes));
  const int m = a.HWo, n = a.O, k = a.K;
  const long long stride_col = (long long)a.K * a.HWo;
  const long long stride_y = (long long)a.O * a.HWo;
  const float alpha = 1.f, beta = 0.f;
  if (P == Precision::kFp32) {
    GIE_CUBLAS_CHECK(cublasSgemmStridedBatched(
        st->handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, reinterpret_cast<const float*>(col), m,
        stride_col, reinterpret_cast<const float*>(la.w), k, 0LL, &beta, reinterpret_cast<float*>(la.y), m,
        stride_y, a.B));
  } else {
    const cudaDataType_t dt = P == Precision::kFp16 ? CUDA_R_16F : (P == Precision::kBf16 ? CUDA_R_16BF : CUDA_R_32F);
    const cublasComputeType_t ct = P == Precision::kTf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    GIE_CUBLAS_CHECK(cublasGemmStridedBatchedEx(st->handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, col, dt,
                                                m, stride_col, la.w, dt, k, 0LL, &beta, la.y, dt, m, stride_y,
                                                a.B, ct, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  if (la.stage_events) GIE_CUDA_CHECK(cudaEventRecord(la.stage_events[2], la.stream));

  launch_bias_act(prob.p, P, la.bias, a.relu != 0, la.y, la.stream);
}

template <Precision P>
std::shared_ptr<void> create_state(const ConvProblem&) {
  auto st = std::make_shared<Im2colState>();
  GIE_CUBLAS_CHECK(cublasCreate(&st->handle));
  // FP32: strict (no TF32). 16-bit/TF32: tensor-op GEMMs but with reduced-precision split-K reductions
  // disallowed, so the documented policy "FP32 accumulation" really holds (by default cuBLAS may
  // accumulate split-K partial sums in 16 bits, which cost ~2 ulp of accuracy on K=4608 reductions here).
  GIE_CUBLAS_CHECK(cublasSetMathMode(st->handle, P == Precision::kFp32
                                                     ? CUBLAS_PEDANTIC_MATH
                                                     : static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
  return st;
}

template <Precision P>
bool supports(const ConvProblem& prob, const DeviceInfo& dev, std::string* why) {
  if (prob.precision != P) {
    if (why) *why = std::string("kernel is ") + to_string(P) + " only";
    return false;
  }
  if ((P == Precision::kBf16 || P == Precision::kTf32) && !dev.supports_tf32_tensor_cores()) {
    if (why) *why = "requires compute capability >= 8.0";
    return false;
  }
  if (align256(col_bytes<P>(prob)) + kCublasWorkspaceBytes > dev.global_mem_bytes / 2) {
    if (why) *why = "im2col workspace exceeds half of device memory";
    return false;
  }
  return true;
}

template <Precision P>
size_t workspace(const ConvProblem& prob) {
  return align256(col_bytes<P>(prob)) + kCublasWorkspaceBytes;  // unrolled matrix + cuBLAS scratch
}

int64_t blocks(const ConvProblem& prob) {
  return ceil_div(int64_t(prob.p.out_h()) * prob.p.out_w(), kThreads) * prob.p.gemm_k() * prob.p.batch;
}

template <Precision P>
void add(std::vector<KernelDesc>& out) {
  using T = typename PrecTraits<P>::T;
  KernelDesc k;
  k.name = std::string("im2col_gemm_") + to_string(P);
  k.backend = Backend::kIm2colGemm;
  k.params = {{"threads", kThreads}};
  k.threads_per_block = kThreads;
  k.uses_tensor_cores = P != Precision::kFp32;  // cuBLAS may select Tensor Core GEMMs for these
  k.function = reinterpret_cast<const void*>(&im2col_kernel<T>);
  k.supports = &supports<P>;
  k.workspace_bytes = &workspace<P>;
  k.dynamic_smem_bytes = [](const ConvProblem&) -> size_t { return 0; };
  k.num_blocks = &blocks;
  k.create_state = &create_state<P>;
  k.launch = &launch_im2col_gemm<P>;
  out.push_back(std::move(k));
}

}  // namespace

void register_im2col(std::vector<KernelDesc>& out) {
  add<Precision::kFp32>(out);
  add<Precision::kFp16>(out);
  add<Precision::kBf16>(out);
  add<Precision::kTf32>(out);
}

}  // namespace kernels
}  // namespace gie
