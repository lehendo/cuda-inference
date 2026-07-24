// Backend F: custom Tensor Core implicit-GEMM convolution (WMMA, FP16/BF16 in, FP32 accumulate).
//
// Same GEMM view as the SIMT implicit-GEMM backend (M=C_out, N=B*Ho*Wo, K=C_in*Kh*Kw), executed
// on Tensor Cores with nvcuda::wmma m16n16k16 fragments:
//   * A tile (weights, BM x BK, row-major) is loaded with 16-byte vector loads when K % 8 == 0 and
//     the base pointer is 16-byte aligned, otherwise with predicated scalar loads. Rows/columns
//     outside M or K are zero-filled, so irregular channel counts (e.g. C_in=3, K=147) need no
//     separate padded copy of the weights.
//   * B tile (BK x BN) is gathered from NCHW X with the same per-k lookup table as the SIMT
//     kernel (zero padding by predication); each thread keeps one output column so the
//     n -> (b,oh,ow) decomposition happens once.
//   * Shared memory holds two stages of [BM][BK+8] and [BK][BN+8] tiles (the +8-element pad keeps
//     wmma ldm a multiple of 16 B and spreads banks); global loads for the next K tile are issued
//     into registers before the MMAs of the current tile and committed to the other stage, so one
//     __syncthreads per K step suffices. (cp.async cannot express the 2-byte predicated gather.)
//   * Each warp owns a (BM/WM) x (BN/WN) sub-tile = FM x FN accumulators of 16x16 FP32.
//   * Epilogue: each accumulator fragment is staged through a per-warp shared-memory scratch
//     (reusing the pipeline buffers), then bias + ReLU are applied and the result is rounded to
//     the output type and stored directly in NCHW -- no separate permutation or elementwise kernel.
// Requires compute capability >= 8.0 here (BF16 WMMA); FP16 WMMA itself needs >= 7.0.
#include <mma.h>

#include <cstring>

#include "conv_common.cuh"
#include "fused_epilogue.cuh"

namespace gie {
namespace kernels {
namespace {

using namespace nvcuda;

template <int BM, int BN, int BK, int WM, int WN>
struct TcShape {
  static constexpr int NW = WM * WN;
  static constexpr int NT = NW * 32;
  static constexpr int WARP_M = BM / WM;
  static constexpr int WARP_N = BN / WN;
  static constexpr int FM = WARP_M / 16;
  static constexpr int FN = WARP_N / 16;
  static constexpr int LDA = BK + 8;
  static constexpr int LDB = BN + 8;
  static constexpr int LA = (BM * BK / 8) / NT;  // 16-byte A vectors per thread
  static constexpr int KROWS = NT / BN;
  static constexpr int LB = BK / KROWS;          // B gather elements per thread
  static_assert(BM % (WM * 16) == 0 && BN % (WN * 16) == 0, "warp tile must be a multiple of 16x16");
  static_assert(BK % 16 == 0, "BK must be a multiple of the wmma K (16)");
  static_assert((BM * BK / 8) % NT == 0 && (BM * BK / 8) >= NT, "A vectors must divide over threads");
  static_assert(NT % BN == 0 && BK % KROWS == 0, "gather mapping");
  static constexpr size_t kPipelineElems = size_t(2) * (BM * LDA + BK * LDB);
  static constexpr size_t kScratchBytes = size_t(NW) * 16 * 16 * sizeof(float);
  static size_t smem_bytes(size_t elem) {
    const size_t pipe = kPipelineElems * elem;
    return pipe > kScratchBytes ? pipe : kScratchBytes;
  }
};

// ABL (ablation, only instantiated with -DGIE_ENABLE_ABLATIONS, output is WRONG by construction):
//   0 = real kernel, 1 = no global loads (constants instead), 2 = no MMA instructions,
//   3 = no shared-memory stores of the staged tiles. Used to attribute time to gather / MMA / smem.
template <typename T, int BM, int BN, int BK, int WM, int WN, int ABL = 0>
__global__ void __launch_bounds__(TcShape<BM, BN, BK, WM, WN>::NT)
    tc_conv_kernel(const T* __restrict__ x, const T* __restrict__ w, const T* __restrict__ bias,
                   T* __restrict__ y, const LutEntry* __restrict__ lut, ConvArgs a, int vec_a) {
  using S = TcShape<BM, BN, BK, WM, WN>;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  T* sA = reinterpret_cast<T*>(smem_raw);
  T* sB = sA + size_t(2) * BM * S::LDA;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int wm = warp / WN, wn = warp % WN;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;

  // hoisted gather-column decomposition
  const int n_l = tid % BN;
  const int k_row0 = tid / BN;
  const int n_g = n0 + n_l;
  const bool n_ok = n_g < a.N;
  int hbase = 0, wbase = 0;
  const T* xb = x;
  if (n_ok) {
    const int b_img = n_g / a.HWo;
    const int p = n_g - b_img * a.HWo;
    const int oh = p / a.OW;
    const int ow = p - oh * a.OW;
    hbase = oh * a.SH - a.PH;
    wbase = ow * a.SW - a.PW;
    xb = x + int64_t(b_img) * a.C * a.H * a.W;
  }
  const T zero = from_f32<T>(0.f);

  uint4 ra[S::LA];
  T rb[S::LB];

  auto load_tiles = [&](int k0) {
    if (ABL == 1) {
#pragma unroll
      for (int i = 0; i < S::LA; ++i) ra[i] = make_uint4(k0, 1, 2, 3);
#pragma unroll
      for (int i = 0; i < S::LB; ++i) rb[i] = zero;
      return;
    }
#pragma unroll
    for (int i = 0; i < S::LA; ++i) {
      const int idx = tid + i * S::NT;
      const int row = idx / (BK / 8);
      const int kc = (idx % (BK / 8)) * 8;
      const int m = m0 + row, k = k0 + kc;
      if (vec_a) {
        ra[i] = (m < a.M && k < a.K) ? *reinterpret_cast<const uint4*>(w + int64_t(m) * a.K + k) : make_uint4(0, 0, 0, 0);
      } else {
        T tmp[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) tmp[e] = (m < a.M && k + e < a.K) ? w[int64_t(m) * a.K + k + e] : zero;
        memcpy(&ra[i], tmp, sizeof(uint4));
      }
    }
#pragma unroll
    for (int i = 0; i < S::LB; ++i) {
      const int k = k0 + k_row0 + i * S::KROWS;
      T v = zero;
      if (n_ok && k < a.K) {
        const LutEntry e = lut[k];
        const int hi = hbase + e.r_off;
        const int wi = wbase + e.s_off;
        if (hi >= 0 && hi < a.H && wi >= 0 && wi < a.W) v = xb[int64_t(e.c_off) + int64_t(hi) * a.W + wi];
      }
      rb[i] = v;
    }
  };
  auto store_tiles = [&](int buf) {
    if (ABL == 3) return;
    T* dA = sA + size_t(buf) * BM * S::LDA;
    T* dB = sB + size_t(buf) * BK * S::LDB;
#pragma unroll
    for (int i = 0; i < S::LA; ++i) {
      const int idx = tid + i * S::NT;
      *reinterpret_cast<uint4*>(dA + (idx / (BK / 8)) * S::LDA + (idx % (BK / 8)) * 8) = ra[i];
    }
#pragma unroll
    for (int i = 0; i < S::LB; ++i) dB[(k_row0 + i * S::KROWS) * S::LDB + n_l] = rb[i];
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[S::FM][S::FN];
#pragma unroll
  for (int i = 0; i < S::FM; ++i)
#pragma unroll
    for (int j = 0; j < S::FN; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  const int nk = (a.K + BK - 1) / BK;
  load_tiles(0);
  store_tiles(0);
  __syncthreads();

  int buf = 0;
  for (int kt = 0; kt < nk; ++kt) {
    if (kt + 1 < nk) load_tiles((kt + 1) * BK);

    const T* a_base = sA + size_t(buf) * BM * S::LDA + size_t(wm * S::WARP_M) * S::LDA;
    const T* b_base = sB + size_t(buf) * BK * S::LDB + wn * S::WARP_N;
#pragma unroll
    for (int ks = 0; ks < BK / 16; ++ks) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fa[S::FM];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> fb[S::FN];
#pragma unroll
      for (int i = 0; i < S::FM; ++i) wmma::load_matrix_sync(fa[i], a_base + i * 16 * S::LDA + ks * 16, S::LDA);
#pragma unroll
      for (int j = 0; j < S::FN; ++j) wmma::load_matrix_sync(fb[j], b_base + ks * 16 * S::LDB + j * 16, S::LDB);
#pragma unroll
      for (int i = 0; i < S::FM; ++i)
#pragma unroll
        for (int j = 0; j < S::FN; ++j) {
          if (ABL == 2) acc[i][j].x[0] += float(fa[i].x[0]) + float(fb[j].x[0]);  // keep fragment loads alive, no HMMA
          else wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
    }

    if (kt + 1 < nk) store_tiles(buf ^ 1);
    __syncthreads();
    buf ^= 1;
  }

  // Epilogue. All warps passed the final barrier, so the pipeline buffers can be reused as scratch.
  float* scratch = reinterpret_cast<float*>(smem_raw) + warp * 256;
  const int col = lane & 15, rbase = lane >> 4;
#pragma unroll
  for (int fn = 0; fn < S::FN; ++fn) {
    const int n = n0 + wn * S::WARP_N + fn * 16 + col;
    const bool nv = n < a.N;
    const int b_img = nv ? n / a.HWo : 0;
    const int64_t yb = int64_t(b_img) * a.M * a.HWo + (n - b_img * a.HWo);
#pragma unroll
    for (int fm = 0; fm < S::FM; ++fm) {
      wmma::store_matrix_sync(scratch, acc[fm][fn], 16, wmma::mem_row_major);
      __syncwarp();
#pragma unroll
      for (int rr = 0; rr < 8; ++rr) {
        const int row = rbase + 2 * rr;
        const int m = m0 + wm * S::WARP_M + fm * 16 + row;
        if (nv && m < a.M) y[yb + int64_t(m) * a.HWo] = from_f32<T>(apply_epilogue(scratch[row * 16 + col], bias, m, a.relu));
      }
      __syncwarp();
    }
  }
}

template <Precision P, int BM, int BN, int BK, int WM, int WN>
size_t dyn_smem(const ConvProblem&) {
  using T = typename PrecTraits<P>::T;
  return TcShape<BM, BN, BK, WM, WN>::smem_bytes(sizeof(T));
}

template <Precision P, int BM, int BN, int BK, int WM, int WN>
bool supports(const ConvProblem& prob, const DeviceInfo& dev, std::string* why) {
  if (prob.precision != P) {
    if (why) *why = std::string("kernel is ") + to_string(P) + " only";
    return false;
  }
  if (!dev.supports_bf16_tensor_cores()) {
    if (why) *why = "requires compute capability >= 8.0 (device is sm_" + std::to_string(dev.cc()) + ")";
    return false;
  }
  const size_t need = dyn_smem<P, BM, BN, BK, WM, WN>(prob);
  if (need > dev.smem_per_block_optin) {
    if (why) *why = "needs " + std::to_string(need / 1024) + " KiB shared memory (limit " + std::to_string(dev.smem_per_block_optin / 1024) + " KiB)";
    return false;
  }
  if (ceil_div(prob.p.out_channels, BM) > 65535) {
    if (why) *why = "grid.y limit (65535) exceeded";
    return false;
  }
  return true;
}

template <Precision P, int BM, int BN, int BK, int WM, int WN, int ABL = 0>
void launch(const LaunchArgs& la) {
  using T = typename PrecTraits<P>::T;
  const ConvArgs a = make_args(*la.problem);
  const size_t smem = dyn_smem<P, BM, BN, BK, WM, WN>(*la.problem);
  auto kern = &tc_conv_kernel<T, BM, BN, BK, WM, WN, ABL>;
  set_max_dynamic_smem(reinterpret_cast<const void*>(kern), smem);
  const int vec_a = (a.K % 8 == 0 && reinterpret_cast<uintptr_t>(la.w) % 16 == 0) ? 1 : 0;
  dim3 grid(unsigned(ceil_div(a.N, BN)), unsigned(ceil_div(a.M, BM)));
  kern<<<grid, TcShape<BM, BN, BK, WM, WN>::NT, smem, la.stream>>>(
      static_cast<const T*>(la.x), static_cast<const T*>(la.w), static_cast<const T*>(la.bias),
      static_cast<T*>(la.y), static_cast<const LutEntry*>(la.plan_data), a, vec_a);
  GIE_CUDA_CHECK(cudaGetLastError());
}

template <int BM, int BN>
int64_t num_blocks(const ConvProblem& prob) {
  return ceil_div(prob.p.gemm_n(), BN) * ceil_div(prob.p.gemm_m(), BM);
}

template <Precision P, int BM, int BN, int BK, int WM, int WN, int ABL = 0>
void add(std::vector<KernelDesc>& out) {
  using T = typename PrecTraits<P>::T;
  using S = TcShape<BM, BN, BK, WM, WN>;
  KernelDesc k;
  k.name = std::string(ABL ? "ablation" + std::to_string(ABL) + "_tensor_core_" : "tensor_core_") + to_string(P) + "_" + std::to_string(BM) + "x" + std::to_string(BN) + "x" +
           std::to_string(BK) + "_w" + std::to_string(WM) + "x" + std::to_string(WN);
  k.backend = Backend::kTensorCore;
  k.params = {{"block_m", BM}, {"block_n", BN}, {"block_k", BK}, {"warps_m", WM}, {"warps_n", WN}, {"stages", 2}};
  k.threads_per_block = S::NT;
  k.tile_m = BM;
  k.tile_n = BN;
  k.tile_k = BK;
  k.uses_tensor_cores = true;
  k.needs_lut = true;
  k.function = reinterpret_cast<const void*>(&tc_conv_kernel<T, BM, BN, BK, WM, WN, ABL>);
  k.supports = &supports<P, BM, BN, BK, WM, WN>;
  k.workspace_bytes = [](const ConvProblem&) -> size_t { return 0; };
  k.dynamic_smem_bytes = &dyn_smem<P, BM, BN, BK, WM, WN>;
  k.num_blocks = &num_blocks<BM, BN>;
  k.launch = &launch<P, BM, BN, BK, WM, WN, ABL>;
  out.push_back(std::move(k));
}

template <Precision P>
void add_all(std::vector<KernelDesc>& out) {
  add<P, 128, 128, 32, 2, 4>(out);
  add<P, 128, 64, 32, 4, 2>(out);
  add<P, 64, 128, 32, 2, 4>(out);
  add<P, 64, 64, 32, 2, 2>(out);
  add<P, 256, 128, 32, 4, 2>(out);
  add<P, 128, 256, 32, 2, 4>(out);
  add<P, 32, 64, 32, 1, 2>(out);
  add<P, 64, 128, 64, 2, 4>(out);
  add<P, 128, 128, 64, 2, 4>(out);
}

}  // namespace

void register_tensor_core(std::vector<KernelDesc>& out) {
  add_all<Precision::kFp16>(out);
  add_all<Precision::kBf16>(out);
#ifdef GIE_ENABLE_ABLATIONS
  // Diagnostic variants with deliberately wrong output (see ABL above). They live in the registry only in
  // ablation builds, where the autotuner's validation rejects them; they exist to attribute kernel time.
  add<Precision::kFp16, 64, 128, 32, 2, 4, 1>(out);
  add<Precision::kFp16, 64, 128, 32, 2, 4, 2>(out);
  add<Precision::kFp16, 64, 128, 32, 2, 4, 3>(out);
  add<Precision::kFp16, 128, 128, 32, 2, 4, 1>(out);
  add<Precision::kFp16, 128, 128, 32, 2, 4, 2>(out);
  add<Precision::kFp16, 128, 128, 32, 2, 4, 3>(out);
#endif
}

}  // namespace kernels
}  // namespace gie
