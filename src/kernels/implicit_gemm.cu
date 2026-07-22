// Backend E: custom implicit-GEMM convolution on CUDA cores.
//
// Mapping (see ALGORITHMS): convolution is the matrix product C[M,N] = A[M,K] * B[K,N] with
//   M = C_out,  N = B*Ho*Wo,  K = C_in*Kh*Kw
//   A[m,k]  = W[m, c, r, s]                         (the OIHW weight tensor, already row-major MxK)
//   B[k,n]  = X[b, c, oh*Sh-Ph+r*Dh, ow*Sw-Pw+s*Dw]  (zero outside the image)  with k=(c,r,s), n=(b,oh,ow)
//   C[m,n]  -> Y[b, m, oh, ow]
// B is never materialized: each block gathers its BK x BN tile of B straight from X while filling
// shared memory. The (c,r,s) -> address decomposition is a per-k lookup table built once per plan
// (LutEntry), so the main loop has no integer division; the n -> (b,oh,ow) decomposition is hoisted
// out of the K loop because each thread always handles the same output column.
//
// Kernel structure: classic register-blocked SGEMM (BM x BN x BK block tile, TM x TN thread tile,
// outer-product accumulation) with two shared-memory stages and register-staged prefetch of the
// next tile, one __syncthreads per K step. M/N/K need not divide the tile sizes (predicated, zero-
// filled loads and guarded stores). Bias + ReLU are applied in the store.
#include "conv_common.cuh"
#include "fused_epilogue.cuh"

namespace gie {
namespace kernels {
namespace {

template <int BM, int BN, int BK, int TM, int TN>
struct IgShape {
  static constexpr int TX = BN / TN;           // thread columns
  static constexpr int TY = BM / TM;           // thread rows
  static constexpr int NT = TX * TY;           // threads per block
  static constexpr int KROWS = NT / BN;        // k-rows covered per gather pass
  static constexpr int LB = BK / KROWS;        // B elements per thread per tile
  static constexpr int LA = (BM * BK) / NT;    // A elements per thread per tile
  static_assert(BN % TN == 0 && BM % TM == 0, "thread tile must divide block tile");
  static_assert(NT % BN == 0, "gather mapping needs threads to be a multiple of BN");
  static_assert(BK % KROWS == 0, "BK must be a multiple of the gather row count");
  static_assert((BM * BK) % NT == 0, "A tile must divide evenly over threads");
  static_assert(TM % 4 == 0 && TN % 4 == 0, "float4 fragment loads");
};

// ABL (ablation; only instantiated with -DGIE_ENABLE_ABLATIONS, output is WRONG by construction):
//   0 = real kernel, 1 = no global loads (constants), 2 = serial add chain on one accumulator (NOT an FMA-free
//   kernel: it is latency-bound, so it overstates the cost), 3 = no global stores (FMAs kept alive).
// MINB: minBlocksPerMultiprocessor for __launch_bounds__ (0 = unconstrained, the original kernels). The 8x8-tile kernels
//   compile to ~140 registers, which leaves one block (8 warps) per SM; MINB=2 caps registers at 128 for two resident blocks.
// IL: interleaved thread tile. Original mapping gives thread (ty,tx) the contiguous columns [tx*TN, tx*TN+TN); with TN=8 the
//   two float4 fragment loads of a quarter-warp then stride 32 bytes and hit 2-way shared-memory bank conflicts. IL=1 splits the
//   thread tile into chunks of 4 that are TX*4 (TY*4) apart, so every float4 load of 8 adjacent lanes is a contiguous 128 bytes.
template <typename T, int BM, int BN, int BK, int TM, int TN, int ABL = 0, int MINB = 0, int IL = 0>
__global__ void __launch_bounds__(IgShape<BM, BN, BK, TM, TN>::NT, MINB)
    implicit_gemm_kernel(const T* __restrict__ x, const T* __restrict__ w, const T* __restrict__ bias,
                         T* __restrict__ y, const LutEntry* __restrict__ lut, ConvArgs a) {
  using S = IgShape<BM, BN, BK, TM, TN>;
  constexpr int PAD = 4;
  __shared__ __align__(16) float As[2][BK][BM + PAD];
  __shared__ __align__(16) float Bs[2][BK][BN + PAD];

  const int tid = threadIdx.x;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int ty = tid / S::TX, tx = tid % S::TX;

  // --- hoisted decomposition of this thread's gather column n ---
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

  // first row / column (within the block tile) of fragment chunk i/j (i, j multiples of 4)
  auto row_of = [&](int i) { return IL ? (i / 4) * (S::TY * 4) + ty * 4 : ty * TM + i; };
  auto col_of = [&](int j) { return IL ? (j / 4) * (S::TX * 4) + tx * 4 : tx * TN + j; };

  float ra[S::LA];
  float rb[S::LB];

  auto load_tiles = [&](int k0) {
    if (ABL == 1) {
#pragma unroll
      for (int i = 0; i < S::LA; ++i) ra[i] = float(k0 + i);
#pragma unroll
      for (int i = 0; i < S::LB; ++i) rb[i] = float(k0 - i);
      return;
    }
#pragma unroll
    for (int i = 0; i < S::LA; ++i) {
      const int idx = tid + i * S::NT;
      const int ak = idx % BK;
      const int am = idx / BK;
      const int m = m0 + am, k = k0 + ak;
      ra[i] = (m < a.M && k < a.K) ? to_f32(w[int64_t(m) * a.K + k]) : 0.f;
    }
#pragma unroll
    for (int i = 0; i < S::LB; ++i) {
      const int k = k0 + k_row0 + i * S::KROWS;
      float v = 0.f;
      if (n_ok && k < a.K) {
        const LutEntry e = lut[k];
        const int hi = hbase + e.r_off;
        const int wi = wbase + e.s_off;
        if (hi >= 0 && hi < a.H && wi >= 0 && wi < a.W) v = to_f32(xb[int64_t(e.c_off) + int64_t(hi) * a.W + wi]);
      }
      rb[i] = v;
    }
  };
  auto store_tiles = [&](int buf) {
#pragma unroll
    for (int i = 0; i < S::LA; ++i) {
      const int idx = tid + i * S::NT;
      As[buf][idx % BK][idx / BK] = ra[i];
    }
#pragma unroll
    for (int i = 0; i < S::LB; ++i) Bs[buf][k_row0 + i * S::KROWS][n_l] = rb[i];
  };

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

  const int nk = (a.K + BK - 1) / BK;
  load_tiles(0);
  store_tiles(0);
  __syncthreads();

  int buf = 0;
  for (int kt = 0; kt < nk; ++kt) {
    if (kt + 1 < nk) load_tiles((kt + 1) * BK);  // global loads in flight during the FMAs below

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float af[TM], bf[TN];
#pragma unroll
      for (int i = 0; i < TM; i += 4) *reinterpret_cast<float4*>(&af[i]) = *reinterpret_cast<const float4*>(&As[buf][kk][row_of(i)]);
#pragma unroll
      for (int j = 0; j < TN; j += 4) *reinterpret_cast<float4*>(&bf[j]) = *reinterpret_cast<const float4*>(&Bs[buf][kk][col_of(j)]);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          if (ABL == 2) acc[0][0] += af[i] + bf[j];
          else acc[i][j] = fmaf(af[i], bf[j], acc[i][j]);
        }
    }

    if (kt + 1 < nk) store_tiles(buf ^ 1);  // other buffer was last read before the previous barrier
    __syncthreads();
    buf ^= 1;
  }

  if (ABL == 3) {  // keep every FMA alive but skip all global stores
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) sum += acc[i][j];
    if (sum == 12345.678f) y[0] = from_f32<T>(sum);
    return;
  }
  // --- fused epilogue ---
  int64_t ybase[TN];
  bool nvalid[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = n0 + col_of(j & ~3) + (j & 3);
    nvalid[j] = n < a.N;
    const int b_img = nvalid[j] ? n / a.HWo : 0;
    const int p = n - b_img * a.HWo;
    ybase[j] = int64_t(b_img) * a.M * a.HWo + p;
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int m = m0 + row_of(i & ~3) + (i & 3);
    if (m >= a.M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      if (nvalid[j]) y[ybase[j] + int64_t(m) * a.HWo] = from_f32<T>(apply_epilogue(acc[i][j], bias, m, a.relu));
    }
  }
}

template <Precision P, int BM, int BN, int BK, int TM, int TN>
bool supports(const ConvProblem& prob, const DeviceInfo&, std::string* why) {
  if (prob.precision != P) {
    if (why) *why = std::string("kernel is ") + to_string(P) + " only";
    return false;
  }
  if (ceil_div(prob.p.out_channels, BM) > 65535) {
    if (why) *why = "grid.y limit (65535) exceeded";
    return false;
  }
  return true;
}

template <Precision P, int BM, int BN, int BK, int TM, int TN, int ABL = 0, int MINB = 0, int IL = 0>
void launch(const LaunchArgs& la) {
  using T = typename PrecTraits<P>::T;
  const ConvArgs a = make_args(*la.problem);
  dim3 grid(unsigned(ceil_div(a.N, BN)), unsigned(ceil_div(a.M, BM)));
  implicit_gemm_kernel<T, BM, BN, BK, TM, TN, ABL, MINB, IL><<<grid, IgShape<BM, BN, BK, TM, TN>::NT, 0, la.stream>>>(
      static_cast<const T*>(la.x), static_cast<const T*>(la.w), static_cast<const T*>(la.bias),
      static_cast<T*>(la.y), static_cast<const LutEntry*>(la.plan_data), a);
  GIE_CUDA_CHECK(cudaGetLastError());
}

template <int BM, int BN>
int64_t num_blocks(const ConvProblem& prob) {
  return ceil_div(prob.p.gemm_n(), BN) * ceil_div(prob.p.gemm_m(), BM);
}

template <Precision P, int BM, int BN, int BK, int TM, int TN, int ABL = 0, int MINB = 0, int IL = 0>
void add(std::vector<KernelDesc>& out) {
  using T = typename PrecTraits<P>::T;
  KernelDesc k;
  k.name = std::string(ABL ? "ablation" + std::to_string(ABL) + "_implicit_gemm_" : "implicit_gemm_") + to_string(P) + "_" + std::to_string(BM) + "x" + std::to_string(BN) + "x" +
           std::to_string(BK) + "_t" + std::to_string(TM) + "x" + std::to_string(TN) +
           (MINB ? "_mb" + std::to_string(MINB) : std::string()) + (IL ? "_il" : "");
  k.backend = Backend::kImplicitGemm;
  k.params = {{"block_m", BM}, {"block_n", BN}, {"block_k", BK}, {"thread_m", TM}, {"thread_n", TN}};
  k.threads_per_block = IgShape<BM, BN, BK, TM, TN>::NT;
  k.tile_m = BM;
  k.tile_n = BN;
  k.tile_k = BK;
  k.needs_lut = true;
  k.function = reinterpret_cast<const void*>(&implicit_gemm_kernel<T, BM, BN, BK, TM, TN, ABL, MINB, IL>);
  k.supports = &supports<P, BM, BN, BK, TM, TN>;
  k.workspace_bytes = [](const ConvProblem&) -> size_t { return 0; };
  k.dynamic_smem_bytes = [](const ConvProblem&) -> size_t { return 0; };
  k.num_blocks = &num_blocks<BM, BN>;
  k.launch = &launch<P, BM, BN, BK, TM, TN, ABL, MINB, IL>;
  out.push_back(std::move(k));
}

template <Precision P>
void add_all(std::vector<KernelDesc>& out) {
  add<P, 128, 128, 8, 8, 8>(out);
  add<P, 128, 128, 16, 8, 8>(out);
  add<P, 128, 64, 8, 8, 4>(out);
  add<P, 64, 128, 8, 4, 8>(out);
  add<P, 64, 64, 8, 4, 4>(out);
  add<P, 64, 64, 16, 4, 4>(out);
  add<P, 32, 128, 8, 4, 4>(out);
  add<P, 128, 32, 8, 4, 4>(out);
  // Occupancy-constrained variants of the large-tile kernels (separate entries; the originals above are unchanged).
  add<P, 128, 128, 8, 8, 8, 0, 2>(out);
  add<P, 128, 128, 16, 8, 8, 0, 2>(out);
  add<P, 128, 64, 8, 8, 4, 0, 3>(out);
  add<P, 64, 128, 8, 4, 8, 0, 3>(out);
  // Interleaved thread tiles (bank-conflict-free fragment loads), with and without the occupancy cap.
  add<P, 128, 128, 8, 8, 8, 0, 0, 1>(out);
  add<P, 128, 128, 8, 8, 8, 0, 2, 1>(out);
  add<P, 128, 128, 16, 8, 8, 0, 2, 1>(out);
  add<P, 128, 64, 8, 8, 4, 0, 0, 1>(out);
  add<P, 128, 64, 8, 8, 4, 0, 3, 1>(out);
  add<P, 64, 128, 8, 4, 8, 0, 0, 1>(out);
  add<P, 64, 128, 8, 4, 8, 0, 3, 1>(out);
}

}  // namespace

void register_implicit_gemm(std::vector<KernelDesc>& out) {
  add_all<Precision::kFp32>(out);
  add_all<Precision::kFp16>(out);
#ifdef GIE_ENABLE_ABLATIONS
  add<Precision::kFp32, 64, 64, 8, 4, 4, 1>(out);
  add<Precision::kFp32, 64, 64, 8, 4, 4, 2>(out);
  add<Precision::kFp32, 64, 64, 8, 4, 4, 3>(out);
  add<Precision::kFp32, 128, 128, 8, 8, 8, 1>(out);
  add<Precision::kFp32, 128, 128, 8, 8, 8, 2>(out);
  add<Precision::kFp32, 128, 128, 8, 8, 8, 3>(out);
#endif
}

}  // namespace kernels
}  // namespace gie
