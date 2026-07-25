// Backend F (second kernel family): Tensor Core implicit GEMM with mma.sync + ldmatrix + multi-stage cp.async.
//
// Why a second family. The WMMA kernel (tensor_core_conv.cu) gathers the B operand from NCHW with scalar,
// predicated 2-byte loads -- many instructions per useful byte, and no cp.async possible. This family trades
// one extra pass for 16-byte vector gathers: per call it
//   1. transposes X  NCHW -> NHWC and W OIHW -> [O][R][S][C]   (two bandwidth-bound kernels; part of the
//      timed pipeline and of the reported workspace, unlike a library baseline that is handed NHWC),
//   2. runs the implicit-GEMM main kernel on those buffers,
//   3. (split-K only) reduces FP32 partial sums and applies the fused epilogue.
// GEMM view: M = C_out, N = B*Ho*Wo, K = R*S*C (k ordered (r,s,c) so 8 consecutive k are 8 consecutive
// channels of one input pixel = one 16-byte vector). A = W[M][K] and the gathered B^T[N][K] both have K
// contiguous, which is exactly the .row.col operand layout of mma.m16n8k16 and what ldmatrix reads.
//
// Main kernel: BMxBNxBK block tile, WMxWN warps, STAGES-deep cp.async pipeline (zero-fill for padding and
// out-of-range rows via the cp.async src-size operand), padded shared tiles (BK+8 halves) so ldmatrix is
// bank-conflict free, FP32 accumulators in mma fragments, epilogue (bias, ReLU, rounding) fused into the NCHW
// store. Split-K (grid.z) parallelizes long reductions on small-N problems; it is a registered search dimension.
// Requirements: C_in % 8 == 0, sm_80+. FP16 and BF16 storage, FP32 accumulate.
#include "conv_common.cuh"
#include "fused_epilogue.cuh"

namespace gie {
namespace kernels {
namespace {

// ---- PTX helpers ------------------------------------------------------------------------------------------
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gmem_src, bool pred) {
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  const int src_size = pred ? 16 : 0;  // 0 => nothing is read, destination is zero-filled
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(gmem_src), "r"(src_size));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* smem_ptr) {
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

template <typename T> struct Mma;
template <> struct Mma<__half> {
  static __device__ __forceinline__ void run(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
};
template <> struct Mma<__nv_bfloat16> {
  static __device__ __forceinline__ void run(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
};

// ---- layout conversion kernels ----------------------------------------------------------------------------
// src [B][C][HW] -> dst [B][HW][C] (16-bit elements); 32x32 shared-memory tile, coalesced both ways.
__global__ void transpose_c_hw(const unsigned short* __restrict__ src, unsigned short* __restrict__ dst, int C, int HW) {
  __shared__ unsigned short tile[32][33];
  const int b = blockIdx.z;
  const int hw0 = blockIdx.x * 32, c0 = blockIdx.y * 32;
  const unsigned short* s = src + int64_t(b) * C * HW;
  unsigned short* d = dst + int64_t(b) * C * HW;
  for (int j = threadIdx.y; j < 32; j += 8) {
    const int c = c0 + j, hw = hw0 + threadIdx.x;
    if (c < C && hw < HW) tile[j][threadIdx.x] = s[int64_t(c) * HW + hw];
  }
  __syncthreads();
  for (int j = threadIdx.y; j < 32; j += 8) {
    const int hw = hw0 + j, c = c0 + threadIdx.x;
    if (hw < HW && c < C) d[int64_t(hw) * C + c] = tile[threadIdx.x][j];
  }
}

void launch_transpose(const void* src, void* dst, int batch, int C, int HW, cudaStream_t s) {
  dim3 grid(unsigned((HW + 31) / 32), unsigned((C + 31) / 32), unsigned(batch));
  transpose_c_hw<<<grid, dim3(32, 8), 0, s>>>(static_cast<const unsigned short*>(src), static_cast<unsigned short*>(dst), C, HW);
  GIE_CUDA_CHECK(cudaGetLastError());
}

// ---- main kernel ------------------------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, int WM, int WN>
struct MmaShape {
  static constexpr int NT = WM * WN * 32;
  static constexpr int WARP_M = BM / WM, WARP_N = BN / WN;
  static constexpr int FM = WARP_M / 16, FN = WARP_N / 8;
  static constexpr int LDS = BK + 8;                 // padded row length (halves)
  static constexpr int CPR = BK / 8;                 // 16-byte chunks per tile row
  static constexpr int LA = BM * CPR / NT, LB = BN * CPR / NT;
  static constexpr size_t kSmemBytes = size_t(STAGES) * (BM + BN) * LDS * 2;
  static_assert(BM % (WM * 16) == 0 && BN % (WN * 16) == 0, "warp tile must be a multiple of 16x16");
  static_assert((BM * CPR) % NT == 0 && (BN * CPR) % NT == 0 && NT % CPR == 0, "tile chunks must divide evenly over threads");
  static_assert(BK % 16 == 0 && STAGES >= 2, "BK multiple of 16, at least 2 stages");
};

template <typename T, int BM, int BN, int BK, int STAGES, int WM, int WN>
__global__ void __launch_bounds__(MmaShape<BM, BN, BK, STAGES, WM, WN>::NT)
    tcmma_kernel(const T* __restrict__ xn, const T* __restrict__ wk, const T* __restrict__ bias, T* __restrict__ y,
                 float* __restrict__ partial, const LutEntry* __restrict__ lutv, ConvArgs a, int split) {
  using S = MmaShape<BM, BN, BK, STAGES, WM, WN>;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  T* sA = reinterpret_cast<T*>(smem_raw);                       // [STAGES][BM][LDS]
  T* sB = sA + size_t(STAGES) * BM * S::LDS;                    // [STAGES][BN][LDS]

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

  const int ktiles = (a.K + BK - 1) / BK;
  const int kt_begin = int((int64_t(ktiles) * blockIdx.z) / split);
  const int kt_end = int((int64_t(ktiles) * (blockIdx.z + 1)) / split);
  const int nk = kt_end - kt_begin;
  const int kchunks = a.K / 8;  // K % 8 == 0 by construction (C % 8 == 0)

  // Thread-fixed chunk coordinates. A: row ra_i, B: row rb_i (a tile row = one pixel for B); kc is the same for all i.
  const int kc = tid % S::CPR;
  int64_t b_xoff[S::LB];
  int b_hb[S::LB], b_wb[S::LB];
  bool b_ok[S::LB];
#pragma unroll
  for (int i = 0; i < S::LB; ++i) {
    const int row = (tid + i * S::NT) / S::CPR;
    const int n = n0 + row;
    b_ok[i] = n < a.N;
    int bi = 0, hb = 0, wb = 0;
    if (b_ok[i]) {
      bi = n / a.HWo;
      const int p = n - bi * a.HWo;
      const int oh = p / a.OW;
      const int ow = p - oh * a.OW;
      hb = oh * a.SH - a.PH;
      wb = ow * a.SW - a.PW;
    }
    b_xoff[i] = int64_t(bi) * a.H * a.W * a.C;
    b_hb[i] = hb;
    b_wb[i] = wb;
  }

  auto load_stage = [&](int stage, int kt) {
    T* dA = sA + size_t(stage) * BM * S::LDS;
    T* dB = sB + size_t(stage) * BN * S::LDS;
    const int kv = kt * S::CPR + kc;  // 8-wide k chunk index
#pragma unroll
    for (int i = 0; i < S::LA; ++i) {
      const int row = (tid + i * S::NT) / S::CPR;
      const int m = m0 + row;
      const bool ok = m < a.M && kv < kchunks;
      cp_async16(dA + row * S::LDS + kc * 8, wk + (ok ? int64_t(m) * a.K + kv * 8 : 0), ok);
    }
    LutEntry e{0, 0, 0, 0};
    if (kv < kchunks) e = lutv[kv];
#pragma unroll
    for (int i = 0; i < S::LB; ++i) {
      const int row = (tid + i * S::NT) / S::CPR;
      const int hi = b_hb[i] + e.r_off, wi = b_wb[i] + e.s_off;
      const bool ok = b_ok[i] && kv < kchunks && hi >= 0 && hi < a.H && wi >= 0 && wi < a.W;
      cp_async16(dB + row * S::LDS + kc * 8, xn + (ok ? b_xoff[i] + (int64_t(hi) * a.W + wi) * a.C + e.c_off : 0), ok);
    }
  };

  float acc[S::FM][S::FN][4];
#pragma unroll
  for (int i = 0; i < S::FM; ++i)
#pragma unroll
    for (int j = 0; j < S::FN; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < nk) load_stage(s, kt_begin + s);
    cp_async_commit();
  }

  for (int it = 0; it < nk; ++it) {
    cp_async_wait<STAGES - 2>();  // tile `it` has landed
    __syncthreads();              // ... for every thread, and everyone finished reading the stage we refill next
    {
      const int nxt = it + STAGES - 1;
      if (nxt < nk) load_stage(nxt % STAGES, kt_begin + nxt);
      cp_async_commit();
    }
    const T* tA = sA + size_t(it % STAGES) * BM * S::LDS;
    const T* tB = sB + size_t(it % STAGES) * BN * S::LDS;
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      unsigned af[S::FM][4], bf[S::FN][2];
#pragma unroll
      for (int i = 0; i < S::FM; ++i)
        ldmatrix_x4(af[i], tA + (wm * S::WARP_M + i * 16 + (lane & 15)) * S::LDS + kk * 16 + (lane >> 4) * 8);
#pragma unroll
      for (int j = 0; j < S::FN / 2; ++j) {
        unsigned r[4];
        ldmatrix_x4(r, tB + (wn * S::WARP_N + j * 16 + (lane & 7) + (lane >> 4) * 8) * S::LDS + kk * 16 + ((lane >> 3) & 1) * 8);
        bf[2 * j][0] = r[0]; bf[2 * j][1] = r[1]; bf[2 * j + 1][0] = r[2]; bf[2 * j + 1][1] = r[3];
      }
#pragma unroll
      for (int i = 0; i < S::FM; ++i)
#pragma unroll
        for (int j = 0; j < S::FN; ++j) Mma<T>::run(acc[i][j], af[i], bf[j][0], bf[j][1]);
    }
  }
  cp_async_wait<0>();

  // ---- epilogue: accumulator (i,j): rows lane/4 and lane/4+8, columns 2*(lane%4)+{0,1} of the 16x8 tile ----
  const int gid = lane >> 2, tig = lane & 3;
  const int64_t out_elems = int64_t(a.B) * a.M * a.HWo;
#pragma unroll
  for (int j = 0; j < S::FN; ++j) {
    int nn[2], bi[2], pp[2];
    bool nv[2];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      nn[c] = n0 + wn * S::WARP_N + j * 8 + tig * 2 + c;
      nv[c] = nn[c] < a.N;
      bi[c] = nv[c] ? nn[c] / a.HWo : 0;
      pp[c] = nn[c] - bi[c] * a.HWo;
    }
    const bool pair_ok = nv[0] && nv[1] && bi[0] == bi[1];
#pragma unroll
    for (int i = 0; i < S::FM; ++i) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int m = m0 + wm * S::WARP_M + i * 16 + gid + h * 8;
        if (m >= a.M) continue;
        const int64_t base0 = (int64_t(bi[0]) * a.M + m) * a.HWo + pp[0];
        const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
        if (split > 1) {
          float* pz = partial + int64_t(blockIdx.z) * out_elems;
          if (nv[0]) pz[base0] = v0;
          if (nv[1]) pz[(int64_t(bi[1]) * a.M + m) * a.HWo + pp[1]] = v1;
        } else {
          const T o0 = from_f32<T>(apply_epilogue(v0, bias, m, a.relu));
          const T o1 = from_f32<T>(apply_epilogue(v1, bias, m, a.relu));
          // 4-byte paired store: needs the actual address (not just the element offset) to be 4-byte aligned,
          // because the caller-provided y pointer may itself be offset by 2 bytes.
          if (pair_ok && (reinterpret_cast<uintptr_t>(y + base0) & 3) == 0) {
            const unsigned short u0 = *reinterpret_cast<const unsigned short*>(&o0), u1 = *reinterpret_cast<const unsigned short*>(&o1);
            reinterpret_cast<unsigned*>(y + base0)[0] = (unsigned(u1) << 16) | unsigned(u0);
          } else {
            if (nv[0]) y[base0] = o0;
            if (nv[1]) y[(int64_t(bi[1]) * a.M + m) * a.HWo + pp[1]] = o1;
          }
        }
      }
    }
  }
}

// Sums split-K partials, applies bias/ReLU, rounds once to T.
template <typename T>
__global__ void splitk_reduce(const float* __restrict__ partial, const T* __restrict__ bias, T* __restrict__ y, int64_t total,
                              int split, int M, int HWo, int relu) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x) {
    float s = 0.f;
    for (int z = 0; z < split; ++z) s += partial[int64_t(z) * total + i];
    y[i] = from_f32<T>(apply_epilogue(s, bias, int((i / HWo) % M), relu));
  }
}

// ---- workspace layout ------------------------------------------------------------------------------------------
size_t align256(size_t v) { return (v + 255) & ~size_t(255); }
struct WsLayout {
  size_t xn_off, wk_off, part_off, total;
};
WsLayout ws_layout(const ConvProblem& prob, int split) {
  const Conv2DParams& p = prob.p;
  const size_t esz = dtype_size(prob.dtype());
  WsLayout l;
  l.xn_off = 0;
  l.wk_off = align256(size_t(p.input_shape().numel()) * esz);
  l.part_off = l.wk_off + align256(size_t(p.weight_shape().numel()) * esz);
  l.total = l.part_off + (split > 1 ? align256(size_t(split) * size_t(p.output_shape().numel()) * 4) : 0);
  return l;
}

template <Precision P, int BM, int BN, int BK, int STAGES, int WM, int WN, int SPLIT>
bool supports(const ConvProblem& prob, const DeviceInfo& dev, std::string* why) {
  auto no = [&](const std::string& m) { if (why) *why = m; return false; };
  if (prob.precision != P) return no(std::string("kernel is ") + to_string(P) + " only");
  if (!dev.supports_bf16_tensor_cores()) return no("requires compute capability >= 8.0");
  if (prob.p.in_channels % 8 != 0) return no("needs C_in % 8 == 0 for 16-byte gathers");
  if (MmaShape<BM, BN, BK, STAGES, WM, WN>::kSmemBytes > dev.smem_per_block_optin)
    return no("needs " + std::to_string(MmaShape<BM, BN, BK, STAGES, WM, WN>::kSmemBytes / 1024) + " KiB shared memory (limit " +
              std::to_string(dev.smem_per_block_optin / 1024) + " KiB)");
  if (ceil_div(prob.p.out_channels, BM) > 65535) return no("grid.y limit exceeded");
  // the layout-conversion kernels use grid.z = batch (X) and grid.z = C_out (W)
  if (prob.p.batch > 65535 || prob.p.out_channels > 65535) return no("grid.z limit (65535) exceeded by the layout transposes");
  if (SPLIT > 1 && ceil_div(prob.p.gemm_k(), BK) < 2 * SPLIT) return no("reduction too short for this split-K factor");
  if (ws_layout(prob, SPLIT).total > dev.global_mem_bytes / 4) return no("workspace exceeds a quarter of device memory");
  return true;
}

template <Precision P, int BM, int BN, int BK, int STAGES, int WM, int WN, int SPLIT>
void launch(const LaunchArgs& la) {
  using T = typename PrecTraits<P>::T;
  using S = MmaShape<BM, BN, BK, STAGES, WM, WN>;
  const ConvProblem& prob = *la.problem;
  const ConvArgs a = make_args(prob);
  const WsLayout l = ws_layout(prob, SPLIT);
  char* ws = static_cast<char*>(la.workspace);
  T* xn = reinterpret_cast<T*>(ws + l.xn_off);
  T* wk = reinterpret_cast<T*>(ws + l.wk_off);
  float* part = SPLIT > 1 ? reinterpret_cast<float*>(ws + l.part_off) : nullptr;

  launch_transpose(la.x, xn, a.B, a.C, a.H * a.W, la.stream);                    // NCHW -> NHWC
  launch_transpose(la.w, wk, a.O, a.C, a.KH * a.KW, la.stream);                  // [O][C][RS] -> [O][RS][C]
  auto kern = &tcmma_kernel<T, BM, BN, BK, STAGES, WM, WN>;
  set_max_dynamic_smem(reinterpret_cast<const void*>(kern), S::kSmemBytes);
  dim3 grid(unsigned(ceil_div(a.N, BN)), unsigned(ceil_div(a.M, BM)), unsigned(SPLIT));
  kern<<<grid, S::NT, S::kSmemBytes, la.stream>>>(xn, wk, static_cast<const T*>(la.bias), static_cast<T*>(la.y), part,
                                                  static_cast<const LutEntry*>(la.plan_data), a, SPLIT);
  GIE_CUDA_CHECK(cudaGetLastError());
  if (SPLIT > 1) {
    const int64_t total = int64_t(a.B) * a.M * a.HWo;
    int64_t blocks = (total + 255) / 256;
    if (blocks > 1 << 20) blocks = 1 << 20;
    splitk_reduce<T><<<unsigned(blocks), 256, 0, la.stream>>>(part, static_cast<const T*>(la.bias), static_cast<T*>(la.y), total, SPLIT, a.M, a.HWo, a.relu);
    GIE_CUDA_CHECK(cudaGetLastError());
  }
}

template <Precision P, int SPLIT>
size_t workspace(const ConvProblem& prob) { return ws_layout(prob, SPLIT).total; }

template <int BM, int BN, int SPLIT>
int64_t num_blocks(const ConvProblem& prob) { return ceil_div(prob.p.gemm_n(), BN) * ceil_div(prob.p.gemm_m(), BM) * SPLIT; }

template <int SPLIT>
double extra_traffic(const ConvProblem& prob) {
  // transposes read + write X and W once more; split-K writes and re-reads FP32 partials
  const double esz = double(dtype_size(prob.dtype()));
  const double xw = (double(prob.p.input_shape().numel()) + double(prob.p.weight_shape().numel())) * esz;
  const double part = SPLIT > 1 ? 2.0 * SPLIT * double(prob.p.output_shape().numel()) * 4.0 : 0.0;
  return 2.0 * xw + part;
}

template <Precision P, int BM, int BN, int BK, int STAGES, int WM, int WN, int SPLIT>
void add(std::vector<KernelDesc>& out) {
  using T = typename PrecTraits<P>::T;
  using S = MmaShape<BM, BN, BK, STAGES, WM, WN>;
  KernelDesc k;
  k.name = std::string("tcmma_") + to_string(P) + "_" + std::to_string(BM) + "x" + std::to_string(BN) + "x" + std::to_string(BK) + "_s" +
           std::to_string(STAGES) + "_w" + std::to_string(WM) + "x" + std::to_string(WN) + "_sk" + std::to_string(SPLIT);
  k.backend = Backend::kTensorCore;
  k.params = {{"block_m", BM}, {"block_n", BN}, {"block_k", BK}, {"warps_m", WM}, {"warps_n", WN}, {"stages", STAGES}, {"split_k", SPLIT}};
  k.threads_per_block = S::NT;
  k.tile_m = BM;
  k.tile_n = BN;
  k.tile_k = BK;
  k.uses_tensor_cores = true;
  k.needs_lut = true;
  k.lut_chunked = true;
  k.launches = 3 + (SPLIT > 1 ? 1 : 0);
  k.efficiency_prior = 0.5;  // multi-stage cp.async + ldmatrix + mma.sync: expected to reach a larger fraction of peak than WMMA
  k.extra_traffic_bytes = &extra_traffic<SPLIT>;
  k.function = reinterpret_cast<const void*>(&tcmma_kernel<T, BM, BN, BK, STAGES, WM, WN>);
  k.supports = &supports<P, BM, BN, BK, STAGES, WM, WN, SPLIT>;
  k.workspace_bytes = &workspace<P, SPLIT>;
  k.dynamic_smem_bytes = [](const ConvProblem&) -> size_t { return S::kSmemBytes; };
  k.num_blocks = &num_blocks<BM, BN, SPLIT>;
  k.launch = &launch<P, BM, BN, BK, STAGES, WM, WN, SPLIT>;
  out.push_back(std::move(k));
}

template <Precision P>
void add_all(std::vector<KernelDesc>& out) {
  add<P, 128, 128, 32, 3, 2, 4, 1>(out);
  add<P, 128, 128, 32, 2, 2, 4, 1>(out);  // 2 stages: less shared memory => more resident blocks on 99 KiB parts
  add<P, 64, 128, 32, 3, 2, 4, 1>(out);
  add<P, 128, 64, 32, 3, 4, 2, 1>(out);
  add<P, 128, 64, 32, 4, 4, 2, 1>(out);
  add<P, 64, 128, 32, 4, 2, 4, 1>(out);
  add<P, 64, 64, 32, 4, 2, 2, 1>(out);
  add<P, 256, 128, 32, 3, 4, 2, 1>(out);
  add<P, 128, 128, 64, 3, 2, 4, 1>(out);
  add<P, 64, 64, 64, 3, 2, 2, 1>(out);
  // split-K variants for long reductions with few output tiles (small batch, deep layers)
  add<P, 64, 64, 32, 4, 2, 2, 2>(out);
  add<P, 64, 64, 32, 4, 2, 2, 4>(out);
  add<P, 64, 64, 32, 4, 2, 2, 8>(out);
  add<P, 64, 128, 32, 4, 2, 4, 4>(out);
  add<P, 128, 64, 32, 4, 4, 2, 4>(out);
  add<P, 128, 128, 32, 3, 2, 4, 2>(out);
}

}  // namespace

void register_tensor_core_mma(std::vector<KernelDesc>& out) {
  add_all<Precision::kFp16>(out);
  add_all<Precision::kBf16>(out);
}

}  // namespace kernels
}  // namespace gie
