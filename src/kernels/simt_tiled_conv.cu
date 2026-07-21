// Backend D: shared-memory tiled, register-blocked direct convolution on CUDA cores.
//
// Decomposition. A thread block produces a TC x (TH x TW) tile of the output of one image.
// It loops over input channels in chunks of CC:
//   1. cooperatively stages the input patch  sIn[CC][IH][IW]  (IH = (TH-1)*Sh + (Kh-1)*Dh + 1,
//      IW likewise) with consecutive threads reading consecutive pixels, zero-filling the
//      padding halo and channel tail, and the weight slab sW[TC][CC*Kh*Kw];
//   2. every thread accumulates an RC x RP register tile (RC output channels x RP adjacent
//      pixels) over the chunk, reading RC weights and RP input values from shared memory per
//      (c,r,s) tap for RC*RP FMAs.
// Reuse: each staged input value is used by TC channels and by up to Kh*Kw overlapping windows;
// each staged weight is reused across the TH*TW pixels. Compared with the naive kernel, global
// traffic drops by roughly min(Kh*Kw, window overlap) for inputs and TH*TW for weights.
// Shared-memory loads per FMA are (RC+RP)/(RC*RP). Bias+ReLU are fused into the store.
//
// Template parameters (exactly the autotuner's search dimensions):
//   TC channels per block, TH x TW pixels per block, RC/RP per-thread register tile, CC chunk.
#include "conv_common.cuh"
#include "fused_epilogue.cuh"

namespace gie {
namespace kernels {
namespace {

template <int TC, int TH, int TW, int RC, int RP>
struct SimtShape {
  static constexpr int NTX = TW / RP;
  static constexpr int kThreads = NTX * TH * (TC / RC);
  static_assert(TW % RP == 0 && TC % RC == 0, "register tile must divide block tile");
  static_assert((NTX * TH) % 32 == 0, "a warp must not straddle output-channel groups (weight broadcast)");
};

template <typename T, int TC, int TH, int TW, int RC, int RP, int CC>
__global__ void __launch_bounds__(SimtShape<TC, TH, TW, RC, RP>::kThreads)
    simt_tiled_kernel(const T* __restrict__ x, const T* __restrict__ w, const T* __restrict__ bias,
                      T* __restrict__ y, ConvArgs a, int tiles_w) {
  constexpr int NTX = SimtShape<TC, TH, TW, RC, RP>::NTX;
  constexpr int NT = SimtShape<TC, TH, TW, RC, RP>::kThreads;
  extern __shared__ __align__(16) float smem[];

  const int IH = (TH - 1) * a.SH + (a.KH - 1) * a.DH + 1;
  const int IW = (TW - 1) * a.SW + (a.KW - 1) * a.DW + 1;
  const int KHW = a.KH * a.KW;
  const int WK = CC * KHW;
  float* sIn = smem;
  float* sW = smem + CC * IH * IW;

  const int tid = threadIdx.x;
  const int tx = tid % NTX;
  const int ty = (tid / NTX) % TH;
  const int tc = tid / (NTX * TH);
  const int tile_x = blockIdx.x % tiles_w;
  const int tile_y = blockIdx.x / tiles_w;
  const int oh0 = tile_y * TH, ow0 = tile_x * TW;
  const int o0 = blockIdx.y * TC;
  const int b = blockIdx.z;
  const int ih0 = oh0 * a.SH - a.PH, iw0 = ow0 * a.SW - a.PW;
  const T* xb = x + int64_t(b) * a.C * a.H * a.W;

  float acc[RC][RP];
#pragma unroll
  for (int i = 0; i < RC; ++i)
#pragma unroll
    for (int j = 0; j < RP; ++j) acc[i][j] = 0.f;

  for (int c0 = 0; c0 < a.C; c0 += CC) {
    for (int idx = tid; idx < CC * IH * IW; idx += NT) {
      const int iw_l = idx % IW;
      const int t = idx / IW;
      const int ih_l = t % IH;
      const int cc = t / IH;
      const int c = c0 + cc, ih = ih0 + ih_l, iw = iw0 + iw_l;
      float v = 0.f;
      if (c < a.C && ih >= 0 && ih < a.H && iw >= 0 && iw < a.W)
        v = to_f32(xb[(int64_t(c) * a.H + ih) * a.W + iw]);
      sIn[idx] = v;
    }
    for (int idx = tid; idx < TC * WK; idx += NT) {
      const int kk = idx % WK;
      const int oc = idx / WK;
      const int cc = kk / KHW;
      const int rs = kk - cc * KHW;
      const int o = o0 + oc, c = c0 + cc;
      float v = 0.f;
      if (o < a.O && c < a.C) v = to_f32(w[(int64_t(o) * a.C + c) * KHW + rs]);
      sW[idx] = v;
    }
    __syncthreads();

    for (int cc = 0; cc < CC; ++cc) {
      for (int r = 0; r < a.KH; ++r) {
        const float* inrow = sIn + (cc * IH + ty * a.SH + r * a.DH) * IW + tx * RP * a.SW;
        const float* wrow = sW + (tc * RC) * WK + cc * KHW + r * a.KW;
        for (int s = 0; s < a.KW; ++s) {
          float wv[RC];
          float xv[RP];
#pragma unroll
          for (int i = 0; i < RC; ++i) wv[i] = wrow[i * WK + s];
#pragma unroll
          for (int j = 0; j < RP; ++j) xv[j] = inrow[j * a.SW + s * a.DW];
#pragma unroll
          for (int i = 0; i < RC; ++i)
#pragma unroll
            for (int j = 0; j < RP; ++j) acc[i][j] = fmaf(wv[i], xv[j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  const int oh = oh0 + ty;
  if (oh >= a.OH) return;
#pragma unroll
  for (int i = 0; i < RC; ++i) {
    const int o = o0 + tc * RC + i;
    if (o >= a.O) continue;
    T* yrow = y + ((int64_t(b) * a.O + o) * a.OH + oh) * a.OW;
#pragma unroll
    for (int j = 0; j < RP; ++j) {
      const int ow = ow0 + tx * RP + j;
      if (ow < a.OW) yrow[ow] = from_f32<T>(apply_epilogue(acc[i][j], bias, o, a.relu));
    }
  }
}

template <int TC, int TH, int TW, int CC>
size_t smem_bytes(const ConvProblem& prob) {
  const Conv2DParams& p = prob.p;
  const int ih = (TH - 1) * p.stride_h + (p.kernel_h - 1) * p.dilation_h + 1;
  const int iw = (TW - 1) * p.stride_w + (p.kernel_w - 1) * p.dilation_w + 1;
  return (size_t(CC) * ih * iw + size_t(TC) * CC * p.kernel_h * p.kernel_w) * sizeof(float);
}

template <int TC, int TH, int TW, int CC>
bool fits(const ConvProblem& prob, const DeviceInfo& dev, std::string* why) {
  if (smem_bytes<TC, TH, TW, CC>(prob) > dev.smem_per_block_optin) {
    if (why)
      *why = "needs " + std::to_string(smem_bytes<TC, TH, TW, CC>(prob) / 1024) + " KiB shared memory per block (limit " +
             std::to_string(dev.smem_per_block_optin / 1024) + " KiB)";
    return false;
  }
  const Conv2DParams& p = prob.p;
  if (ceil_div(p.out_channels, TC) > 65535 || p.batch > 65535) {
    if (why) *why = "grid.y/grid.z limit (65535) exceeded";
    return false;
  }
  return true;
}

template <Precision P, int TC, int TH, int TW, int RC, int RP, int CC>
bool supports(const ConvProblem& prob, const DeviceInfo& dev, std::string* why) {
  if (prob.precision != P) {
    if (why) *why = std::string("kernel is ") + to_string(P) + " only";
    return false;
  }
  return fits<TC, TH, TW, CC>(prob, dev, why);
}

template <Precision P, int TC, int TH, int TW, int RC, int RP, int CC>
void launch(const LaunchArgs& la) {
  using T = typename PrecTraits<P>::T;
  const ConvArgs a = make_args(*la.problem);
  const int tiles_w = int(ceil_div(a.OW, TW));
  const int tiles_h = int(ceil_div(a.OH, TH));
  const size_t smem = smem_bytes<TC, TH, TW, CC>(*la.problem);
  auto kern = &simt_tiled_kernel<T, TC, TH, TW, RC, RP, CC>;
  set_max_dynamic_smem(reinterpret_cast<const void*>(kern), smem);
  dim3 grid(unsigned(tiles_w * tiles_h), unsigned(ceil_div(a.O, TC)), unsigned(a.B));
  kern<<<grid, SimtShape<TC, TH, TW, RC, RP>::kThreads, smem, la.stream>>>(
      static_cast<const T*>(la.x), static_cast<const T*>(la.w), static_cast<const T*>(la.bias),
      static_cast<T*>(la.y), a, tiles_w);
  GIE_CUDA_CHECK(cudaGetLastError());
}

template <int TC, int TH, int TW, int CC>
int64_t num_blocks(const ConvProblem& prob) {
  const Conv2DParams& p = prob.p;
  return ceil_div(p.out_w(), TW) * ceil_div(p.out_h(), TH) * ceil_div(p.out_channels, TC) * p.batch;
}

template <int TC, int TH, int TW, int CC>
double tile_eff(const ConvProblem& prob) {
  const Conv2DParams& p = prob.p;
  const double o = double(p.out_channels) / (ceil_div(p.out_channels, TC) * TC);
  const double s = double(p.out_h()) * p.out_w() /
                   (double(ceil_div(p.out_h(), TH) * TH) * double(ceil_div(p.out_w(), TW) * TW));
  const double c = double(p.in_channels) / (ceil_div(p.in_channels, CC) * CC);
  return o * s * c;
}

template <Precision P, int TC, int TH, int TW, int RC, int RP, int CC>
void add(std::vector<KernelDesc>& out) {
  using T = typename PrecTraits<P>::T;
  KernelDesc k;
  k.name = std::string("simt_tiled_") + to_string(P) + "_tc" + std::to_string(TC) + "_t" + std::to_string(TH) + "x" +
           std::to_string(TW) + "_r" + std::to_string(RC) + "x" + std::to_string(RP) + "_cc" + std::to_string(CC);
  k.backend = Backend::kSimtTiled;
  k.params = {{"tile_channels", TC}, {"tile_h", TH}, {"tile_w", TW}, {"reg_channels", RC}, {"reg_pixels", RP}, {"chan_chunk", CC}};
  k.threads_per_block = SimtShape<TC, TH, TW, RC, RP>::kThreads;
  k.tile_m = TC;
  k.tile_n = TH * TW;
  k.tile_k = CC;
  k.function = reinterpret_cast<const void*>(&simt_tiled_kernel<T, TC, TH, TW, RC, RP, CC>);
  k.supports = &supports<P, TC, TH, TW, RC, RP, CC>;
  k.workspace_bytes = [](const ConvProblem&) -> size_t { return 0; };
  k.dynamic_smem_bytes = &smem_bytes<TC, TH, TW, CC>;
  k.num_blocks = &num_blocks<TC, TH, TW, CC>;
  k.tile_efficiency = &tile_eff<TC, TH, TW, CC>;
  k.launch = &launch<P, TC, TH, TW, RC, RP, CC>;
  out.push_back(std::move(k));
}

template <Precision P, int CC>
void add_all(std::vector<KernelDesc>& out) {
  add<P, 64, 8, 16, 8, 4, CC>(out);
  add<P, 32, 8, 8, 4, 2, CC>(out);
  add<P, 16, 8, 8, 2, 2, CC>(out);
  add<P, 32, 4, 32, 4, 4, CC>(out);
  add<P, 64, 8, 8, 8, 2, CC>(out);
}

}  // namespace

void register_simt_tiled(std::vector<KernelDesc>& out) {
  add_all<Precision::kFp32, 4>(out);
  add_all<Precision::kFp32, 8>(out);
  add_all<Precision::kFp16, 4>(out);
  add_all<Precision::kFp16, 8>(out);
}

}  // namespace kernels
}  // namespace gie
