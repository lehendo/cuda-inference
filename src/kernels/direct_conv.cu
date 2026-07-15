// Backend B: naive direct convolution.
//
// Mapping: one thread per output element, flat index over (b, o, oh, ow) with ow fastest,
// so a warp reads consecutive input pixels (coalesced for stride 1) and the same weights
// (broadcast). There is no data reuse across threads except through the hardware caches:
// every output re-reads its C*KH*KW input window and C*KH*KW weights from global memory,
// i.e. up to KH*KW times redundant input traffic and O(B*OH*OW) times redundant weight
// traffic. This kernel is the correctness baseline for the optimized backends.
#include "conv_common.cuh"
#include "fused_epilogue.cuh"

namespace gie {
namespace kernels {
namespace {

constexpr int kThreads = 256;

template <typename T>
__global__ void __launch_bounds__(kThreads)
    direct_conv_kernel(const T* __restrict__ x, const T* __restrict__ w, const T* __restrict__ bias,
                       T* __restrict__ y, ConvArgs a) {
  const int64_t total = int64_t(a.B) * a.O * a.OH * a.OW;
  for (int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += int64_t(gridDim.x) * blockDim.x) {
    const int ow = int(idx % a.OW);
    const int oh = int((idx / a.OW) % a.OH);
    const int o = int((idx / a.HWo) % a.O);
    const int b = int(idx / (int64_t(a.HWo) * a.O));
    const int hb = oh * a.SH - a.PH;
    const int wb = ow * a.SW - a.PW;
    const T* xb = x + int64_t(b) * a.C * a.H * a.W;
    const T* wo = w + int64_t(o) * a.C * a.KH * a.KW;
    float acc = 0.f;
    for (int c = 0; c < a.C; ++c) {
      const T* xc = xb + int64_t(c) * a.H * a.W;
      const T* wc = wo + c * a.KH * a.KW;
      for (int r = 0; r < a.KH; ++r) {
        const int ih = hb + r * a.DH;
        if (ih < 0 || ih >= a.H) continue;
        for (int s = 0; s < a.KW; ++s) {
          const int iw = wb + s * a.DW;
          if (iw < 0 || iw >= a.W) continue;
          acc = fmaf(to_f32(xc[int64_t(ih) * a.W + iw]), to_f32(wc[r * a.KW + s]), acc);
        }
      }
    }
    y[idx] = from_f32<T>(apply_epilogue(acc, bias, o, a.relu));
  }
}

template <Precision P>
void launch_direct(const LaunchArgs& la) {
  using T = typename PrecTraits<P>::T;
  const ConvArgs a = make_args(*la.problem);
  const int64_t total = int64_t(a.B) * a.O * a.OH * a.OW;
  int64_t blocks = ceil_div(total, kThreads);
  if (blocks > 2147483647LL) blocks = 2147483647LL;
  direct_conv_kernel<T><<<unsigned(blocks), kThreads, 0, la.stream>>>(
      static_cast<const T*>(la.x), static_cast<const T*>(la.w), static_cast<const T*>(la.bias),
      static_cast<T*>(la.y), a);
  GIE_CUDA_CHECK(cudaGetLastError());
}

template <Precision P>
bool supports_direct(const ConvProblem& prob, const DeviceInfo&, std::string* why) {
  if (prob.precision != P) {
    if (why) *why = std::string("kernel is ") + to_string(P) + " only";
    return false;
  }
  return true;
}

int64_t direct_blocks(const ConvProblem& prob) {
  return ceil_div(int64_t(prob.p.batch) * prob.p.out_channels * prob.p.out_h() * prob.p.out_w(), kThreads);
}

template <Precision P>
void add(std::vector<KernelDesc>& out) {
  using T = typename PrecTraits<P>::T;
  KernelDesc k;
  k.name = std::string("direct_") + to_string(P);
  k.backend = Backend::kDirect;
  k.params = {{"threads", kThreads}};
  k.threads_per_block = kThreads;
  k.function = reinterpret_cast<const void*>(&direct_conv_kernel<T>);
  k.supports = &supports_direct<P>;
  k.workspace_bytes = [](const ConvProblem&) -> size_t { return 0; };
  k.dynamic_smem_bytes = [](const ConvProblem&) -> size_t { return 0; };
  k.num_blocks = &direct_blocks;
  k.launch = &launch_direct<P>;
  out.push_back(std::move(k));
}

}  // namespace

void register_direct(std::vector<KernelDesc>& out) {
  add<Precision::kFp32>(out);
  add<Precision::kFp16>(out);
  add<Precision::kBf16>(out);
}

}  // namespace kernels
}  // namespace gie
