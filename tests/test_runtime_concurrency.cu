// Runtime concurrency and lifetime properties of ExecutionPlan:
//  * creating a plan must not wait for (or delay) unrelated work on other streams,
//  * a plan used immediately after creation on a brand-new non-blocking stream must see its uploaded tables,
//  * one plan reused concurrently on several streams / host threads with distinct workspaces stays correct,
//  * changing inputs between executions of the same plan is handled,
//  * unaligned user pointers and workspaces are either handled or rejected, never silently misused.
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>

#include "conv_test_common.h"

using namespace gie;

namespace {

__global__ void spin_kernel(long long cycles) {
  const long long start = clock64();
  while (clock64() - start < cycles) {
  }
}

struct Stream {
  cudaStream_t s = nullptr;
  Stream() { GIE_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(s); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
};

// Problem whose K >= 256 so every split-K variant is valid; C % 8 == 0 so the mma.sync family applies.
ConvProblem make_prob(Precision prec, int variant = 0) {
  Conv2DParams p = gt::P(2, 32 + 8 * (variant % 3), 14 + variant % 4, 15 + variant % 3, 32 + 8 * (variant % 2), 3, 3, 1, 1, 1, 1);
  return gt::make_problem(p, prec, true, true);
}

std::vector<std::string> kernels_for(Precision prec) {
  if (prec == Precision::kFp16)
    return {"direct_fp16", "im2col_gemm_fp16", "simt_tiled_fp16_tc32_t8x8_r4x2_cc8", "implicit_gemm_fp16_64x64x8_t4x4",
            "tensor_core_fp16_64x64x32_w2x2", "tcmma_fp16_128x128x32_s3_w2x4_sk1", "tcmma_fp16_64x64x32_s4_w2x2_sk4"};
  return {"direct_fp32", "im2col_gemm_fp32", "simt_tiled_fp32_tc32_t8x8_r4x2_cc8", "implicit_gemm_fp32_64x64x8_t4x4"};
}

bool needs_tc(const std::string& n) { return n.rfind("tensor_core", 0) == 0 || n.rfind("tcmma", 0) == 0; }

bool double_round_for(const KernelDesc& k, const ConvProblem& prob) {
  return k.backend == Backend::kIm2colGemm && prob.precision != Precision::kFp32 && (prob.has_bias || prob.activation != Activation::kNone);
}

}  // namespace

TEST(plan_creation_does_not_wait_for_other_streams) {
  // compute-sanitizer serializes and instruments API calls, so a wall-clock overlap assertion is meaningless under it.
  // The test still runs (and must pass) natively; scripts/run_sanitizers.sh sets GIE_UNDER_SANITIZER.
  if (std::getenv("GIE_UNDER_SANITIZER") != nullptr) return;
  const DeviceInfo& dev = DeviceInfo::current();
  const ConvProblem prob = make_prob(Precision::kFp32);
  const KernelDesc* k = find_kernel("implicit_gemm_fp32_64x64x8_t4x4");
  Stream busy;
  const long long cycles = static_cast<long long>(double(dev.max_sm_clock_khz) * 1e3 * 0.8);  // ~0.8 s of device time
  spin_kernel<<<1, 1, 0, busy.s>>>(cycles);
  GIE_CUDA_CHECK(cudaGetLastError());
  const auto t0 = std::chrono::steady_clock::now();
  auto plan = ExecutionPlan::create(prob, *k);  // uploads the lookup table
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  CHECK(plan != nullptr);
  const cudaError_t pending = cudaStreamQuery(busy.s);  // the spinner must still be running: create() did not wait for it
  CHECK_MSG(pending == cudaErrorNotReady, "plan creation took " << ms << " ms and the unrelated 0.8 s kernel had already finished");
  CHECK_MSG(ms < 400.0, "plan creation took " << ms << " ms (should not wait for unrelated streams)");
  GIE_CUDA_CHECK(cudaStreamSynchronize(busy.s));
  (void)cudaGetLastError();
}

TEST(create_then_use_on_fresh_stream_is_race_free) {
  // Many fresh plans (fresh lookup-table uploads) used immediately on brand-new non-blocking streams with no other
  // synchronization than waiting on that stream. A missing completion guarantee would surface as wrong output or
  // an illegal address in the lookup-table gather.
  struct Case { std::string kernel; ConvProblem prob; gt::RefBundle ref; };
  std::vector<Case> cases;
  const char* names[] = {"implicit_gemm_fp16_64x64x8_t4x4", "tensor_core_fp16_64x64x32_w2x2", "tcmma_fp16_64x64x32_s4_w2x2_sk1",
                         "tcmma_fp16_64x64x32_s4_w2x2_sk4", "implicit_gemm_fp32_64x64x8_t4x4"};
  for (const char* n : names) {
    const KernelDesc* k = find_kernel(n);
    CHECK(k != nullptr);
    if (!k) continue;
    if (needs_tc(n) && !DeviceInfo::current().supports_bf16_tensor_cores()) continue;
    for (int v = 0; v < 4; ++v) {
      const ConvProblem prob = make_prob(std::string(n).find("fp32") != std::string::npos ? Precision::kFp32 : Precision::kFp16, v);
      if (!k->supports(prob, DeviceInfo::current(), nullptr)) continue;
      cases.push_back({n, prob, gt::make_bundle(prob, 1000 + uint64_t(v), Distribution::kNormal)});
    }
  }
  int bad = 0, runs = 0;
  for (int iter = 0; iter < 400; ++iter) {
    Case& c = cases[size_t(iter) % cases.size()];
    const KernelDesc& k = *find_kernel(c.kernel);
    auto plan = ExecutionPlan::create(c.prob, k);
    DeviceBuffer ws(plan->workspace_bytes());
    Stream st;
    GIE_CUDA_CHECK(cudaMemsetAsync(c.ref.data.dy.data(), 0xFF, c.ref.data.dy.size(), st.s));
    plan->execute_unchecked(c.ref.data.dx.data(), c.ref.data.dw.data(), c.ref.data.db.data(), c.ref.data.dy.data(), ws.data(), ws.size(), st.s);
    GIE_CUDA_CHECK(cudaStreamSynchronize(st.s));
    ++runs;
    if (!gt::compare_device_output(c.ref.data.dy.data(), c.ref, double_round_for(k, c.prob)).passed()) ++bad;
  }
  CHECK_MSG(bad == 0, bad << " of " << runs << " create-then-use runs produced a wrong result");
}

TEST(one_plan_concurrent_streams_changing_inputs) {
  const bool tc = DeviceInfo::current().supports_bf16_tensor_cores();
  for (Precision prec : {Precision::kFp16, Precision::kFp32}) {
    const ConvProblem prob = make_prob(prec);
    for (const std::string& name : kernels_for(prec)) {
      if (needs_tc(name) && !tc) continue;
      const KernelDesc* k = find_kernel(name);
      CHECK_MSG(k != nullptr, name);
      if (!k || !k->supports(prob, DeviceInfo::current(), nullptr)) continue;
      auto plan = ExecutionPlan::create(prob, *k);  // ONE plan shared by all streams below
      constexpr int S = 4;
      std::vector<std::unique_ptr<Stream>> streams;
      std::vector<gt::RefBundle> bufs, even, odd;  // bufs: tensors the plan reads/writes; even/odd: two datasets + references
      std::vector<DeviceBuffer> ws;
      for (int s = 0; s < S; ++s) {
        streams.push_back(std::make_unique<Stream>());
        bufs.push_back(gt::make_bundle(prob, 1 + uint64_t(s), Distribution::kUniform));
        even.push_back(gt::make_bundle(prob, 11 + uint64_t(s), Distribution::kNormal));
        odd.push_back(gt::make_bundle(prob, 91 + uint64_t(s), Distribution::kSparseSigned));
        ws.emplace_back(plan->workspace_bytes());
      }
      for (int round = 0; round < 10; ++round) {
        for (int s = 0; s < S; ++s) {  // new data for this round, ordered on the stream that will consume it
          const gt::RefBundle& src = (round % 2) ? odd[size_t(s)] : even[size_t(s)];
          gt::copy_inputs_async(bufs[size_t(s)], src, streams[size_t(s)]->s);
          GIE_CUDA_CHECK(cudaMemsetAsync(bufs[size_t(s)].data.dy.data(), 0xFF, bufs[size_t(s)].data.dy.size(), streams[size_t(s)]->s));
        }
        for (int s = 0; s < S; ++s) {  // all streams execute the same plan concurrently
          gt::RefBundle& d = bufs[size_t(s)];
          plan->execute_unchecked(d.data.dx.data(), d.data.dw.data(), d.data.db.data(), d.data.dy.data(), ws[size_t(s)].data(), ws[size_t(s)].size(), streams[size_t(s)]->s);
        }
        for (int s = 0; s < S; ++s) GIE_CUDA_CHECK(cudaStreamSynchronize(streams[size_t(s)]->s));
        for (int s = 0; s < S; ++s) {
          const gt::RefBundle& src = (round % 2) ? odd[size_t(s)] : even[size_t(s)];
          const ErrorMetrics m = gt::compare_device_output(bufs[size_t(s)].data.dy.data(), src, double_round_for(*k, prob));
          CHECK_MSG(m.passed(), name << " stream " << s << " round " << round << ": " << m.summary());
        }
      }
    }
  }
}

TEST(one_plan_shared_by_host_threads) {
  // The cuBLAS-backed plan serializes handle use with a mutex and takes its scratch from the caller's workspace.
  for (const char* name : {"im2col_gemm_fp16", "im2col_gemm_fp32", "implicit_gemm_fp32_64x64x8_t4x4"}) {
    const KernelDesc* k = find_kernel(name);
    const Precision prec = std::string(name).find("fp16") != std::string::npos ? Precision::kFp16 : Precision::kFp32;
    const ConvProblem prob = make_prob(prec);
    auto plan = ExecutionPlan::create(prob, *k);
    constexpr int T = 4;
    std::vector<gt::RefBundle> bundles;
    std::vector<DeviceBuffer> ws;
    for (int t = 0; t < T; ++t) {
      bundles.push_back(gt::make_bundle(prob, 300 + uint64_t(t), Distribution::kNormal));
      ws.emplace_back(plan->workspace_bytes());
    }
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < T; ++t) {
      threads.emplace_back([&, t] {
        try {
          GIE_CUDA_CHECK(cudaSetDevice(DeviceInfo::current().device));
          Stream st;
          gt::RefBundle& d = bundles[size_t(t)];
          for (int i = 0; i < 25; ++i) {
            GIE_CUDA_CHECK(cudaMemsetAsync(d.data.dy.data(), 0xFF, d.data.dy.size(), st.s));
            plan->execute_unchecked(d.data.dx.data(), d.data.dw.data(), d.data.db.data(), d.data.dy.data(), ws[size_t(t)].data(), ws[size_t(t)].size(), st.s);
            GIE_CUDA_CHECK(cudaStreamSynchronize(st.s));
            if (!gt::compare_device_output(d.data.dy.data(), d, double_round_for(*k, prob)).passed()) ++errors;
          }
        } catch (const std::exception&) {
          ++errors;
        }
      });
    }
    for (auto& th : threads) th.join();
    CHECK_MSG(errors.load() == 0, name << ": " << errors.load() << " wrong results or errors across threads");
  }
}

TEST(unaligned_user_pointers_are_correct_or_rejected) {
  if (!DeviceInfo::current().supports_bf16_tensor_cores()) return;
  const ConvProblem prob = make_prob(Precision::kFp16);
  gt::RefBundle b = gt::make_bundle(prob, 77, Distribution::kNormal);
  const size_t ybytes = b.data.dy.size(), xbytes = b.data.dx.size();
  for (const char* name : {"tcmma_fp16_128x128x32_s3_w2x4_sk1", "tcmma_fp16_64x64x32_s4_w2x2_sk4", "tensor_core_fp16_64x64x32_w2x2"}) {
    const KernelDesc* k = find_kernel(name);
    auto plan = ExecutionPlan::create(prob, *k);
    DeviceBuffer ws(plan->workspace_bytes());
    // output pointer offset by 2 bytes (not 4-byte aligned): the paired 32-bit epilogue store must not be used
    DeviceBuffer yb(ybytes + 64), xb(xbytes + 64);
    char* y = static_cast<char*>(yb.data()) + 2;
    char* x = static_cast<char*>(xb.data()) + 2;
    GIE_CUDA_CHECK(cudaMemset(yb.data(), 0xFF, yb.size()));
    GIE_CUDA_CHECK(cudaMemcpy(x, b.data.dx.data(), xbytes, cudaMemcpyDeviceToDevice));
    plan->execute_unchecked(x, b.data.dw.data(), b.data.db.data(), y, ws.data(), ws.size(), nullptr);
    GIE_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_MSG(gt::compare_device_output(y, b).passed(), name << " with 2-byte offset input and output pointers");
  }
  // workspace alignment is enforced (16-byte vector buffers live inside it)
  const KernelDesc* k = find_kernel("tcmma_fp16_128x128x32_s3_w2x4_sk1");
  auto plan = ExecutionPlan::create(prob, *k);
  DeviceBuffer ws(plan->workspace_bytes() + 64);
  const ConstTensorRef bias = b.data.b_ref();
  CHECK_THROWS_CODE(plan->execute(b.data.x_ref(), b.data.w_ref(), &bias, b.data.y_ref(), static_cast<char*>(ws.data()) + 16, plan->workspace_bytes(), nullptr),
                    StatusCode::kInvalidArgument);
}

TEST(grid_limit_capability_checks) {
  // The mma.sync family launches transposes with grid.z = batch / C_out; larger values are rejected up front.
  const DeviceInfo& dev = DeviceInfo::current();
  ConvProblem big = gt::make_problem(gt::P(70000, 8, 4, 4, 8, 1, 1, 1, 1, 0, 0), Precision::kFp16, false, false);
  std::string why;
  const KernelDesc* k = find_kernel("tcmma_fp16_64x64x32_s4_w2x2_sk1");
  CHECK(!k->supports(big, dev, &why));
  CHECK(why.find("grid.z") != std::string::npos);
}

GT_MAIN()
