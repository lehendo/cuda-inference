// Minimal use of the public API: run a fused conv+bias+ReLU with automatic backend selection.
#include <iostream>

#include "gpu_inference/conv2d.h"
#include "gpu_inference/testing.h"

int main() {
  using namespace gie;
  try {
    ConvProblem prob;
    prob.p = {};
    prob.p.batch = 8; prob.p.in_channels = 64; prob.p.in_h = 56; prob.p.in_w = 56;
    prob.p.out_channels = 64; prob.p.kernel_h = 3; prob.p.kernel_w = 3; prob.p.pad_h = 1; prob.p.pad_w = 1;
    prob.precision = Precision::kFp16;   // FP16 storage, FP32 accumulation
    prob.has_bias = true;
    prob.activation = Activation::kReLU;

    ConvData d = make_conv_data(prob, /*seed=*/1);   // synthetic, deterministic tensors on the device
    cudaStream_t stream;
    GIE_CUDA_CHECK(cudaStreamCreate(&stream));

    const size_t ws_bytes = Conv2D::workspace_size(prob);
    DeviceBuffer workspace(ws_bytes);
    const ConstTensorRef bias = d.b_ref();
    Conv2D::forward(d.x_ref(), d.w_ref(), &bias, prob, d.y_ref(), workspace.data(), workspace.size(), stream);
    GIE_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> y = download_as_float(d.dy.data(), prob.dtype(), prob.p.output_shape().numel());
    std::cout << "output " << prob.p.output_shape().to_string() << ", y[0]=" << y[0] << ", workspace=" << ws_bytes << " B\n";
    cudaStreamDestroy(stream);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
