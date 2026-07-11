// Standalone elementwise operators used (a) as the epilogue of the explicit
// im2col+GEMM pipeline and (b) as the *unfused* reference path in the fusion
// experiment. The custom implicit-GEMM / Tensor Core kernels do not call these:
// they fuse bias+ReLU into their output stage.
#pragma once

#include <cuda_runtime.h>

#include "gpu_inference/conv2d.h"

namespace gie {

// y[b,o,:,:] += bias[o]   (in place, NCHW)
void launch_bias_add(const Conv2DParams& p, Precision prec, const void* bias, void* y, cudaStream_t stream);
// y = max(y, 0)           (in place)
void launch_relu(const Conv2DParams& p, Precision prec, void* y, cudaStream_t stream);
// y = act(y + bias) in one pass (either part optional)
void launch_bias_act(const Conv2DParams& p, Precision prec, const void* bias_or_null, bool relu, void* y,
                     cudaStream_t stream);

}  // namespace gie
