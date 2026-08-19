// Public, independently defined correctness shapes (ResNet/VGG-style layers plus adversarial
// geometries). No course-specific workloads are used.
#pragma once

#include <string>
#include <vector>

#include "gpu_inference/conv2d.h"

namespace gt {

struct Shape {
  const char* name;
  gie::Conv2DParams p;
};

inline gie::Conv2DParams P(int B, int C, int H, int W, int O, int Kh, int Kw, int Sh, int Sw, int Ph, int Pw) {
  gie::Conv2DParams p;
  p.batch = B; p.in_channels = C; p.in_h = H; p.in_w = W; p.out_channels = O;
  p.kernel_h = Kh; p.kernel_w = Kw; p.stride_h = Sh; p.stride_w = Sw; p.pad_h = Ph; p.pad_w = Pw;
  return p;
}

// Batch sizes 1,2,8,16,32; kernels 1x1,3x3,5x5,7x7 (+ non-square); stride 1/2; zero and nonzero
// padding; odd and unequal H/W; dimensions not divisible by any tile size; tiny outputs;
// boundary-dominated geometries.
inline const std::vector<Shape>& correctness_shapes() {
  static const std::vector<Shape> s = {
      {"tiny_1ch_5x5_k3", P(1, 1, 5, 5, 1, 3, 3, 1, 1, 0, 0)},
      {"scalar_1x1_k1", P(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0)},
      {"rgb_8x8_k3p1", P(1, 3, 8, 8, 4, 3, 3, 1, 1, 1, 1)},
      {"rgb_32x32_k3p1_b2", P(2, 3, 32, 32, 16, 3, 3, 1, 1, 1, 1)},
      {"odd_unequal_k5s2", P(1, 3, 33, 31, 8, 5, 5, 2, 2, 2, 2)},
      {"pointwise_odd_b2", P(2, 16, 15, 17, 24, 1, 1, 1, 1, 0, 0)},
      {"resnet_14x14_b8_k3", P(8, 64, 14, 14, 64, 3, 3, 1, 1, 1, 1)},
      {"resnet_56x56_1x1", P(1, 64, 56, 56, 64, 1, 1, 1, 1, 0, 0)},
      {"resnet_down_28_s2_b2", P(2, 64, 28, 28, 128, 3, 3, 2, 2, 1, 1)},
      {"k7_b16_c32", P(16, 32, 20, 20, 48, 7, 7, 1, 1, 3, 3)},
      {"deep_512ch_7x7", P(1, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1)},
      {"b32_c16_16x16", P(32, 16, 16, 16, 32, 3, 3, 1, 1, 1, 1)},
      {"irregular_c7_o9_s2", P(2, 7, 11, 13, 9, 3, 3, 2, 2, 0, 0)},
      {"kernel_eq_map_7x7", P(1, 3, 7, 7, 5, 7, 7, 1, 1, 3, 3)},
      {"c130_o130_9x9", P(2, 130, 9, 9, 130, 3, 3, 1, 1, 1, 1)},
      {"o255_pointwise_13", P(1, 64, 13, 13, 255, 1, 1, 1, 1, 0, 0)},
      {"k5_b4_c96_27", P(4, 96, 27, 27, 192, 5, 5, 1, 1, 2, 2)},
      {"stem_7x7_s2_64", P(2, 3, 64, 64, 64, 7, 7, 2, 2, 3, 3)},
      {"down_14_s2_c256", P(1, 256, 14, 14, 256, 3, 3, 2, 2, 1, 1)},
      {"small_b8_4x4", P(8, 8, 4, 4, 8, 3, 3, 1, 1, 1, 1)},
      {"boundary_pad2_3x3", P(1, 2, 3, 3, 3, 3, 3, 1, 1, 2, 2)},
      {"boundary_pad4_k5", P(1, 16, 10, 10, 16, 5, 5, 1, 1, 4, 4)},
      {"aniso_k3x5_s2x1", P(2, 8, 17, 23, 12, 3, 5, 2, 1, 1, 2)},
      {"k1x3", P(1, 4, 12, 12, 6, 1, 3, 1, 1, 0, 1)},
      {"k3x1", P(1, 4, 12, 12, 6, 3, 1, 1, 1, 1, 0)},
      {"wide_1x1_b32", P(32, 24, 7, 7, 40, 1, 1, 1, 1, 0, 0)},
  };
  return s;
}

}  // namespace gt
