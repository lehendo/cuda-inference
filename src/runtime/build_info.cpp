#include "gpu_inference/build_info.h"

#include "gie_build_info.h"

namespace gie {

const BuildInfo& build_info() {
  static const BuildInfo info = [] {
    BuildInfo b;
    b.source_fingerprint = GIE_SOURCE_FINGERPRINT;
    b.build_type = GIE_BUILD_TYPE;
    b.cuda_archs = GIE_CUDA_ARCHS;
    b.nvcc_version = GIE_NVCC_VERSION;
    b.cxx_compiler = GIE_CXX_COMPILER;
    b.cuda_flags = GIE_CUDA_FLAGS_STR;
    b.cxx_flags = GIE_CXX_FLAGS_STR;
#ifdef GIE_HAS_NVTX
    b.has_nvtx = true;
#endif
    return b;
  }();
  return info;
}

}  // namespace gie
