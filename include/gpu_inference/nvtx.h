// Optional NVTX range annotations (header-only NVTX3 shipped with the CUDA toolkit). With no profiler attached the
// calls are no-ops with negligible cost; without the header (GIE_HAS_NVTX undefined) they compile to nothing.
#pragma once

#ifdef GIE_HAS_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace gie {

class NvtxRange {
 public:
#ifdef GIE_HAS_NVTX
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
#else
  explicit NvtxRange(const char*) {}
#endif
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace gie
