// Build provenance recorded in tuning-cache keys and benchmark metadata. Git commits are not required: the
// fingerprint is a SHA-256 over the library/benchmark sources taken at CMake configure time (a non-mutating
// identifier that changes whenever any of those files change).
#pragma once

#include <string>

namespace gie {

struct BuildInfo {
  std::string source_fingerprint;  // first 16 hex chars of SHA-256 over src/, include/, benchmarks/ sources
  std::string build_type;
  std::string cuda_archs;          // e.g. "80,86,89,90"
  std::string nvcc_version;
  std::string cxx_compiler;
  std::string cuda_flags;          // CMAKE_CUDA_FLAGS + per-config flags
  std::string cxx_flags;
  bool has_cudnn_baseline = false;
  bool has_cutlass_baseline = false;
  bool has_nvtx = false;
};

const BuildInfo& build_info();

}  // namespace gie
