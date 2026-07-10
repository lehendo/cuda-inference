// Error handling for the GPU inference engine.
//
// All recoverable failures (invalid arguments, unsupported configurations,
// CUDA/cuBLAS errors) are reported by throwing gie::Error with a StatusCode.
// The core execution path never swallows an error and never returns a result
// that was not computed.
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace gie {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,   // caller error: bad shape, null pointer, aliasing, ...
  kUnsupported,       // valid request that the chosen backend/hardware can't run
  kWorkspaceTooSmall,
  kCudaError,
  kCublasError,
  kCudnnError,
  kInternal,
  kIo,
};

const char* to_string(StatusCode code);

class Error : public std::runtime_error {
 public:
  Error(StatusCode code, const std::string& message)
      : std::runtime_error(std::string(gie::to_string(code)) + ": " + message), code_(code) {}
  StatusCode code() const noexcept { return code_; }

 private:
  StatusCode code_;
};

[[noreturn]] void throw_cuda_error(cudaError_t err, const char* expr, const char* file, int line);

}  // namespace gie

#define GIE_CUDA_CHECK(expr)                                                     \
  do {                                                                           \
    cudaError_t gie_err_ = (expr);                                               \
    if (gie_err_ != cudaSuccess) ::gie::throw_cuda_error(gie_err_, #expr, __FILE__, __LINE__); \
  } while (0)

// Validates a caller-supplied condition; throws gie::Error(code, msg) on failure.
#define GIE_REQUIRE(cond, code, msg)                                             \
  do {                                                                           \
    if (!(cond)) throw ::gie::Error((code), (msg));                              \
  } while (0)
