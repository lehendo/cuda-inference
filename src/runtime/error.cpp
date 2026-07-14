#include "gpu_inference/error.h"

#include <sstream>

namespace gie {

const char* to_string(StatusCode code) {
  switch (code) {
    case StatusCode::kOk: return "Ok";
    case StatusCode::kInvalidArgument: return "InvalidArgument";
    case StatusCode::kUnsupported: return "Unsupported";
    case StatusCode::kWorkspaceTooSmall: return "WorkspaceTooSmall";
    case StatusCode::kCudaError: return "CudaError";
    case StatusCode::kCublasError: return "CublasError";
    case StatusCode::kCudnnError: return "CudnnError";
    case StatusCode::kInternal: return "Internal";
    case StatusCode::kIo: return "Io";
  }
  return "Unknown";
}

void throw_cuda_error(cudaError_t err, const char* expr, const char* file, int line) {
  // Clear the (non-sticky) error state so a caller that catches can keep using the context.
  (void)cudaGetLastError();
  std::ostringstream os;
  os << expr << " failed with " << cudaGetErrorName(err) << " (" << cudaGetErrorString(err) << ") at "
     << file << ":" << line;
  throw Error(StatusCode::kCudaError, os.str());
}

}  // namespace gie
