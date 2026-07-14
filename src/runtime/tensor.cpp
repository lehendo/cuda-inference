#include "gpu_inference/tensor.h"

#include <sstream>

namespace gie {

const char* to_string(DataType t) {
  switch (t) {
    case DataType::kFloat32: return "fp32";
    case DataType::kFloat16: return "fp16";
    case DataType::kBFloat16: return "bf16";
  }
  return "?";
}

std::string TensorShape::to_string() const {
  std::ostringstream os;
  os << "[" << n << "," << c << "," << h << "," << w << "]";
  return os.str();
}

DeviceBuffer::DeviceBuffer(size_t bytes) : bytes_(bytes) {
  if (bytes == 0) return;
  GIE_CUDA_CHECK(cudaMalloc(&ptr_, bytes));
}

DeviceBuffer::~DeviceBuffer() { reset(); }

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& o) noexcept {
  if (this != &o) {
    reset();
    ptr_ = o.ptr_;
    bytes_ = o.bytes_;
    o.ptr_ = nullptr;
    o.bytes_ = 0;
  }
  return *this;
}

void DeviceBuffer::reset() {
  if (ptr_ != nullptr) {
    cudaFree(ptr_);  // destructor path: errors here cannot be propagated
    ptr_ = nullptr;
  }
  bytes_ = 0;
}

void DeviceBuffer::memset_async(int value, cudaStream_t stream) {
  if (ptr_ != nullptr) GIE_CUDA_CHECK(cudaMemsetAsync(ptr_, value, bytes_, stream));
}

}  // namespace gie
