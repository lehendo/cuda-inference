// Minimal tensor vocabulary: data types, RAII device memory, non-owning tensor views.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "gpu_inference/error.h"

namespace gie {

enum class DataType { kFloat32, kFloat16, kBFloat16 };

inline size_t dtype_size(DataType t) { return t == DataType::kFloat32 ? 4 : 2; }
const char* to_string(DataType t);

// Owning, move-only device allocation (cudaMalloc / cudaFree).
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes);
  ~DeviceBuffer();
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_) {
    o.ptr_ = nullptr;
    o.bytes_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept;

  void* data() const { return ptr_; }
  size_t size() const { return bytes_; }
  explicit operator bool() const { return ptr_ != nullptr; }
  void memset_async(int value, cudaStream_t stream = nullptr);
  void reset();

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

// Logical NCHW extents (a bias vector is {1, C_out, 1, 1}).
struct TensorShape {
  int64_t n = 1, c = 1, h = 1, w = 1;
  int64_t numel() const { return n * c * h * w; }
  bool operator==(const TensorShape& o) const { return n == o.n && c == o.c && h == o.h && w == o.w; }
  bool operator!=(const TensorShape& o) const { return !(*this == o); }
  std::string to_string() const;
};

// Non-owning views over device memory. Layout is always contiguous NCHW
// (weights: OIHW). The shape is carried so forward() can validate it.
struct ConstTensorRef {
  const void* data = nullptr;
  DataType dtype = DataType::kFloat32;
  TensorShape shape;
};

struct TensorRef {
  void* data = nullptr;
  DataType dtype = DataType::kFloat32;
  TensorShape shape;
  operator ConstTensorRef() const { return {data, dtype, shape}; }
};

}  // namespace gie
