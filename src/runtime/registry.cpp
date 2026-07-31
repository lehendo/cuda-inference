#include <set>
#include <sstream>

#include "../kernels/conv_common.cuh"
#include "gpu_inference/kernel_registry.h"

namespace gie {

std::string KernelDesc::describe() const {
  std::ostringstream os;
  os << name << " [" << to_string(backend) << ", " << threads_per_block << " thr";
  for (const auto& kv : params) os << ", " << kv.first << "=" << kv.second;
  os << "]";
  return os.str();
}

namespace {
std::vector<KernelDesc>& registry_storage() {
  static std::vector<KernelDesc> registry = [] {
    std::vector<KernelDesc> r;
    r.reserve(1024);  // stable element addresses even if tests inject extra entries
    kernels::register_direct(r);
    kernels::register_im2col(r);
    kernels::register_simt_tiled(r);
    kernels::register_implicit_gemm(r);
    kernels::register_tensor_core(r);
    kernels::register_tensor_core_mma(r);
    std::set<std::string> names;
    for (const auto& k : r) {
      if (!names.insert(k.name).second) throw Error(StatusCode::kInternal, "duplicate kernel name " + k.name);
      if (!k.supports || !k.launch || !k.workspace_bytes)
        throw Error(StatusCode::kInternal, "incomplete kernel descriptor " + k.name);
    }
    return r;
  }();
  return registry;
}
}  // namespace

const std::vector<KernelDesc>& kernel_registry() { return registry_storage(); }
std::vector<KernelDesc>& kernel_registry_for_testing() { return registry_storage(); }

const KernelDesc* find_kernel(const std::string& name) {
  for (const auto& k : kernel_registry())
    if (k.name == name) return &k;
  return nullptr;
}

std::vector<const KernelDesc*> kernels_for_backend(Backend b) {
  std::vector<const KernelDesc*> out;
  for (const auto& k : kernel_registry())
    if (b == Backend::kAuto || k.backend == b) out.push_back(&k);
  return out;
}

}  // namespace gie
