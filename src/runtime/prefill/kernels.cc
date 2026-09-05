#include "runtime/prefill/kernels.h"

#include <stdexcept>

#include "kernels/kernels.h"

namespace runtime::prefill {

l0::Kernel& KernelCache::get(const std::string& variant, const char* entry) {
  auto m = modules_.find(variant);
  if (m == modules_.end()) {
    try {
      m = modules_.emplace(variant, std::make_unique<l0::Module>(ctx_, kernels::path(variant)))
              .first;
    } catch (const std::exception& e) {
      // The variant name IS the file name (`kernels::path`), so a missing
      // binary is a build-graph bug - an `add_ocloc_kernel` row that does not
      // exist, or a test target missing its `add_dependencies`. Say which.
      throw std::runtime_error("prefill::KernelCache: cannot load variant '" + variant +
                               "' (" + kernels::path(variant) + "): " + e.what());
    }
  }
  const std::string key = variant + "/" + entry;
  auto k = kernels_.find(key);
  if (k == kernels_.end()) {
    try {
      k = kernels_.emplace(key, std::make_unique<l0::Kernel>(*m->second, entry)).first;
    } catch (const std::exception& e) {
      throw std::runtime_error("prefill::KernelCache: variant '" + variant +
                               "' has no entry point '" + std::string(entry) + "': " + e.what());
    }
  }
  return *k->second;
}

}  // namespace runtime::prefill
