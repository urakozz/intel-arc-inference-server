#pragma once
#include <cstdint>
#include "l0/error.h"

namespace l0 {
class Module;
// One kernel function with its argument state. Arguments are set by value;
// pointers are passed as the pointer value (sizeof(void*)).
class Kernel {
 public:
  Kernel(Module& m, const char* name);
  ~Kernel();
  Kernel(Kernel&& o) noexcept : k_(o.k_) { o.k_ = nullptr; }
  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;

  template <class T>
  void arg(uint32_t index, const T& value) {
    ZE_CHECK(zeKernelSetArgumentValue(k_, index, sizeof(T), &value));
  }
  void arg_ptr(uint32_t index, const void* p) { arg<const void*>(index, p); }
  void group_size(uint32_t x, uint32_t y = 1, uint32_t z = 1) {
    ZE_CHECK(zeKernelSetGroupSize(k_, x, y, z));
  }
  ze_kernel_handle_t handle() const { return k_; }

 private:
  ze_kernel_handle_t k_ = nullptr;
};
}  // namespace l0
