#pragma once
#include <string>
#include "l0/context.h"

namespace l0 {
class Kernel;
// A native (ocloc-built) device binary loaded into the context.
class Module {
 public:
  Module(Context& ctx, const std::string& bin_path);
  ~Module();
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  Kernel kernel(const char* name);
  ze_module_handle_t handle() const { return m_; }

 private:
  ze_module_handle_t m_ = nullptr;
};
}  // namespace l0
