#pragma once
#include <level_zero/ze_api.h>
#include <cstdint>
#include <string>

namespace l0 {
// Driver init + one GPU device + one context. device_index counts GPU
// devices of the first driver (ONEAPI_DEVICE_SELECTOR is not consulted).
class Context {
 public:
  explicit Context(uint32_t device_index = 0);
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  ze_context_handle_t handle() const { return ctx_; }
  ze_driver_handle_t driver() const { return driver_; }
  ze_device_handle_t device() const { return dev_; }
  const ze_device_properties_t& props() const { return props_; }
  const ze_device_compute_properties_t& compute() const { return compute_; }
  std::string name() const { return props_.name; }
  uint32_t eu_count() const {
    return props_.numSlices * props_.numSubslicesPerSlice * props_.numEUsPerSubslice;
  }

 private:
  ze_driver_handle_t driver_ = nullptr;
  ze_device_handle_t dev_ = nullptr;
  ze_context_handle_t ctx_ = nullptr;
  ze_device_properties_t props_{};
  ze_device_compute_properties_t compute_{};
};
}  // namespace l0
