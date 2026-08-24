#pragma once
#include <level_zero/ze_api.h>
#include <cstdint>
#include <string>

namespace l0 {
// Driver init + one GPU device + one context. `device_index` counts GPU
// devices of the first driver, in enumeration order.
//
// Device selection (doc 04, "Device selection"), top wins:
//   1. an explicit index - the CLI's `--device N`, and every probe and test,
//      which pass 0 and are unaffected by the environment;
//   2. `ONEAPI_DEVICE_SELECTOR`, consulted only by the default `kFromEnv` and
//      parsed by `device_index_from_env()` below. Level Zero does not read
//      that variable - only the SYCL runtime does - so the raw-L0 path parses
//      it itself, and both paths land on the same card.
//   3. device 0.
// `ZE_AFFINITY_MASK` sits *underneath* all three: it is a driver-level filter
// deciding which devices are enumerated at all. We never set it and nothing
// here reads it; under a mask, `--device N` and the selector index both refer
// to the masked (visible) view - the same renumbering the driver gives
// everyone.
class Context {
 public:
  // Resolve the index from ONEAPI_DEVICE_SELECTOR instead of taking it here.
  static constexpr uint32_t kFromEnv = 0xFFFFFFFFu;

  explicit Context(uint32_t device_index = kFromEnv);
  ~Context();

  // Parses ONEAPI_DEVICE_SELECTOR: unset or empty and `level_zero:*` mean
  // device 0 (there is no multi-device execution until P/D disaggregation);
  // `level_zero:N` means GPU N. Anything else - another backend, a list,
  // garbage - throws std::runtime_error naming the value and both accepted
  // forms, rather than silently binding card 0. Pure parse: no driver, no
  // device, no GPU needed, so it is testable anywhere.
  static uint32_t device_index_from_env();

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
