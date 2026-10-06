#pragma once
#include <level_zero/ze_api.h>
#include <cstddef>
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
  // Spec 16b (pipeline parallel): a view of GPU `device_index` - an explicit index, so
  // the environment is not consulted - inside `primary`'s Level Zero context. Spec 16
  // decision 1 is ONE context holding both devices: every allocation, module, list and
  // event made through either view lives in that context, so a cross-device copy, a peer
  // pointer or an event one device signals and the other waits on needs no handle
  // export. The view shares `primary`'s driver and context handle and does not destroy
  // the context; `primary` must outlive it. (zeContextCreate's context already spans
  // every device of the driver - this constructor only binds a second device to it.)
  Context(const Context& primary, uint32_t device_index);
  ~Context();

  // Spec 16b: the GPU devices of the first driver, in the order an index (`--device N`,
  // the view above) counts them - what `ZE_AFFINITY_MASK` leaves visible. Initialises the
  // driver; no context is created.
  static uint32_t gpu_count();
  // Spec 16b: zeDeviceCanAccessPeer(this view's device, other's device) - whether this
  // device can read and write `other`'s device memory (P4: refused, never assumed).
  bool can_access_peer(const Context& other) const;

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
  // The device's total memory (zeDeviceGetMemoryProperties' totalSize, summed): the
  // "of <GB>" of Engine::memory_line() and the budget --max-len auto plans against.
  size_t memory_bytes() const;

 private:
  ze_driver_handle_t driver_ = nullptr;
  ze_device_handle_t dev_ = nullptr;
  ze_context_handle_t ctx_ = nullptr;
  ze_device_properties_t props_{};
  ze_device_compute_properties_t compute_{};
  bool owns_ = true;   // false for a view (spec 16b): the primary destroys the context
};
}  // namespace l0
