#pragma once
#include <level_zero/ze_api.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "l0/gpu_info.h"

namespace l0 {
// Driver init + one GPU device + one context. `device_index` counts GPU
// devices of the first driver, in enumeration order (default 0).
//
// Device selection (the zero-config cleanup, operator 2026-11-06): the CLIs run on
// the supported cards they find - device 0, or devices 0 and 1 under --parallel pp;
// probes and tests pass their index. `ZE_AFFINITY_MASK` sits underneath: it is a
// driver-level filter deciding which devices are enumerated at all. We never set
// it and nothing here reads it; under a mask the index refers to the masked
// (visible) view - the same renumbering the driver gives everyone.
// (ONEAPI_DEVICE_SELECTOR and --device were removed with the cleanup.)
class Context {
 public:
  explicit Context(uint32_t device_index = 0);
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

  // Spec 16b: the GPU devices of the first driver, in the order an index (the view
  // above) counts them - what `ZE_AFFINITY_MASK` leaves visible. Initialises the
  // driver; no context is created.
  static uint32_t gpu_count();
  // Spec 16b: zeDeviceCanAccessPeer(this view's device, other's device) - whether this
  // device can read and write `other`'s device memory (P4: refused, never assumed).
  bool can_access_peer(const Context& other) const;

  // The zero-config hardware gate's view (cli/hardware_check.h): every GPU the first driver shows -
  // what ZE_AFFINITY_MASK leaves visible - read without creating a context.
  static std::vector<GpuInfo> gpu_infos();
  // zeDeviceCanAccessPeer(visible GPU a, visible GPU b), without a context.
  static bool peer_access(uint32_t a, uint32_t b);

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
  uint32_t xe_cores() const { return props_.numSlices * props_.numSubslicesPerSlice; }
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
