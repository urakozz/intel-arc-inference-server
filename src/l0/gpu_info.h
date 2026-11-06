#pragma once
// What the hardware gate (cli/hardware_check.h) reads of each visible GPU - l0::Context::gpu_infos fills it
// without creating a context. A plain struct, so the gate's rule is testable on any host.
#include <cstddef>
#include <cstdint>
#include <string>

namespace l0 {
struct GpuInfo {
  uint32_t vendor_id = 0, device_id = 0;   // PCI ids (ze_device_properties_t)
  std::string name;
  uint32_t xe_cores = 0;                   // slices x sub-slices per slice: an Xe2 sub-slice is one Xe-core
  size_t memory_bytes = 0;                 // as Context::memory_bytes()
};
}  // namespace l0
