#pragma once
// The hardware gate on the devices the driver shows (cli/hardware_check.h holds the rule and its reasons).
#include <vector>

#include "cli/hardware_check.h"
#include "l0/context.h"

namespace cli {

// The gate on the devices the driver shows. Opens no context and reads no model file.
inline Hardware hardware_gate() {
  const std::vector<l0::GpuInfo> gpus = l0::Context::gpu_infos();
  const bool peer = gpus.size() == 2 && l0::Context::peer_access(0, 1) && l0::Context::peer_access(1, 0);
  return check_hardware(gpus, peer);
}

}  // namespace cli
