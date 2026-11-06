#pragma once
// The hardware gate (the zero-config cleanup, operator 2026-11-06): b70-serve and b70-decode run on one or two
// Intel Arc Pro B70 cards, with peer access between them when there are two - checked once at startup, before
// a model file is read, and refused by name otherwise. What it reads comes from the devices themselves
// (l0::Context::gpu_infos: PCI ids, Xe-cores, memory), never from a constant. This header is the rule
// (pure, host-tested); cli/hardware_gate.h runs it on the driver.
//
//   vendor        Intel (0x8086)
//   device id     the table below: the B70 (0xE223) runs; the B65 (0xE222) and B60 (0xE211) are Xe2 cards of
//                 the same family that have never run here, refused as "not yet tested"; anything else is
//                 not a supported card
//   Xe-cores      what the kernels were built for (kernels::kBuiltXeCores: the grids, e.g. gemv_bf16's
//                 tiny-N rule, assume it)
//   cards         one or two visible (ZE_AFFINITY_MASK picks which); more is refused
//   peer access   both ways when two are visible (the pipeline's hand-offs and the load-time copy write the
//                 other card's memory)
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "l0/gpu_info.h"

namespace cli {

struct CardModel {
  uint32_t device_id;
  const char* name;
  bool tested;   // has run on this box
};
inline constexpr uint32_t kIntelVendor = 0x8086;
inline constexpr CardModel kCardModels[] = {
    {0xE223, "Intel Arc Pro B70", true},
    {0xE222, "Intel Arc Pro B65", false},
    {0xE211, "Intel Arc Pro B60", false},
};

struct Hardware {
  uint32_t cards = 0;       // visible supported cards: 1 or 2
  size_t card_bytes = 0;    // the smallest visible card's memory (zeDeviceGetMemoryProperties)
};

inline std::string hex4(uint32_t v) {
  char b[16];
  std::snprintf(b, sizeof b, "0x%04X", v);
  return b;
}

// The rule, on what the driver reported (pure: the host test drives it). `peer` is the peer access both ways
// between GPUs 0 and 1, read only when two are visible.
inline Hardware check_hardware(const std::vector<l0::GpuInfo>& gpus, bool peer) {
  if (gpus.empty())
    throw std::runtime_error("no GPU: Level Zero shows none (is ZE_AFFINITY_MASK hiding the cards?) - b70 runs on one "
                             "or two Intel Arc Pro B70s");
  if (gpus.size() > 2)
    throw std::runtime_error(std::to_string(gpus.size()) + " GPUs visible: b70 runs on one or two Intel Arc Pro B70s - "
                             "pick them with ZE_AFFINITY_MASK (e.g. ZE_AFFINITY_MASK=0,1)");
  Hardware hw;
  hw.cards = uint32_t(gpus.size());
  for (size_t i = 0; i < gpus.size(); ++i) {
    const l0::GpuInfo& g = gpus[i];
    const std::string which = "GPU " + std::to_string(i) + " (" + g.name + ", device id " + hex4(g.device_id) + ")";
    if (g.vendor_id != kIntelVendor)
      throw std::runtime_error(which + " is not an Intel GPU (vendor " + hex4(g.vendor_id) +
                               "): b70 runs on the Intel Arc Pro B70 only");
    const CardModel* m = nullptr;
    for (const CardModel& c : kCardModels)
      if (c.device_id == g.device_id) m = &c;
    if (m == nullptr)
      throw std::runtime_error(which + " is not a supported card: b70 runs on the Intel Arc Pro B70 (device id " +
                               hex4(kCardModels[0].device_id) + ") only");
    if (!m->tested)
      throw std::runtime_error(which + " is an " + m->name + ": not yet tested on this box - b70 runs on the Intel "
                               "Arc Pro B70 only");
    if (g.xe_cores != kernels::kBuiltXeCores)
      throw std::runtime_error(which + " shows " + std::to_string(g.xe_cores) + " Xe-cores; the kernels are built for " +
                               std::to_string(kernels::kBuiltXeCores) + " (the B70's)");
    if (hw.card_bytes == 0 || g.memory_bytes < hw.card_bytes) hw.card_bytes = g.memory_bytes;
  }
  if (gpus.size() == 2 && !peer)
    throw std::runtime_error("two GPUs visible but they cannot access each other's memory (zeDeviceCanAccessPeer is "
                             "false; docs/10-the-box.md) - b70 needs peer access between its two cards; run on one "
                             "with ZE_AFFINITY_MASK=0");
  return hw;
}

}  // namespace cli
