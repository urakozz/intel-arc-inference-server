#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/muse_glimmer.h"

// Spec 23b Task 5: Muse-Glimmer-30B's device memory as device-free arithmetic - the planner over the weights (the
// loader's one formula, loader/muse_layout.h), the attention state (13 full layers' KV at max_len, 39 sliding
// rings) and the scratch / drafter reservations, `--max-len auto` (spec 23 decision 6: up to 131072, never above)
// and the split for two cards by bytes (spec 16b's runtime::pp_balance - not a second rule; decision 7: one card is
// the default, `--pp 2` for capacity). Host only (tests/runtime/muse_plan_test.cc); the loader runs it before the
// first allocation (loader::load_muse_glimmer -> require_fits).
//
// **The scratch terms are placeholders here**: 23c fills the decode scratch, 23d the prefill scratch, 23f the
// drafter's bytes; this plan adds whatever MuseReserve says to every device (the drafter to the last one, where
// the head it reads lives) and the planner's answers move with them.
namespace runtime::muse {

// What the caller reserves beside the weights and the attention state (bytes; 0 = none).
struct MuseReserve {
  size_t drafter = 0;           // the DFlash drafter's weights + ring (spec 23 §8.2: 1.358 GB int4, 5.112 GB bf16)
  size_t prefill_scratch = 0;   // per device (23d)
  size_t decode_scratch = 0;    // per device (23c)
};

// One device's plan. total() = weights + kv + rings + scratch + drafter + reserve (the 1.5 GB of every planner,
// runtime::kDefaultReserveGb: the driver, modules, command lists and slack - an estimate).
struct MuseDevicePlan {
  uint32_t device = 0, first = 0, end = 0;   // layers [first, end)
  size_t weights = 0, kv = 0, rings = 0, scratch = 0, reserve = 0, drafter = 0;
  size_t capacity = 0;                       // the card's bytes (the caller's)
  size_t total() const { return weights + kv + rings + scratch + drafter + reserve; }
};

struct MusePlan {
  model::MusePlacement placement;
  uint32_t max_len = 0;
  bool int8_head = true;
  std::vector<MuseDevicePlan> devices;
  bool fits = false;
  std::string why;              // empty when it fits; else each device's bytes against its capacity
  std::string describe() const; // "Muse-Glimmer plan at max_len N, one card | split s: ..." + a line per device
};

inline constexpr size_t kMuseReserveBytes = 1500000000ull;   // runtime::kDefaultReserveGb x 1e9

// The bytes one layer puts on its device at max_len: its weights + its ring (sliding) or its K and V rows (full,
// max_len x 1024 B) - the split's per-layer term.
std::vector<size_t> pp_layer_bytes(const model::MuseGlimmerDesc& d, uint32_t max_len);

// The plan at a fixed placement and length (no search).
MusePlan plan_at(const model::MuseGlimmerDesc& d, const model::MusePlacement& p, uint32_t max_len, bool int8_head,
                 const MuseReserve& r, const std::vector<size_t>& capacity);

// `devices` 1 or 2; `max_len` 0 = auto: the largest multiple of 256 up to d.max_pos (decision 6) whose plan fits
// every device, not below 4096 (runtime::kMinAutoMaxLen; nothing fits: `fits` false at 4096, `why` names the
// bytes). An explicit max_len above d.max_pos is refused (std::invalid_argument naming decision 6), as is one that
// is not a multiple of 256; one that does not fit returns `fits` false. Two devices: `split` 0 = by bytes
// (runtime::pp_balance over pp_layer_bytes at that length, the embedding + scratch on device 0's side, the final
// norm + head + scratch + drafter on device 1's), else the given split (validated). `capacity[i]` is device i's
// bytes (capacity.size() >= devices).
MusePlan plan_muse(const model::MuseGlimmerDesc& d, uint32_t devices, uint32_t max_len, bool int8_head,
                   const MuseReserve& r, const std::vector<size_t>& capacity, uint32_t split = 0);

// Throws std::runtime_error naming each device's bytes and capacity unless p.fits.
void require_fits(const MusePlan& p);

}  // namespace runtime::muse
