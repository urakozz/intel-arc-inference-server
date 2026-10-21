#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "loader/bolt_layout.h"
#include "model/bolt.h"
#include "runtime/memory_plan.h"     // kMaxLenQuantum, kMinAutoMaxLen, kDefaultReserveGb
#include "runtime/pipeline_plan.h"   // pp_balance

// Spec 25b Task 5: Nemotron 3.5 Bolt's memory plan as device-free arithmetic - the split by bytes (runtime::pp_balance, spec
// 16b's rule, not a second one) and `--max-len auto` (the largest multiple of 256 that fits EVERY device and is at most
// max_position_embeddings, 262144 - Decision 14). Host only (tests/runtime/bolt_plan_test.cc). 25c adds the decode scratch and
// the launch counts, 25d the prefill scratch, spec 26 the drafter and the verify slots: the BoltReserve below carries them
// as caller-given terms (0 until then) so this plan's shape does not change under them.
//
// Per device, the terms (all derived from model::BoltDesc and loader/bolt_layout.h - the plan IS the allocation, the loader
// asserts `weights` against the same formula):
//   weights         loader::bolt::device_weight_bytes (its blocks, the embedding on device 0, norm_f + the head on the last)
//   kv              its attention blocks' K and V at max_len (bf16; 8,192 B a position over all 8 blocks)
//   ssm             one SSM state (fp32) per Mamba block it holds, x (1 + verify_slots) (spec 19c / 26's per-row slots)
//   conv_rings      its Mamba blocks' raw-xBC ring (16 slots)
//   decode_scratch, prefill_scratch   per device (25c / 25d fill them)
//   drafter, drafter_embed_replica    on the LAST device only (spec 26: the drafter and the 1.07 GB embedding replica its
//                                     gather needs there)
//   reserve         runtime::kDefaultReserveGb (1.5 GB) unless given
namespace runtime::bolt {

struct BoltReserve {
  size_t decode_scratch = 0, prefill_scratch = 0;     // bytes, charged to every device
  size_t drafter = 0, drafter_embed_replica = 0;      // bytes, charged to the last device
  uint32_t verify_slots = 0;                          // EXTRA SSM state slots (spec 26 / 19c: 7), charged to every device
};

struct DevicePlan {
  size_t weights = 0, kv = 0, ssm = 0, conv_rings = 0, decode_scratch = 0, prefill_scratch = 0, drafter = 0,
         drafter_embed_replica = 0, reserve = 0;
  size_t total() const {
    return weights + kv + ssm + conv_rings + decode_scratch + prefill_scratch + drafter + drafter_embed_replica + reserve;
  }
};

struct BoltPlan {
  model::BoltPlacement place;       // devices, split (the first block of device 1; layers on one device), layers
  uint32_t max_len = 0;
  std::vector<DevicePlan> device;   // one per device, in category bytes
};

// The plan on `device_free.size()` (1 or 2) devices. `max_len` 0 = auto (the largest multiple of 256, at most
// d.max_pos, that fits every device; throws when not even min(4096, max_pos) does); otherwise it must be a multiple of 256
// and at most d.max_pos. `split` (two devices only; default = pp_balance's, by block weights alone - the hand-off is the
// residual row, so there is no per-length KV asymmetry to balance against). Throws std::runtime_error naming the device, every
// term and its bytes when the plan does not fit - and, for the real model on one card, Decision 4 (spec 25 §4.2).
BoltPlan plan(const model::BoltDesc& d, uint32_t max_len, const std::vector<size_t>& device_free,
              const BoltReserve& extra = {}, std::optional<uint32_t> split = {},
              size_t reserve_bytes = size_t(kDefaultReserveGb * 1e9), bool int8_head = true);

// One device's terms at `max_len` under placement `p` (what plan() fits and the loader allocates).
DevicePlan device_plan(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev, uint32_t max_len, bool int8_head,
                       const BoltReserve& extra, size_t reserve_bytes);

// Throws naming every term of `p` when its total exceeds device_bytes (the loader calls it before the first allocation).
void require_fits(const DevicePlan& p, uint32_t dev, uint32_t max_len, size_t device_bytes);

// The two-card split by BLOCK WEIGHTS alone (runtime::pp_balance): the split whose heavier device holds the fewest bytes,
// device 0's embedding and the last device's norm_f / head counted as each side's fixed overhead. 48 on the real model.
uint32_t pp_split(const model::BoltDesc& d, bool int8_head = true);

// "weights W GB, kv K GB, ssm S GB, conv rings C B, ... total T GB" (+ " of D GB" when `device_bytes` is given).
std::string describe(const DevicePlan& p, size_t device_bytes = 0);

}  // namespace runtime::bolt
