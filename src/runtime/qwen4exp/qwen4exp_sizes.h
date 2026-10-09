#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "loader/qwen4exp_ple_hash.h"   // Q4PleScale (header-only)
#include "model/qwen4exp.h"
#include "runtime/memory_plan.h"
#include "runtime/pipeline_plan.h"     // PpBalance, pp_balance, kPpDevices

// Spec 21b Task 5: Qwen3.8-Flash-Next's device memory as device-free arithmetic - the weights (the loader's
// one formula, loader/qwen4exp_layout.h) and the persistent state, the planner over them (`--layers auto`,
// `--max-len auto`) and the split by bytes (spec 16b's runtime::pp_balance - not a second rule). Host only
// (tests/runtime/qwen4exp_plan_test.cc). 21c adds the decode scratch, the hand-off and the launch counts to
// this file, 21d the prefill scratch behind a `prefill` flag, 21e the MTP buffers (Kolibri's arrangement);
// until then a plan here is the weights and the persistent state only.
//
// **The full model does not fit two B70s** (spec 21 §3): its routed experts alone are 64.17 GB at int4 g64
// against ~62 GB usable, so `require_fits` refuses it by bytes, naming spec 22 (the expert-offload tier);
// the development mode `--layers N` (the first N layers, then the final mixer and the head) is planned per
// card by `layers_that_fit`.
namespace runtime::qwen4exp {

// One device's persistent state at max_len (zeroed by reset):
//   kv          bf16 [its QSA layers][max_len][2 kv heads][256], K and V    2048 B a position a layer
//   idx_keys    bf16 [its QSA layers][max_len / 4][128]                    the compressed indexer keys, 64 B a
//                                                                          position a layer
//   idx_tail    bf16 [its QSA layers][4][128]                              the raw keys of the open block
//   gdn_state   fp32 [its GDN layers][48][128][128]                        3,145,728 B a layer
//   conv_ring   bf16 [its GDN layers][16][10240]                           Qwen3.8's ring (gdn_step reads it)
//   ple         u32 [2] ids + bf16 [9][10240] conv rows                    the device holding the PLE layer
// An empty group (a device with no QSA / no GDN layer, the PLE state off the PLE layer's device) still
// allocates one 64-byte line.
inline constexpr uint32_t kConvRing = 16;   // = PersistentDims::kConvRing (Qwen3.8's gdn_step ring)
inline constexpr uint32_t kIdxTail = 4;     // the open block's raw keys (spec 21 §4.2; 21c may widen it)
struct PersistentSizes {
  size_t control = 0, kv = 0, idx_keys = 0, idx_tail = 0, gdn_state = 0, conv_ring = 0, ple = 0;
  size_t kv_total() const { return kv + idx_keys + idx_tail; }
  size_t state() const { return control + gdn_state + conv_ring + ple; }
  size_t total() const { return kv_total() + state(); }
};
PersistentSizes persistent_sizes(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev,
                                 uint32_t max_len);
// The KV and compressed keys of one position over the descriptor's QSA layers: 12 x (2048 + 64) = 25,344 B.
size_t kv_bytes_per_pos(const model::Qwen4ExpDesc& d);
size_t gdn_state_bytes_per_layer(const model::Qwen4ExpDesc& d);   // 3,145,728
size_t conv_ring_bytes_per_layer(const model::Qwen4ExpDesc& d);   // 327,680
size_t ple_state_bytes(const model::Qwen4ExpDesc& d);             // 8 + 9 x 10240 x 2
// The host table spec 21 §4.3 pins (16 ranges of int8 rows + scales): 51.84 GB with bf16 scales (derived).
size_t host_ple_bytes(const model::Qwen4ExpDesc& d, loader::Q4PleScale s);

// --- the split by bytes (16b's rule) ---------------------------------------------------------------------
// layer_bytes[l] = loader::q4_layer_bytes(d, l).total() + its state at max_len (QSA: KV, keys, tail; GDN:
// state, ring; the PLE layer: + the PLE state).
std::vector<size_t> pp_layer_bytes(const model::Qwen4ExpDesc& d, uint32_t max_len);
// dev0_fixed = the embedding + the RoPE table + control; dev1_fixed = the final mixer + lm_head (+ the MTP
// head) + the RoPE table + control - then runtime::pp_balance.
PpBalance pp_split(const model::Qwen4ExpDesc& d, uint32_t max_len, bool int8_head, bool mtp);

// --- the planner ---------------------------------------------------------------------------------------
// One device's plan in memory_line()'s components: model = weights + RoPE; kv = KV + indexer keys + tails;
// decode_state = control + GDN state + conv ring + PLE state (21c adds the scratch).
struct DevicePlan : MemoryComponents {
  uint32_t device = 0, first = 0, end = 0;   // its layers [first, end)
  size_t weights = 0, rope = 0, state = 0;
  bool mtp = false;                          // the MTP head's weights are in `weights` (the last device)
  bool whole = false;                        // the descriptor is the published 48 layers (not --layers N)
};
std::vector<DevicePlan> plan(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t max_len,
                             bool int8_head, bool mtp);
// Whether every device's plan + reserve fits its capacity.
bool fits(const std::vector<DevicePlan>& p, const std::array<size_t, kPpDevices>& device_bytes, size_t reserve);
// `--layers auto`: the largest N in [ple_layer + 1, d.layers] whose truncated model (layers [0, N), the final
// mixer, lm_head, with mtp the head) fits `devices` cards of `device_bytes` at max_len - on two cards under
// the best split for that N (any split that fits); 0 when not even N = ple_layer + 1 does.
uint32_t layers_that_fit(const model::Qwen4ExpDesc& d, uint32_t devices, uint32_t max_len, bool int8_head, bool mtp,
                         size_t device_bytes, size_t reserve);
// The truncated descriptor and the placement layers_that_fit plans: N layers; two cards at the pp_split
// balance when it fits, else the first split that does.
model::Qwen4ExpDesc truncated(const model::Qwen4ExpDesc& d, uint32_t layers);
model::Q4Placement placement_for(const model::Qwen4ExpDesc& truncated, uint32_t devices, uint32_t max_len,
                                 bool int8_head, bool mtp, size_t device_bytes, size_t reserve);
// The largest multiple of kMaxLenQuantum, at most min(cap, 262144), whose plan + reserve fits every device
// (`--max-len auto`); 0 when not even min(kMinAutoMaxLen, cap) does.
uint32_t max_len_that_fits(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, bool int8_head, bool mtp,
                           const std::array<size_t, kPpDevices>& device_bytes, size_t reserve, uint32_t cap = 0);
// Throws std::runtime_error naming the bytes, each device's capacity and spec 22 when the placement does not
// fit - the full model's refusal ("Qwen3.8-Flash-Next holds ~69 GB of weights at int4 g64 and two B70s
// ~62 GB: it runs whole only with spec 22's expert-offload tier; use --layers N").
void require_fits(const std::vector<DevicePlan>& p, const std::array<size_t, kPpDevices>& device_bytes, size_t reserve);
// "plan at max_len N (one card, layers [0, L)):" or "pipeline plan at max_len N, split s (...):" + one
// memory_line()-style line per device.
std::string describe(const std::vector<DevicePlan>& p, const model::Q4Placement& pl, uint32_t max_len,
                     const std::array<size_t, kPpDevices>& device_bytes, size_t reserve);

}  // namespace runtime::qwen4exp
