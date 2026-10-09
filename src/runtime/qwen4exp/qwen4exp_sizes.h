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
// this file, 21d the prefill scratch behind a `prefill` flag, 21e the MTP buffers (Kolibri's arrangement).
// Spec 21c (built): the decode scratch, the debug tap, the two-card hand-off and the launch counts below; a plan
// is the weights, the persistent state and the decode scratch (+ the link on two cards).
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
//   idx_tail    bf16 [its QSA layers][8][128]                              the raw keys of the open block, slot
//                                                                          p % 8 (21c: 8, not spec 21 §4.2's 4 -
//                                                                          plan 21c Review Focus 3)
//   gdn_state   fp32 [its GDN layers][48][128][128]                        3,145,728 B a layer
//   conv_ring   bf16 [its GDN layers][16][10240]                           Qwen3.8's ring (gdn_step reads it)
//   ple         u32 [16] id ring + bf16 [16][10240] conv ring              the device holding the PLE layer: slot
//                                                                          p % 16 (21c: the history is 2 ids and 9
//                                                                          rows; 16-slot rings are q4_ple.cl's
//                                                                          addressing, 21b planned [2] + [9])
// An empty group (a device with no QSA / no GDN layer, the PLE state off the PLE layer's device) still
// allocates one 64-byte line.
inline constexpr uint32_t kConvRing = 16;   // = PersistentDims::kConvRing (Qwen3.8's gdn_step ring)
inline constexpr uint32_t kIdxTail = 8;     // the open block's raw keys (kernels::qwen4exp::kTailSlots)
inline constexpr uint32_t kPleRing = 16;    // the PLE id and conv rings (kernels::qwen4exp::kPleRing)
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
size_t ple_state_bytes(const model::Qwen4ExpDesc& d);             // 16 x 4 + 16 x 10240 x 2
// The PLE state's layout inside its allocation: the id ring at 0, the conv ring at ple_conv_off (64-aligned).
inline constexpr size_t kPleConvOff = 64;
// The host table spec 21 §4.3 pins (16 ranges of int8 rows + scales): 51.84 GB with bf16 scales (derived).
size_t host_ple_bytes(const model::Qwen4ExpDesc& d, loader::Q4PleScale s);

// --- spec 21c: the decode list ----------------------------------------------------------------------------
inline constexpr uint32_t kM = 1;   // decode builds M = 1 (21e's verify adds M = 2..4)
// Which decode attention every capture binds: q4_qsa_attn.cl's flash form or q4_qsa_attn_eager.cl's (the
// reference's bf16 chain). `B70_Q4_ATTN=flash|eager`, read at each call; unset or empty is the default;
// anything else throws (spec 18 §10.1's switch).
enum class Q4Attn { Flash, Eager };
inline constexpr Q4Attn kDefaultQ4Attn = Q4Attn::Flash;   // until the box A/B (spec 18 §10.1's rule)
Q4Attn q4_attn();
const char* q4_attn_name(Q4Attn a);
// The QSA selection row (kernels::qwen4exp: kListMax / kListRow / kCountWord), the route row, the slices.
inline constexpr uint32_t kListMax = 2052, kListRow = 2064, kCountWord = 2052;
inline constexpr uint32_t kRouteWords = 32, kAttnTgt = 32, kAttnPart = 258, kPleConsts = 35;
inline constexpr uint32_t kMaxS = 4;   // the widest split-K a mixer GEMV writes (out / o_proj S 4)
// One device's decode scratch, sized for kM rows and the whole descriptor (every device holds the whole set:
// the lists are per device and scratch is per list; the per-layer rows - routes, selections, diagnostics - are
// indexed by absolute layer / QSA index):
//   H, xn        bf16 [M][10240]        the 4-stream residual (materialised by the combines), the HC norm's output
//   x            bf16 [M][2560]         a block's input (up_mix's), the embedding row (embed_gather's)
//   partials     fp32 [max S x N][M]    the dense GEMVs' output (q||gate||k||v S2 x 13312 at most), the mixers'
//                                       split-K slices the next combine folds
//   down_f32     fp32 [M][336]          the HC down||inject GEMV (and the final mixer's 320)
//   inj          fp32 [M][4]            the pending inject weights (up_mix writes, the next combine reads)
//   ab, gdn_o    fp32 [M][128], fp32 [M][48][128]    a||b, gdn_step's output
//   idx_f32, idx_q   fp32 [M][640], fp32 [M][4][128] the indexer GEMV, the roped query heads
//   scores       fp32 [M][max_len / 4]  a row's block scores (the term that scales: 1 MiB at 262144)
//   list         u32 [qsa layers][M][kListRow]  every QSA layer's selection (positions, count) - the gate S rows
//   diag         fp32 [qsa layers][M][2]        the 512th / 513th scores
//   attn_q, attn_gate  fp32 [M][24][256]  attn_prep's
//   attn_part    fp32 [24][kAttnTgt][M][258]  flash's slices
//   attn_out     bf16 [M][6144]         the mixer's output into out_proj / o_proj (GDN: prep_gated_head's)
//   logits_r     fp32 [M][528]          the router GEMV (+ the shared gate)
//   routes       u32 [layers][M][32]    every layer's route row (the routing diagnostic)
//   moe_h, y     bf16 [M][11][640], bf16 [M][2560]
//   ple_e, ple_kv, ple_ids   bf16 [M][2560], fp32 [M][12800], u64 [M][16]   the PLE gather / projections / ids
//   ple_consts   u64 [35]               the hash constants (multipliers, sizes, offsets)
//   logits, argmax_part   fp32 [M][248320], fp32 [M][243][2]
struct ScratchSizes {
  size_t H = 0, xn = 0, x = 0, partials = 0, down_f32 = 0, inj = 0, ab = 0, gdn_o = 0, idx_f32 = 0, idx_q = 0,
         scores = 0, list = 0, diag = 0, attn_q = 0, attn_gate = 0, attn_part = 0, attn_out = 0, logits_r = 0,
         routes = 0, moe_h = 0, y = 0, ple_e = 0, ple_kv = 0, ple_ids = 0, ple_consts = 0, logits = 0,
         argmax_part = 0;
  size_t total() const {
    return H + xn + x + partials + down_f32 + inj + ab + gdn_o + idx_f32 + idx_q + scores + list + diag + attn_q +
           attn_gate + attn_part + attn_out + logits_r + routes + moe_h + y + ple_e + ple_kv + ple_ids + ple_consts +
           logits + argmax_part;
  }
};
ScratchSizes scratch_sizes(const model::Qwen4ExpDesc& d, uint32_t max_len, Q4Attn a = q4_attn());
size_t partials_floats(const model::Qwen4ExpDesc& d);
inline size_t route_at(uint32_t layer) { return size_t(layer) * kM * kRouteWords * 4; }
inline size_t list_at(uint32_t qsa_index) { return size_t(qsa_index) * kM * kListRow * 4; }
inline size_t diag_at(uint32_t qsa_index) { return size_t(qsa_index) * kM * 2 * 4; }
// The residual tap (debug): bf16 [layers][M][10240] on every device (its layers' rows written).
inline size_t tap_bytes(const model::Qwen4ExpDesc& d) { return size_t(d.layers) * kM * d.hc_n() * 2; }
// The injected selection (spec 21 F3's debug input): host USM [qsa layers][M][kListRow] u32 per device.
inline size_t injected_bytes(const model::Qwen4ExpDesc& d) { return size_t(d.qsa_before(d.layers)) * kM * kListRow * 4; }

// The hand-off (spec 16b's buffers through the descriptor-free layout): the MATERIALISED 4-stream H crosses -
// 10240 bf16, 20 KB, as vLLM's PP carries it - and no norm sums (pp_landing_layout's second region empty; the
// stamp word still follows it, the flag its own page).
PpLandingLayout landing_layout(const model::Qwen4ExpDesc& d);
size_t link_bytes(const model::Qwen4ExpDesc& d, uint32_t dev);

// The decode list's launches on one device (the hand-off's copies and barriers excluded; peer's pp_send /
// pp_recv included), and their sum:
//   device 0     embed_gather
//   a GDN layer  15: combine_norm, hc down, up_mix, qkv||z GEMV, a||b, gdn_step, prep_gated_head _SIG, out_proj
//                GEMV, combine_norm, hc down, up_mix, router GEMV, q4_route, q4_moe_gate_up, q4_moe_down
//   a QSA layer  19 (eager 18): combine_norm, hc down, up_mix, q||gate||k||v GEMV, indexer GEMV, attn_prep,
//                q4_qsa_prep / _score / _select, q4_qsa_attn + _reduce (eager: q4_qsa_attn_eager), o_proj GEMV, the
//                MLP side's 8 as a GDN layer's
//   the PLE layer +4: combine_norm _Y_NN (materialise H), q4_ple_gather, key||value GEMV, q4_ple_block (3 when it
//                is device 1's first layer: the landed H is already materialised)
//   last device  the final mixer's combine_norm, its down GEMV, up_mix, lm_head, argmax x 2
// 779 at 48 layers on one card (1 + 36 x 15 + 12 x 19 + 4 + 6), 767 under eager; two cards +1 (device 0 ends with
// combine_norm _Y_NN: the materialised H crosses; device 1 starts at its first layer's norm-only _X), peer +2
// (pp_send, pp_recv). `injected` (spec 21 F3's run with the reference's selections fed in): every QSA layer
// without q4_qsa_score / _select, 2 fewer.
size_t device_launches(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t dev, Q4Attn a,
                       PpHandoff h, bool injected = false);
size_t decode_launches(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, Q4Attn a, PpHandoff h,
                       bool injected = false);

// --- the split by bytes (16b's rule) ---------------------------------------------------------------------
// layer_bytes[l] = loader::q4_layer_bytes(d, l).total() + its state at max_len (QSA: KV, keys, tail; GDN:
// state, ring; the PLE layer: + the PLE state).
std::vector<size_t> pp_layer_bytes(const model::Qwen4ExpDesc& d, uint32_t max_len);
// dev0_fixed = the embedding + the RoPE table + control; dev1_fixed = the final mixer + lm_head (+ the MTP
// head) + the RoPE table + control - then runtime::pp_balance.
PpBalance pp_split(const model::Qwen4ExpDesc& d, uint32_t max_len, bool int8_head, bool mtp);

// --- the planner ---------------------------------------------------------------------------------------
// One device's plan in memory_line()'s components: model = weights + RoPE; kv = KV + indexer keys + tails;
// decode_state = control + GDN state + conv ring + PLE state + (21c) the decode scratch (+ the tap with
// debug_tap) + the hand-off on two cards.
struct DevicePlan : MemoryComponents {
  uint32_t device = 0, first = 0, end = 0;   // its layers [first, end)
  size_t weights = 0, rope = 0, state = 0, scratch = 0, link = 0;
  bool mtp = false;                          // the MTP head's weights are in `weights` (the last device)
  bool whole = false;                        // the descriptor is the published 48 layers (not --layers N)
};
std::vector<DevicePlan> plan(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t max_len,
                             bool int8_head, bool mtp, bool debug_tap = false);
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
