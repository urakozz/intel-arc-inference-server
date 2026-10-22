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
  bool bf16_state = false;                            // spec 25c, Decision 2's arm: the SSM state in bf16 (half the bytes)
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

// ==== spec 25c: the decode list's switches, scratch and launch counts (begin) ==========================================
inline constexpr uint32_t kM = 1;   // the decode list is M = 1 (the M-row verify lists are spec 26c's)

// Which decode attention every capture binds - Muse's flash mg_attn (the default, held to fp64) or its eager form (the
// reference's bf16 chain, ONE launch): `B70_BOLT_ATTN=flash|eager`; unset or empty is the default; anything else throws.
enum class BoltAttn { Flash, Eager };
BoltAttn parse_attn(const std::string& v);
BoltAttn bolt_attn();                       // B70_BOLT_ATTN
const char* bolt_attn_name(BoltAttn a);
// The fused arms (spec 25 §6.1, Decision 16) - each bitwise the as-built list: `router` (the E router rows inside the MoE
// block's norm launch: -1 a MoE block), `route` (the rank recomputed in every nb_moe_up work-group: -1 a MoE block), `conv`
// (the conv inside nb_ssm: -1 a Mamba block). `B70_BOLT_FUSE=none|all|<comma list of router,route,conv>`; unset or empty is
// none (Decision 16's default until Task 8 records the box's A/B); anything else throws.
struct BoltFuse {
  bool router = false, route = false, conv = false;
  bool any() const { return router || route || conv; }
};
BoltFuse parse_fuse(const std::string& v);
BoltFuse bolt_fuse();                       // B70_BOLT_FUSE
std::string bolt_fuse_name(BoltFuse f);     // "none" | "all" | "router,conv" ...
// The SSM state's precision (Decision 2): fp32 the default (the config's mamba_ssm_cache_dtype), bf16 the measured arm:
// `B70_BOLT_SSM=fp32|bf16`; anything else throws.
enum class BoltState { Fp32, Bf16 };
BoltState parse_state(const std::string& v);
BoltState bolt_state();                     // B70_BOLT_SSM
const char* bolt_state_name(BoltState s);

// The route row's words (= kernels::bolt::route::kWords, checked in bolt_capture.cc) and a block's row in `routes`.
inline constexpr uint32_t kRouteWords = 48;
inline size_t route_at(uint32_t layer) { return size_t(layer) * kM * kRouteWords * 4; }
// The residual tap (debug): bf16 [layers][M][hidden] on every device.
inline size_t tap_bytes(const model::BoltDesc& d) { return size_t(d.layers) * kM * d.hidden * 2; }

// The split-K slices block l leaves for the next norm to fold (no block folds its own output - spec 25 §6.1): {s1, s2} =
// Mamba out_proj's S, attention o_proj's S, a MoE block's fc2 and shared_down S (the latter's slices right after the
// former's in `slices`).
struct BoltPending {
  uint32_t s1 = 0, s2 = 0;
  uint32_t total() const { return s1 + s2; }
};
BoltPending pending_of(const model::BoltDesc& d, uint32_t layer);
// The hand-off at the cut (devices 2): h (kM x hidden bf16) and ALWAYS the pending slices of block split - 1 ((s1 + s2) x
// kM x hidden fp32 in the landing's sums region - spec 25 §4.1); device 1's first norm folds them.
PpLandingLayout landing_layout(const model::BoltDesc& d, const model::BoltPlacement& p);
BoltPending cut_pending(const model::BoltDesc& d, const model::BoltPlacement& p);

// One device's decode scratch at kM rows (every device has the whole set: the lists are per device), never zeroed (no step
// reads scratch it has not written first), but `ones` (filled once with bf16 1.0 - Decision 15's gate):
//   h0, h1, x        bf16 [M][4096]          the residual (ping-ponged between norm sites), every norm's output
//   slices           fp32 [widest][M]        every int4 GEMV's split-K output: the widest S x N of the linears, and a MoE
//                                            block's fc2 + shared_down pair ((s1 + s2) x 4096) side by side
//   xc, y, g         bf16 [M][10240], [M][8192], [M][8192]   nb_conv's, nb_ssm's, nb_gnorm's outputs
//   logits_r         fp32 [M][max E]         the router's logits (or the arm router's)
//   routes           u32  [layers][M][48]    every MoE block's route row
//   lat, sh          bf16 [M][1024], [M][5376]   nb_sh_act's
//   moe_a, moe_r     bf16 [M][9][max I], [M][1024]   the experts' relu2 activations, the combine
//   attn_q           fp32 [M][32][128];  attn_part fp32 [32][64][M][130];  attn_out, ones bf16 [M][4096]
//   attn_scores      fp32 [M][32][max_len]  eager only (0 under flash) - the term that scales with max_len
//   logits           fp32 [M][131072];  argmax_part fp32 [M][128][2]
// An empty term (a descriptor without that block kind) is one 64-byte line.
struct ScratchSizes {
  size_t h0 = 0, h1 = 0, x = 0, slices = 0, xc = 0, y = 0, g = 0, logits_r = 0, routes = 0, lat = 0, sh = 0, moe_a = 0,
         moe_r = 0, attn_q = 0, attn_part = 0, attn_out = 0, ones = 0, attn_scores = 0, logits = 0, argmax_part = 0;
  size_t total() const {
    return h0 + h1 + x + slices + xc + y + g + logits_r + routes + lat + sh + moe_a + moe_r + attn_q + attn_part + attn_out +
           ones + attn_scores + logits + argmax_part;
  }
};
ScratchSizes scratch_sizes(const model::BoltDesc& d, uint32_t max_len, BoltAttn a);
// One device's persistent group: control (shared), its attention blocks' K and V [max_len][2][128] each, its Mamba blocks'
// raw-xBC rings [16][10240] and SSM states [128][64][128] (fp32, or bf16 under the arm). An empty group is one 64-byte
// line (bolt_decode_scratch counts the lines plan()'s exact terms leave out).
struct PersistentSizes {
  size_t control = 0, kv_k = 0, kv_v = 0, conv_ring = 0, ssm = 0;
  size_t total() const { return control + kv_k + kv_v + conv_ring + ssm; }
};
PersistentSizes persistent_sizes(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev, uint32_t max_len,
                                 BoltState s);
inline size_t kv_layer_bytes(const model::BoltDesc& d, uint32_t max_len) { return size_t(max_len) * d.kv_n() * 2; }
inline size_t ring_layer_bytes(const model::BoltDesc& d) { return size_t(loader::bolt::kConvRing) * d.m_conv_dim * 2; }
inline size_t ssm_layer_bytes(const model::BoltDesc& d, BoltState s) {
  return size_t(d.m_heads) * d.m_head_dim * d.m_state * (s == BoltState::Bf16 ? 2 : 4);
}
// What BoltReserve::decode_scratch must hold for device `dev`'s decode list: the control block, scratch_sizes' total and
// the 64-byte lines of any EMPTY persistent group (plan()'s kv / ssm / conv_rings terms are exact and 0 there) - so plan()
// IS the allocation (bolt_decode_test checks it on the card).
size_t decode_scratch_bytes(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev, uint32_t max_len,
                            BoltAttn a, BoltState s);

// The decode list's launches on one device (the hand-off's copies and barriers excluded; peer's pp_send / pp_recv
// included), and their sum (spec 25 §6.1):
//   device 0     nb_embed_gather
//   a block      Mamba 6 (norm, in_proj, conv, ssm, gnorm, out_proj); attention 6 (norm, q||k||v, prep, decode, reduce,
//                o) - eager 5 (Muse's mg_attn_eager is ONE launch, Decision 15); MoE 9 (norm, router, route,
//                fc1||shared_up, sh_act, up, down, fc2, shared_down); the arms: conv -1 a Mamba block, router -1 and route
//                -1 a MoE block
//   last device  the final norm (norm_f, folding the last block), the head GEMV, the two argmax stages
// 653 on the real model (40 x 6 + 8 x 6 + 40 x 9 + 5), 645 eager, 533 with every arm; two cards +0 (copy) / +2 (peer).
size_t device_launches(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev, BoltAttn a, BoltFuse f,
                       PpHandoff h);
size_t decode_launches(const model::BoltDesc& d, const model::BoltPlacement& p, BoltAttn a, BoltFuse f, PpHandoff h);
// ==== spec 25c (end) ===================================================================================================

}  // namespace runtime::bolt
