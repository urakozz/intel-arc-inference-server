#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/muse_glimmer.h"
#include "runtime/pipeline_plan.h"   // PpHandoff, PpLandingLayout, pp_landing_layout (spec 16b)

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

// ==== Spec 23c: the decode list's buffers and launches ======================================================
// Which decode attention every capture binds - mg_attn.cl's flash form or mg_attn_eager.cl's eager form (the
// reference's bf16 chain). `B70_MUSE_ATTN=flash|eager`, read where called (MuseEngine reads it once, at
// construction); unset or empty is the default; anything else throws (spec 18 §10.1's switch).
enum class MuseAttn { Flash, Eager };
inline constexpr MuseAttn kDefaultMuseAttn = MuseAttn::Flash;   // until the box A/B (spec 18 §10.1's rule)
MuseAttn muse_attn();
const char* muse_attn_name(MuseAttn a);

inline constexpr uint32_t kM = 1;          // 23c builds M = 1 (23f adds the verify widths M = 2..16)
inline constexpr uint32_t kNormG = 26;     // the Σ² chunks at hidden 6656 (= kernels::muse::kNormG)
inline constexpr uint32_t kAttnTgt = 64, kAttnPart = 130, kArgmaxChunk = 1024;

// One device's decode scratch for M rows (every device has the whole set: the lists are per device and scratch is
// per list). bf16 unless noted:
//   resid, x, a      [M][hidden]           the residual, every norm's output, a sub-block's own row (o_proj / down)
//   partials   fp32  [max S x N][M]        the int4 GEMVs' split-K output (gate||up S8 x 39936 at most)
//   sumsq_a, sumsq_r fp32 [kNormG][M]      a sub-block's Σ², the residual's
//   attn_q     fp32  [M][q_n]              mg_attn_prep's q;    gate [M][q_n] its bf16(sigmoid(g))
//   attn_part  fp32  [q_heads][TGT][M][130]   flash's blocks;   attn_out [M][q_n] o_proj's input
//   h                [M][inter]            SiLU(gate) x up, down's input
//   logits     fp32  [M][vocab];  argmax_part fp32 [M][ceil(vocab / 1024)][2]
//   control          runtime::Control (shared memory)
//   rope       fp32  [max_len][head_dim]   the sliding layers' cos / sin (runtime::muse::rope_table)
//   attn_s     fp32  [M][q_heads][max_len] eager's score row (B70_MUSE_ATTN=eager only)
struct MuseScratchSizes {
  size_t resid = 0, x = 0, a = 0, partials = 0, sumsq_a = 0, sumsq_r = 0, attn_q = 0, gate = 0, attn_part = 0,
         attn_out = 0, h = 0, logits = 0, argmax_part = 0, control = 0, rope = 0, attn_s = 0;
  size_t total() const {
    return resid + x + a + partials + sumsq_a + sumsq_r + attn_q + gate + attn_part + attn_out + h + logits +
           argmax_part + control + rope + attn_s;
  }
};
MuseScratchSizes scratch_sizes(const model::MuseGlimmerDesc& d, uint32_t max_len, uint32_t M, MuseAttn a);
size_t partials_floats(const model::MuseGlimmerDesc& d);   // max over the four GEMVs of S x N
// The decode scratch of one device (scratch_sizes(...).total()) - what MuseReserve::decode_scratch carries, plus,
// on two cards, link_bytes (decode_reserve below).
size_t decode_scratch_bytes(const model::MuseGlimmerDesc& d, uint32_t max_len, uint32_t M, MuseAttn a);
// The persistent attention state of one device: its full layers' K and V [max_len][kv_heads][128] and its sliding
// layers' rings [4096][kv_heads][128] (bf16) - the planner's kv / rings (loader/muse_layout.h), exactly.
size_t device_kv_bytes(const model::MuseGlimmerDesc& d, const model::MusePlacement& p, uint32_t dev, uint32_t max_len);
size_t device_ring_bytes(const model::MuseGlimmerDesc& d, const model::MusePlacement& p, uint32_t dev);
// The residual tap (debug, not planned): bf16 [layers][M][hidden] on every device.
inline size_t tap_bytes(const model::MuseGlimmerDesc& d) { return size_t(d.layers) * kM * d.hidden * 2; }

// The hand-off (spec 16b's buffers through the descriptor-free layout): the residual row and the 26 norm sums
// cross. Device 0: pp_send's counter; device 1: the landing allocation + pp_recv's state.
PpLandingLayout landing_layout(const model::MuseGlimmerDesc& d);
size_t link_bytes(const model::MuseGlimmerDesc& d, uint32_t dev);
// What a run reserves per device for 23c's decode (the planner's MuseReserve::decode_scratch): the scratch at
// max_len (0 = the model's 131072: --max-len auto plans with the largest), + the larger device's link on two cards.
MuseReserve decode_reserve(const model::MuseGlimmerDesc& d, uint32_t max_len, uint32_t devices, MuseAttn a);

// The decode list's launches on one device (the hand-off's copies and barriers excluded; peer's pp_send / pp_recv
// included), and their sum (spec 23 §5.2):
//   device 0  mg_embed (the gather, the weight-less norm and layer 0's Σ² in one launch), then its layers
//   a layer   14 (flash) / 13 (eager): mg_norm, q||gate||k||v GEMV, mg_attn_prep, mg_attn_decode + mg_attn_reduce
//             (eager: mg_attn_eager), o_proj GEMV, prep_res_fold _Z, mg_post_add, mg_norm, gate||up GEMV,
//             mg_silu_mul, down GEMV, prep_res_fold _Z, mg_post_add
//   last      mg_final_norm, lm_head, the two argmax stages
// 733 / 681 at 52 layers on one card AND on two (the cut is 16b's: device 0 ends with layer s-1's mg_post_add,
// which already wrote resid and the sums layer s's mg_norm reads); peer adds pp_send and pp_recv: 735 / 683.
// `M` is the rows of a step (23c: 1; the count does not depend on it).
size_t device_launches(const model::MuseGlimmerDesc& d, const model::MusePlacement& p, uint32_t dev, MuseAttn a,
                       PpHandoff h, uint32_t M = 1);
size_t decode_launches(const model::MuseGlimmerDesc& d, const model::MusePlacement& p, MuseAttn a, PpHandoff h);

}  // namespace runtime::muse
