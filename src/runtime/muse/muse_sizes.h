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

// ==== Spec 23d: the prefill chunk ===========================================================================
// One chunk of at most kPfC positions (= kernels::muse::kPfC, checked in muse_prefill.cc). Every allocation
// MusePrefillScratch makes on a device, as device-free arithmetic. None depends on max_len (the flash attention keeps
// no score scratch; the eager build recomputes its scores), so the prefill scratch is one constant per device.
inline constexpr uint32_t kPfC = 2048;
inline constexpr uint32_t kPfSlab = 1024;   // a whole slab of a linear (pf_gemm's 1024 columns)
inline uint32_t pf_pad256(uint32_t n) { return (n + 255) / 256 * 256; }
// A linear's slab walk over N columns: slabs of kPfSlab and one tail of pad256(N - n0) (k2_pf_dequant_slab zero-fills
// its padding); the partials row pitch pad256(N) - 8704, 6656 and 39936 are all whole 256s.
inline uint32_t pf_slabs(uint32_t N) { return (N + kPfSlab - 1) / kPfSlab; }
inline uint32_t pf_slab_width(uint32_t N, uint32_t n0) { return N - n0 >= kPfSlab ? kPfSlab : pf_pad256(N - n0); }
inline uint32_t pf_ld(uint32_t N) { return pf_pad256(N); }

// MusePrefillScratch's allocations on one device (kPfC rows each; both devices hold the whole set - the lists are
// per device):
//   ids       u32  [kPfC]                  host memory: the chunk's ids (read by device 0's mg_pf_embed)
//   resid     bf16 [kPfC][hidden]          the residual stream (crosses the cut on two cards)
//   x, a      bf16 [kPfC][hidden]          every norm's output; a sub-block's own row (o_proj's, down's)
//   partials  fp32 [kPfC][pf_ld(39936)]    the slab GEMMs' output (q||gate||k||v, o_proj, gate||up, down in turn)
//   slab      bf16 [19968][kPfSlab]        one slab of the widest-K linear (down)
//   sumsq_a/r fp32 [kNormG][kPfC]          a sub-block's Σ², the residual's (crosses with resid)
//   attn_q    fp32 [kPfC][q_n]             mg_attn_prep's q;   gate bf16 [kPfC][q_n] its bf16(sigmoid(g))
//   attn_out  bf16 [kPfC][q_n]             the attention's output, gated in place, o_proj's A
//   h         bf16 [kPfC][inter]           SiLU(gate) x up, down's A
struct PrefillSizes {
  size_t ids = 0, resid = 0, x = 0, a = 0, partials = 0, slab = 0, sumsq_a = 0, sumsq_r = 0, attn_q = 0, gate = 0,
         attn_out = 0, h = 0;
  size_t total() const {
    return ids + resid + x + a + partials + slab + sumsq_a + sumsq_r + attn_q + gate + attn_out + h;
  }
};
PrefillSizes prefill_sizes(const model::MuseGlimmerDesc& d);
// The two-card prefill hand-off: a second runtime::PipelineLink in `copy` mode over spec 16b's layout at chunk size -
// kPfC rows of the residual and the [kNormG][kPfC] sums (27.3 MB + 213 KB).
PpLandingLayout pf_landing_layout(const model::MuseGlimmerDesc& d);
size_t pf_link_bytes(const model::MuseGlimmerDesc& d, uint32_t dev);
// What a prefilling run reserves per device (MuseReserve::prefill_scratch): the scratch, + the larger device's prefill
// link on two cards.
size_t prefill_reserve(const model::MuseGlimmerDesc& d, uint32_t devices);
// decode_reserve, plus the prefill's when the run prefills (MuseOptions::prefill).
MuseReserve run_reserve(const model::MuseGlimmerDesc& d, uint32_t max_len, uint32_t devices, MuseAttn a, bool prefill);

// The chunk's launches (the hand-off's copies, the event and the head excluded), the same at every C and under either
// attention form:
//   device 0  mg_pf_embed (the gather, the weight-less norm and layer 0's Σ² in one launch), then its layers
//   a layer   mg_norm · q||gate||k||v pf_slabs(8704) x (slab + pf_gemm_T0) · mg_attn_prep · mg_pf_flash_attn ·
//             mg_pf_attn_gate · o_proj 7 x 2 · pf_res_fold _Z + mg_post_add · mg_norm · gate||up 39 x 2 · mg_silu_mul ·
//             down 7 x 2 · pf_res_fold _Z + mg_post_add = 134 at Muse's shapes
// 6969 at 52 layers on one card AND on two (the cut is 16b's: device 0 ends with layer s-1's mg_post_add, whose rows
// and sums cross; device 1 starts at layer s's norm).
size_t prefill_layer_launches(const model::MuseGlimmerDesc& d);
size_t prefill_device_launches(const model::MuseGlimmerDesc& d, const model::MusePlacement& p, uint32_t dev);
size_t prefill_chunk_launches(const model::MuseGlimmerDesc& d, const model::MusePlacement& p);
inline constexpr size_t kPrefillHeadLaunches = 4;   // mg_final_norm on the last row, lm_head, argmax x 2

// The bound a prefill chunk's lists are waited for (ms), `depth` = the chunk's end (pos + C): the configured bound
// (MuseOptions::prefill_timeout_ms) under flash; under B70_MUSE_ATTN=eager - the portable three-pass build, whose cost
// grows with the keys a row reads (a full layer's row at depth d reads d keys, three times) - one bound per kPfC keys
// of the chunk's deepest row, so a long prompt's eager chunk is not reported as a lost hand-off.
inline uint64_t prefill_wait_ms(uint32_t base_ms, MuseAttn a, uint32_t depth) {
  const uint64_t steps = a == MuseAttn::Eager ? (uint64_t(depth) + kPfC - 1) / kPfC : 1;
  return uint64_t(base_ms) * (steps ? steps : 1);
}

// ==== Spec 23e: prefix-cache snapshots (spec 7's PrefixCache on Muse-Glimmer, spec 23 §5.4) ======================
// What a session leaves behind between replays, besides `pos`: the 13 full layers' growing KV (the cache's BLOCKS, as
// every model's) and the 39 sliding layers' rings. A query at position p reads the keys (p - 2048, p] - its own key
// it writes itself - so the ring rows of positions [p - 2047, p) are the whole recurrent "state": 39 x 2047 x (K + V)
// x 256 x 2 B = 81,748,992 B at the real shapes (derived), the same at every pos. Positions before 0 (a snapshot at p
// < 2047) are zero rows on the host, written back as zeros: the window never reads them, and a cold run's ring holds
// zeros there too (reset). The RoPE table and the scratch are per step or constant.
//
// The HOST layouts (spec 16b's rule: --pp 2's is --pp 1's byte for byte, so an entry moves between one card and two
// unchanged - every layer is addressed through the placement, never by device order):
//   state  [K | V][sliding layer, in layer order][state_positions(d)][kv_n] bf16, positions ascending
//   kv     [K | V][full layer, in layer order][end - begin][kv_n] bf16 (Kolibri's / runtime::Engine's order)
// A run is one contiguous range of one device allocation (`tensor`: the device's full_k, full_v, ring_k, ring_v), in
// host order. `zero` runs cover positions before 0: save_state writes zeros to the host for them (the device slots are
// not read), load_state copies the host's zeros into their slots.
inline uint32_t state_positions(const model::MuseGlimmerDesc& d) { return d.window - 1; }   // 2047
enum class SnapTensor : uint32_t { FullK, FullV, RingK, RingV };
struct SnapRun {
  uint32_t device = 0;
  SnapTensor tensor = SnapTensor::FullK;
  size_t offset = 0, bytes = 0;   // within the device allocation
  bool zero = false;              // positions before 0 (state runs only)
};
// One device's four persistent allocations (MuseBuffers' full_k / full_v / ring_k / ring_v), bytes.
struct PersistentSizes {
  size_t full_k = 0, full_v = 0, ring_k = 0, ring_v = 0;
};
PersistentSizes persistent_sizes(const model::MuseGlimmerDesc& d, const model::MusePlacement& p, uint32_t dev,
                                 uint32_t max_len);
// 39 x 2047 x 2 x 512 B = 81,748,992 at the real shapes.
size_t state_snapshot_bytes(const model::MuseGlimmerDesc& d);
// 13,312 B a position at the real shapes (13 full layers x (K + V) x 256 x 2 B).
size_t kv_snapshot_bytes(const model::MuseGlimmerDesc& d, uint32_t n_pos);
// The ring rows of positions [pos - state_positions, pos), every sliding layer, K then V. Throws when the ring cannot
// hold them (kRing < state_positions).
std::vector<SnapRun> state_runs(const model::MuseGlimmerDesc& d, const model::MusePlacement& p, uint32_t pos);
// Positions [begin, end) of every full layer, K then V. Throws unless begin <= end <= max_len.
std::vector<SnapRun> kv_runs(const model::MuseGlimmerDesc& d, const model::MusePlacement& p, uint32_t max_len,
                             uint32_t begin, uint32_t end);

}  // namespace runtime::muse
