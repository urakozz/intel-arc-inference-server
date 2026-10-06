#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/kolibri1.h"
#include "runtime/memory_plan.h"
#include "runtime/pipeline_plan.h"   // PpBalance, PpChoice, PpHandoff, kPpDevices, the landing layout

// Spec 20c: Kolibri-1's runtime allocations as device-free arithmetic, the planner over them
// (`--max-len auto`, spec 6 §10) and the split by bytes (`--pipeline-split auto`, spec 16b's
// runtime::pp_balance - not a second rule) - runtime/k2/k2_sizes.h's arrangement for Kolibri: the
// KolibriBuffers constructor allocates exactly what these return, per device, and the plan adds the
// same numbers up before anything is allocated. Host only (tests/runtime/kolibri_plan_test.cc).
namespace runtime::kolibri {

inline constexpr uint32_t kM = 1;   // Kolibri builds M = 1 only (no MTP head, spec 20 §1)

// Which decode attention every capture binds - kol_attn.cl's flash form or kol_attn_eager.cl's
// eager form (the reference's bf16 chain). `B70_KOLIBRI_ATTN=flash|eager`, read at each call;
// unset or empty is the default; anything else throws (spec 18 §10.1's switch).
enum class KolAttn { Flash, Eager };
inline constexpr KolAttn kDefaultKolAttn = KolAttn::Flash;   // until the box A/B (spec 18 §10.1's rule)
KolAttn kolibri_attn();
const char* kol_attn_name(KolAttn a);

// One device's persistent group (zeroed by KolibriBuffers::zero, KolibriEngine::reset):
//   control      runtime::Control, shared memory
//   full_k/v     bf16 [its full layers][max_len][kv_heads][head_dim] each - the growing KV, 20 KiB a
//                position for the model's 10 full layers (K and V)
//   ring_k/v     bf16 [its sliding layers][kRing][kv_heads][head_dim] each - 8 MiB a layer (K and V),
//                335.5 MB for the 40 sliding layers (derived)
// An empty group (a device with no full / no sliding layer) still allocates one 64-byte line.
struct PersistentSizes {
  size_t control = 0, full_k = 0, full_v = 0, ring_k = 0, ring_v = 0;
  size_t kv() const { return full_k + full_v + ring_k + ring_v; }
  size_t total() const { return control + kv(); }
};
PersistentSizes persistent_sizes(const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t dev,
                                 uint32_t max_len);
inline size_t full_kv_bytes_per_pos(const model::Kolibri1Desc& d) {
  return size_t(d.full_before(d.layers)) * 2 * d.kv_n() * 2;   // 20480 at 50 layers
}
inline size_t ring_bytes_per_layer(const model::Kolibri1Desc& d) {
  return size_t(model::Kolibri1Desc::kRing) * 2 * d.kv_n() * 2;   // 8 MiB (K and V)
}
inline size_t full_bytes_per_layer(const model::Kolibri1Desc& d, uint32_t max_len) {
  return size_t(max_len) * 2 * d.kv_n() * 2;   // K and V
}
// The rows of one layer of either kind in its group (bytes, K or V alone).
inline size_t full_layer_rows_bytes(const model::Kolibri1Desc& d, uint32_t max_len) {
  return size_t(max_len) * d.kv_n() * 2;
}
inline size_t ring_layer_rows_bytes(const model::Kolibri1Desc& d) {
  return size_t(model::Kolibri1Desc::kRing) * d.kv_n() * 2;
}

// The decode scratch of one device, sized for kM rows (every device has the whole set: the lists are
// per device and scratch is per list):
//   resid, x, a, mo  bf16 [M][hidden]        the residual, every norm's output, o_proj's row, the MoE's
//   partials         fp32 [max S x N][M]     the int4 / bf16 GEMVs' output (q||k||v S2 x 7168 at most)
//   sumsq_a, sumsq_r fp32 [kNormG][M]        a sub-block's Σ², the residual's
//   attn_q           fp32 [M][q_n]
//   attn_part        fp32 [q_heads][TGT][M][130]
//   attn_out         bf16 [M][q_n]           o_proj's input
//   logits_r         fp32 [M][router_n]      the router GEMV's fp32 logits
//   routes           u32  [layers][M][32]    every layer's route row (the routing diagnostic)
//   moe_h            bf16 [M][7][moe_inter]  SiLU(gate) x up of the 6 routed + the shared slot
//   logits           fp32 [M][vocab]
//   argmax_part      fp32 [M][ceil(vocab / 1024)][2]
struct ScratchSizes {
  size_t resid = 0, x = 0, a = 0, mo = 0, partials = 0, sumsq_a = 0, sumsq_r = 0, attn_q = 0, attn_part = 0,
         attn_out = 0, logits_r = 0, routes = 0, moe_h = 0, logits = 0, argmax_part = 0;
  size_t total() const {
    return resid + x + a + mo + partials + sumsq_a + sumsq_r + attn_q + attn_part + attn_out + logits_r + routes +
           moe_h + logits + argmax_part;
  }
};
ScratchSizes scratch_sizes(const model::Kolibri1Desc& d);
size_t partials_floats(const model::Kolibri1Desc& d);
// Eager's score row, fp32 [M][q_heads][max_len]; 0 under flash.
size_t attn_scores_bytes(const model::Kolibri1Desc& d, uint32_t max_len, KolAttn a = kolibri_attn());
inline constexpr uint32_t kRouteWords = 32;   // = kernels::kolibri::route::kWords
inline size_t route_at(uint32_t layer) { return size_t(layer) * kM * kRouteWords * 4; }
// The residual tap (debug): bf16 [layers][M][hidden] on every device (its layers' rows written).
inline size_t tap_bytes(const model::Kolibri1Desc& d) { return size_t(d.layers) * kM * d.hidden * 2; }

// The hand-off (spec 16b's buffers through the descriptor-free layout): the residual row and the
// 20 norm sums cross. Device 0: pp_send's counter; device 1: the landing allocation + pp_recv's state.
PpLandingLayout landing_layout(const model::Kolibri1Desc& d);
size_t link_bytes(const model::Kolibri1Desc& d, uint32_t dev);

// The decode list's launches on one device (the hand-off's copies and barriers excluded; peer's
// pp_send / pp_recv included), and their sum:
//   device 0  embed_gather + prep_res_fold SP0, then its layers
//   a layer   15 (flash) / 17 (eager): kol_norm_finish, q||k||v GEMV, kol_attn_prep, decode + reduce
//             (eager: score, softmax, P·V, reduce), o_proj GEMV, prep_res_fold _Z (a, Σa²), kol_post_add
//             (post_attn_norm), kol_norm_finish, router GEMV, kol_route, kol_moe_gate_up, kol_moe_down,
//             prep_res_fold SP0 (Σmo²), kol_post_add (post_ffn_norm)
//   last      kol_norm_finish (model.norm), lm_head, the two argmax stages
// 756 / 856 at 50 layers on one card AND on two (the cut is 16b's: device 0 ends with layer s-1's
// kol_post_add, which already wrote resid and the sums layer s's norm reads); peer adds pp_send and
// pp_recv: 758 / 858.
size_t device_launches(const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t dev, KolAttn a,
                       PpHandoff h);
size_t decode_launches(const model::Kolibri1Desc& d, const model::KolPlacement& p, KolAttn a, PpHandoff h);

// --- spec 20d: the prefill chunk ------------------------------------------------------------------
// One chunk of at most kPfC positions (= kernels::kolibri::kPfC, checked in kolibri_prefill.cc): every
// allocation KolibriPrefillScratch makes on a device, as device-free arithmetic. None depends on max_len
// (the flash attention keeps no score scratch), so the prefill scratch is one constant per device.
inline constexpr uint32_t kPfC = 2048;
inline constexpr uint32_t kPfTm = 32;                          // the grouped GEMM's tile rows (pf_moe_gemm TM)
inline constexpr uint32_t kPfSlab = 1024;                      // a whole slab of an attention linear
inline constexpr size_t kPfBatchBytes = size_t(512) << 20;     // one weight batch of bf16 expert blocks
inline uint32_t pad256(uint32_t n) { return (n + 255) / 256 * 256; }
// The attention linears' slab walk over N columns: slabs of kPfSlab and one tail of pad256(N - n0)
// (k2_pf_dequant_slab / kol_pf_bf16_slab zero-fill its padding); the partials row pitch pad256(N).
inline uint32_t pf_slabs(uint32_t N) { return (N + kPfSlab - 1) / kPfSlab; }
inline uint32_t pf_slab_width(uint32_t N, uint32_t n0) { return N - n0 >= kPfSlab ? kPfSlab : pad256(N - n0); }
inline uint32_t pf_ld(uint32_t N) { return pad256(N); }
// The grouped GEMMs' padded tile count of a C-row chunk (kol_pf_moe.cl's header has the bound):
//   floor((C x top_k + experts x (TM - 1)) / TM) + ceil(C / TM)        820 at C = 2048
uint32_t pf_tiles(const model::Kolibri1Desc& d, uint32_t C);
// The weight batches: 385 blocks (384 experts + the bf16 shared expert, block 384) of bf16 [K][N] in
// kPfBatchBytes regions - gate||up 5,242,880 B a block, 102 a batch, 4 batches; down 2,621,440 B, 204,
// 2 batches.
size_t pf_block_gu_bytes(const model::Kolibri1Desc& d);
size_t pf_block_dn_bytes(const model::Kolibri1Desc& d);
uint32_t pf_batch_blocks_gu(const model::Kolibri1Desc& d);
uint32_t pf_batch_blocks_dn(const model::Kolibri1Desc& d);
uint32_t pf_batches_gu(const model::Kolibri1Desc& d);
uint32_t pf_batches_dn(const model::Kolibri1Desc& d);

// KolibriPrefillScratch's allocations on one device (kPfC rows each; both devices hold the whole set -
// the lists are per device):
//   ids       u32  [kPfC]                       host memory: the chunk's ids (read by device 0's embed)
//   resid     bf16 [kPfC][hidden]               the residual stream (crosses the cut on two cards)
//   x, a, mo  bf16 [kPfC][hidden]               every norm's output, o_proj's own row, the MoE's output
//   partials  fp32 [kPfC][pf_ld(7168)]          the slab GEMMs' output (q||k||v, then o_proj at 2560)
//   slab      bf16 [6144][kPfSlab]              one slab of the widest-K attention linear
//   sumsq_a/r fp32 [kNormG][kPfC]               a sub-block's Σ², the residual's (crosses with resid)
//   attn_q    fp32 [kPfC][q_n]                  kol_attn_prep's q
//   attn_out  bf16 [kPfC][q_n]                  the flash attention's output, o_proj's A
//   logits    fp32 [kPfC][router_n]             the router GEMV
//   routes    u32  [layers][kPfC][32]           every layer's route rows of the LAST chunk
//   hdr       u32  [pf_hdr words]               the sort's header
//   tiles     u32  [tmax(kPfC)][2]              the tile table
//   row_tok   u32  [tmax x TM]                  the sorted rows' tokens
//   pair_row  u32  [kPfC][top_k]                every (token, slot)'s sorted row
//   xg        bf16 [tmax x TM][hidden]          the gathered A, then down's y
//   h         bf16 [tmax x TM][moe_inter]       gate||up's SiLU h
//   w         bf16 kPfBatchBytes                the expert weight batch
struct PrefillSizes {
  size_t ids = 0, resid = 0, x = 0, a = 0, mo = 0, partials = 0, slab = 0, sumsq_a = 0, sumsq_r = 0, attn_q = 0,
         attn_out = 0, logits = 0, routes = 0, hdr = 0, tiles = 0, row_tok = 0, pair_row = 0, xg = 0, h = 0, w = 0;
  size_t total() const {
    return ids + resid + x + a + mo + partials + slab + sumsq_a + sumsq_r + attn_q + attn_out + logits + routes + hdr +
           tiles + row_tok + pair_row + xg + h + w;
  }
};
PrefillSizes prefill_sizes(const model::Kolibri1Desc& d);
inline size_t pf_route_at(uint32_t layer) { return size_t(layer) * kPfC * kRouteWords * 4; }
// The two-card prefill hand-off: a second runtime::PipelineLink in `copy` mode over spec 16b's layout at
// chunk size - kPfC rows of the residual and the [kNormG][kPfC] sums (10.5 MB + 160 KB).
PpLandingLayout pf_landing_layout(const model::Kolibri1Desc& d);
size_t pf_link_bytes(const model::Kolibri1Desc& d, uint32_t dev);

// The chunk's launches on one device and their sum, the same at every C (the hand-off's copies, the
// event and the head excluded):
//   device 0  pf_embed_gather + pf_res_fold SP0, then its layers
//   a layer   45: kol_norm_finish, q||k||v 7 slabs x (dequant | copy + pf_gemm), kol_attn_prep,
//             kol_pf_flash_attn, o_proj 3 slabs x 2, pf_res_fold _Z + kol_post_add, kol_norm_finish,
//             the router GEMV, kol_route, sort, gather, gate||up 4 batches x (dequant + grouped GEMM),
//             down 2 x 2, the combine, pf_res_fold SP0 + kol_post_add
// 2252 at 50 layers on one card AND on two (the cut is 16b's: device 0 ends with layer s-1's
// kol_post_add, whose rows and sums cross; device 1 starts at layer s's norm - nothing recomputed).
size_t prefill_device_launches(const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t dev);
size_t prefill_chunk_launches(const model::Kolibri1Desc& d, const model::KolPlacement& p);
inline constexpr size_t kPrefillHeadLaunches = 5;   // fold + norm over the last row, lm_head, argmax x 2

// --- the split by bytes (16b's rule) -------------------------------------------------------------
// layer_bytes[l] = kol_layer_bytes(d).total() + (is_sliding(l) ? ring_bytes_per_layer : max_len x 2 x kv_n x 2)
std::vector<size_t> pp_layer_bytes(const model::Kolibri1Desc& d, uint32_t max_len);
// dev0_fixed = embedding + RoPE + decode scratch + control + link (device 0); dev1_fixed = final norm +
// lm_head + RoPE + decode scratch + control + link (device 1) - then runtime::pp_balance. `prefill`
// (spec 20d): both devices' fixed bytes also hold the prefill scratch and the prefill link.
PpBalance pp_split(const model::Kolibri1Desc& d, uint32_t max_len, bool int8_head, PpHandoff h,
                   KolAttn a = kolibri_attn(), bool prefill = false);

// --- the planner -----------------------------------------------------------------------------------
// One device's plan in memory_line()'s components: model = its weights + its RoPE table; kv = its
// full KV + its rings; decode_state = control + scratch (+ eager's score row) (+ the tap) + its link;
// prefill_scratch (spec 20d, `prefill`) = prefill_sizes + its prefill link.
struct DevicePlan : MemoryComponents {
  uint32_t device = 0;
  size_t weights = 0, rope = 0, link = 0, full_kv = 0, rings = 0;
};
std::vector<DevicePlan> plan(const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t max_len,
                             bool int8_head, bool debug_tap = false, KolAttn a = kolibri_attn(), bool prefill = false);
// The largest multiple of kMaxLenQuantum, at most min(cap, trained 262144), whose plan + reserve fits
// EVERY device; 0 when not even min(kMinAutoMaxLen, cap) does.
uint32_t max_len_that_fits(const model::Kolibri1Desc& d, const model::KolPlacement& p, bool int8_head,
                           const std::array<size_t, kPpDevices>& device_bytes, size_t reserve, uint32_t cap = 0,
                           bool prefill = false);
// `--pipeline-split auto` with `--max-len auto`: pp_auto_split_and_len's rule over pp_split - the split
// whose min-over-devices max_len is the largest, ties broken by pp_balance at that length.
PpChoice pp_split_and_len(const model::Kolibri1Desc& d, bool int8_head,
                          const std::array<size_t, kPpDevices>& device_bytes, size_t reserve, uint32_t cap = 0,
                          bool prefill = false);
// Whether `--pp 1` may run: the one-card plan at min(kMinAutoMaxLen, trained) fits device_bytes - reserve.
bool fits_one_card(const model::Kolibri1Desc& d, bool int8_head, size_t device_bytes, size_t reserve,
                   bool prefill = false);
// "pipeline plan at max_len N, split s (layers [0, s) | [s, L)):" + one line per device
// (runtime::pp_describe's text; "plan at max_len N (one card):" for one device).
std::string describe(const std::vector<DevicePlan>& p, const model::KolPlacement& pl, uint32_t max_len,
                     const std::array<size_t, kPpDevices>& device_bytes, size_t reserve);

// --- spec 20e: prefix-cache snapshots (spec 7's PrefixCache on Kolibri) ----------------------------
// What a session leaves behind between replays, besides `pos`: the 10 full layers' growing KV (the
// cache's BLOCKS, as every model's) and the 40 sliding layers' rings. A query at position p reads the
// keys (p - window, p] - its own key it writes itself - so the ring rows of positions [p - 512, p) are
// the whole recurrent "state" (spec 7's word): 40 x 512 x (K + V) x kv_n x 2 B = 41,943,040 B at the
// real shapes (derived), the same at every pos. Positions before 0 (a snapshot at p < 512) are zero
// rows on the host, written back as zeros: the window never reads them, and a cold run's ring holds
// zeros there too (reset). The routes, the RoPE table and the scratch are per step or constant.
//
// The HOST layouts (spec 16b's rule: --pp 2's is --pp 1's byte for byte, so an entry moves between one
// card and two unchanged - every layer is addressed through the placement, never by device order):
//   state  [K | V][sliding layer, in layer order][state_positions(d)][kv_n] bf16, positions ascending
//   kv     [K | V][full layer, in layer order][end - begin][kv_n] bf16 (runtime::Engine's / K2's order)
// A run is one contiguous range of one device allocation (`tensor`: the device's full_k, full_v,
// ring_k, ring_v), in host order. `zero` runs cover positions before 0: save_state writes zeros to the
// host for them (the device slots are not read), load_state copies the host's zeros into their slots.
inline uint32_t state_positions(const model::Kolibri1Desc& d) { return d.window - 1; }   // 512
enum class SnapTensor : uint32_t { FullK, FullV, RingK, RingV };
struct SnapRun {
  uint32_t device = 0;
  SnapTensor tensor = SnapTensor::FullK;
  size_t offset = 0, bytes = 0;   // within the device allocation
  bool zero = false;              // positions before 0 (state runs only)
};
// 40 x 512 x 2 x 1024 B = 41,943,040 at the real shapes.
inline size_t state_snapshot_bytes(const model::Kolibri1Desc& d) {
  return size_t(2) * (d.layers - d.full_before(d.layers)) * state_positions(d) * d.kv_n() * 2;
}
// 20,480 B a position at the real shapes (= full_kv_bytes_per_pos).
inline size_t kv_snapshot_bytes(const model::Kolibri1Desc& d, uint32_t n_pos) {
  return full_kv_bytes_per_pos(d) * n_pos;
}
// The ring rows of positions [pos - state_positions, pos), every sliding layer, K then V. Throws when
// the ring cannot hold them (kRing < state_positions).
std::vector<SnapRun> state_runs(const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t pos);
// Positions [begin, end) of every full layer, K then V. Throws unless begin <= end <= max_len.
std::vector<SnapRun> kv_runs(const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t max_len,
                             uint32_t begin, uint32_t end);

}  // namespace runtime::kolibri
