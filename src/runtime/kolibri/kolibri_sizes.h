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

// --- the split by bytes (16b's rule) -------------------------------------------------------------
// layer_bytes[l] = kol_layer_bytes(d).total() + (is_sliding(l) ? ring_bytes_per_layer : max_len x 2 x kv_n x 2)
std::vector<size_t> pp_layer_bytes(const model::Kolibri1Desc& d, uint32_t max_len);
// dev0_fixed = embedding + RoPE + decode scratch + control + link (device 0); dev1_fixed = final norm +
// lm_head + RoPE + decode scratch + control + link (device 1) - then runtime::pp_balance.
PpBalance pp_split(const model::Kolibri1Desc& d, uint32_t max_len, bool int8_head, PpHandoff h,
                   KolAttn a = kolibri_attn());

// --- the planner -----------------------------------------------------------------------------------
// One device's plan in memory_line()'s components: model = its weights + its RoPE table; kv = its
// full KV + its rings; decode_state = control + scratch (+ eager's score row) (+ the tap) + its link.
struct DevicePlan : MemoryComponents {
  uint32_t device = 0;
  size_t weights = 0, rope = 0, link = 0, full_kv = 0, rings = 0;
};
std::vector<DevicePlan> plan(const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t max_len,
                             bool int8_head, bool debug_tap = false, KolAttn a = kolibri_attn());
// The largest multiple of kMaxLenQuantum, at most min(cap, trained 262144), whose plan + reserve fits
// EVERY device; 0 when not even min(kMinAutoMaxLen, cap) does.
uint32_t max_len_that_fits(const model::Kolibri1Desc& d, const model::KolPlacement& p, bool int8_head,
                           const std::array<size_t, kPpDevices>& device_bytes, size_t reserve, uint32_t cap = 0);
// `--pipeline-split auto` with `--max-len auto`: pp_auto_split_and_len's rule over pp_split - the split
// whose min-over-devices max_len is the largest, ties broken by pp_balance at that length.
PpChoice pp_split_and_len(const model::Kolibri1Desc& d, bool int8_head,
                          const std::array<size_t, kPpDevices>& device_bytes, size_t reserve, uint32_t cap = 0);
// Whether `--pp 1` may run: the one-card plan at min(kMinAutoMaxLen, trained) fits device_bytes - reserve.
bool fits_one_card(const model::Kolibri1Desc& d, bool int8_head, size_t device_bytes, size_t reserve);
// "pipeline plan at max_len N, split s (layers [0, s) | [s, L)):" + one line per device
// (runtime::pp_describe's text; "plan at max_len N (one card):" for one device).
std::string describe(const std::vector<DevicePlan>& p, const model::KolPlacement& pl, uint32_t max_len,
                     const std::array<size_t, kPpDevices>& device_bytes, size_t reserve);

}  // namespace runtime::kolibri
