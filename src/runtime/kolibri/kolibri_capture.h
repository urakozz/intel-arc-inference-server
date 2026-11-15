#pragma once
#include <cstdint>
#include <vector>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/kolibri1_loader.h"
#include "runtime/capture.h"   // CapturedStep, StageLink - the same records
#include "runtime/kolibri/kolibri_buffers.h"

// Spec 20c: one device's Kolibri-1 decode step, captured once and replayed per token - a
// straight walk (every per-token value read from runtime::Control at execution, the expert ids
// produced and consumed on the device) over the device's layers [first, end):
//
//   device 0     kol_embed_gather, prep_res_fold SP0 (Σ resid² for layer 0's norm)
//   every layer  kol_norm_finish (input_layernorm), q||k||v GEMV (gemv int4 S2 | gemv_bf16),
//                kol_attn_prep (q/k head norm, RoPE in sliding layers; k, v -> the ring or the cache),
//                kol_attn_decode + kol_attn_reduce (eager: score, softmax, P·V, reduce), o_proj GEMV,
//                prep_res_fold _Z (o_proj's row `a` and Σa²), kol_post_add (post_attn_norm, resid +=,
//                Σ resid²), kol_norm_finish (post_attention_layernorm), router gemv_bf16 (fp32 logits),
//                kol_route, kol_moe_gate_up, kol_moe_down (into `mo`), prep_res_fold SP0 (Σmo²),
//                kol_post_add (post_ffn_norm)                                       15 (eager 17)
//   last device  kol_norm_finish (model.norm), lm_head (gemv_bf16 | gemv_i8w), the two argmax stages
//
// 756 launches at 50 layers on one card (runtime::kolibri::decode_launches), 856 under
// B70_KOLIBRI_ATTN=eager; asserted per device at capture.
//
// **Two cards** (`link` non-null; spec 16b's cut and hand-offs): device 0's list ends after layer s-1's
// kol_post_add, which already wrote `resid` and `sumsq_r` - exactly what layer s's kol_norm_finish
// reads - and hands those off: two device-to-device copies into the landing buffer and a barrier
// signalling the cross-device event. Device 1's list starts with the wait, the copies into its own
// `resid` / `sumsq_r`, then layer s. The compute
// launches of the two lists add up to the one-card count.
//
// tap (debug): `resid` copied to tap row l after each layer's last launch (layer l's output), or after its
// attention half's kol_post_add (tap_at AfterAttn).
namespace runtime::kolibri {

CapturedStep build(l0::Context& ctx, const loader::KolDevicePart& part, const model::Kolibri1Desc& d,
                   KolibriBuffers& b, const runtime::StageLink* link, l0::Mem* tap = nullptr,
                   TapAt tap_at = TapAt::LayerOut);

// Kolibri DSpark: the verify lists - the same walk at M = 1..8 rows over the device's KolibriVerifyBuffers scratch (the
// persistent group, KV and rings, is KolibriBuffers'), the anchor and up to 7 drafts at positions pos .. pos + M - 1
// (n_active = M; the gather reads Control::cur_token[0..M-1] on device 0), with these changes:
//   every M-row launch runs exactly M real rows (one captured list per M); the attention is the flash form only, and at
//   M > 4 runs twice (kol_attn.cl's M <= 4 builds) - rows 0..3 over the Control block, rows 4..M-1 over control_hi (pos
//   + 4, M - 4 rows), each with its own attn_q / attn_out rows (verify_device_launches counts the second pair);
//   the head at M rows writes [M][vocab] fp32 logits; argmax stage 2 writes out_token[0..M-1] - and advances the last
//   device's Control::pos / cur_token[0], which KolibriEngine::verify puts back (the host commits);
//   with a drafter (`taps`), after layer l's last kol_post_add (l in tap_layers) every row's residual is copied into taps
//   [row][tap][hidden] - device 0's tap layers write the last device's buffer, in its list BEFORE the hand-off's barrier;
//   two cards: the hand-off moves M rows and their [kNormG][M] sums through `link` (the verify link: a landing for 8).
// The launch count is runtime::kolibri::verify_device_launches' (asserted).
struct VerifyBind {
  KolibriVerifyBuffers* vb = nullptr;                 // the device's verify scratch and control_hi
  uint32_t M = 1;
  l0::Mem* taps = nullptr;                            // a drafter's taps() bf16 [8][taps][hidden] (the last device's), or null
  const std::vector<uint32_t>* tap_layers = nullptr;  // ascending, each < the target's layers
};
CapturedStep build_verify(l0::Context& ctx, const loader::KolDevicePart& part, const model::Kolibri1Desc& d,
                          KolibriBuffers& b, const runtime::StageLink* link, const VerifyBind& v);

}  // namespace runtime::kolibri
