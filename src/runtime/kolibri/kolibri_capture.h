#pragma once
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/kolibri1_loader.h"
#include "runtime/capture.h"   // CapturedStep, StageLink - the same records
#include "runtime/kolibri/kolibri_buffers.h"

// Spec 20c: one device's Kolibri-1 decode step, captured once and replayed per token - K2's
// arrangement (runtime/k2/k2_capture.h: a straight walk, every per-token value read from
// runtime::Control at execution, the expert ids produced and consumed on the device) over the
// device's layers [first, end):
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
// reads - and hands those off: `copy` two device-to-device copies into the landing buffer and a
// barrier signalling the cross-device event; `peer` pp_send. Device 1's list starts with the wait
// (copy) or pp_recv (peer), the copies into its own `resid` / `sumsq_r`, then layer s. The compute
// launches of the two lists add up to the one-card count.
//
// tap (debug): `resid` copied to tap row l after each layer's last launch (layer l's output).
namespace runtime::kolibri {

CapturedStep build(l0::Context& ctx, const loader::KolDevicePart& part, const model::Kolibri1Desc& d,
                   KolibriBuffers& b, const runtime::StageLink* link, l0::Mem* tap = nullptr);

}  // namespace runtime::kolibri
