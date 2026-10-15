#pragma once
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/muse_loader.h"
#include "runtime/capture.h"   // CapturedStep, StageLink - the same records
#include "runtime/muse/muse_buffers.h"

// Spec 23c: one device's Muse-Glimmer decode step, captured once and replayed per token - Kolibri's arrangement
// (runtime/kolibri/kolibri_capture.h: a straight walk, every per-token value read from runtime::Control at execution)
// with Muse's chains (spec 23 §5.2) over the device's layers [first, end):
//
//   device 0     mg_embed (the gather, the weight-less norm, layer 0's Σ²)
//   every layer  mg_norm (input_layernorm), q||gate||k||v GEMV (gemv int4 S2), mg_attn_prep (q / k norms, q x 3.87,
//                RoPE on sliding layers only, the gate's sigmoid; k, v -> the ring or the NoPE cache),
//                mg_attn_decode + mg_attn_reduce (the gate fused; eager: mg_attn_eager), o_proj GEMV, prep_res_fold _Z
//                (o_proj's row `a` and Σa²), mg_post_add (post_attention_layernorm, resid +=, Σ resid²), mg_norm
//                (pre_feedforward_layernorm), gate||up GEMV, mg_silu_mul, down GEMV, prep_res_fold _Z (down's row
//                and Σ²), mg_post_add (post_feedforward_layernorm)                      14 (eager 13)
//   last device  mg_final_norm (model.norm, plain w), lm_head (gemv_i8w | gemv_bf16), the two argmax stages
//
// 733 launches at 52 layers on one card (runtime::muse::decode_launches), 681 under B70_MUSE_ATTN=eager; asserted per
// device at capture. The row count M is a parameter of every name and grid (kM = 1 here; spec 23f's verify lists are
// this walk at M = 2..16).
//
// **Two cards** (`link` non-null; spec 16b's cut and hand-offs, Kolibri's): device 0's list ends after layer s-1's
// mg_post_add, which already wrote `resid` and `sumsq_r` - exactly what layer s's mg_norm reads - and hands those off:
// `copy` two device-to-device copies into the landing buffer and a barrier signalling the cross-device event; `peer`
// pp_send. Device 1's list starts with the wait (copy) or pp_recv (peer), the copies into its own `resid` /
// `sumsq_r`, then layer s. The compute launches of the two lists add up to the one-card count.
//
// tap (debug): `resid` copied to tap row l after each layer's last launch (layer l's output: the reference's H.L<l>).
namespace runtime::muse {

CapturedStep build(l0::Context& ctx, const loader::MuseDevicePart& part, const model::MuseGlimmerDesc& d, MuseBuffers& b,
                   const runtime::StageLink* link, l0::Mem* tap = nullptr);

}  // namespace runtime::muse
