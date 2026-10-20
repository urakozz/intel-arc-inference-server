#pragma once
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/lfm2moe_loader.h"
#include "runtime/capture.h"   // CapturedStep, StageLink, ProfileEvents - the same records
#include "runtime/lfm2moe/lfm2moe_buffers.h"

// Spec 24c: one device's LFM2.5-24B-A2B decode step, captured once and replayed per token - Kolibri's arrangement
// (runtime/kolibri/kolibri_capture.h: a straight walk, every per-token value read from runtime::Control at execution,
// the expert ids produced and consumed on the device) over the device's layers [first, end) (spec 24 §4.5):
//
//   device 0           lfm_embed_gather (cur_token's row of the embedding tiles -> the residual h)
//   dense conv, 8      res_norm · in_proj GEMV · lfm_conv · out_proj GEMV · res_norm · gate||up GEMV · lfm_silu_mul ·
//                      down GEMV (its slices left for the next norm to fold)
//   conv + MoE, 9      res_norm · in_proj GEMV · lfm_conv · out_proj GEMV · res_norm · lfm_router · lfm_route ·
//                      lfm_moe_gate_up · lfm_moe_down (folds y into h)
//   attention + MoE, 11  res_norm · q||k||v GEMV · lfm_attn_prep · lfm_attn decode + reduce (eager: score, softmax,
//                      P·V, reduce) · o GEMV · res_norm · lfm_router · lfm_route · lfm_moe_gate_up · lfm_moe_down
//   last device        res_norm (embedding_norm) · the head (gemv_i8w | gemv_bf16 over the embedding tiles) · the two
//                      argmax stages
//
// **The norm sites and the residual's ping-pong** (plan 24c Review Focus 3): a norm site folds whatever split-K
// slices are pending (out_proj's / o_proj's at every ffn_norm; a dense layer's down at the next operator norm - layers
// 1 and 2, and the final norm or the cut when such a layer is last) - `_S<s>`, reading h[cur] and writing h[1 - cur]
// (then cur flips) - or, with nothing pending (after the gather, after lfm_moe_down folded), only normalises (`_NF`,
// h stays). The walk asserts that every fold writes the buffer the site did not read, and every reader reads the one
// the last writer wrote.
// **The fused arms** (`fuse`, spec 24 §4.5, decision 6): `route` binds lfm_moe_M1_..._R (the route recomputed in
// every gate||up work-group) for lfm_route + lfm_moe_gate_up; `router` binds lfm_res_norm_..._R at a MoE layer's
// ffn_norm for that norm + lfm_router; `conv` binds lfm_conv_in (in_proj's GEMV on the conv-tile layout - the model
// must be loaded with conv_tiles - with the conv as its epilogue) for in_proj + lfm_conv. Each arm is the same
// arithmetic in another launch: bitwise the as-built list (lfm2moe_decode_test --fuse).
// **Two cards** (`link` non-null; spec 16b's cut): device 0's list ends after layer s-1 and hands off h[cur] - plus,
// when layer s-1 is dense, its down GEMV's pending slices (the receiving norm folds them) - by `copy` (device-to-device
// copies into the landing buffer and a barrier signalling the cross-device event) or `peer` (pp_send). Device 1's
// list starts with the wait (copy) or pp_recv (peer) into its own h0 (and slices), then layer s. The compute launches
// of the two lists add up to the one-card count (runtime::lfm2moe::device_launches, asserted per device).
//
// conv_tiles: the loaded model's in_proj layout (loader::LfmLoaded::conv_tiles) - true exactly under the arm conv.
// tap (debug): layer l's whole output copied to tap row l - h[cur] after its last launch, or, for a dense layer (its
// down slices still pending), after the next norm's fold (at a dense two-card cut: on device 1). prof (--profile):
// launch i signals prof->events[i] (runtime/capture.h).
namespace runtime::lfm2moe {

// The hand-off's landing layout for the cut at `p.split` (devices 2): h (kM x hidden bf16) and - when layer split - 1
// is dense - its down GEMV's slices (S x kM x hidden fp32) in the "sums" region. pp_landing_layout(4096, 0) at a
// MoE cut (h only).
PpLandingLayout landing_layout(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p);
// The slices pending at the cut (the dense down's S; 0 at a MoE cut).
uint32_t cut_pending_s(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p);

CapturedStep build(l0::Context& ctx, const loader::LfmDevicePart& part, const model::Lfm2MoeDesc& d,
                   Lfm2MoeBuffers& b, LfmFuse fuse, bool conv_tiles, const runtime::StageLink* link,
                   l0::Mem* tap = nullptr, ProfileEvents* prof = nullptr);

}  // namespace runtime::lfm2moe
