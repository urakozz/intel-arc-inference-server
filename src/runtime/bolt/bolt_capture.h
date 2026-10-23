#pragma once
#include <cstdint>
#include <vector>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/bolt_loader.h"
#include "runtime/bolt/bolt_buffers.h"
#include "runtime/capture.h"   // CapturedStep, StageLink, ProfileEvents - the same records

// Spec 25c: one device's Nemotron 3.5 Bolt decode step, captured once and replayed per token - Kolibri's / LFM2's
// arrangement (a straight walk, every per-token value read from runtime::Control at execution, the expert ids produced and
// consumed on the device) over the device's blocks [first, end) (spec 25 §6.1):
//
//   device 0         nb_embed_gather (cur_token's row of the embedding tiles -> the residual h)
//   Mamba, 6         nb_res_norm · in_proj GEMV · nb_conv · nb_ssm · nb_gnorm · out_proj GEMV
//   attention, 6     nb_res_norm · q||k||v GEMV · nb_attn_prep · mg_attn decode + reduce (eager: mg_attn_eager, 1) · o GEMV
//   MoE, 9           nb_res_norm · nb_router · nb_route · fc1||shared_up GEMV · nb_sh_act · nb_moe_up · nb_moe_down ·
//                    fc2 GEMV · shared_down GEMV
//   last device      nb_res_norm (norm_f) · the head (gemv_i8w | gemv_bf16 over lm_head's tiles) · the two argmax stages
//
// **No block folds its own output.** Every block leaves its last GEMV's split-K slices (a MoE block two sets: fc2's, then
// shared_down's right after them in `slices`) for the NEXT norm site, which folds them - `_S<s1>` one source, `_S<s1>_T<s2>`
// two (spec 25 §2.4's three roundings), `_NF` none (block 0 over the gathered embedding) - reading h[cur] and writing
// h[1 - cur] (then cur flips: the residual's ping-pong, plan 25c Review Focus 4). The walk asserts every fold writes the
// buffer the site did not read. Block l's residual output exists only after that fold, so the debug tap row l is written
// right after the NEXT site's norm (norm_f for the last block; device 1's first norm for the cut's block).
// **The fused arms** (`fuse`, spec 25 §6.1, Decision 16): `router` binds nb_res_norm ..._R at a MoE block's norm for that
// norm + nb_router; `route` binds nb_moe_up ..._R for nb_route + nb_moe_up; `conv` binds nb_ssm ..._C for nb_conv + nb_ssm.
// Each arm is the same arithmetic in another launch: bitwise the as-built list (bolt_decode_test --fuse).
// **Two cards** (`link` non-null; spec 16b's cut, spec 25 §4.1): device 0's list ends after block s-1 and hands off h[cur]
// AND block s-1's pending slices (always: no block folds its own) by `copy` (device-to-device copies into the landing
// buffer and a barrier signalling the cross-device event) or `peer` (pp_send). Device 1's list starts with the wait (copy)
// or pp_recv (peer) into its own h0 and slices, then block s, whose norm folds them. The compute launches of the two lists
// add up to the one-card count (runtime::bolt::device_launches, asserted per device).
// tap (debug): block l's output -> tap row l. prof (--profile): launch i signals prof->events[i] (runtime/capture.h).
namespace runtime::bolt {

CapturedStep build(l0::Context& ctx, const loader::BoltDevicePart& part, const model::BoltDesc& d, BoltBuffers& b,
                   BoltFuse fuse, bool int8_head, const runtime::StageLink* link, l0::Mem* tap = nullptr,
                   ProfileEvents* prof = nullptr);

// Spec 26c: the verify lists (spec 26 §5.3) - the same walk at M = 1..8 rows over the device's BoltVerifyBuffers scratch (the
// persistent group, KV / rings / slot 0 of the states, is BoltBuffers'), the anchor and up to 7 drafts at positions pos ..
// pos + M - 1 (n_active = M; the gather reads Control::cur_token[0..M-1] on device 0), with these changes:
//   every M-row launch runs exactly M real rows (one captured list per M, never a padded M - vllm-xpu-kernels 8b95ea6);
//   nb_ssm is the slot build `_SL8` (the state after row m into slot (Control::gdn_live + m) % 8; its last argument the
//   block's slots 1..7) - the commit is the host moving gdn_live on both cards;
//   the routed experts are the union-aware form at M > 1 (`union_moe`: nb_moe_up_U, nb_moe_down_U into the scratch's
//   moe_d, nb_moe_comb_U - Decision 10) or the plain M-row form; no fused arm (each is bitwise the as-built list);
//   the head at M rows writes [M][vocab] fp32 logits; argmax stage 2 (the M-free binary) writes out_token[0..M-1] - and
//   advances the last device's Control::pos / cur_token[0], which BoltEngine::verify puts back (the host commits);
//   with a drafter (`taps`), after the fold that produces block l's output (l in tap_layers) every row's residual is copied
//   into taps [row][tap][hidden] - device 0's three sites write device 1's buffer, in its list BEFORE the hand-off's
//   barrier signals (19d: the copy hand-off only; peer is refused by name);
//   two cards: the hand-off moves M rows of h and of the cut's pending slices through `link` (the verify link: a landing
//   sized for kVerifyRows).
// The launch count is runtime::bolt::verify_device_launches' (asserted).
struct VerifyBind {
  BoltVerifyBuffers* vb = nullptr;                    // the device's verify scratch, slots and union rows
  uint32_t M = 1;
  bool union_moe = true;                              // VerifyMoe::Union (used at M > 1)
  l0::Mem* taps = nullptr;                            // a drafter's taps() bf16 [8][taps][hidden] (the last device's), or null
  const std::vector<uint32_t>* tap_layers = nullptr;  // ascending, each < the target's blocks
};
CapturedStep build_verify(l0::Context& ctx, const loader::BoltDevicePart& part, const model::BoltDesc& d, BoltBuffers& b,
                          bool int8_head, const runtime::StageLink* link, const VerifyBind& v);

}  // namespace runtime::bolt
