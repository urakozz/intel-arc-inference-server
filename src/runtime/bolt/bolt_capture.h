#pragma once
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/bolt_loader.h"
#include "runtime/bolt/bolt_buffers.h"
#include "runtime/capture.h"   // CapturedStep, StageLink, ProfileEvents - the same records

// Spec 25c: one device's Nemotron 3.5 Bolt decode step, captured once and replayed per token - Kolibri's
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

}  // namespace runtime::bolt
