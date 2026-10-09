#pragma once
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/qwen4exp_loader.h"
#include "runtime/capture.h"   // CapturedStep, StageLink - the same records
#include "runtime/qwen4exp/qwen4exp_buffers.h"

// Spec 21c: one device's Qwen3.8-Flash-Next decode step, captured once and replayed per token - Kolibri's
// arrangement (runtime/kolibri/kolibri_capture.h: a straight walk, every per-token value read from
// runtime::Control at execution, the expert ids and the QSA selections produced and consumed on the device,
// the PLE rows read from host USM by an index the device computes) over the device's layers [first, end).
//
// **The pending combine.** The residual H (4 streams x 2560 bf16) is materialised only by the combines: a
// block's output y and its inject weights are folded into H by the NEXT q4_hc_combine_norm (vLLM's fusion), so
// the walk carries what is pending - the embedding (layer 0: _E), a mixer's GEMV slices (_S<S>), the MoE's y
// (_Y) or nothing (_X) - into the next combine; the PLE layer and device 0's end on two cards materialise it
// with a combine-only _Y_NN. One gated residual = combine_norm, the down||inject gemv_bf16, up_mix:
//
//   device 0     embed_gather (into x; layer 0's combine _E repeats it into the 4 streams)
//   GDN layer    hc(attn) [3] - qkv||z GEMV, a||b gemv_bf16, gdn_step, prep_gated_head _SIG, out_proj GEMV -
//                hc(mlp) [3] - router gemv_bf16, q4_route, q4_moe_gate_up, q4_moe_down                     15
//   QSA layer    hc(attn) [3] - q||gate||k||v GEMV, indexer gemv_bf16, attn_prep _Q24KV2, q4_qsa_prep,
//                q4_qsa_score, q4_qsa_select, q4_qsa_attn + q4_qsa_reduce (eager: q4_qsa_attn_eager), o_proj GEMV -
//                hc(mlp) [3] - the MoE's 4                                                                19 (18)
//   PLE layer    + combine_norm _Y_NN, q4_ple_gather, key||value gemv_bf16, q4_ple_block; its attn side's
//                combine is _X                                                                            + 4
//   last device  the final mixer (combine_norm _Y, gemv_bf16 {10240, 320}, up_mix), lm_head, argmax x 2       6
// 779 launches at 48 layers on one card (runtime::qwen4exp::decode_launches), asserted per device at capture.
//
// **Injected selections** (`injected` non-null; spec 21 F3's debug run): every QSA layer reads its list from the
// host-USM rows [qsa layers][M][kListRow] the caller writes before each replay, and q4_qsa_score / _select are not
// in the list (2 launches fewer a QSA layer).
//
// **Two cards** (`link` non-null; spec 16b's cut and hand-offs): device 0's list ends with a combine-only _Y_NN
// (the MATERIALISED H, as vLLM's PP carries it) and hands H off - `copy`: a device-to-device copy into the
// landing buffer and a barrier signalling the cross-device event; `peer`: pp_send with 0 norm-sum words. Device
// 1's list starts with the wait (copy) or pp_recv (peer), the copy into its own H, then its first layer with
// nothing pending (_X; the PLE layer's prologue skips its _Y_NN when it is device 1's first layer).
//
// tap (debug): H copied to tap row l right after the combine that materialises layer l's output (the next
// layer's first combine, the PLE prologue's, device 0's last, or the final mixer's) - the reference's H.L<l>.
namespace runtime::qwen4exp {

CapturedStep build(l0::Context& ctx, const loader::Q4LoadedModel& m, const loader::Q4DevicePart& part,
                   Qwen4ExpBuffers& b, const runtime::StageLink* link, l0::Mem* tap = nullptr,
                   const l0::Mem* injected = nullptr);

}  // namespace runtime::qwen4exp
