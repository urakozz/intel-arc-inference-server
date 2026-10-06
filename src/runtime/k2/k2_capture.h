#pragma once
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/k2_loader.h"
#include "runtime/capture.h"   // CapturedStep - the same record (list, modules, kernels, labels)
#include "runtime/k2/k2_buffers.h"

// Spec 18b: K2-Horizon's decode step, captured once and replayed per token - runtime::build's
// model (capture.cc: a straight walk, every per-token value read from runtime::Control at
// execution, the expert ids produced and consumed on the device) over K2's layers:
//
//   embed_gather
//   dense layer   prep_res_fold + k2_norm_finish (input), gemv q||k||gate||v,
//   (0-2)         k2_attn_prep (k, v -> cache; q, gate), k2_attn_decode + k2_attn_reduce,
//                 gemv o_proj, prep_res_fold + k2_norm_finish (post), gemv gate||up,
//                 k2_silu_mul, gemv down                                         12 launches
//   MoVA/MoE      prep_res_fold + k2_norm_finish (input), gemv q||k||gate||v_router,
//   layer (3-47)  k2_route (MoVA, from the fused row's v_router columns), k2_mova_value
//                 (v -> cache), k2_attn_prep (k -> cache; q, gate), k2_attn_decode +
//                 k2_attn_reduce, gemv o_proj, prep_res_fold + k2_norm_finish (post),
//                 gemv_bf16 router, k2_route (MoE), k2_moe_gate_up, k2_moe_down (folds the
//                 block into the residual itself)                                15 launches
//   boundary      prep_res_fold + k2_norm_finish (final), lm_head (gemv_bf16 / gemv_i8w),
//                 argmax_stage1, argmax_stage2                                   5 launches
//
// 717 launches on K2 (runtime::k2::decode_launches), asserted at capture (Review Focus 5).
// B70_K2_ATTN=eager (spec 18 §10.1) binds k2_attn_eager.cl's score / softmax / P·V / reduce in
// place of every layer's decode + reduce: 813.
//
// Spec 18e, buffers built with the int8 KV cache (K2Buffers::kv_lay): the same walk and the same
// launch counts with k2_kv8.cl's binaries in three places - MoVA's value experts write the routed
// mix to a staging row (attn_out[m][kv_n], k2_mova MOVA_STAGE), k2_attn_prep_kv8 rotates q and
// rotates + quantises K and V (V from the fused row or that staging row) into the int8 rows and
// fp16 scales, and the attention pair (or eager's score / P·V / reduce) reads int8 and un-rotates
// each head before the softplus gate.
//
// The residual stream: a dense layer's down leaves its partials for the next layer's fold
// (S_PREV = down's S); a MoE layer's k2_moe_down folds into `resid` itself, so the fold that
// follows it is SP0. tap (debug): `resid` copied after each layer's last launch - layer L's
// OUTPUT on a MoE layer; on a dense layer the MLP is still un-folded in `partials` (as on
// Qwen3.8, runtime/capture.h).
namespace runtime::k2 {

CapturedStep build(l0::Context& ctx, const loader::K2LoadedModel& m, K2Buffers& b,
                   l0::Mem* tap = nullptr);

}  // namespace runtime::k2
