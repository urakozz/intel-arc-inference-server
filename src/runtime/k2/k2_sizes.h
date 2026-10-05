#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

#include "model/k2_horizon.h"
#include "runtime/memory_plan.h"

// Spec 18b: K2-Horizon's runtime allocations as device-free arithmetic, and the memory
// planner over them (`--max-len auto`, spec 6 §10) - runtime/buffer_sizes.h's and
// runtime/memory_plan.h's arrangement for K2: the K2Buffers constructor allocates exactly
// what these return, and the plan adds the same numbers up before anything is allocated.
namespace runtime::k2 {

// The decode list's rows in flight. K2 builds M = 1 only (no MTP head, spec 18 §2; batching
// is spec 13's), so the scratch is sized for one row and every K2 binary is `_M1`.
inline constexpr uint32_t kM = 1;

// The persistent group: what survives a token boundary and Engine reset() zeroes.
//   control   runtime::Control, shared memory (the host writes cur_token, reads out_token)
//   kv_k/kv_v bf16 [layers][max_len][kv_heads][head_dim] each: 192 KiB per position for K and
//             V together on K2 (48 x 8 x 128 x 2 B x 2), 6.44 GB at 32k (spec 18 §1)
struct PersistentSizes {
  size_t control = 0, kv_k = 0, kv_v = 0;
  size_t kv() const { return kv_k + kv_v; }
  size_t total() const { return control + kv_k + kv_v; }
};
PersistentSizes persistent_sizes(const model::K2Desc& d, uint32_t max_len);
// One layer's K (or V) rows: max_len x kv_heads x head_dim bf16.
inline size_t kv_layer_bytes(const model::K2Desc& d, uint32_t max_len) {
  return size_t(max_len) * d.kv_n() * 2;
}

// The decode scratch, sized for kM rows:
//   resid       bf16 [M][hidden]                        the residual stream
//   x           bf16 [M][max(hidden, dense_inter)]      norm outputs, SiLU x up
//   partials    fp32 [max S x N over the int4 linears][M]   every int4 GEMV's split-K output
//   norm_sumsq  fp32 [kNormG][M]                         prep_res_fold -> k2_norm_finish
//   attn_q, attn_gate  fp32 [M][q_n]
//   attn_part   fp32 [q_heads][TGT][M][130]              k2_attn_decode -> k2_attn_reduce
//   attn_out    bf16 [M][q_n]                            o_proj's input
//   router      fp32 [M][router_n]                       the MoE router GEMV's output
//   routes      u32  [layers][2][M][32]                  every layer's MoVA (0) and MoE (1)
//                                                        route rows - kept per layer so a step
//                                                        leaves the whole routing readable
//                                                        (the routing diagnostic, K2)
//   moe_h       bf16 [M][top_k + 1][moe_inter]           SiLU(gate) x up of every slot
//   logits      fp32 [M][vocab]
//   argmax_part fp32 [M][ceil(vocab / 1024)][2]
struct ScratchSizes {
  size_t resid = 0, x = 0, partials = 0, norm_sumsq = 0, attn_q = 0, attn_gate = 0,
         attn_part = 0, attn_out = 0, router = 0, routes = 0, moe_h = 0, logits = 0,
         argmax_part = 0;
  size_t total() const {
    return resid + x + partials + norm_sumsq + attn_q + attn_gate + attn_part + attn_out + router +
           routes + moe_h + logits + argmax_part;
  }
};
ScratchSizes scratch_sizes(const model::K2Desc& d);
// The largest S x N over the int4 linears: the partials row the capture binds every GEMV to.
size_t partials_floats(const model::K2Desc& d);

// The per-layer route rows inside `routes` (bytes).
inline constexpr uint32_t kRouteWords = 32;   // = kernels::k2::route::kWords (k2_capture.cc)
inline size_t mova_route_at(uint32_t layer) { return (size_t(layer) * 2 + 0) * kM * kRouteWords * 4; }
inline size_t moe_route_at(uint32_t layer) { return (size_t(layer) * 2 + 1) * kM * kRouteWords * 4; }

// The golden gate's per-layer tap: bf16 [layers][M][hidden].
inline size_t tap_bytes(const model::K2Desc& d) { return size_t(d.layers) * kM * d.hidden * 2; }

// The decode list's launch count (Review Focus 5; k2_capture.cc asserts its walk against it):
//   1 embed_gather
//   dense layer  12: fold + norm, q||k||gate||v GEMV, attn prep, decode + reduce, o_proj GEMV,
//                    fold + norm, gate||up GEMV, SiLU x up, down GEMV
//   sparse layer 15: fold + norm, q||k||gate||v_router GEMV, MoVA route, MoVA value experts,
//                    attn prep, decode + reduce, o_proj GEMV, fold + norm, router GEMV, MoE
//                    route, gate||up (8 + shared), down + combine + residual
//   5 at the boundary: the final fold + norm, lm_head, the two argmax stages
// K2: 1 + 3 x 12 + 45 x 15 + 5 = 717 - against spec 18 §2's ~1000 estimate and spec 4's
// 2067 (fixed slot launches).
size_t decode_launches(const model::K2Desc& d);

// --- the planner ------------------------------------------------------------------------
// `model_bytes`: what load_k2 allocated except the RoPE table (K2LoadReport: bytes.total(),
// = loader::k2_weight_bytes). The components are runtime::MemoryComponents' (memory_line's
// format): model (+ the RoPE table at max_len), kv, decode state (control + scratch); a K2
// engine has no prefill scratch (spec 18c) and no int8 prefill state.
struct Plan : MemoryComponents {
  uint32_t max_len = 0;
  size_t rope = 0;
};
Plan plan(const model::K2Desc& d, uint32_t max_len, size_t model_bytes, bool debug_tap = false);
// The largest multiple of runtime::kMaxLenQuantum up to `cap` whose plan + reserve fits;
// 0 when not even min(kMinAutoMaxLen, cap) does. Throws std::invalid_argument below a quantum.
uint32_t max_len_that_fits(const model::K2Desc& d, size_t model_bytes, size_t device_bytes,
                           size_t reserve_bytes, uint32_t cap);
std::string describe(const Plan& p, size_t device_bytes, size_t reserve_bytes);

}  // namespace runtime::k2
