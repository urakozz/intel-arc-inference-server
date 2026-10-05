#pragma once
#include <cstdint>
#include <memory>

#include "l0/context.h"
#include "l0/memory.h"
#include "model/k2_horizon.h"
#include "runtime/k2/k2_sizes.h"

namespace l0 {
class CmdList;
}

// Spec 18b: K2-Horizon's decode allocations (runtime/k2/k2_sizes.h sizes every one; the
// constructor allocates exactly those). The K2 counterpart of runtime::DecodeBuffers - no
// GDN state, no conv ring, no MTP, no prefill scratch; K2's KV is 48 full-attention layers.
namespace runtime::k2 {

struct K2Buffers {
  // `attn`: the decode attention the capture will bind; by default the one B70_K2_ATTN
  // selects, which is what capture reads, so an engine built in one process agrees.
  K2Buffers(l0::Context& ctx, const model::K2Desc& d, uint32_t max_len, K2Attn attn = k2_attn());

  // --- persistent: zeroed by zero() (K2Engine::reset) ---------------------------------
  l0::Mem control;    // runtime::Control, shared memory
  l0::Mem kv_k, kv_v; // bf16 [layers][max_len][kv_heads][head_dim] each
  // --- decode scratch: never zeroed (no step reads scratch it has not written first) ---
  l0::Mem resid, x, partials, norm_sumsq, attn_q, attn_gate, attn_part, attn_out, router,
      routes, moe_h, logits, argmax_part;
  // B70_K2_ATTN=eager only (null under flash): the score row, attn_scores_bytes().
  std::unique_ptr<l0::Mem> attn_scores;

  const model::K2Desc& desc;
  uint32_t max_len;

  void* kv_k_layer(uint32_t layer) const;
  void* kv_v_layer(uint32_t layer) const;
  size_t persistent_bytes() const { return control.size() + kv_k.size() + kv_v.size(); }
  size_t scratch_bytes() const;
  // control (then n_active = 1), kv_k, kv_v - in construction order.
  void zero(l0::CmdList& imm);
};

}  // namespace runtime::k2
