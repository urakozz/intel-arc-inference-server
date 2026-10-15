#pragma once
#include <cstdint>
#include <memory>

#include "l0/context.h"
#include "l0/memory.h"
#include "model/muse_glimmer.h"
#include "runtime/muse/muse_sizes.h"

namespace l0 {
class CmdList;
}

// Spec 23c: one device's Muse-Glimmer decode allocations (runtime/muse/muse_sizes.h sizes every one; the
// constructor allocates exactly those, so plan == allocation - muse_decode_test asserts it): the persistent group of
// the device's layers - its full layers' growing KV and its sliding layers' rings, each group allocated only when the
// device holds such a layer - and the whole decode scratch (with the control block and the RoPE table, uploaded here).
namespace runtime::muse {

struct MuseBuffers {
  // `attn`: the decode attention the capture will bind.
  MuseBuffers(l0::Context& ctx, const model::MuseGlimmerDesc& d, const model::MusePlacement& p, uint32_t device,
              uint32_t max_len, MuseAttn attn);

  // --- persistent: zeroed by zero() (MuseEngine::reset) ----------------------------------------------------
  l0::Mem control;                           // runtime::Control, shared memory
  std::unique_ptr<l0::Mem> full_k, full_v;   // bf16 [its full layers][max_len][kv_heads][128]; null: none here
  std::unique_ptr<l0::Mem> ring_k, ring_v;   // bf16 [its sliding layers][4096][kv_heads][128]; null: none here
  // --- decode scratch: never zeroed (no step reads scratch it has not written first) --------------------------
  l0::Mem resid, x, a, partials, sumsq_a, sumsq_r, attn_q, gate, attn_part, attn_out, h, logits, argmax_part;
  l0::Mem rope;                              // fp32 [max_len][2][64]: runtime::muse::rope_table, uploaded
  std::unique_ptr<l0::Mem> attn_s;           // B70_MUSE_ATTN=eager only: fp32 [M][q_heads][max_len]

  const model::MuseGlimmerDesc& desc;
  model::MusePlacement placement;
  uint32_t device, max_len;
  MuseAttn attn;

  // Layer `layer`'s K / V rows on this device (it must live here): a full layer's [max_len][kv_heads][128] or a
  // sliding layer's ring [4096][kv_heads][128].
  void* k_layer(uint32_t layer) const;
  void* v_layer(uint32_t layer) const;
  bool holds(uint32_t layer) const { return layer >= placement.first(device) && layer < placement.end(device); }
  size_t kv_bytes() const;        // the full layers' K + V (= device_kv_bytes)
  size_t ring_bytes() const;      // the rings' K + V (= device_ring_bytes)
  size_t scratch_bytes() const;   // control + scratch + rope (+ eager's score row) (= decode_scratch_bytes)
  // control, the full KV and the rings - in construction order.
  void zero(l0::CmdList& imm);
};

}  // namespace runtime::muse
