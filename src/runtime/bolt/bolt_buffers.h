#pragma once
#include <cstdint>
#include <memory>

#include "l0/context.h"
#include "l0/memory.h"
#include "model/bolt.h"
#include "runtime/bolt/bolt_sizes.h"

namespace l0 {
class CmdList;
}

// Spec 25c: one device's Nemotron 3.5 Bolt decode allocations (runtime/bolt/bolt_sizes.h sizes every one; the constructor
// allocates exactly those): the persistent group of the device's blocks - the attention blocks' growing KV, the Mamba
// blocks' raw-xBC rings and SSM states - and the whole decode scratch. Every base is a 64-byte multiple (the KV rows 256 B,
// a ring 327,680 B, a state 4 MiB / 2 MiB - asserted by the capture, plan 25c's alignment rule).
namespace runtime::bolt {

struct BoltBuffers {
  // `attn`: the decode attention the capture binds (eager allocates the score row); `state`: the SSM state's precision.
  BoltBuffers(l0::Context& ctx, const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t device, uint32_t max_len,
              BoltAttn attn, BoltState state);

  // --- persistent: zeroed by zero() (BoltEngine::reset) -----------------------------------------------------------------
  l0::Mem control;                 // runtime::Control, shared memory
  l0::Mem kv_k, kv_v;              // bf16 [its attention blocks][max_len][2][128]
  l0::Mem conv_ring;               // bf16 [its Mamba blocks][16][10240] (raw xBC; positions < 0 read 0, never the ring)
  l0::Mem ssm;                     // fp32 / bf16 [its Mamba blocks][128][64][128]
  // --- decode scratch: never zeroed (no step reads scratch it has not written first) but `ones` (bf16 1.0, filled at
  // construction: Decision 15's gate) ---------------------------------------------------------------------------------
  l0::Mem h0, h1, x, slices, xc, y, g, logits_r, routes, lat, sh, moe_a, moe_r, attn_q, attn_part, attn_out, ones, logits,
      argmax_part;
  std::unique_ptr<l0::Mem> attn_scores;   // eager only: fp32 [M][32][max_len]

  const model::BoltDesc& desc;
  model::BoltPlacement placement;
  uint32_t device, max_len;
  BoltAttn attn;
  BoltState state;

  // Block `layer`'s K / V rows, raw-xBC ring, SSM state on this device (it must live here and be of that kind).
  void* k_layer(uint32_t layer) const;
  void* v_layer(uint32_t layer) const;
  void* ring_layer(uint32_t layer) const;
  void* ssm_layer(uint32_t layer) const;
  bool holds(uint32_t layer) const { return layer >= placement.first(device) && layer < placement.end(device); }
  size_t persistent_bytes() const { return control.size() + kv_k.size() + kv_v.size() + conv_ring.size() + ssm.size(); }
  size_t scratch_bytes() const;
  // Zeroes the control block (n_active 0 - ingest / generate set it before every replay), the KV, the rings and the
  // states, in construction order.
  void zero(l0::CmdList& imm);
};

// Spec 26c: one device's verify allocations (runtime/bolt/bolt_sizes.h verify_sizes sizes every one): the verify lists'
// scratch at kVerifyRows rows - the same names as BoltBuffers' decode scratch, so the capture binds either -, the SSM state
// slots 1..7 of each Mamba block it holds (slot 0 is BoltBuffers::ssm's own: nb_ssm's `_SL8` build reads the live slot and
// writes slot (live + m) % 8) and the union MoE's down rows. Allocated only with --spec dflash / lookup.
struct BoltVerifyBuffers {
  BoltVerifyBuffers(l0::Context& ctx, const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t device,
                    uint32_t max_len, BoltAttn attn, BoltState state);
  l0::Mem h0, h1, x, slices, xc, y, g, logits_r, routes, lat, sh, moe_a, moe_r, attn_q, attn_part, attn_out, ones, logits,
      argmax_part;
  std::unique_ptr<l0::Mem> attn_scores;   // eager only: fp32 [8][32][max_len]
  l0::Mem slots;                          // [its Mamba blocks][7][128][64][128] (fp32, or bf16 under the arm)
  l0::Mem moe_d;                          // bf16 [8][9][1024]: the union's per-pair down rows
  const model::BoltDesc& desc;
  model::BoltPlacement placement;
  uint32_t device, max_len;
  BoltAttn attn;
  BoltState state;
  static constexpr uint32_t rows = kVerifyRows;

  // Block `layer`'s slots 1..7 (nb_ssm's `state_spec`: slot s at + (s - 1) x one state); it must be a Mamba block here.
  void* slots_layer(uint32_t layer) const;
  size_t scratch_bytes() const;
  size_t bytes() const { return scratch_bytes() + slots.size() + moe_d.size(); }
};

}  // namespace runtime::bolt
