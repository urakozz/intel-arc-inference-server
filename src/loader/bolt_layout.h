#pragma once
#include <cstddef>
#include <cstdint>

#include "loader/int4_layout.h"
#include "loader/lm_head_int8.h"
#include "model/bolt.h"

// Spec 25b Task 4: Nemotron 3.5 Bolt on the card as device-free arithmetic - the ONE home of every weight allocation's
// size, header-only so the planner (runtime/bolt/bolt_sizes.h, no loader library) and the loader (src/loader/
// bolt_loader.cc allocates exactly these) share it. Each int4 linear's {S, layout} is PROVISIONAL and PER LINEAR
// (model::BoltDesc::linears: layout 1 for the routed experts and every N = 4096 linear, layout 0 for in_proj and the
// fused q||k||v / fc1||shared_up; 25c sweeps them); at N % 16 == 0 and K % 64 == 0 (true everywhere) both layouts are
// the same byte count (loader/int4_layout.h), so this choice changes no total. `layer_weight_bytes` reads the descriptor
// for every shape and layout - one source of truth, not a second hardcoded copy.
//
// Heterogeneous MoE (spec 25 §1.1): E and I per layer. One MoE layer's routed experts are ONE allocation of E blocks, a
// block = the expert's up (K latent, N I) then its down (K I, N latent), both layout 1:
//
//   block(I)          = up(I) + dn(I) = 2 x I x 544 B at latent 1024 (up: I/16 n-tiles x 16 k-groups x 544 B; down:
//                       1024/16 n-tiles x I/64 k-groups x 544 B - the same I x 544)
//   expert (layer, id) at  id x block(I) (up) and id x block(I) + up(I) (down), from the layer's OWN I
//
// Every block base is a multiple of 64 B (block = 1088 I, I a multiple of 64) and so is every small-block offset and every
// allocation (the LSC 2D-load rule, vllm-xpu-kernels 6f5935c) - tests/loader/bolt_repack_test.cc asserts it.
//
// Every number is derived (model::bolt()'s widths) and printed by tests/loader/bolt_repack_test.cc and
// tests/runtime/bolt_plan_test.cc; no test types a grand total.
namespace loader::bolt {

inline constexpr size_t kTileBytes = 544;   // layout 1: 128 u32 nibbles + 8 u32 f16 scales
inline constexpr uint32_t kConvRing = 16;   // the raw-xBC ring's slots per Mamba layer (spec 25 §1.2)
inline constexpr size_t kAlign = 64;        // every device allocation base and every offset inside one

inline constexpr size_t int4_l1_bytes(uint32_t K, uint32_t N) { return int4_layout1_bytes(K, N); }
// A linear's device bytes at its OWN layout (0: words + separate scales; 1: inline).
inline size_t int4_bytes(const model::GemvShape& s) {
  return s.layout == 0 ? int4_layout0_bytes(s.K, s.N) : int4_l1_bytes(s.K, s.N);
}

// --- the routed experts ------------------------------------------------------------------------------------------
inline constexpr size_t expert_up_bytes(uint32_t latent, uint32_t I) { return int4_l1_bytes(latent, I); }
inline constexpr size_t expert_dn_bytes(uint32_t latent, uint32_t I) { return int4_l1_bytes(I, latent); }
inline constexpr size_t expert_block_bytes(uint32_t latent, uint32_t I) {
  return expert_up_bytes(latent, I) + expert_dn_bytes(latent, I);
}
inline constexpr size_t expert_up_offset(uint32_t latent, uint32_t I, uint32_t id) {
  return size_t(id) * expert_block_bytes(latent, I);
}
inline constexpr size_t expert_dn_offset(uint32_t latent, uint32_t I, uint32_t id) {
  return expert_up_offset(latent, I, id) + expert_up_bytes(latent, I);
}
inline constexpr size_t layer_expert_bytes(uint32_t latent, uint32_t E, uint32_t I) {
  return size_t(E) * expert_block_bytes(latent, I);
}

// bf16 tiles {hidden, E} (gemv_bf16's layout: E is a multiple of 16 in every layer - no padding rows).
inline constexpr size_t router_bytes(uint32_t hidden, uint32_t E) { return size_t(hidden) * E * 2; }

// --- the per-block fp32 small block -------------------------------------------------------------------------------
// norm (the block's input norm, fp32 [hidden]) always; then per kind
//   Mamba   conv taps fp32 [conv_dim][conv taps] (the checkpoint's [c][1][j], j = 0 the OLDEST position, widened),
//           conv bias [conv_dim], A = -exp(A_log) [heads], D [heads], dt_bias [heads], gated-norm weight [d_inner]
//   Moe     e_score_correction_bias fp32 [E]
//   Attention  nothing
struct SmallBlock {
  size_t norm = 0, conv_w = 0, conv_b = 0, A = 0, D = 0, dt_bias = 0, gnorm = 0, e_bias = 0, bytes = 0;
};
inline SmallBlock small_block(const model::BoltDesc& d, uint32_t layer) {
  SmallBlock b;
  size_t off = 0;
  b.norm = off;
  off += size_t(d.hidden) * 4;
  if (d.is_mamba(layer)) {
    b.conv_w = off;
    off += size_t(d.m_conv_dim) * d.m_conv * 4;
    b.conv_b = off;
    off += size_t(d.m_conv_dim) * 4;
    b.A = off;
    off += size_t(d.m_heads) * 4;
    b.D = off;
    off += size_t(d.m_heads) * 4;
    b.dt_bias = off;
    off += size_t(d.m_heads) * 4;
    b.gnorm = off;
    off += size_t(d.m_inner) * 4;
  } else if (d.is_moe(layer)) {
    b.e_bias = off;
    off += size_t(d.experts.at(layer)) * 4;
  }
  b.bytes = off;
  return b;
}
inline size_t norm_bytes(const model::BoltDesc& d) { return size_t(d.hidden) * 4; }
// What a Mamba block's small block adds to its norm: taps, bias, A / D / dt_bias, the gated-norm weight.
inline size_t mamba_small_bytes(const model::BoltDesc& d) {
  return (size_t(d.m_conv_dim) * d.m_conv + d.m_conv_dim + 3 * size_t(d.m_heads) + d.m_inner) * 4;
}

// --- the embedding and the head ------------------------------------------------------------------------------------
inline size_t embed_tiled_bytes(const model::BoltDesc& d) { return size_t(d.hidden) * d.vocab * 2; }
inline size_t head_int8_bytes(const model::BoltDesc& d) { return lm_head_int8_bytes(d.hidden, d.vocab); }
// The untied bf16 head (`--lm-head bf16`): its own tiles, the same size as the embedding's.
inline size_t head_bf16_bytes(const model::BoltDesc& d) { return size_t(d.hidden) * d.vocab * 2; }
inline size_t final_norm_bytes(const model::BoltDesc& d) { return size_t(d.hidden) * 4; }

// Element (k, n) of a bf16 [N][K] row-major tensor inside gemv_bf16's tiles (common::repack_bf16_tiled's layout:
// [n_tile][k_octet][8 k][16 n]) - the gather kernel's index.
inline size_t embed_row_index(uint32_t K, uint32_t k, uint32_t n) {
  const uint32_t K8 = K / 8;
  return ((size_t(n / 16) * K8 + k / 8) * 8 + k % 8) * 16 + n % 16;
}

// --- state ---------------------------------------------------------------------------------------------------------
// 8 attention layers x 2 kv heads x 128 x {K, V} x 2 B (bf16) = 8,192 B a position at the real model.
inline size_t kv_bytes_per_position(const model::BoltDesc& d) {
  return size_t(d.attn_layers()) * d.kv_heads * d.head_dim * 2 * 2;
}
// One SSM state per Mamba layer: heads x head_dim x state fp32 (4 MiB), one slot.
inline size_t ssm_state_bytes(const model::BoltDesc& d, uint32_t layers) {
  return size_t(layers) * d.m_heads * d.m_head_dim * d.m_state * 4;
}
// The raw xBC ring: layers x 16 slots x conv_dim x 2 B (bf16).
inline size_t conv_ring_bytes(const model::BoltDesc& d, uint32_t layers) {
  return size_t(layers) * kConvRing * d.m_conv_dim * 2;
}

// One block's weights: its non-expert linears (at EACH ONE's own layout, model::BoltDesc::linears - the single source of
// shapes) + its router (bf16) + its routed experts + its small block - what the two-card split balances.
inline size_t layer_weight_bytes(const model::BoltDesc& d, uint32_t layer) {
  size_t b = 0;
  for (const model::BoltLinear& ln : d.linears(layer)) {
    switch (ln.id) {
      case model::BoltLinearId::ExpertUp:
      case model::BoltLinearId::ExpertDown:
        break;   // the expert allocation below
      case model::BoltLinearId::Router:
        b += router_bytes(ln.shape.K, ln.shape.N);
        break;
      default:
        b += int4_bytes(ln.shape);
    }
  }
  if (d.is_moe(layer)) b += layer_expert_bytes(d.latent, d.experts.at(layer), d.inter.at(layer));
  return b + small_block(d, layer).bytes;
}

// Every block's weights + the embedding + norm_f + the head (int8, or the bf16 tiles when asked).
inline size_t weight_bytes(const model::BoltDesc& d, bool int8_head) {
  size_t b = embed_tiled_bytes(d) + final_norm_bytes(d) + (int8_head ? head_int8_bytes(d) : head_bf16_bytes(d));
  for (uint32_t l = 0; l < d.layers; ++l) b += layer_weight_bytes(d, l);
  return b;
}

// What `load_bolt` uploads onto device `dev` under placement `p` (loader::load_bolt asserts its own total equals this -
// the ONE formula the loader and the planner share): its blocks' weights, the embedding on device 0, norm_f and the head
// on the last device.
inline size_t device_weight_bytes(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev,
                                  bool int8_head) {
  size_t b = 0;
  for (uint32_t l = p.first(dev); l < p.end(dev); ++l) b += layer_weight_bytes(d, l);
  if (dev == 0) b += embed_tiled_bytes(d);
  if (dev + 1 == p.devices) b += final_norm_bytes(d) + (int8_head ? head_int8_bytes(d) : head_bf16_bytes(d));
  return b;
}

}  // namespace loader::bolt
