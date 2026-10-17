#pragma once
#include <cstddef>
#include <cstdint>

#include "loader/int4_layout.h"
#include "loader/lm_head_int8.h"
#include "model/lfm2moe.h"

// Spec 24b Task 3: LFM2.5-24B-A2B on the card as device-free arithmetic - the ONE home of every
// weight allocation's size, header-only so the planner (runtime/lfm2moe/lfm2moe_sizes.h, no loader
// library) and the loader (src/loader/lfm2moe_loader.cc allocates exactly these) share it. Every
// int4 linear (per-layer and per-expert) is the shared subgroup-tiled layout 1 (loader/int4_layout.h,
// Review Focus 2: `repack_in_proj`'s false form is exactly `repack_int4_layout1_cols`).
//
// Every number below is derived (model::lfm2moe()'s widths) and cross-checked against spec 24 §3's
// table: routed experts 12,192,841,728 B (38 MoE layers x 320,864,256), the other int4 linears
// 399,932,928 B (conv 267,386,880 + attention 55,705,600 + dense 76,840,448), routers 9,961,472 B,
// the embedding (also the bf16 head) 524,288,000 B, the int8 head 262,656,000 B: weight_bytes(lfm2moe(),
// true) = 13,391,087,616 B (13.39 GB, spec §3's figure, small blocks included). KV 20,480 B a
// position (2.684 GB at 131072); the conv rings 1,966,080 B.
namespace loader::lfm {

inline constexpr size_t kTileBytes = 544;                 // layout 1: 128 u32 nibbles + 8 u32 f16 scales
inline constexpr uint32_t kConvRing = 16;

inline constexpr size_t int4_l1_bytes(uint32_t K, uint32_t N) { return int4_layout1_bytes(K, N); }

inline size_t expert_gate_up_bytes(const model::Lfm2MoeDesc& d) {
  return int4_l1_bytes(d.hidden, 2 * d.moe_inter);   // 3,342,336 at the real widths
}
inline size_t expert_down_bytes(const model::Lfm2MoeDesc& d) {
  return int4_l1_bytes(d.moe_inter, d.hidden);   // 1,671,168
}
inline size_t expert_gate_up_offset(const model::Lfm2MoeDesc& d, uint32_t e) {
  return size_t(e) * expert_gate_up_bytes(d);
}
inline size_t expert_down_offset(const model::Lfm2MoeDesc& d, uint32_t e) {
  return size_t(e) * expert_down_bytes(d);
}
inline size_t moe_layer_bytes(const model::Lfm2MoeDesc& d) {
  return size_t(d.experts) * (expert_gate_up_bytes(d) + expert_down_bytes(d));   // 320,864,256
}
// bf16 tiles {hidden, experts} (gemv_bf16's layout: experts (64) is already a multiple of 16).
inline size_t router_bytes(const model::Lfm2MoeDesc& d) { return size_t(d.hidden) * d.experts * 2; }

// The per-layer fp32 small block: operator_norm | ffn_norm always; conv_taps (conv layers) OR
// q_norm | k_norm (attention layers); expert_bias (MoE layers, i.e. !is_dense). Every offset is a
// multiple of 64 B (hidden, head_dim and experts are all multiples of 16, x 4 B), so the fields that
// apply to a given layer never overlap and the block itself needs no padding to stay 64-aligned.
struct SmallBlock {
  size_t operator_norm = 0, ffn_norm = 0, conv_taps = 0, q_norm = 0, k_norm = 0, expert_bias = 0, bytes = 0;
};
inline SmallBlock small_block(const model::Lfm2MoeDesc& d, uint32_t layer) {
  SmallBlock b;
  size_t off = 0;
  b.operator_norm = off;
  off += size_t(d.hidden) * 4;
  b.ffn_norm = off;
  off += size_t(d.hidden) * 4;
  if (d.is_attention(layer)) {
    b.q_norm = off;
    off += size_t(d.head_dim) * 4;
    b.k_norm = off;
    off += size_t(d.head_dim) * 4;
  } else {
    b.conv_taps = off;
    off += size_t(d.hidden) * d.conv_taps * 4;
  }
  if (!d.is_dense(layer)) {
    b.expert_bias = off;
    off += size_t(d.experts) * 4;
  }
  b.bytes = off;
  return b;
}

inline size_t embed_tiled_bytes(const model::Lfm2MoeDesc& d) { return size_t(d.hidden) * d.vocab * 2; }
inline size_t head_int8_bytes(const model::Lfm2MoeDesc& d) { return lm_head_int8_bytes(d.hidden, d.vocab); }

// Element (k, n) of the bf16 embedding's [vocab][hidden] row-major tensor, inside gemv_bf16's tiles
// (common::repack_bf16_tiled's layout: [n_tile][k_octet][8 k][16 n]) - the gather kernel's index.
inline size_t lfm_embed_row_index(uint32_t K, uint32_t k, uint32_t n) {
  const uint32_t K8 = K / 8;
  return ((size_t(n / 16) * K8 + k / 8) * 8 + k % 8) * 16 + n % 16;
}

// fp32 [max_len][2][head_dim / 2] (cos then sin, bf16-valued; loader::lfm_rope_table).
inline size_t rope_table_bytes(uint32_t max_len) { return size_t(max_len) * 2 * 32 * 4; }
// 10 attention layers x 8 kv heads x 64 x 2 (K, V) x 2 B (bf16) = 20,480 B a position.
inline size_t kv_bytes_per_position(const model::Lfm2MoeDesc& d) {
  return size_t(d.attn_layers()) * d.kv_heads * d.head_dim * 2 * 2;
}
// 30 conv layers x 16 ring slots x hidden x 2 B (bf16 Bx) = 1,966,080 B.
inline size_t conv_ring_bytes(const model::Lfm2MoeDesc& d) {
  return size_t(d.conv_layers()) * kConvRing * d.hidden * 2;
}

// One layer's int4 linears + (router + its expert block, if MoE) + its small block - what the
// two-card split (runtime::lfm2moe::pp_split) balances.
inline size_t layer_weight_bytes(const model::Lfm2MoeDesc& d, uint32_t layer) {
  size_t b = 0;
  if (d.is_attention(layer))
    b += int4_l1_bytes(d.hidden, d.qkv_n()) + int4_l1_bytes(d.q_n(), d.hidden);
  else
    b += int4_l1_bytes(d.hidden, 3 * d.hidden) + int4_l1_bytes(d.hidden, d.hidden);
  if (d.is_dense(layer))
    b += int4_l1_bytes(d.hidden, 2 * d.dense_inter) + int4_l1_bytes(d.dense_inter, d.hidden);
  else
    b += router_bytes(d) + moe_layer_bytes(d);
  b += small_block(d, layer).bytes;
  return b;
}

// Every layer's weights + the embedding (also the bf16 head) + the final norm + (the int8 head, when
// asked). 13,391,087,616 B at the real widths with the int8 head (spec §3's 13.39 GB).
inline size_t weight_bytes(const model::Lfm2MoeDesc& d, bool int8_head) {
  size_t b = embed_tiled_bytes(d) + size_t(d.hidden) * 4 /* embedding_norm */;
  if (int8_head) b += head_int8_bytes(d);
  for (uint32_t l = 0; l < d.layers; ++l) b += layer_weight_bytes(d, l);
  return b;
}

}  // namespace loader::lfm
