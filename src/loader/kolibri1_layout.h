#pragma once
#include <cstddef>
#include <cstdint>

#include "model/kolibri1.h"

// Spec 20c: Kolibri-1 on the card as device-free arithmetic - the ONE home of every weight
// allocation's size, header-only so the planner (runtime/kolibri/kolibri_sizes.h, no loader
// library) and the loader (loader/kolibri1_loader.cc allocates exactly these) share it.
//
// Per layer (model::Kolibri1Desc's shapes; derived, the loader asserts them):
//
//   qkv, oproj   int4 arm: GPTQ layout 0 (words [K/8][N] + f16 scales [K/64][N]) -
//                  q||k||v 2560 x 7168 (9,175,040 + 573,440 B), o_proj 6144 x 2560
//                  (7,864,320 + 491,520 B): 18,104,320 B
//                bf16 arm: gemv_bf16's tiles, 36,700,160 + 31,457,280 = 68,157,440 B
//   router       bf16 tiled {hidden, router_n 512}, rows 384..511 zero: 2,621,440 B
//   bias         fp32 [router_n]: expert_bias widened, 0 past the experts: 2,048 B
//   norms        fp32 [4 x hidden + 2 x head_dim] (Kolibri1Desc::norm_off_*): 41,984 B
//   gate_up      int4 layout-1 blocks [experts][hidden x 2I] (gate||up interleave16):
//                  384 x 1,392,640 = 534,773,760 B
//   down         int4 layout-1 blocks [experts][I x hidden]: 384 x 696,320 = 267,386,880 B
//   shared       bf16 tiled {hidden, 2I} (gate||up interleave16) 5,242,880 B and
//                  {I, hidden} 2,621,440 B: 7,864,320 B
//   total        830,794,752 B (int4 attention), 880,847,872 B (bf16 attention)
//
// Top level: embed bf16 [vocab][hidden] 655,360,000 B (device 0, gathered); lm_head bf16 tiled
// 655,360,000 B or int8 + fp32 row scales 328,192,000 B, and the final norm fp32 [hidden]
// (the last device); the RoPE table fp32 [max_len][2][64] on every device holding a sliding
// layer (Kolibri1Desc::rope_table_bytes - the one term that scales with max_len).
// The real model, int4 attention, int8 head, split 25: device 0 21,425,228,800 B, device 1
// 21,098,071,040 B (derived; kolibri1_repack_test pins it).
namespace loader {

inline constexpr uint32_t kKolTileU32 = 136;   // layout 1: 128 nibble + 8 scale u32 per tile

inline constexpr size_t kol_layout1_bytes(uint32_t K, uint32_t N) {
  return size_t(N / 16) * (K / 64) * kKolTileU32 * 4;
}
inline constexpr size_t kol_layout0_bytes(uint32_t K, uint32_t N) {
  return size_t(K / 8) * N * 4 + size_t(K / 64) * N * 2;
}

struct KolLayerBytes {
  size_t qkv = 0, oproj = 0, router = 0, bias = 0, norms = 0, gate_up = 0, down = 0, shared_gate_up = 0,
         shared_down = 0;
  size_t total() const { return qkv + oproj + router + bias + norms + gate_up + down + shared_gate_up + shared_down; }
  size_t experts() const { return gate_up + down; }
  size_t attention() const { return qkv + oproj; }
};

inline size_t kol_linear_bytes(const model::Kolibri1Desc& d, model::KolLinearId id) {
  const model::GemvShape s = d.linear(id).shape;
  return d.attn == model::KolAttnForm::Int4 ? kol_layout0_bytes(s.K, s.N) : size_t(s.K) * s.N * 2;
}

inline KolLayerBytes kol_layer_bytes(const model::Kolibri1Desc& d) {
  KolLayerBytes b;
  b.qkv = kol_linear_bytes(d, model::KolLinearId::Qkv);
  b.oproj = kol_linear_bytes(d, model::KolLinearId::OProj);
  b.router = size_t(d.router_n()) * d.hidden * 2;
  b.bias = size_t(d.router_n()) * 4;
  b.norms = size_t(d.norm_floats()) * 4;
  b.gate_up = kol_layout1_bytes(d.hidden, 2 * d.moe_inter) * d.experts;
  b.down = kol_layout1_bytes(d.moe_inter, d.hidden) * d.experts;
  b.shared_gate_up = size_t(d.hidden) * 2 * d.shared_inter * 2;
  b.shared_down = size_t(d.shared_inter) * d.hidden * 2;
  return b;
}
// One expert's block in `gate_up` / `down` (block e at e x this).
inline size_t kol_gate_up_block_bytes(const model::Kolibri1Desc& d) {
  return kol_layout1_bytes(d.hidden, 2 * d.moe_inter);
}
inline size_t kol_down_block_bytes(const model::Kolibri1Desc& d) { return kol_layout1_bytes(d.moe_inter, d.hidden); }

inline size_t kol_embed_bytes(const model::Kolibri1Desc& d) { return size_t(d.vocab) * d.hidden * 2; }
inline size_t kol_lm_head_bytes(const model::Kolibri1Desc& d, bool int8) {
  return int8 ? size_t(d.vocab) * d.hidden + size_t(d.vocab) * 4 : size_t(d.vocab) * d.hidden * 2;
}
inline size_t kol_final_norm_bytes(const model::Kolibri1Desc& d) { return size_t(d.hidden) * 4; }

// A device's weights under a placement (the RoPE table excluded): its layers, the embedding on
// device 0, the final norm and lm_head on the last device.
inline size_t kol_device_weight_bytes(const model::Kolibri1Desc& d, const model::KolPlacement& p,
                                      uint32_t dev, bool int8_head) {
  size_t b = size_t(p.count(dev)) * kol_layer_bytes(d).total();
  if (dev == 0) b += kol_embed_bytes(d);
  if (dev + 1 == p.devices) b += kol_final_norm_bytes(d) + kol_lm_head_bytes(d, int8_head);
  return b;
}
// Whether device `dev` holds a sliding layer (and so a RoPE table).
inline bool kol_device_has_sliding(const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t dev) {
  for (uint32_t l = p.first(dev); l < p.end(dev); ++l)
    if (d.is_sliding(l)) return true;
  return false;
}

// What ONE decode token reads of a layer (derived): attention, router, bias, norms, top_k
// expert blocks and the shared expert.
inline size_t kol_layer_read_per_token(const model::Kolibri1Desc& d) {
  const KolLayerBytes b = kol_layer_bytes(d);
  return b.qkv + b.oproj + b.router + b.bias + b.norms +
         size_t(d.top_k) * (kol_gate_up_block_bytes(d) + kol_down_block_bytes(d)) + b.shared_gate_up +
         b.shared_down;
}

}  // namespace loader
