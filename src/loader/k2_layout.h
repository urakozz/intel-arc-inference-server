#pragma once
#include <cstddef>
#include <cstdint>

#include "model/k2_horizon.h"

// Spec 18b: K2-Horizon on the card, as device-free arithmetic - the ONE home of every
// weight allocation's size, header-only so the planner (runtime/k2/k2_plan.h, no loader
// library) and the loader (loader/k2_loader.cc allocates exactly these) share it.
//
// Per layer (model::K2Desc's shapes; K2's numbers derived from the checkpoint headers):
//
//   linears   int4 g64 layout 0 (GPTQ-native words [K/8][N] + f16 scales [K/64][N]):
//               dense:  q||k||gate||v 2560 x 10240, o_proj 4096 x 2560,
//                       gate||up 2560 x 12288 (interleave16), down 6144 x 2560
//               sparse: q||k||gate||v_router 2560 x 9280, o_proj
//   router    sparse: bf16 [router_n 128][2560] in gemv_bf16's tiling, rows 100..127 zero
//   experts   sparse, int4 g64 layout-1 blocks (common/repack.h), block e at e x stride:
//               value    64 blocks of 2560 x 1024                  1,392,640 B each
//               gate_up 101 blocks of 2560 x 1536 (interleave16)   2,088,960 B each
//               down    101 blocks of  768 x 2560                  1,044,480 B each
//             (the MoE groups' block 100 is the shared expert)
//   small     norms fp32 [2][2560] every layer; route fp32 [128 + 64] sparse layers
//
// Top level: embed bf16 [250624][2560] (gathered, row-major), lm_head bf16 tiled (or int8
// rows + fp32 scales, spec 9), the final norm fp32 [2560], the RoPE table fp32
// [max_len][2][64] (K2Desc::rope_table_bytes, the one term that scales with max_len).
//
// Derived totals at K2's shape (tests/model/k2_horizon_test.cc pins them):
//   per sparse layer experts 405,606,400 B (value 89,128,960 + gate_up 210,984,960 +
//   down 105,492,480); over 45 layers 18,252,288,000 B.
namespace loader {

inline constexpr uint32_t kK2TileU32 = 136;   // layout 1: 128 nibble + 8 scale u32 per tile

// Layout-1 bytes of a K x N int4 g64 linear: (N / 16) x (K / 64) tiles of 136 u32.
inline constexpr size_t k2_layout1_bytes(uint32_t K, uint32_t N) {
  return size_t(N / 16) * (K / 64) * kK2TileU32 * 4;
}
// Layout-0 bytes: words [K/8][N] u32, scales [K/64][N] f16.
inline constexpr size_t k2_layout0_words_bytes(uint32_t K, uint32_t N) { return size_t(K / 8) * N * 4; }
inline constexpr size_t k2_layout0_scales_bytes(uint32_t K, uint32_t N) { return size_t(K / 64) * N * 2; }

struct K2ExpertBytes {
  size_t value_block = 0, gate_up_block = 0, down_block = 0;
  uint32_t value_blocks = 0, moe_blocks = 0;
  size_t router = 0;   // bf16 [router_n][hidden]
  size_t value() const { return value_block * value_blocks; }
  size_t gate_up() const { return gate_up_block * moe_blocks; }
  size_t down() const { return down_block * moe_blocks; }
  size_t experts() const { return value() + gate_up() + down(); }
  // What ONE decode token reads of a sparse layer's routed weights (derived): the router,
  // top-k routed + the shared expert's gate||up and down, the top value experts.
  size_t per_token(const model::K2Desc& d) const {
    return router + size_t(d.top_k + 1) * (gate_up_block + down_block) +
           size_t(d.value_top_k) * value_block;
  }
};

inline K2ExpertBytes k2_expert_bytes(const model::K2Desc& d) {
  K2ExpertBytes b;
  b.value_block = k2_layout1_bytes(d.hidden, d.kv_n());
  b.gate_up_block = k2_layout1_bytes(d.hidden, 2 * d.moe_inter);
  b.down_block = k2_layout1_bytes(d.moe_inter, d.hidden);
  b.value_blocks = d.value_experts;
  b.moe_blocks = d.moe_blocks();
  b.router = size_t(d.router_n()) * d.hidden * 2;
  return b;
}

// The int4 bytes (words + scales) of one non-expert linear on the card.
inline size_t k2_linear_bytes(const model::K2Desc& d, model::K2LinearId id) {
  const model::GemvShape s = d.linear(id).shape;
  return k2_layout0_words_bytes(s.K, s.N) + k2_layout0_scales_bytes(s.K, s.N);
}

// lm_head's device bytes in its form: bf16 tiled, or int8 rows + fp32 row scales (spec 9).
inline size_t k2_lm_head_bytes(const model::K2Desc& d, bool int8) {
  return int8 ? size_t(d.vocab) * d.hidden + size_t(d.vocab) * 4 : size_t(d.vocab) * d.hidden * 2;
}

// Everything load_k2 allocates EXCEPT the RoPE table (which scales with max_len): the
// planner's `model_bytes` (runtime/k2/k2_plan.h) and the loader's own cross-check.
struct K2WeightBytes {
  size_t linears = 0;    // non-expert int4 words + scales
  size_t routers = 0;    // bf16 tiled, zero rows included
  size_t experts = 0;    // value + gate_up + down blocks
  size_t small = 0;      // norms + route blocks + final norm, fp32
  size_t embed = 0, lm_head = 0;
  size_t total() const { return linears + routers + experts + small + embed + lm_head; }
};
inline K2WeightBytes k2_weight_bytes(const model::K2Desc& d, bool int8_head) {
  K2WeightBytes w;
  const K2ExpertBytes e = k2_expert_bytes(d);
  for (uint32_t l = 0; l < d.layers; ++l) {
    for (model::K2LinearId id : d.layer_linears(l)) w.linears += k2_linear_bytes(d, id);
    w.small += d.norms_bytes();
    if (!d.is_dense(l)) {
      w.routers += e.router;
      w.experts += e.experts();
      w.small += d.route_bytes();
    }
  }
  w.small += size_t(d.hidden) * 4;   // the final norm
  w.embed = size_t(d.vocab) * d.hidden * 2;
  w.lm_head = k2_lm_head_bytes(d, int8_head);
  return w;
}

}  // namespace loader
