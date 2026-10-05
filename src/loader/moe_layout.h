#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

#include "model/model_desc.h"

// Spec 15c: a mixture-of-experts layer on the card, as device-free arithmetic and
// names - the ONE home of both, header-only so runtime/memory_plan.h (b70_plan, no
// loader library) and the loader (loader/moe.cc allocates exactly these) share it.
//
// Per MoE layer, three allocations (loader::MoeLayer):
//
//   router   bf16 [router_n][hidden] in gemv_bf16's tiled layout
//            (common::repack_bf16_tiled): rows 0..E-1 the router `mlp.gate.weight`
//            [E][hidden], row E the shared expert's gate `mlp.shared_expert_gate.weight`
//            [1][hidden], zero rows to router_n (Ornith 256 + 1 -> 272). One GEMV gives
//            the 256 router logits and the shared gate's logit (spec 15 §4.1).
//   gate_up  int4 g64, `blocks` = E + 1 contiguous layout-1 blocks (common/repack.h:
//            per (n-tile of 16, k-group of 64) one 136-u32 tile, scales inline), block b
//            at byte b x gate_up_block: K = hidden, N = 2 x expert_intermediate, gate and
//            up interleaved in 16-column blocks exactly as the dense gate||up is
//            (common::cols_interleave16). Blocks 0..E-1 are the routed experts, block E
//            the shared expert (MoeDesc::shared_block), so a kernel reads expert e by
//            offset e x stride and the shared expert is just the last slot.
//   down     int4 g64, E + 1 layout-1 blocks of K = expert_intermediate, N = hidden.
//
// Ornith: router 1,114,112 B; gate_up 257 x 1,114,112 = 286,326,784 B; down 257 x
// 557,056 = 143,163,392 B; 430,604,288 B per layer, 17.224 GB over 40 layers (derived).
//
// **The checkpoint naming this loader reads (the assumption 15a confirms).** No int4
// Ornith exists yet (spec 15 decision 1). The proposed source is the operator's
// AutoRound W4A16 g64 sym export with GPTQ packing. AutoRound quantises nn.Linear
// modules, so it UNFUSES transformers 5's 3D `mlp.experts.gate_up_proj [E][2I][H]` /
// `mlp.experts.down_proj [E][H][I]` into one Qwen3_5MoeMLP per expert before
// quantising (auto_round/modeling/fused_moe/qwen3_5_moe.py: gate = rows [0, I) of
// gate_up_proj, up = rows [I, 2I); the transformers-5 linear_loop path names them the
// same) and exports per expert, GPTQ-packed:
//
//   model.language_model.layers.L.mlp.experts.E.{gate,up,down}_proj.{qweight,scales,qzeros}
//
// - the names vLLM's per-expert mapping loads (fused_moe/routed_experts.py,
// build_expert_params_mapping: `experts.{e}.gate_proj.` -> w1, `up_proj` -> w3,
// `down_proj` -> w2). vLLM also accepts a per-expert FUSED `experts.E.gate_up_proj`
// (gate the first half of its output columns); so does this loader. The 3D bf16
// tensors of the published checkpoint are NOT read: a bf16 checkpoint has no
// quantization_config and is refused before this (RTN at load, decision 1's option B,
// is not built). The router (a Qwen3_5MoeTopKRouter parameter, not an nn.Linear) and
// the [1][hidden] shared gate stay bf16 in an AutoRound export; an int4 one is refused
// by name. The shared expert's three linears are the descriptor's GateUp / Down rows'
// parts (`mlp.shared_expert.{gate,up,down}_proj`), int4 like the experts.
namespace loader {

inline constexpr uint32_t kInt4TileU32 = 136;   // common/repack.h layout 1: 128 nibble + 8 scale u32

// The checkpoint's per-layer names, layer-relative (the loader prefixes them).
inline constexpr const char* kMoeRouter = "mlp.gate";
inline constexpr const char* kMoeSharedGate = "mlp.shared_expert_gate";
inline std::string moe_expert_part(uint32_t e, const char* proj) {
  return "mlp.experts." + std::to_string(e) + "." + proj;
}

struct MoeLayerBytes {
  size_t router = 0;           // bf16 tiled [router_n][hidden]
  size_t gate_up_block = 0;    // one expert's gate||up layout-1 block
  size_t down_block = 0;       // one expert's down layout-1 block
  uint32_t blocks = 0;         // E + 1
  size_t gate_up() const { return gate_up_block * blocks; }
  size_t down() const { return down_block * blocks; }
  size_t total() const { return router + gate_up() + down(); }
  // The bytes ONE token reads in this layer (derived, spec 15 §2): the router GEMV,
  // top_k routed experts and the shared expert - the read-per-token figure the loader
  // reports for a MoE model (a decode step never touches the other experts).
  size_t per_token(uint32_t top_k) const {
    return router + size_t(top_k + 1) * (gate_up_block + down_block);
  }
};

// Layout-1 bytes of a K x N int4 g64 linear: (N / 16) x (K / 64) tiles of 136 u32.
inline constexpr size_t int4_layout1_bytes(uint32_t K, uint32_t N) {
  return size_t(N / 16) * (K / 64) * kInt4TileU32 * 4;
}

// Zero on a dense model.
inline MoeLayerBytes moe_layer_bytes(const model::ModelDesc& d) {
  MoeLayerBytes b;
  if (!d.is_moe()) return b;
  const model::MoeDesc& m = d.moe;
  b.router = size_t(m.router_n()) * d.hidden * 2;
  b.gate_up_block = int4_layout1_bytes(d.hidden, 2 * m.expert_intermediate);
  b.down_block = int4_layout1_bytes(m.expert_intermediate, d.hidden);
  b.blocks = m.blocks();
  return b;
}
// Every MoE layer of the main model (all `layers` on Ornith; the MTP head's MoE layer
// is spec 15e's and not counted).
inline size_t moe_bytes(const model::ModelDesc& d) {
  return d.is_moe() ? size_t(d.layers) * moe_layer_bytes(d).total() : 0;
}

}  // namespace loader
