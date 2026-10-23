#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/dflash_layout.h"
#include "loader/lilicorr_layout.h"
#include "loader/loader.h"   // DeviceWeight
#include "model/dflash.h"

// Spec 26b Task 2's device half: the LiLiCorr drafter on the card beside Bolt's last stage (spec 26 Decision 8: device 1,
// which holds the target's lm_head) - its backbone in the chosen format, its head always exact, its small tensors; the
// target's embedding and head are SHARED (the embedding through the row-major replica lilicorr_embed_replica uploads, the
// head the target loader's), never copied here. The mask row is the target's embedding row 990 (copied once at load).
//
// The layouts are loader/lilicorr_layout.h's (the planner's formula); load_lilicorr asserts its allocations add up to
// loader::lilicorr_weight_bytes exactly (at the checkpoint's own NVFP4 set).
namespace loader {

// A backbone linear: its device weight and, for an exact NVFP4 linear, its per-column output scale (ll_gemv_bf16_g).
struct LiliLinear {
  DeviceWeight w;
  std::unique_ptr<l0::Mem> gs;   // fp32 [N], null for every other linear
};
struct LiliLayer {
  LiliLinear qkv, o, gate_up, down, attn_kproj, mlp_kproj;
  l0::Mem norms;                 // fp32 [input_layernorm | post_attention_layernorm][hidden]
};
// A head linear: W'^T bf16 [K][N] and [gs N | bias N] fp32 (ll_gemm's two arguments).
struct LiliHeadWeight {
  l0::Mem wt, gb;
  uint32_t K = 0, N = 0;
};

struct LiliWeights {
  model::DFlashDesc desc;
  DraftFormat format = DraftFormat::Int8;
  uint32_t target_hidden = 0;
  std::unique_ptr<LiliLinear> fc;
  std::unique_ptr<l0::Mem> hidden_norm, final_norm;   // fp32 [hidden]
  std::vector<LiliLayer> layers;
  std::unique_ptr<l0::Mem> qk_norm;                   // fp32 [layers][q | k][head_dim]
  std::unique_ptr<l0::Mem> conv_base;                 // fp32 [layers][attn, mlp][side][tap][hidden]
  std::unique_ptr<l0::Mem> sinks;                     // fp32 [layers][q heads]
  std::unique_ptr<l0::Mem> mask_row;                  // fp32 [hidden]
  // the LiLiCorr head, by loader::LiliHead (TokenProj .. AnchorOut: the per-layer ones hold layer 0's, `head_layers` the
  // rest by [layer][InProj, OutProj, MlpUp, MlpDown])
  std::vector<std::unique_ptr<LiliHeadWeight>> head;
  std::vector<std::vector<std::unique_ptr<LiliHeadWeight>>> head_layers;
  std::unique_ptr<l0::Mem> head_small;                // fp32 [lili_head_small(desc).total]
  uint64_t nvfp4_rows = 0;                            // the backbone's NVFP4 output rows (the formula's)
  size_t bytes = 0;                                   // == lilicorr_weight_bytes(desc, target_hidden, format, nvfp4_rows)
  double seconds = 0, quant_seconds = 0;
  std::string line;
  const LiliHeadWeight& head_at(LiliHead w) const { return *head.at(size_t(w)); }
  const LiliHeadWeight& head_layer(uint32_t l, LiliHead w) const;
};

// Loads the LiLiCorr drafter at `snapshot_or_repo` (resolve_snapshot rules) onto `ctx`'s device for a target of
// `target`'s shape whose embedding `target_embed` (bf16 ROW-MAJOR [vocab][hidden], on this device - the replica) gives the
// mask row. Before a byte is allocated: the host half's checks (lilicorr_repack.h), model::check_lilicorr (the kernels'
// shapes) and the drafter held to the target. `format`: Exact / Int8 / Int4G64 (Bf16 is refused: exact is its spelling).
// One load line:
//   LiLiCorr drafter: 5 layers, taps 1/18/35/51/68/85, block 8, int8 (decoded + quantised at load 6.1 s), 25 NVFP4 linears,
//   the head exact, 1,186,xxx,xxx B, 7.0 s
LiliWeights load_lilicorr(l0::Context& ctx, const std::string& snapshot_or_repo, const model::DFlashTargetShape& target,
                          const l0::Mem& target_embed, DraftFormat format, unsigned threads = 0);

// The drafter's embedding replica on its device (spec 26 Decision 8): the target's bf16 [vocab][hidden] rows, row-major
// (df_embed and ll_gemm_gather read row `id` at id x hidden) - Bolt's own embedding lives on device 0 in gemv_bf16's
// tiles. `rows` is the checkpoint's tensor (loader::BoltCheckpoint::embed(), mmapped).
l0::Mem lilicorr_embed_replica(l0::Context& ctx, const uint16_t* rows, uint32_t vocab, uint32_t hidden);

}  // namespace loader
