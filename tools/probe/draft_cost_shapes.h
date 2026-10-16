// draft_cost_shapes.h - spec 19a (plan 19a Task 4): the GEMVs one DFlash2 block costs on
// Qwen3.8, as tools/probe/probe_draft_cost times them and as src/kernels/CMakeLists.txt's
// B70_VERIFY_M8 block builds them (tests/kernels/verify_m8_names_test.cc checks the two
// against each other).
//
// From z-lab/Qwen3.8-27B-DFlash2's config.json and safetensors header (DFlash2DraftModel,
// bf16, 1.924 G parameters): 5 layers, hidden 5120, 32 q / 8 kv heads x 128 (q 4096, k / v
// 1024), intermediate 17408, `fc` 25600 -> 5120 (the 5 target taps x 5120), two conv kernel
// projections per layer (attention_conv / mlp_conv .kernel_projection, 5120 -> 1280), the
// candidate selector's hidden projection 5120 -> 256; its two 248320 x 256 codebooks are
// gathered, not streamed, and are not GEMVs. block_size 8, so a block is M = 8 rows (the
// anchor and 7 masks) and the target head runs on its K = 7 draft rows.
//
// Fused as the engine fuses Qwen3.8's own linears: q||k||v (6144) and gate||up (34816). The
// commit list's context projection re-reads each layer's k||v (5120 -> 2048) for the j + 1
// committed rows (at most 8) and `fc` once (spec 19 §5, "Commit list"); both are here.
#pragma once

namespace draft_cost {

struct Linear {
  const char* name;
  unsigned K, N;
  bool per_layer;   // once per drafter layer (kLayers launches), or once per block
  unsigned int4_s;  // gemv.cl's split-K S at layout 0 (PROVISIONAL picks but gate||up / down)
};

inline constexpr unsigned kLayers = 5;      // num_hidden_layers
inline constexpr unsigned kRows = 8;        // dflash_config.block_size
inline constexpr unsigned kHeadRows = 7;    // the block's draft rows (K = block - 1)
inline constexpr unsigned kHidden = 5120;

inline constexpr Linear kLinears[] = {
    {"attn conv proj", 5120, 1280, true, 16},
    {"q||k||v", 5120, 6144, true, 4},
    {"o_proj", 4096, 5120, true, 4},
    {"mlp conv proj", 5120, 1280, true, 16},
    {"gate||up", 5120, 34816, true, 8},     // Qwen3.8's production cell
    {"down", 17408, 5120, true, 4},         // Qwen3.8's production cell
    {"ctx k||v", 5120, 2048, true, 8},      // the commit's context projection
    {"fc", 25600, 5120, false, 8},          // the commit's context projection
    {"selector proj", 5120, 256, false, 16},
};
inline constexpr unsigned kNumLinears = sizeof(kLinears) / sizeof(kLinears[0]);

// The target's lm_head over the K draft rows, int8 (spec 9): the full vocabulary and a
// 32k draft vocabulary (spec 8 §11's compact head; P0's ranked V' = 32768).
inline constexpr unsigned kHeadVocab[] = {248320, 32768};

// Spec 23f Task 4 (`probe_draft_cost --shapes muse`): Muse-Glimmer-30B's verify and its drafter's block, from the two
// config.json files (model::MuseGlimmerDesc, meta-models/Muse-Glimmer-30B-assistant @ e8192f3a; spec 23 §2 / §8.2):
//   the verify at M = 1..16 rows   52 layers x the four int4 g64 layout-0 GEMVs at their PROVISIONAL S (23b / 23c:
//                                  q||gate||k||v 6656 x 8704 S2, o_proj 4096 x 6656 S4, gate||up 6656 x 39936 S8,
//                                  down 19968 x 6656 S4 - 13.37 GB), the int8 lm_head 6656 x 202048 and the two
//                                  argmax stages (mg_argmax.cl) at M rows
//   one drafter block at 16 rows   5 layers x q||k||v 6656 x 6144, o 4096 x 6656, gate||up 6656 x 39936, down
//                                  19968 x 6656 and the commit's ctx k||v 6656 x 2048; fc 33280 x 6656 once; the
//                                  target's int8 head over the 15 draft rows - in bf16, int8 and int4 g64 (layout 1,
//                                  S 1: the engine's form - the [M][N] rows the df_* kernels read)
namespace muse {
inline constexpr unsigned kLayers = 52, kHidden = 6656, kVocab = 202048, kMaxM = 16;
inline constexpr Linear kTarget[] = {
    {"q||gate||k||v", 6656, 8704, true, 2},
    {"o_proj", 4096, 6656, true, 4},
    {"gate||up", 6656, 39936, true, 8},
    {"down", 19968, 6656, true, 4},
};
inline constexpr unsigned kDraftLayers = 5, kDraftRows = 16, kDraftHeadRows = 15;
inline constexpr Linear kDrafter[] = {
    {"q||k||v", 6656, 6144, true, 1},
    {"o_proj", 4096, 6656, true, 1},
    {"gate||up", 6656, 39936, true, 1},
    {"down", 19968, 6656, true, 1},
    {"ctx k||v", 6656, 2048, true, 1},     // the commit's context projection
    {"fc", 33280, 6656, false, 1},         // the commit's context projection (5 taps x 6656)
};
inline constexpr unsigned kNumDrafter = sizeof(kDrafter) / sizeof(kDrafter[0]);
// The attention term (ESTIMATED from bytes, not timed): 13 full layers' K + V rows (13,312 B a position) and the 39
// windows of 2048 rows (1024 B a row) - read once per verify whatever M (the rows of a kv head share the reads).
inline constexpr double kFullKvBytesPerPos = 13312.0, kWindowBytes = 39.0 * 2048 * 1024;
}  // namespace muse

}  // namespace draft_cost
