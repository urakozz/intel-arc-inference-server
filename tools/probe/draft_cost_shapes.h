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

}  // namespace draft_cost
