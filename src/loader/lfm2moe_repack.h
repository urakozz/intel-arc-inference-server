#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/lfm2moe.h"

// Spec 24b Task 3: the host half of the LFM2.5-24B-A2B loader - a checkpoint's tensors repacked into
// the device layouts loader/lfm2moe_layout.h sizes, with NO Level Zero, so
// tests/loader/lfm2moe_repack_test.cc runs it on any host over a crafted / synthetic checkpoint.
// loader::load_lfm2moe (lfm2moe_loader.h, Task 4) drives these functions per layer.
//
// **Both expert spellings** (spec 24 §5): `experts.E.{w1,w3,w2}` (the checkpoint's own, as
// transformers fuses them) or `experts.E.{gate_proj,up_proj,down_proj}` (AutoRound's generic
// unfusing). `detect_expert_names` reads which one a checkpoint ships, all-or-nothing - a mix is an
// error naming the first tensor that does not fit.
namespace loader {

// Every tensor of an LFM2.5-24B-A2B checkpoint of `d` in our int4 g64 export, by name - int4 linears
// as {base}.qweight / .qzeros / .scales (never .weight), everything else bf16 / fp32 as shipped. No
// `lm_head.weight` (the tied head is never required; an optional one is the loader's own check,
// Task 4).
std::vector<std::string> lfm_expected_names(const model::Lfm2MoeDesc& d, model::LfmExpertNames names);

// The expert spelling a checkpoint's tensor names use (the first `.experts.` tensor found). Throws
// naming the key when neither spelling is present, or when both are (a mix: one format, two name
// sets, never both in the same checkpoint).
model::LfmExpertNames detect_expert_names(const SafetensorsSet& set);

// One routed expert's device blocks: gate_up = cols_interleave16(w1, w3) then layout-1 (Review Focus
// 1: w1's tiles then w3's, interleaved in 16s - so a sign-flipped w3 lands in the up tiles alone);
// down = w2's layout-1. `w1`, `w3` share N (moe_inter); `w1.K == w3.K == w2.N == hidden`.
struct ExpertRepacked {
  std::vector<uint32_t> gate_up, down;
};
ExpertRepacked repack_expert(const LinearSrc& w1, const LinearSrc& w3, const LinearSrc& w2);

// A conv layer's in_proj (K = hidden, N = 3 x hidden: B | C | x in that column order) into layout-1
// tiles. `conv_tiles` false: the plain column order (B's 2048, then C's, then x's) - ordinary
// `repack_int4_layout1_cols` of one tensor. `conv_tiles` true: spec 24 §4.1's arm-c permutation -
// 16-channel tiles `[B 16 | C 16 | x 16]` (channel tile t's B, C and x columns, t = 0..hidden/16 - 1,
// in that order) - a tile-granularity permutation of the false layout (Review Focus 2: applying it,
// then its inverse, gives the false layout back word for word - lfm2moe_repack_test.cc checks this
// both ways).
std::vector<uint32_t> repack_in_proj(const uint32_t* qweight, const uint16_t* scales, uint32_t K, uint32_t N,
                                     bool conv_tiles);

// A bf16 tensor widened to fp32, element for element, exact (norms, conv taps, embedding rows read
// one at a time).
std::vector<float> widen_bf16(const uint16_t* w, size_t n);

// RoPE table fp32 [max_len][2][head_dim / 2], cos at [p][0][i], sin at [p][1][i], each a bf16 value -
// kolibri1_rope.cc's method (dim 64, theta 1e7): inv[i] = 1 / theta^(2i / head_dim); ang = p x inv[i];
// bf16(cos / sin(ang)).
std::vector<float> lfm_rope_table(const model::Lfm2MoeDesc& d, uint32_t max_len);

}  // namespace loader
