#pragma once
#include <string>
#include <vector>

#include "kernels/kernels.h"

// Spec 19b: the DFlash drafter's device binaries by name - the host half of the "Spec 19b" block in
// src/kernels/CMakeLists.txt, in its own header so kernels.h (shared with every branch in flight) is not edited.
// Every drafter-only binary is prefixed `df_` and carries in its name the shapes its code bakes, so a host / device
// disagreement names a binary that does not exist and throws at capture. The GEMVs are the reused gemv_i8w /
// gemv_bf16 (kernels.h's names: K, N and M carry every shape).
//
// The shapes built are z-lab/Qwen3.8-27B-DFlash2's beside Qwen3.8 (hidden 5120, 32 / 8 heads x 128, intermediate
// 17408, the 2048 window over a 2048-slot ring, conv group 16 x 2 taps, selector rank 256, top 16, block 8; the
// target's 248320-row head of which 248077 are tokens). runtime/dflash/dflash_capture.cc checks the drafter's
// descriptor against them before it names a binary.
namespace kernels::dflash {

inline constexpr unsigned kM = 8;          // the block's rows (the list's M): anchor + up to 7 drafts
inline constexpr unsigned kR = kM - 1;     // the head's rows (mask rows 1..7)
inline constexpr unsigned kTopK = 16, kQs = 16;
inline constexpr unsigned kWg = 256;       // df_prep's, df_topk's, df_select's work-group
inline constexpr unsigned kAttnPb = 128, kAttnWg = 128;   // df_attn_part: ring offsets per block, its work-group
inline constexpr unsigned kTopkChunk = 4096;              // df_topk_stage1's logits per work-group

// The DFlash control block's word indices (runtime/dflash/dflash_control.h holds the struct to them).
inline constexpr unsigned kDctlK = 0, kDctlProjPos = 1, kDctlProjN = 2, kDctlTemp = 3, kDctlSeedLo = 4,
                          kDctlSeedHi = 5;

struct Shape {
  unsigned H = 5120, I = 17408, NH = 32, NKV = 8, HD = 128, ring = 2048, win = 2048, group = 16, ctaps = 2;
  unsigned rank = 256, vocab = 248320, used = 248077;
};
inline constexpr Shape kQwen38{};

inline std::string s_(unsigned v) { return std::to_string(v); }
inline std::string prep_variant(const Shape& s = kQwen38) {
  return "df_prep_M" + s_(kM) + "_H" + s_(s.H) + "_I" + s_(s.I) + "_Q" + s_(s.NH) + "KV" + s_(s.NKV) + "_D" +
         s_(s.HD) + "_G" + s_(s.group) + "T" + s_(s.ctaps) + "_R" + s_(s.ring) + "_V" + s_(s.vocab);
}
inline std::string attn_variant(const Shape& s = kQwen38) {
  return "df_attn_M" + s_(kM) + "_Q" + s_(s.NH) + "KV" + s_(s.NKV) + "_D" + s_(s.HD) + "_R" + s_(s.ring) + "_W" +
         s_(s.win) + "_P" + s_(kAttnPb);
}
// N: the head's rows (the full head, or a draft vocabulary's |V'|); used: indices below it are candidates.
inline std::string topk_variant(unsigned N, unsigned used, bool compact) {
  return "df_topk_R" + s_(kR) + "_N" + s_(N) + "_U" + s_(used) + (compact ? "_C" : "");
}
inline std::string select_variant(bool v2, const Shape& s = kQwen38) {
  return v2 ? "df_select_M" + s_(kM) + "_K" + s_(kTopK) + "_Q" + s_(kQs) + "_RK" + s_(s.rank) + "_V" + s_(s.vocab)
            : "df_select_v1_M" + s_(kM) + "_K" + s_(kTopK);
}

// The drafter's linears at the block's M rows, in their form (`int8`: gemv_i8w; else gemv_bf16 at the tiling
// kernels::gemv_bf16_tiling gives N).
inline std::string linear_variant(bool int8, unsigned M, unsigned K, unsigned N) {
  return int8 ? gemv_i8w_variant(M, K, N) : gemv_bf16_variant(M, K, N, gemv_bf16_tiling(N));
}

// The draft vocabulary sizes (spec 8 §11's kDraftVocabSizes) the compact head's binaries exist for.
inline const std::vector<unsigned>& draft_vocab_sizes() {
  static const std::vector<unsigned> v = {32768, 65536, 131072};
  return v;
}

// Every binary the draft list and the projection list bind for Qwen3.8's drafter: `v2` (DFlash 2 or v1), `int8`
// (the drafter's linears), `int8_head` (the target's lm_head int8, else bf16), `dv` (|V'|, 0 = the full head).
// Capture checks each exists; tests/kernels/dflash_variant_names_test.cc holds the union to the CMake block.
inline std::vector<std::string> draft_variants(bool v2, bool int8, bool int8_head, unsigned dv) {
  const Shape& s = kQwen38;
  std::vector<std::string> v = {prep_variant(), attn_variant(), select_variant(v2)};
  for (auto [K, N] : {std::pair<unsigned, unsigned>{s.H, s.NH * s.HD + 2 * s.NKV * s.HD},   // q||k||v
                      {s.NH * s.HD, s.H},                                                    // o
                      {s.H, 2 * s.I},                                                        // gate||up
                      {s.I, s.H},                                                            // down
                      {5 * s.H, s.H},                                                        // fc (5 taps)
                      {s.H, 2 * s.NKV * s.HD}})                                              // the context's k||v
    v.push_back(linear_variant(int8, kM, K, N));
  if (v2) {
    v.push_back(linear_variant(int8, kM, s.H, 2 * s.ctaps * (s.H / s.group)));   // the conv kernel projections
    v.push_back(linear_variant(int8, kM, s.H, s.rank));                          // the selector's projection
  }
  const unsigned N = dv ? dv : s.vocab;
  v.push_back(int8_head ? gemv_i8w_variant(kR, s.H, N) : gemv_bf16_variant(kR, s.H, N, gemv_bf16_tiling(N)));
  v.push_back(topk_variant(N, dv ? dv : s.used, dv != 0));
  return v;
}

}  // namespace kernels::dflash
