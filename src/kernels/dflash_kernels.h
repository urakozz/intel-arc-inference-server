#pragma once
#include <string>
#include <vector>

#include "kernels/dflash_dims.h"
#include "kernels/kernels.h"

// Spec 19b: the DFlash drafter's device binaries by name - the host half of the "Spec 19b" block in
// src/kernels/CMakeLists.txt, in its own header so kernels.h (shared with every branch in flight) is not edited.
// Every drafter-only binary is prefixed `df_` and carries in its name the shapes its code bakes, so a host / device
// disagreement names a binary that does not exist and throws at capture. The linears are the reused gemm_sm_i8w
// (spec 28d; gemv_i8w under the test override) / gemv_bf16 (kernels.h's names: K, N and M carry every shape).
//
// The shapes built are z-lab/Qwen3.8-27B-DFlash2's beside Qwen3.8 (hidden 5120, 32 / 8 heads x 128, intermediate
// 17408, the 2048 window over a 2048-slot ring, conv group 16 x 2 taps, selector rank 256, top 16, block 8; the
// target's 248320-row head of which 248077 are tokens). runtime/dflash/dflash_capture.cc checks the drafter's
// descriptor against them before it names a binary.
namespace kernels::dflash {

// kM, kR, kTopK, kQs, kWg, kAttnPb, kAttnWg, kTopkChunk and the control block's indices: kernels/dflash_dims.h.

// `M` the lists' rows (the block), `pb` df_attn's ring offsets a work-group, `win` df_attn's WIN define (its window
// rule is |q - k| <= WIN - 1).
struct Shape {
  unsigned H = 5120, I = 17408, NH = 32, NKV = 8, HD = 128, ring = 2048, win = 2048, group = 16, ctaps = 2;
  unsigned rank = 256, vocab = 248320, used = 248077;
  unsigned M = kM, pb = kAttnPb;
  constexpr unsigned R() const { return M - 1; }
};
inline constexpr Shape kQwen38{};
// Spec 19 v3c: one TP card's half of that drafter (runtime::dflash::TpDFlashDrafter) - 16 q / 4 kv heads, the MLP's
// 8704; everything on the hidden size, the ring's slots and the selector is the whole drafter's.
inline constexpr Shape kQwen38Tp{5120, 8704, 16, 4};
inline constexpr unsigned kTpVocabHalf = 124160;   // a card's rows of the 248320-row head (card 1's real: 123917)

inline std::string s_(unsigned v) { return std::to_string(v); }
inline std::string prep_variant(const Shape& s = kQwen38, bool split = false) {
  return "df_prep_M" + s_(kM) + "_H" + s_(s.H) + "_I" + s_(s.I) + "_Q" + s_(s.NH) + "KV" + s_(s.NKV) + "_D" +
         s_(s.HD) + "_G" + s_(s.group) + "T" + s_(s.ctaps) + "_R" + s_(s.ring) + "_V" + s_(s.vocab) +
         (split ? "_F8Q4R4U8C16H16K8" : "");
}
inline std::string attn_variant(const Shape& s = kQwen38, unsigned heads = 0, unsigned rows = 0) {
  return "df_attn_M" + s_(s.M) + "_Q" + s_(s.NH) + "KV" + s_(s.NKV) + "_D" + s_(s.HD) + "_R" + s_(s.ring) + "_W" +
         s_(s.win) + "_P" + s_(s.pb) + (heads ? "_H" + s_(heads) : "") +
         (rows ? "_B" + s_(rows) : "");
}
// Spec 19 v3d Task 6: df_attn_part on DPAS (df_attn_dpas.cl), one sub-group a q head; its reduce is attn_variant(s, 1, 4)'s.
// ks > 1: the occupancy form, a head's block keys split ks ways (df_attn_dpas.cl's KS).
// Plan 19v3b: the one-card KS 4 part is the 256-GRF build (_g256: the same arithmetic, bitwise, faster).
inline std::string attn_dpas_variant(const Shape& s = kQwen38, unsigned ks = 1) {
  const bool g256 = ks == 4 && s.NH == kQwen38.NH && s.NKV == kQwen38.NKV;
  return "df_attn_M" + s_(s.M) + "_Q" + s_(s.NH) + "KV" + s_(s.NKV) + "_D" + s_(s.HD) + "_R" + s_(s.ring) + "_W" +
         s_(s.win) + "_P" + s_(s.pb) + "_dpas" + (ks > 1 ? "_k" + s_(ks) : "") + (g256 ? "_g256" : "");
}
// Plan 19v3b: the batched draft's row-table part (df_attn_dpas.cl SEQS, KS 1, 256 GRF) and its reduce.
inline std::string attn_seqs_part_variant(const Shape& s = kQwen38) {
  return "df_attn_M" + s_(s.M) + "_Q" + s_(s.NH) + "KV" + s_(s.NKV) + "_D" + s_(s.HD) + "_R" + s_(s.ring) + "_W" +
         s_(s.win) + "_P" + s_(s.pb) + "_dpas_g256_SEQS";
}
inline std::string attn_seqs_reduce_variant(const Shape& s = kQwen38) { return attn_variant(s, 1, 4) + "_SEQS"; }
// The context projection at prefill width (df_proj_wide.cl; runtime::dflash::DFlashDrafter::project_wide).
inline std::string proj_wide_variant(const Shape& s = kQwen38) {
  return "df_proj_wide_H" + s_(s.H) + "_KV" + s_(s.NKV) + "_D" + s_(s.HD) + "_R" + s_(s.ring);
}
// N: the head's rows (the full head, or a draft vocabulary's |V'|); used: indices below it are candidates.
inline std::string topk_variant(unsigned N, unsigned used, bool compact) {
  return "df_topk_R" + s_(kR) + "_N" + s_(N) + "_U" + s_(used) + (compact ? "_C" : "");
}
inline std::string candidate_topk_variant(unsigned c = 64) {
  return "df_candidate_R7_N248320_U248077_K" + std::to_string(c);
}
inline std::string rerank_variant(unsigned c = 64) { return "df_rerank_R7_K5120_U248077_C" + std::to_string(c) + "_Q4"; }
// Spec 19 v3c: a TP card's vocabulary half - its top C of `used` real rows, the rerank's ids offset by the half's first.
inline std::string candidate_half_variant(unsigned used, unsigned c) {
  return "df_candidate_R7_N" + s_(kTpVocabHalf) + "_U" + s_(used) + "_K" + s_(c);
}
inline std::string rerank_half_variant(unsigned used, unsigned base, unsigned c) {
  return "df_rerank_R7_K5120_U" + s_(used) + "_C" + s_(c) + "_Q4_B" + s_(base);
}
// tp.cl's send and sum at the drafter's 8 rows over a crossing of `s` slices (1: gemm_sm_i8w; 4: the int4 split-K).
inline std::string tp_sum_variant(unsigned s, unsigned rank) {
  return "tp_cross_M8_K5120_G20_S" + s_(s) + "_F_R" + s_(rank) + "_sum";
}
inline std::string select_variant(bool v2, const Shape& s = kQwen38, bool split = false) {
  return v2 ? "df_select_M" + s_(kM) + "_K" + s_(kTopK) + "_Q" + s_(kQs) + "_RK" + s_(s.rank) + "_V" + s_(s.vocab) +
                   (split ? "_H16" : "")
            : "df_select_v1_M" + s_(kM) + "_K" + s_(kTopK);
}

// The drafter's linears at the block's M rows, in their form (`int8`: gemm_sm_i8w at kSmallMTuning's row of the shape
// since spec 28d - gemv_i8w under `k` Gemv, the test / probe override runtime::CaptureOptions::force_linear; else
// gemv_bf16 at the tiling kernels::gemv_bf16_tiling gives N). The target's int8 head at the 7 draft rows: the same.
inline std::string linear_variant(unsigned fmt, unsigned M, unsigned K, unsigned N, LinearKind k = LinearKind::SmallM) {
  if (fmt == 2) return int4_linear_variant(k, M, K, N, int4_slices(K, N), 1);
  return fmt == 1 ? int8_linear_variant(k, M, K, N) : gemv_bf16_variant(M, K, N, gemv_bf16_tiling(N));
}

// The draft vocabulary sizes (spec 8 §11's kDraftVocabSizes) the compact head's binaries exist for.
inline const std::vector<unsigned>& draft_vocab_sizes() {
  static const std::vector<unsigned> v = {32768, 65536, 131072};
  return v;
}

// Every binary the draft list and the projection list bind for Qwen3.8's drafter: `v2` (DFlash 2 or v1), `int8`
// (the drafter's linears), `int8_head` (the target's lm_head int8, else bf16), `dv` (|V'|, 0 = the full head).
// Capture checks each exists; tests/kernels/dflash_variant_names_test.cc holds the union to the CMake block.
inline std::vector<std::string> draft_variants(bool v2, unsigned fmt, bool int8_head, unsigned dv,
                                               LinearKind k, LinearKind head, unsigned attn_heads = 1, unsigned attn_rows = 4,
                                               bool candidate_head = false,
                                               bool attn_dpas = false, unsigned attn_ks = 1,
                                               unsigned candidate_bits = 2, unsigned candidates = 80) {
  const Shape& s = kQwen38;
  std::vector<std::string> v = {prep_variant(s, fmt == 2), attn_variant(s, attn_heads, attn_rows), select_variant(v2, s, fmt == 2)};
  if (attn_dpas) v.push_back(attn_dpas_variant(s, attn_ks));   // its part; the reduce stays attn_variant's
  for (auto [K, N] : {std::pair<unsigned, unsigned>{s.H, s.NH * s.HD + 2 * s.NKV * s.HD},   // q||k||v
                      {s.NH * s.HD, s.H},                                                    // o
                      {s.H, 2 * s.I},                                                        // gate||up
                      {s.I, s.H},                                                            // down
                      {5 * s.H, s.H},                                                        // fc (5 taps)
                      {s.H, 2 * s.NKV * s.HD}})                                              // the context's k||v
    v.push_back(linear_variant(fmt, kM, K, N, k));
  if (v2) {
    v.push_back(linear_variant(fmt, kM, s.H, 2 * s.ctaps * (s.H / s.group), k));   // the conv kernel projections
    v.push_back(linear_variant(fmt, kM, s.H, s.rank, k));                          // the selector's projection
  }
  const unsigned N = dv ? dv : s.vocab;
  if (candidate_head) {   // spec 19 v3d: the 2-bit head (gemm_sm's layout 3) with 80 candidates by default
    if (candidate_bits == 2)
      v.push_back(gemm_sm_variant(kR, *small_m_tuning(s.H, s.vocab, 1, 3)));
    else
      v.push_back(linear_variant(2u, kR, s.H, s.vocab, head));
    v.push_back(candidate_topk_variant(candidates));
    v.push_back(rerank_variant(candidates));
  } else {
    v.push_back(linear_variant(int8_head, kR, s.H, N, head));
    v.push_back(topk_variant(N, dv ? dv : s.used, dv != 0));
  }
  return v;
}
inline std::vector<std::string> draft_variants(bool v2, unsigned fmt, bool int8_head, unsigned dv,
                                               LinearKind k = LinearKind::SmallM) {
  return draft_variants(v2, fmt, int8_head, dv, k, k);
}

// Spec 19 v3c: every binary a TP card's draft, tail and projection lists bind (runtime::dflash::TpDFlashDrafter) - the
// candidate head's path only (the production head), `fmt` 1 int8 or 2 int4 g64; `rank` the card.
inline std::vector<std::string> tp_draft_variants(unsigned fmt, unsigned rank, unsigned candidate_bits = 2,
                                                  unsigned candidates = 80, unsigned attn_ks = 4) {
  const Shape& s = kQwen38Tp;
  const unsigned used = rank == 0 ? kTpVocabHalf : kQwen38.used - kTpVocabHalf;
  std::vector<std::string> v = {prep_variant(s, fmt == 2), attn_variant(s, 1, 4), attn_dpas_variant(s, attn_ks),
                                select_variant(true, kQwen38, fmt == 2), tp_sum_variant(fmt == 2 ? 4 : 1, rank)};
  for (auto [K, N] : {std::pair<unsigned, unsigned>{s.H, s.NH * s.HD + 2 * s.NKV * s.HD},   // q||k||v
                      {s.NH * s.HD, s.H},                                                    // o
                      {s.H, 2 * s.I},                                                        // gate||up
                      {s.I, s.H},                                                            // down
                      {5 * s.H, s.H},                                                        // fc (replicated)
                      {s.H, 2 * s.NKV * s.HD},                                               // the context's k||v
                      {s.H, 2 * s.ctaps * (s.H / s.group)},                                  // conv kernel projections
                      {s.H, s.rank}})                                                        // the selector's
    v.push_back(linear_variant(fmt, kM, K, N));
  v.push_back(candidate_bits == 2 ? gemm_sm_variant(kR, *small_m_tuning(s.H, kTpVocabHalf, 1, 3))
                                  : linear_variant(2u, kR, s.H, kTpVocabHalf));
  v.push_back(candidate_half_variant(used, candidates));
  v.push_back(rerank_half_variant(used, rank * kTpVocabHalf, candidates));
  return v;
}

// ==== Kolibri DSpark: the DSpark drafter (intuit/Kolibri-1-dspark beside Kolibri-1) ====
// kDSpark: hidden 2560, 20 / 4 heads x 128, FF 6144, the 2048 window (|q - k| <= 2047) over a 2048-slot ring, the 8-row
// block, the Markov rank 256, Kolibri's 128000-row head of which 127998 are tokens (the drafts never name 127998 - the mask
// row - or 127999). df_prep is reused at these defines (the conv defines unused: no convolution is bound), df_attn with
// CAUSAL=1 (the name's `_C`), df_topk over the head's rows 0..6 (anchor-first: block row r predicts p + r + 1), df_select's
// V2 walk with the Markov head as its codebooks and a hidden of ones (model/dflash.h). The linears bf16 at M 8 (gemv_bf16),
// the target's head at the 7 rows (gemv_i8w / gemv_bf16).
// TODO(card): the Markov bias over the top 16 (vLLM's dspark_draft_topk form) against the full-vocabulary walk (vLLM's
// default) - acceptance only, never the output; measure the gap on the first run before a wider top-k.
inline constexpr Shape kDSpark{2560, 6144, 20, 4, 128, 2048, 2048, 16, 2, 256, 128000, 127998, kM, kAttnPb};
inline std::string dspark_attn_variant() { return attn_variant(kDSpark) + "_C"; }
inline std::string dspark_topk_variant() { return topk_variant(kDSpark.vocab, kDSpark.used, false); }
// Every binary the DSpark draft and projection lists bind for the target's head form (bf16 drafter linears).
// tests/kernels/dspark_names_test.cc holds the union to the CMake Kolibri DSpark block.
inline std::vector<std::string> dspark_draft_variants(bool int8_head) {
  const Shape& s = kDSpark;
  std::vector<std::string> v = {prep_variant(s), dspark_attn_variant(), dspark_topk_variant(), select_variant(true, s)};
  for (const auto& [K, N] : {std::pair<unsigned, unsigned>{s.H, s.NH * s.HD + 2 * s.NKV * s.HD},   // q||k||v
                      {s.NH * s.HD, s.H},                                                    // o
                      {s.H, 2 * s.I},                                                        // gate||up
                      {s.I, s.H},                                                            // down
                      {5 * s.H, s.H},                                                        // fc (5 taps)
                      {s.H, 2 * s.NKV * s.HD}})                                              // the context's k||v
    v.push_back(gemv_bf16_variant(kM, K, N, gemv_bf16_tiling(N)));
  v.push_back(int8_head ? gemv_i8w_variant(kR, s.H, s.vocab) : gemv_bf16_variant(kR, s.H, s.vocab, gemv_bf16_tiling(s.vocab)));
  return v;
}

// ==== PERF-53 (plan 21-QFN-v3b): Qwen3.8-Flash-Next's DFlash 2 drafter (incoai/Qwen3.8-Flash-Next-DFlash2) ====
// kFlashNext: hidden 2560, 32 / 8 heads x 128 (Qwen3.8's drafter's attention: df_attn's binaries are shared), FF 7680,
// the 2048 window over a 2048-slot ring, conv group 16 x 2 taps, selector rank 256, top 16, block 8; the target's
// 248320-row head of which 248077 are tokens. New binaries only where the hidden or the MLP is baked (df_prep,
// df_proj_wide) and the linears at its K / N; df_attn, df_topk and df_select are Qwen3.8's (the same defines). The
// drafter's linears are bf16 in this first form (the packed int4 export is plan 21-QFN-v3b Task 1); the target's head
// at the 7 draft rows int8 (ours, quantised at load) or bf16. A check verifies at most 3 drafts (the 4-row lists).
inline constexpr Shape kFlashNext{2560, 7680, 32, 8, 128, 2048, 2048, 16, 2, 256, 248320, 248077, kM, kAttnPb};
// Every binary its draft and projection lists bind (one card; the attention in its production form: the H1 / B4 reduce
// and the KS 4 DPAS part). tests/kernels/flash_next_dflash_names_test.cc holds the union to the CMake PERF-53 block.
inline std::vector<std::string> flash_next_draft_variants(bool int8_head) {
  const Shape& s = kFlashNext;
  std::vector<std::string> v = {prep_variant(s), proj_wide_variant(s), attn_variant(s, 1, 4), attn_dpas_variant(s, 4),
                                select_variant(true, s), topk_variant(s.vocab, s.used, false)};
  for (const auto& [K, N] : {std::pair<unsigned, unsigned>{s.H, s.NH * s.HD + 2 * s.NKV * s.HD},   // q||k||v
                             {s.NH * s.HD, s.H},                                                    // o
                             {s.H, 2 * s.I},                                                        // gate||up
                             {s.I, s.H},                                                            // down
                             {5 * s.H, s.H},                                                        // fc (5 taps)
                             {s.H, 2 * s.NKV * s.HD},                                               // context k||v
                             {s.H, 2 * s.ctaps * (s.H / s.group)},                                  // conv kernels
                             {s.H, s.rank}})                                                        // selector
    v.push_back(gemv_bf16_variant(kM, K, N, gemv_bf16_tiling(N)));
  v.push_back(int8_head ? gemv_i8w_variant(kR, s.H, s.vocab)
                        : gemv_bf16_variant(kR, s.H, s.vocab, gemv_bf16_tiling(s.vocab)));
  return v;
}

}  // namespace kernels::dflash
