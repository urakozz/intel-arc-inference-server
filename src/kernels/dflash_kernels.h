#pragma once
#include <string>
#include <vector>

#include "kernels/dflash_dims.h"
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

// kM, kR, kTopK, kQs, kWg, kAttnPb, kAttnWg, kTopkChunk and the control block's indices: kernels/dflash_dims.h.

// `M` the lists' rows (the block), `pb` df_attn's ring offsets a work-group, `win` df_attn's WIN define (its window
// rule is |q - k| <= WIN - 1). Spec 23f: `muse` - meta-models/Muse-Glimmer-30B-assistant beside Muse-Glimmer-30B
// (kMuse): DFlash v1 (no convolutions, no selector), eps 1e-5 (the "_E5" in its names), the block of 16, the window
// |q - k| <= 2048 as WIN 2049 over a 2112-slot ring of PB 64 blocks (kernels/dflash_dims.h muse_ring), the target's
// 202048-row head of which every row is a token; its norm and its per-row pick are mg_dflash.cl's (mg_df_norm,
// mg_df_select: the assistant's two-rounding RMSNorm and a 16-row readback that writes the verify's ids buffer).
struct Shape {
  unsigned H = 5120, I = 17408, NH = 32, NKV = 8, HD = 128, ring = 2048, win = 2048, group = 16, ctaps = 2;
  unsigned rank = 256, vocab = 248320, used = 248077;
  unsigned M = kM, pb = kAttnPb;
  bool muse = false;
  constexpr unsigned R() const { return M - 1; }
};
inline constexpr Shape kQwen38{};
inline constexpr Shape kMuse{6656, 19968, 32, 8, 128, muse_ring(2048), 2049, 0, 0, 0, 202048, 202048,
                             kMuseM, kMuseAttnPb, true};
static_assert(kMuse.ring == 2112, "the Muse ring is 33 blocks of 64");

inline std::string s_(unsigned v) { return std::to_string(v); }
inline std::string prep_variant(const Shape& s = kQwen38) {
  if (s.muse)
    return "df_prep_M" + s_(s.M) + "_H" + s_(s.H) + "_I" + s_(s.I) + "_Q" + s_(s.NH) + "KV" + s_(s.NKV) + "_D" +
           s_(s.HD) + "_R" + s_(s.ring) + "_V" + s_(s.vocab) + "_E5";
  return "df_prep_M" + s_(kM) + "_H" + s_(s.H) + "_I" + s_(s.I) + "_Q" + s_(s.NH) + "KV" + s_(s.NKV) + "_D" +
         s_(s.HD) + "_G" + s_(s.group) + "T" + s_(s.ctaps) + "_R" + s_(s.ring) + "_V" + s_(s.vocab);
}
inline std::string attn_variant(const Shape& s = kQwen38) {
  return "df_attn_M" + s_(s.M) + "_Q" + s_(s.NH) + "KV" + s_(s.NKV) + "_D" + s_(s.HD) + "_R" + s_(s.ring) + "_W" +
         s_(s.win) + "_P" + s_(s.pb);
}
// N: the head's rows (the full head, or a draft vocabulary's |V'|); used: indices below it are candidates; R: the
// draft rows (Qwen3.8's 7, Muse's 15).
inline std::string topk_variant(unsigned N, unsigned used, bool compact, unsigned R = kR) {
  return "df_topk_R" + s_(R) + "_N" + s_(N) + "_U" + s_(used) + (compact ? "_C" : "");
}
inline std::string select_variant(bool v2, const Shape& s = kQwen38) {
  if (s.muse) return "mg_dflash_M" + s_(s.M) + "_H" + s_(s.H) + "_K" + s_(kTopK) + "_E5";   // mg_df_norm + mg_df_select
  return v2 ? "df_select_M" + s_(kM) + "_K" + s_(kTopK) + "_Q" + s_(kQs) + "_RK" + s_(s.rank) + "_V" + s_(s.vocab)
            : "df_select_v1_M" + s_(kM) + "_K" + s_(kTopK);
}

// The drafter's linears at the block's M rows, in their form (`int8`: gemv_i8w; else gemv_bf16 at the tiling
// kernels::gemv_bf16_tiling gives N).
inline std::string linear_variant(bool int8, unsigned M, unsigned K, unsigned N) {
  return int8 ? gemv_i8w_variant(M, K, N) : gemv_bf16_variant(M, K, N, gemv_bf16_tiling(N));
}
// Spec 23f: the three formats (`fmt` 0 bf16, 1 int8, 2 int4 g64: gemv.cl layout 1 at S 1, the [M][N] rows the df_*
// kernels read).
inline std::string linear_variant_fmt(unsigned fmt, unsigned M, unsigned K, unsigned N) {
  return fmt == 2 ? gemv_variant(M, K, N, 1, 1) : linear_variant(fmt == 1, M, K, N);
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

// Spec 23f: every binary Muse-Glimmer's assistant's draft and projection lists bind, for a drafter format (`fmt`
// linear_variant_fmt's) and the target's head form. No draft vocabulary (refused for Muse). tests/kernels/
// muse_dflash_names_test.cc holds the union to the CMake Spec 23f block's B70_MUSE_DFLASH list.
inline std::vector<std::string> muse_draft_variants(unsigned fmt, bool int8_head) {
  const Shape& s = kMuse;
  std::vector<std::string> v = {prep_variant(s), attn_variant(s), select_variant(false, s)};
  for (auto [K, N] : {std::pair<unsigned, unsigned>{s.H, s.NH * s.HD + 2 * s.NKV * s.HD},   // q||k||v
                      {s.NH * s.HD, s.H},                                                    // o
                      {s.H, 2 * s.I},                                                        // gate||up
                      {s.I, s.H},                                                            // down
                      {5 * s.H, s.H},                                                        // fc (5 taps)
                      {s.H, 2 * s.NKV * s.HD}})                                              // the context's k||v
    v.push_back(linear_variant_fmt(fmt, s.M, K, N));
  v.push_back(int8_head ? gemv_i8w_variant(s.R(), s.H, s.vocab)
                        : gemv_bf16_variant(s.R(), s.H, s.vocab, gemv_bf16_tiling(s.vocab)));
  v.push_back(topk_variant(s.vocab, s.used, false, s.R()));
  return v;
}

// ==== Spec 26b: the LiLiCorr drafter (nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr beside Bolt) ====
// kLiLiCorr: hidden 4096, 32 / 2 heads x 128, FF 12288, the 1024 window (|q - k| <= 1023) over a 1024-slot ring of PB 64
// blocks, conv groups of 16 x 2 taps, the 8-row block, the target's 131072-row head (every row a token). df_prep is reused
// at these defines; the attention (sinks), the top-8 with log-probs, the head and the walk are the ll_* family copies
// (src/kernels/dflash/ll_*.cl) - each name carries the shapes its code bakes. ll_attn_part runs HPW = 8 of a kv head's 16
// q heads a work-group (the local memory of 64 pairs).
inline constexpr Shape kLiLiCorr{4096, 12288, 32, 2, 128, 1024, 1024, 16, 2, 0, 131072, 131072, kM, 64, false};
inline constexpr unsigned kLiliTopK = 8, kLiliSlots = kM - 1, kLiliHidden = 1024, kLiliHeads = 8, kLiliHpw = 8;
inline constexpr unsigned kLiliGemmWg = 64, kLiliSelectWg = 64, kLiliRmax = 64;   // ll_lattice's GW / RMAX, ll_select's WG

inline std::string ll_attn_variant(const Shape& s = kLiLiCorr) {
  return "ll_attn_M" + s_(s.M) + "_Q" + s_(s.NH) + "KV" + s_(s.NKV) + "_D" + s_(s.HD) + "_R" + s_(s.ring) + "_W" +
         s_(s.win) + "_P" + s_(s.pb) + "_H" + s_(kLiliHpw);
}
inline std::string ll_topk_variant(unsigned N = kLiLiCorr.vocab, unsigned used = kLiLiCorr.used) {
  return "ll_topk_R" + s_(kLiliSlots) + "_N" + s_(N) + "_U" + s_(used) + "_K" + s_(kLiliTopK);
}
inline std::string ll_lattice_variant() {
  return "ll_lattice_S" + s_(kLiliSlots) + "_K" + s_(kLiliTopK) + "_HL" + s_(kLiliHidden) + "_LH" + s_(kLiliHeads) +
         "_B" + s_(kM);
}
inline std::string ll_select_variant() { return "ll_select_S" + s_(kLiliSlots) + "_K" + s_(kLiliTopK); }
// The exact arm's NVFP4 linears: ll_gemv_bf16_g at gemv_bf16's tiling for N (its suffixes).
inline std::string ll_gemv_g_variant(unsigned M, unsigned K, unsigned N) {
  const GemvBf16Tiling t = gemv_bf16_tiling(N);
  return "ll_gemv_bf16_g_M" + s_(M) + "_K" + s_(K) + "_N" + s_(N) + (t.cols == kGemvBf16Cols ? "" : "_C" + s_(t.cols)) +
         (t.ksplit == 1 ? "" : "_S" + s_(t.ksplit));
}
// A LiLiCorr backbone linear in `fmt` (0 exact, 1 int8, 2 int4 g64) at M rows; `nvfp4`: an exact NVFP4 linear (its g on the
// output: ll_gemv_bf16_g), else the shared gemv_bf16.
inline std::string lili_linear_variant(unsigned fmt, bool nvfp4, unsigned M, unsigned K, unsigned N) {
  if (fmt == 0) return nvfp4 ? ll_gemv_g_variant(M, K, N) : gemv_bf16_variant(M, K, N, gemv_bf16_tiling(N));
  return linear_variant_fmt(fmt, M, K, N);
}

// Every binary the LiLiCorr draft and projection lists bind for a drafter format (`fmt` as above) and the target's head
// form, at the published checkpoint's NVFP4 set (fc, every layer's gate / up / down - the capture names each weight's own
// by its output scale). tests/kernels/lilicorr_names_test.cc holds the union to the CMake Spec 26b block.
inline std::vector<std::string> lilicorr_draft_variants(unsigned fmt, bool int8_head) {
  const Shape& s = kLiLiCorr;
  std::vector<std::string> v = {prep_variant(s), ll_attn_variant(), ll_topk_variant(), ll_lattice_variant(),
                                ll_select_variant()};
  struct L {
    unsigned K, N;
    bool nvfp4;
  };
  for (const L& l : {L{s.H, s.NH * s.HD + 2 * s.NKV * s.HD, false},   // q||k||v
                     L{s.NH * s.HD, s.H, false},                       // o
                     L{s.H, 2 * s.I, true},                            // gate||up
                     L{s.I, s.H, true},                                // down
                     L{6 * s.H, s.H, true},                            // fc (6 taps)
                     L{s.H, 2 * s.ctaps * (s.H / s.group), false},     // the conv kernel projections
                     L{s.H, 2 * s.NKV * s.HD, false}})                 // the context's k||v
    v.push_back(lili_linear_variant(fmt, l.nvfp4, kM, l.K, l.N));
  v.push_back(int8_head ? gemv_i8w_variant(kLiliSlots, s.H, s.vocab)
                        : gemv_bf16_variant(kLiliSlots, s.H, s.vocab, gemv_bf16_tiling(s.vocab)));
  return v;
}

}  // namespace kernels::dflash
