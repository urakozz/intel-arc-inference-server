#pragma once
#include <array>
#include <string_view>
#include "kernels/kernels.h"
#include "model/inkling2.h"

namespace kernels::inkling2 {

using D = model::Inkling2Desc;
inline constexpr unsigned kPartitions = 32;
// tp's verify rows' attention (PERF-147): a (kv head, block)'s rows over this many work-groups (kol_attn.cl
// ATTN_RSPLIT, the `_RS2` builds at 2..8 rows; grid z), so its half heads keep one card's work-group count. One card
// keeps one (measured at 4k: 2 rows +0.08 ms, 4 / 8 rows -0.12 ms of a 14-23 ms step - not taken).
inline constexpr unsigned kTpAttnRowSplit = 2;
inline constexpr unsigned kWindowRing = 4096, kDraftRing = 2048;
inline constexpr std::array<std::pair<unsigned, unsigned>, 6> kInt8Linears{{
    {D::hidden, D::q_n}, {D::hidden, D::kv_n}, {D::hidden, D::q_heads * D::relative_dim}, {D::hidden, 64},
    {D::hidden, D::head_rows}, {D::q_n, D::hidden}
}};
inline constexpr std::array<std::pair<unsigned, unsigned>, 4> kBf16Linears{{
    {D::hidden, D::experts}, {2 * D::hidden, D::controller_rank}, {D::controller_rank, D::hidden},
    {D::hidden, D::kv_n}
}};
inline constexpr std::array<std::pair<unsigned, unsigned>, 4> kDraftBf16Linears{{
    {D::hidden, D::q_n}, {D::q_n, D::hidden}, {D::hidden, D::draft_inter}, {D::draft_inter, D::hidden}
}};

inline std::string int8(unsigned m, unsigned k, unsigned n) {
  return int8_linear_variant(m == 1 ? LinearKind::Gemv : LinearKind::SmallM, m, k, n);
}
// o_proj at one row on gemm_sm's in-work-group K split (36 column work-groups alone read at ~290 GB/s), the output
// convolution and partial sum in its epilogue (gemm_sm.cl INK_ATTN_POST): 256 work-items, grid N / 32.
inline std::string attn_post() { return "gemm_sm_i8w_M1_K3072_N2304_wg_INKPOST"; }
inline constexpr unsigned kTapColumns = 64;   // the full visits' k / v tap projections (gemv_i8w at every width)
inline constexpr unsigned kTapKSplit = 16;    // their verify rows' K split (gemv_i8w GEMV_KSPLIT, M > 1)
inline std::string qkv(unsigned m) { return int8(m, D::hidden, D::q_n) + "_QKV" + std::to_string(D::kv_n); }
// The down projection (K 4608, N 64): a 16-column tile a work-group, K in 16 slices (256 work-items, grid N / 16);
// the up projection: gemv_bf16's 64 columns a work-group.
inline constexpr GemvBf16Tiling kControllerDown{16, 16};
inline std::string controller(unsigned m, bool down) {
  return gemv_bf16_variant(m, down ? 2 * D::hidden : D::controller_rank,
                         down ? D::controller_rank : D::hidden, down ? kControllerDown : GemvBf16Tiling{64, 1}) +
         "_INKCTRL";
}
// The bf16 linears (the drafter's): outputs up to 3072 wide split K in 16 slices (a 16-column tile a work-group), so
// the launch has the work-groups to keep memory busy; the 9216-wide gate / up keep gemv_bf16's 64 columns.
inline constexpr GemvBf16Tiling bf16_tiling(unsigned n) {
  return n <= 3072 ? GemvBf16Tiling{16, 16} : GemvBf16Tiling{64, 1};
}
inline std::string bf16_linear(unsigned m, unsigned k, unsigned n) { return gemv_bf16_variant(m, k, n, bf16_tiling(n)); }
inline std::string markov_linear() { return bf16_linear(1, 2 * (D::markov_rank + D::hidden), 3 * D::markov_rank); }
inline std::string route() { return "kol_route_E128_T8_N128_L128_RENORM"; }
// The selection the engine binds: only the candidates ranked (PERF-28, ROUTE_CAND), bitwise route()'s kol_route.
inline std::string route_select() { return route() + "_CAND"; }
// PERF-123's greedy head: df_candidate's top 64 a row over the 2-bit logits, head_cand.cl at the hidden width.
inline constexpr std::string_view kHeadCandidates = "df_candidate_R7_N200064_U200058_K64";
inline std::string head_candidates() { return head_cand_variant(D::hidden); }
// The fused route's K split (ROUTE_KS): its work-group is 128 x kRouteSplit.
inline constexpr unsigned kRouteSplit = 8;
inline std::string moe(unsigned m, common::ExpertFpFormat format) {
  return "kol_moe_M" + std::to_string(m) + "_E128_T8_D2304_I1152_CAP_" +
         (format == common::ExpertFpFormat::Nvfp4 ? "NVFP4" : "MXFP8");
}
// bf16: the bf16-glue prompt pass's rows (each sum rounded once to bf16) instead of fp32 sums.
inline std::string prefill_expert(common::ExpertFpFormat format, bool gate_up, bool bf16 = false) {
  return std::string(gate_up ? "pf_moe_q4gemm_K2304_N2304_" : "pf_moe_q4gemm_K1152_N2304_") +
         (bf16 ? "BF16_INK_" : "F32_INK_") + (format == common::ExpertFpFormat::Nvfp4 ? "NVFP4" : "MXFP8");
}
// The expert GEMM's work item (the sort's MI) and work-group: 4 x 4 sub-groups, 64 rows by 32 columns each.
inline constexpr unsigned kPrefillExpertRows = 256, kPrefillExpertGroup = 256;
inline constexpr std::string_view kPrefillSort = "pf_moe_sort_mw_E128_T8_INK";
inline constexpr std::string_view kPrefillMoe = "kol_pf_moe_E128_T8_D2304_I1152_INK";
inline std::string attn(unsigned m, bool local) {
  return "kol_attn_M" + std::to_string(m) + "_T" + std::to_string(kPartitions) + "_Q24KV8_REL16E1024_SINK_GATE_" +
         (local ? "W1024_R4096" : "F");
}
inline std::string prefill_attn(bool local) {
  return std::string("kol_pf_flash_attn_Q24KV8_REL16_SINK_GATE_") + (local ? "W1024_R4096" : "F");
}
// The bound form is prefill_attn's `_PAIR` (PERF-10 / PERF-11): a kv head's three q heads, two sub-groups each, on
// 16 rows - grid (ceil(rows / 16), kv heads, 1).
inline constexpr uint32_t kPrefillAttnGroup = 16 * 2 * (D::q_heads / D::kv_heads);
inline constexpr std::string_view kPrefillPrep = "ink_prep_Q24KV8_D128_REL16_PREFILL";
inline std::string embed(unsigned m) { return "ink_embed_gather_M" + std::to_string(m) + "_D2304_V201024"; }
inline std::string argmax(unsigned m) { return "ink_argmax_stage1_M" + std::to_string(m) + "_N200064_V200058"; }

inline constexpr std::string_view kDraftPrep = "df_prep_M8_H2304_I9216_Q24KV8_D128_R2048";
inline constexpr std::string_view kDraftWindow = "df_attn_M8_Q24KV8_D128_R2048_W1024_P128";
inline constexpr std::string_view kGlue = "ink_glue_H2304";
inline constexpr std::string_view kControllerGlue = "ink_glue_H2304_CTRL";
// Decode's mixes and norms, a row over 8 work-groups of one sub-group (ink_glue.cl INK_GLUE_SPLIT, bitwise).
inline constexpr std::string_view kSplitGlue = "ink_glue_H2304_CTRL_SPLIT";
inline constexpr std::string_view kPrep = "ink_prep_Q24KV8_D128_REL16";
inline constexpr std::string_view kDraftRecurrent = "df_inkling_markov_H2304_R256";
// ink_draft_argmax1's grid: a work-group a 1024 live ids (ink_draft.cl ARGMAX_CHUNK).
inline constexpr unsigned kDraftArgmaxChunks = (D::vocab_used + 1023) / 1024;
inline constexpr std::string_view kDraftGlobal = "df_attn_M8_Q24KV8_D128_FULL";
inline constexpr std::string_view kDraftGlobalDpas = "df_attn_M8_Q24KV8_D128_FULL_dpas";   // its part on DPAS
// `--parallel tp` (plan 41j): a card's half of every visit's heads and of every expert's width. The fused attention
// projections at its rows (q / gate 1,536, k / v 512, relative 192, the sinks' 64 whole), o_proj at its K (gemm_sm's
// in-work-group K split at every row count), the 12 q / 4 kv head attention and prep, the 576-wide experts, the
// commit's 512 k / v channels, and the all-reduce (tp.cl's slices form at one slice: tp_send, then tp_sum) a rank.
inline constexpr unsigned kTpQHeads = D::q_heads / 2, kTpKvHeads = D::kv_heads / 2, kTpInter = D::inter / 2;
inline constexpr unsigned kTpFoldG = 9;   // the all-reduce's chunks a row: 256 columns each
inline std::string tp_qkv(unsigned m) {
  return std::string(m == 1 ? "gemv_i8w" : "gemm_sm_i8w") + "_M" + std::to_string(m) + "_K2304_N1536_QKV512_FULL";
}
inline std::string tp_o_proj(unsigned m) { return "gemm_sm_i8w_M" + std::to_string(m) + "_K1536_N2304_wg"; }
inline std::string tp_attn(unsigned m, bool local) {
  return "kol_attn_M" + std::to_string(m) + "_T" + std::to_string(kPartitions) + "_Q12KV4_REL16E1024_SINK_GATE_" +
         (local ? "W1024_R4096" : "F");
}
inline std::string tp_moe(unsigned m, common::ExpertFpFormat format) {
  return "kol_moe_M" + std::to_string(m) + "_E128_T8_D2304_I576_CAP_" +
         (format == common::ExpertFpFormat::Nvfp4 ? "NVFP4" : "MXFP8");
}
inline std::string tp_cross(unsigned m, unsigned rank) {
  // The verify rows (m > 1) cross on the bf16 wire (tp.cl TP_WIRE_BF16): half the bytes, the order kept, not bitwise
  // the fp32 wire. One row keeps the fp32 wire.
  return "tp_cross_M" + std::to_string(m) + "_K2304_G9_S1_X_R" + std::to_string(rank) + (m > 1 ? "_sum_BF16" : "_sum");
}
inline constexpr std::string_view kTpPrep = "ink_prep_Q12KV4_D128_REL16_FUSED";
inline constexpr std::string_view kTpGlue = "ink_glue_H2304_TP";
// The tp prompt pass's: the glue (with the all-reduce's ink_tp_sum), prep and flash, the experts at 576 columns.
inline std::string tp_prefill_glue(bool bf16) { return bf16 ? "ink_glue_H2304_PREFILL_BF16_TP" : "ink_glue_H2304_PREFILL_TP"; }
inline constexpr std::string_view kTpPrefillPrep = "ink_prep_Q12KV4_D128_REL16_PREFILL";
inline std::string tp_prefill_attn(bool local) {
  return std::string("kol_pf_flash_attn_Q12KV4_REL16_SINK_GATE_") + (local ? "W1024_R4096" : "F");
}
inline std::string tp_prefill_expert(common::ExpertFpFormat format, bool gate_up, bool bf16) {
  return std::string(gate_up ? "pf_moe_q4gemm_K2304_N1152_" : "pf_moe_q4gemm_K576_N2304_") + (bf16 ? "BF16_INK_" : "F32_INK_") +
         (format == common::ExpertFpFormat::Nvfp4 ? "NVFP4" : "MXFP8");
}
inline constexpr std::string_view kTpPrefillMoe = "kol_pf_moe_E128_T8_D2304_I576_INK_TP";
inline constexpr unsigned kTpSwigluGroup = 48;
// The drafter's layers under tp: a card's 12 q / 4 kv heads and 4,608 MLP columns (bf16_linear at those widths, the
// context k / v at 512), their prep and attention; the prompt pass's context store at 4 kv heads.
inline constexpr std::array<std::pair<unsigned, unsigned>, 4> kTpDraftBf16Linears{{
    {D::hidden, D::q_n / 2}, {D::q_n / 2, D::hidden}, {D::hidden, D::draft_inter / 2}, {D::draft_inter / 2, D::hidden}
}};
inline constexpr std::string_view kTpDraftPrep = "df_prep_M8_H2304_I4608_Q12KV4_D128_R2048";
inline constexpr std::string_view kTpDraftWindow = "df_attn_M8_Q12KV4_D128_R2048_W1024_P128";
inline constexpr std::string_view kTpDraftGlobal = "df_attn_M8_Q12KV4_D128_FULL";
inline constexpr std::string_view kTpDraftGlobalDpas = "df_attn_M8_Q12KV4_D128_FULL_dpas";
inline constexpr std::string_view kTpDraftContextPrefill = "df_prep_H2304_INK_CONTEXT_PREFILL_Q12KV4";
// The drafter's head and Markov fold by rows under tp: a card's half (kTpHeadHalf rows from rank x kTpHeadHalf) on the
// whole head's k-major tiles, and its pick (the all-reduce between).
inline constexpr unsigned kTpHeadHalf = D::head_rows / 2;
inline constexpr std::string_view kTpDraftHead = "gemm_sm_i8w_M7_K2304_N100032_HS200064";
inline constexpr std::string_view kTpDraftFold = "gemv_i8w_M1_K256_N100032_HS200064";
inline std::string tp_draft_recurrent(unsigned rank) { return std::string(kDraftRecurrent) + "_TP" + std::to_string(rank); }
// The grouped verify router/gather path is separate from the per-row MoE build.
inline constexpr std::string_view kMoeUnion = "ink_moe_union_E128_T8";

}  // namespace kernels::inkling2
