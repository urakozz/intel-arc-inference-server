#pragma once
#include <array>
#include <string_view>
#include "kernels/kernels.h"
#include "model/inkling2.h"

namespace kernels::inkling2 {

using D = model::Inkling2Desc;
inline constexpr unsigned kPartitions = 32;
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
inline std::string qkv(unsigned m) { return int8(m, D::hidden, D::q_n) + "_QKV" + std::to_string(D::kv_n); }
inline std::string controller(unsigned m, bool down) {
  return gemv_bf16_variant(m, down ? 2 * D::hidden : D::controller_rank,
                         down ? D::controller_rank : D::hidden, {64, 1}) + "_INKCTRL";
}
inline std::string markov_linear() {
  return gemv_bf16_variant(1, 2 * (D::markov_rank + D::hidden), 3 * D::markov_rank, {64, 1});
}
inline std::string route() { return "kol_route_E128_T8_N128_L128_RENORM"; }
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
inline constexpr std::string_view kPrep = "ink_prep_Q24KV8_D128_REL16";
inline constexpr std::string_view kDraftRecurrent = "df_inkling_markov_H2304_R256";
inline constexpr std::string_view kDraftGlobal = "df_attn_M8_Q24KV8_D128_FULL";
// The grouped verify router/gather path is separate from the per-row MoE build.
inline constexpr std::string_view kMoeUnion = "ink_moe_union_E128_T8";

}  // namespace kernels::inkling2
