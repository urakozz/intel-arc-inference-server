#include <set>
#include <string>
#include "check.h"
#include "kernels/inkling2_kernels.h"

int main(int argc, char** argv) {
  namespace ink = kernels::inkling2;
  std::set<std::string> want;
  for (unsigned m = 1; m <= 8; ++m) {
    for (const auto& [k, n] : ink::kInt8Linears) want.insert(ink::int8(m, k, n));
    for (const auto& [k, n] : ink::kBf16Linears) want.insert(kernels::gemv_bf16_variant(m, k, n, {64, 1}));
    want.insert(ink::qkv(m));
    want.insert(ink::qkv(m) + "_FULL");
    want.insert(ink::controller(m, true));
    want.insert(ink::controller(m, false));
    want.insert(ink::embed(m));
    want.insert(ink::argmax(m));
    want.insert(ink::moe(m, common::ExpertFpFormat::Nvfp4));
    want.insert(ink::moe(m, common::ExpertFpFormat::Mxfp8));
    want.insert(ink::attn(m, true));
    want.insert(ink::attn(m, false));
    want.insert(ink::attn(m, false) + "_KV8");
    want.insert(ink::bf16_linear(m, ink::D::hidden, ink::D::kv_n));   // the drafter's context k / v
    want.insert(kernels::gemm_sm_variant(m, *kernels::small_m_tuning(ink::D::hidden, ink::D::head_rows, 1, 3)));
    want.insert(ink::int8(m, ink::D::hidden, 512));                     // the candidate head's rescore
    want.insert(kernels::gemv_i8w_variant(m, ink::D::hidden, ink::kTapColumns));
    if (m == 1) want.insert(kernels::gemv_i8w_variant(m, ink::D::hidden, ink::kTapColumns) + "_KS");
    // `--parallel tp`
    want.insert(ink::tp_qkv(m));
    want.insert(ink::tp_o_proj(m));
    want.insert(ink::tp_moe(m, common::ExpertFpFormat::Nvfp4));
    want.insert(ink::tp_moe(m, common::ExpertFpFormat::Mxfp8));
    want.insert(ink::tp_attn(m, true));
    want.insert(ink::tp_attn(m, false));
    want.insert(ink::tp_attn(m, false) + "_KV8");
    if (m > 1)   // the decode's rows over two work-groups (kTpAttnRowSplit)
      for (const std::string& v : {ink::tp_attn(m, true), ink::tp_attn(m, false), ink::tp_attn(m, false) + "_KV8"})
        want.insert(v + "_RS2");
    for (unsigned rank = 0; rank < 2; ++rank) want.insert(ink::tp_cross(m, rank));
    want.insert(ink::bf16_linear(m, ink::D::hidden, ink::D::kv_n / 2));   // the drafter's context k / v under tp
  }
  want.emplace(ink::kTpPrep);
  want.insert(std::string(ink::kTpPrep) + "_KV8");
  want.emplace(ink::kTpGlue);   // the prompt pass's tp builds are outside this list
  for (const auto& [k, n] : ink::kTpDraftBf16Linears) want.insert(ink::bf16_linear(8, k, n));
  for (auto n : {ink::kTpDraftPrep, ink::kTpDraftWindow, ink::kTpDraftGlobal, ink::kTpDraftGlobalDpas, ink::kTpDraftHead,
                 ink::kTpDraftFold})
    want.emplace(n);
  for (unsigned rank = 0; rank < 2; ++rank) want.insert(ink::tp_draft_recurrent(rank));
  want.insert(ink::attn_post());
  want.emplace(ink::kHeadCandidates);
  want.insert(ink::head_candidates());
  for (const auto format : {common::ExpertFpFormat::Nvfp4, common::ExpertFpFormat::Mxfp8})
    want.insert(ink::moe(1, format) + "_POST");
  want.emplace(ink::kControllerGlue);
  want.emplace(ink::kSplitGlue);   // the decode glue a row over several work-groups
  want.emplace(std::string(ink::kPrep) + "_FUSED");
  want.emplace(std::string(ink::kPrep) + "_FUSED_KV8");
  want.insert(ink::route());
  want.insert(ink::route() + "_M1_FUSED");
  want.insert(ink::route_select());
  want.insert(ink::markov_linear());
  for (const auto& [k, n] : ink::kDraftBf16Linears) want.insert(ink::bf16_linear(8, k, n));
  want.insert(ink::int8(1, 256, ink::D::head_rows));
  for (auto n : {ink::kDraftPrep, ink::kDraftWindow, ink::kGlue, ink::kPrep, ink::kDraftGlobal, ink::kDraftGlobalDpas, ink::kDraftRecurrent}) want.emplace(n);
  std::set<std::string> built(argv + 1, argv + argc);
  for (const auto& n : want)
    if (!built.contains(n)) std::fprintf(stderr, "Missing Inkling kernel %s\n", n.c_str());
  for (const auto& n : built)
    if (!want.contains(n)) std::fprintf(stderr, "Unexpected Inkling kernel %s\n", n.c_str());
  CHECK(want == built);
  std::printf("Inkling kernel names: %zu variants\n", want.size());
}
