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
  }
  for (const auto format : {common::ExpertFpFormat::Nvfp4, common::ExpertFpFormat::Mxfp8})
    want.insert(ink::moe(1, format) + "_POST");
  want.emplace(ink::kControllerGlue);
  want.emplace(std::string(ink::kPrep) + "_FUSED");
  want.insert(ink::route());
  want.insert(ink::route() + "_M1_FUSED");
  want.insert(ink::markov_linear());
  for (const auto& [k, n] : ink::kDraftBf16Linears) want.insert(kernels::gemv_bf16_variant(8, k, n, {64, 1}));
  want.insert(ink::int8(1, 256, ink::D::head_rows));
  for (auto n : {ink::kDraftPrep, ink::kDraftWindow, ink::kGlue, ink::kPrep, ink::kDraftGlobal, ink::kDraftRecurrent}) want.emplace(n);
  std::set<std::string> built(argv + 1, argv + argc);
  for (const auto& n : want)
    if (!built.contains(n)) std::fprintf(stderr, "Missing Inkling kernel %s\n", n.c_str());
  for (const auto& n : built)
    if (!want.contains(n)) std::fprintf(stderr, "Unexpected Inkling kernel %s\n", n.c_str());
  CHECK(want == built);
  std::printf("Inkling kernel names: %zu variants\n", want.size());
}
