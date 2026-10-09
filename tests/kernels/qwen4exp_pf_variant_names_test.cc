// Spec 21d Task 2: the binaries a Qwen3.8-Flash-Next prefill chunk binds (kernels::qwen4exp::prefill_variants - the
// names runtime/qwen4exp/qwen4exp_prefill.cc forms) for both dense arms (ours int4 / Intel's bf16), both attention
// forms and both PLE scale forms, against the ones the build makes for it (passed as arguments: tests/CMakeLists.txt's
// B70_Q4EXP_PREFILL_KERNELS, `kernel_<name>` - src/kernels/CMakeLists.txt's spec 21d block, plus the reused names other
// blocks build: pf_gemm_T0, Qwen3.8's pf_gdn_*, 21c's q4_route_M1, K2's / Kolibri's 6144 x 2560 slab). Host only: a
// name the walk would ask for that no CMake line builds fails here on the Mac, not at prepare_prefill on the box; and
// nothing listed is dead. Also pins the prefill constants the walk and the planner share with the kernels' names.
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/qwen4exp_kernels.h"
#include "model/qwen4exp.h"

int main(int argc, char** argv) {
  namespace kq = kernels::qwen4exp;
  std::set<std::string> built;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    CHECK(a.rfind("kernel_", 0) == 0);
    built.insert(a.substr(7));
  }
  CHECK(!built.empty());
  CHECK_EQ(kq::kPfC, 2048u);
  CHECK_EQ(kq::pf_hdr::words(), 528u);
  CHECK_EQ(kq::kPfTm, kernels::pf_moe::kTileM);
  CHECK_EQ(kq::kPfSlab, kernels::kPfSlabWidth);
  CHECK(kq::kPfIdxLd >= kq::kIdxN && kq::kPfIdxLd % 256 == 0 && kq::kPfHcDownLd >= kq::kHcDownN && kq::kPfHcDownLd % 256 == 0);
  CHECK((kq::kQHeads / kq::kKvHeads) % kq::kPfFlashHpw == 0);   // HPW divides GQA 12 (pf_flash_attn's #error)
  // the weight batches (derived): gate||up 81 blocks a batch, 7 batches; down 163, 4 - of 513 blocks (the shared last)
  const size_t gu = size_t(kq::kHidden) * 2 * kq::kInter * 2, dn = size_t(kq::kInter) * kq::kHidden * 2;
  CHECK(gu == 6553600 && dn == 3276800);
  CHECK(kq::kPfBatchBytes / gu == 81 && (kq::kExperts + 1 + 80) / 81 == 7);
  CHECK(kq::kPfBatchBytes / dn == 163 && (kq::kExperts + 1 + 162) / 163 == 4);
  CHECK_EQ(kq::pf_moe_variant(), std::string("q4_pf_moe_E512_T10_D2560_I640_L256"));
  CHECK_EQ(kq::pf_sparse_attn_variant(false), std::string("q4_pf_sparse_attn_Q24KV2"));
  CHECK_EQ(kq::pf_sparse_attn_variant(true), std::string("q4_pf_sparse_attn_Q24KV2_EAGER"));
  CHECK_EQ(kq::pf_ple_variant(), std::string("q4_pf_ple_C2048"));
  CHECK_EQ(kq::pf_bf16_slab_variant(10240, 336), std::string("q4_pf_bf16_slab_K10240_N336"));
  CHECK_EQ(kq::pf_int4_slab_variant(2560, 16384), std::string("pf_dequant_slab_K2560_N16384_L0"));
  CHECK_EQ(kq::pf_int4_slab_variant(6144, 2560), std::string("k2_pf_dequant_slab_K6144_N2560"));
  CHECK_EQ(kq::pf_qsa_variant(), std::string("q4_qsa_M2048_T512_W1024_PF"));
  CHECK_EQ(kq::pf_ple_gather_variant(true), std::string("q4_ple_gather_M2048_BF16_PF"));
  CHECK_EQ(kq::pf_hc_up_mix_variant(), std::string("q4_hc_up_mix_M2048_I_D512"));
  CHECK_EQ(kq::pf_hc_combine_norm_variant(kq::HcSrc::Slices, true), std::string("q4_hc_combine_norm_M2048_S1"));
  CHECK_EQ(kq::pf_embed_variant(), std::string("pf_embed_gather_D2560"));
  CHECK_EQ(kq::pf_ab_variant(), std::string("pf_ab_proj_D2560"));
  CHECK_EQ(kq::pf_router_variant(), std::string("pf_moe_router_K2560_N528"));
  CHECK_EQ(kq::pf_gemm_gu_variant(), std::string("pf_moe_gemm_K2560_N1280_SILU"));
  CHECK_EQ(kq::pf_gemm_dn_variant(), std::string("pf_moe_gemm_K640_N2560"));
  CHECK_EQ(kq::pf_gated_head_sig_variant(), std::string("pf_gated_head_SIG"));
  CHECK_EQ(kq::pf_flash_q4_variant(), std::string("pf_flash_attn_Q24KV2"));
  CHECK_EQ(kq::pf_attn_prep_q4_variant(), std::string("pf_attn_prep_q16_Q24KV2"));
  CHECK_EQ(kq::pf_gate_q4_variant(), std::string("pf_attn_Q24KV2"));
  std::set<std::string> bound;
  model::Qwen4ExpDesc intel = model::qwen4exp();
  intel.forms = {model::Q4Form::Bf16, model::Q4Form::Bf16, model::Q4Form::Bf16, 128};
  for (const model::Qwen4ExpDesc* f : {&model::qwen4exp(), static_cast<const model::Qwen4ExpDesc*>(&intel)})
    for (bool eager : {false, true})
      for (bool ple_bf16 : {false, true})
        for (const std::string& v : kq::prefill_variants(*f, eager, ple_bf16)) bound.insert(v);
  for (const std::string& v : bound)
    if (!built.count(v)) {
      std::fprintf(stderr, "the Qwen3.8-Flash-Next prefill walk binds %s, which B70_Q4EXP_PREFILL_KERNELS does not list\n",
                   v.c_str());
      return 1;
    }
  for (const std::string& b : built)
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is listed (B70_Q4EXP_PREFILL_KERNELS) but the prefill walk binds nothing of it\n",
                   b.c_str());
      return 1;
    }
  std::printf("qwen4exp_pf_variant_names_test OK: %zu bound names, %zu binaries\n", bound.size(), built.size());
  return 0;
}
