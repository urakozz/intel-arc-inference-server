// Spec 23d: the binaries a Muse-Glimmer prefill chunk binds (kernels::muse::prefill_variants - the names
// runtime/muse/muse_prefill.cc forms) for both attention forms, against the ones the build makes for it (passed as
// arguments: tests/CMakeLists.txt's B70_MUSE_PREFILL_KERNELS, `kernel_<name>` - src/kernels/CMakeLists.txt's spec 23d
// block, plus the shape-free pf_gemm_T0 prefill/ builds). Host only: a name the walk would ask for that no CMake line
// builds fails here on the Mac, not at prepare_prefill on the box; and nothing listed is dead. Also pins the prefill
// constants the walk and the planner share with the kernels' names, and 23c's decode list (MG0: unchanged by 23d).
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/muse_kernels.h"
#include "model/muse_glimmer.h"

int main(int argc, char** argv) {
  namespace km = kernels::muse;
  std::set<std::string> built;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    CHECK(a.rfind("kernel_", 0) == 0);
    built.insert(a.substr(7));
  }
  CHECK(!built.empty());
  CHECK_EQ(km::kPfC, 2048u);
  CHECK_EQ(km::kPfFlashWg, 64u);
  CHECK(model::MuseGlimmerDesc::kRing >= km::kPfC + km::kWindow - 1 && model::MuseGlimmerDesc::kRing % km::kPfKt == 0);
  CHECK_EQ(km::pf_flash_variant(true, false), std::string("mg_pf_flash_attn_Q32KV2_W2048_R4096"));
  CHECK_EQ(km::pf_flash_variant(false, true), std::string("mg_pf_flash_attn_Q32KV2_F_EAGER"));
  CHECK_EQ(km::pf_attn_prep_variant(true), std::string("mg_attn_prep_M2048_N8704_S1_Q32KV2_R4096"));
  CHECK_EQ(km::pf_fold_zero_variant(), std::string("pf_res_fold_K6656_SP1_G26_Z"));
  CHECK_EQ(km::pf_slab_variant(6656, 39936), std::string("k2_pf_dequant_slab_K6656_N39936"));
  CHECK_EQ(km::pf_embed_variant(), std::string("mg_pf_embed_M2048_K6656_G26_V202048"));
  CHECK_EQ(km::pf_gate_variant(), std::string("mg_pf_attn_gate_M2048_N4096"));
  // 23c's decode list is unchanged (MG0's host half): 17 names, the flash form's sliding attention binary 7th.
  CHECK_EQ(km::decode_variants(true, false).size(), size_t(17));
  CHECK_EQ(km::decode_variants(true, false)[6], std::string("mg_attn_M1_T64_Q32KV2_W2048_R4096"));
  std::set<std::string> bound;
  for (bool eager : {false, true})
    for (const std::string& v : km::prefill_variants(eager)) bound.insert(v);
  for (const std::string& v : bound)
    if (!built.count(v)) {
      std::fprintf(stderr, "the Muse prefill walk binds %s, which B70_MUSE_PREFILL_KERNELS does not list\n", v.c_str());
      return 1;
    }
  for (const std::string& b : built)
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is listed (B70_MUSE_PREFILL_KERNELS) but the prefill walk binds nothing of it\n", b.c_str());
      return 1;
    }
  std::printf("muse_pf_variant_names_test OK: %zu bound names, %zu binaries\n", bound.size(), built.size());
  return 0;
}
