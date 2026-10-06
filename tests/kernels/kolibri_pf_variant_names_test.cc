// Spec 20d: the binaries a Kolibri-1 prefill chunk binds (kernels::kolibri::prefill_variants - the names
// runtime/kolibri/kolibri_prefill.cc forms) for both attention arms and both attention forms, against
// the ones the build makes for it (passed as arguments: tests/CMakeLists.txt's
// B70_KOLIBRI_PREFILL_KERNELS, `kernel_<name>` - src/kernels/CMakeLists.txt's spec 20d block, plus the
// reused names other blocks build: pf_gemm_T0, 20c's kol_route, K2's pf_res_fold SP0 and 6144 x 2560
// slab). Host only: a name the walk would ask for that no CMake line builds fails here on the Mac, not at
// prepare_prefill on the box; and nothing listed is dead. Also pins the prefill constants the walk and
// the planner share with the kernels' names.
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/kolibri_kernels.h"
#include "model/kolibri1.h"

int main(int argc, char** argv) {
  namespace kk = kernels::kolibri;
  std::set<std::string> built;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    CHECK(a.rfind("kernel_", 0) == 0);
    built.insert(a.substr(7));
  }
  CHECK(!built.empty());
  CHECK_EQ(kk::kPfC, 2048u);
  CHECK_EQ(kk::pf_hdr::words(), 400u);
  CHECK(model::Kolibri1Desc::kRing >= kk::kPfC + kk::kWindow - 1 && model::Kolibri1Desc::kRing % kk::kPfKt == 0);
  CHECK_EQ(kk::pf_moe_variant(), std::string("kol_pf_moe_E384_T6_D2560_I512_L256"));
  CHECK_EQ(kk::pf_flash_variant(true, false), std::string("kol_pf_flash_attn_Q48KV4_W513_R4096"));
  CHECK_EQ(kk::pf_flash_variant(false, true), std::string("kol_pf_flash_attn_Q48KV4_F_EAGER"));
  CHECK_EQ(kk::pf_bf16_slab_variant(6144, 2560), std::string("kol_pf_bf16_slab_K6144_N2560"));
  std::set<std::string> bound;
  for (model::KolAttnForm a : {model::KolAttnForm::Int4, model::KolAttnForm::Bf16})
    for (bool eager : {false, true})
      for (const std::string& v : kk::prefill_variants(a, eager)) bound.insert(v);
  for (const std::string& v : bound)
    if (!built.count(v)) {
      std::fprintf(stderr, "the Kolibri prefill walk binds %s, which B70_KOLIBRI_PREFILL_KERNELS does not list\n",
                   v.c_str());
      return 1;
    }
  for (const std::string& b : built)
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is listed (B70_KOLIBRI_PREFILL_KERNELS) but the prefill walk binds nothing of it\n",
                   b.c_str());
      return 1;
    }
  std::printf("kolibri_pf_variant_names_test OK: %zu bound names, %zu binaries\n", bound.size(), built.size());
  return 0;
}
