// Kolibri DSpark (host): every binary the Kolibri verify lists (kernels::kolibri::verify_variants, M = 2..8, both attention
// arms and both head forms) and the DSpark drafter's lists (kernels::dflash::dspark_draft_variants, both head forms) bind is
// a compiled file in the build's kernel directory - the CMake Kolibri DSpark block builds exactly what the captures name. And
// the names carry what their code bakes: the drafter's attention is the causal build (`_C`), its top-k leaves out the two
// rows past Kolibri's tokenizer, the verify's attention halves are kol_attn.cl's M <= 4 builds. No device.
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/dflash_kernels.h"
#include "kernels/kernels.h"
#include "kernels/kolibri_kernels.h"

int main() {
  namespace kk = kernels::kolibri;
  namespace kd = kernels::dflash;
  size_t n = 0;
  auto built = [&](const std::string& v) {
    if (!std::ifstream(kernels::path(v)).good()) {
      std::fprintf(stderr, "not compiled: %s (%s)\n", v.c_str(), kernels::path(v).c_str());
      std::exit(1);
    }
    ++n;
  };
  for (unsigned M = 2; M <= kk::kVerifyRows; ++M)
    for (model::KolAttnForm a : {model::KolAttnForm::Int4, model::KolAttnForm::Bf16})
      for (bool int8 : {false, true})
        for (const std::string& v : kk::verify_variants(a, int8, M)) built(v);
  for (bool int8 : {false, true})
    for (const std::string& v : kd::dspark_draft_variants(int8)) built(v);

  CHECK(kk::verify_attn_rows(3, 0) == 3 && kk::verify_attn_rows(3, 1) == 0);
  CHECK(kk::verify_attn_rows(8, 0) == 4 && kk::verify_attn_rows(8, 1) == 4);
  CHECK(kk::verify_attn_rows(5, 0) == 4 && kk::verify_attn_rows(5, 1) == 1);
  CHECK(kd::dspark_attn_variant() == "df_attn_M8_Q20KV4_D128_R2048_W2048_P128_C");
  CHECK(kd::dspark_topk_variant() == "df_topk_R7_N128000_U127998");
  CHECK(kd::select_variant(true, kd::kDSpark) == "df_select_M8_K16_Q16_RK256_V128000");
  CHECK(kd::prep_variant(kd::kDSpark) == "df_prep_M8_H2560_I6144_Q20KV4_D128_G16T2_R2048_V128000");
  std::printf("dspark_names_test: %zu bindings, every binary compiled: OK\n", n);
  return 0;
}
