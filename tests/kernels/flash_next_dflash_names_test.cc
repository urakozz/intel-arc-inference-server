// PERF-53 (plan 21-QFN-v3b, host): every binary Flash-Next's DFlash 2 drafter lists bind (kernels::dflash::
// flash_next_draft_variants, both head forms) is a compiled file in the build's kernel directory - the CMake PERF-53 block
// plus the df_attn / df_topk / df_select binaries spec 19b's block builds for Qwen3.8's drafter (the same 32 / 8 x 128
// attention, the same 248320-row head and selector). And the names carry what their code bakes: df_prep and
// df_proj_wide at hidden 2560 / FF 7680, the fc at K 12800 (5 taps x 2560), the conv kernel projections at N 640, the
// selector's at N 256. No device.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "check.h"
#include "kernels/dflash_kernels.h"
#include "kernels/kernels.h"

int main() {
  namespace kd = kernels::dflash;
  size_t n = 0;
  auto built = [&](const std::string& v) {
    if (!std::ifstream(kernels::path(v)).good()) {
      std::fprintf(stderr, "not compiled: %s (%s)\n", v.c_str(), kernels::path(v).c_str());
      std::exit(1);
    }
    ++n;
  };
  for (bool int8 : {false, true})
    for (const std::string& v : kd::flash_next_draft_variants(int8)) built(v);
  CHECK(kd::prep_variant(kd::kFlashNext) == "df_prep_M8_H2560_I7680_Q32KV8_D128_G16T2_R2048_V248320");
  CHECK(kd::proj_wide_variant(kd::kFlashNext) == "df_proj_wide_H2560_KV8_D128_R2048");
  CHECK(kd::attn_variant(kd::kFlashNext, 1, 4) == kd::attn_variant(kd::kQwen38, 1, 4));
  CHECK(kd::attn_dpas_variant(kd::kFlashNext, 4) == "df_attn_M8_Q32KV8_D128_R2048_W2048_P128_dpas_k4_g256");
  CHECK(kd::select_variant(true, kd::kFlashNext) == "df_select_M8_K16_Q16_RK256_V248320");
  const std::vector<std::string> v = kd::flash_next_draft_variants(true);
  for (const char* want : {"gemv_bf16_M8_K12800_N2560", "gemv_bf16_M8_K2560_N640_C16_S16", "gemv_bf16_M8_K2560_N256_C16_S16",
                           "gemv_bf16_M8_K2560_N15360", "gemv_i8w_M7_K2560_N248320"}) {
    bool found = false;
    for (const std::string& s : v) found = found || s == want;
    CHECK(found);
  }
  std::printf("flash_next_dflash_names_test: %zu bindings, every binary compiled: OK\n", n);
  return 0;
}
