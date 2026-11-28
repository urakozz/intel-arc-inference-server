// Plan 41k Task 1 (host): every binary Inkling 2 Mini's DFlash 2 drafter lists bind (kernels::dflash::
// inkling_dflash2_variants, the packed int4 and int8 linears at its shapes) is a compiled file in the build's kernel directory - the
// CMake plan 41k block plus the df_attn binaries spec 19b's block builds for Qwen3.8's drafter (the same 32 / 8 x 128
// attention over a 2048 ring). And the names carry what their code bakes: df_prep and df_proj_wide at hidden 2304 /
// FF 6912, the fc at K 9216 (four taps x 2304), df_select's cap, df_rerank over the head's 200064 rows. Qwen3.8's and
// Flash-Next's names do not move with the tap count. No device.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "check.h"
#include "kernels/dflash_kernels.h"
#include "kernels/inkling2_kernels.h"
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
  for (bool int4 : {false, true})
    for (const std::string& v : kd::inkling_dflash2_variants(int4)) built(v);
  for (unsigned rank = 0; rank < 2; ++rank) {   // tp: the head by rows
    built(kd::inkling_tp_candidate_variant(rank));
    built(kd::inkling_tp_rerank_variant(rank));
  }
  built(kernels::gemm_sm_variant(7, *kernels::small_m_tuning(2304, kd::kInklingHeadRows / 2, 1, 3)));
  // the int8 linears (L1) as the engine binds them: gemm_sm_i8w at 2..8 rows, gemv_i8w at one
  for (const auto& [K, N] : kd::inkling_linear_shapes()) built(kernels::inkling2::int8(8, K, N));
  for (unsigned m = 1; m <= 8; ++m) {   // the context projection at every committed width
    built(kernels::inkling2::int8(m, kd::inkling_fc_shape().first, kd::inkling_fc_shape().second));
    built(kernels::inkling2::int8(m, kd::inkling_context_kv_shape().first, kd::inkling_context_kv_shape().second));
  }
  CHECK(kd::prep_variant(kd::kInkling) == "df_prep_M8_H2304_I6912_Q32KV8_D128_G16T2_R2048_V201024");
  CHECK(kd::proj_wide_variant(kd::kInkling) == "df_proj_wide_H2304_KV8_D128_R2048");
  CHECK(kd::attn_variant(kd::kInkling, 1, 4) == kd::attn_variant(kd::kQwen38, 1, 4));
  CHECK(kd::attn_dpas_variant(kd::kInkling, 2) == "df_attn_M8_Q32KV8_D128_R2048_W2048_P128_dpas_k2_g256_blk");
  CHECK(kd::inkling_select_variant() == "df_select_M8_K16_Q16_RK256_V201024_CAP30");
  CHECK(kd::inkling_select_variant(true) == "df_select_M8_K16_Q16_RK256_V201024_H12_CAP30");
  CHECK(kd::prep_variant(kd::kInkling, true) == "df_prep_M8_H2304_I6912_Q32KV8_D128_G16T2_R2048_V201024_F8Q4R4U4C12H12K4");
  CHECK(kd::inkling_rerank_variant() == "df_rerank_R7_K2304_U200058_C64_Q4");
  CHECK(kd::inkling_tp_candidate_variant(0) == "df_candidate_R7_N100032_U100032_K64" &&
        kd::inkling_tp_candidate_variant(1) == "df_candidate_R7_N100032_U100026_K64");
  CHECK(kd::inkling_tp_rerank_variant(0) == "df_rerank_R7_K2304_U100032_C64_Q4" &&
        kd::inkling_tp_rerank_variant(1) == "df_rerank_R7_K2304_U100026_C64_Q4");
  CHECK(kd::kInkling.ntaps * kd::kInkling.H == 9216 && kd::kQwen38.ntaps == 5 && kd::kFlashNext.ntaps == 5);
  CHECK(kd::select_variant(true, kd::kFlashNext) == "df_select_M8_K16_Q16_RK256_V248320");
  std::printf("inkling2_dflash_names_test: %zu bindings, every binary compiled: OK\n", n);
  return 0;
}
