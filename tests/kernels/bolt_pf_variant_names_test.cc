// Spec 25d: the binaries a Nemotron 3.5 Bolt prefill chunk binds (kernels::bolt::prefill_variants - the names runtime/bolt/
// bolt_prefill.cc forms) for both attention forms and both SSM state precisions, on the real descriptor and the synthetic
// N 12, plus the prefill head's (prefill_head_variants: decode's binaries), against the ones src/kernels/CMakeLists.txt's
// spec 25d block builds (passed as arguments: tests/CMakeLists.txt's B70_BOLT_PREFILL_KERNELS, `kernel_<name>`; after the
// first `--` the binaries built outside the block that a chunk binds - pf_gemm_T0, Muse's mg_pf_flash_attn pair, 25c's
// decode binaries the head reuses -; after the second `--` the block's test-only builds: the second tiling). Host only.
//
// And 25c's decode names, pinned (plan 25d's B0: 25d adds binaries, it renames none): the decode list's union over every
// head / attention / card / arm / state form is still 41 names, these among them.
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/bolt_kernels.h"
#include "model/bolt.h"

int main(int argc, char** argv) {
  namespace kb = kernels::bolt;
  std::set<std::string> built, reused, test_only;
  int part = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--") {
      ++part;
      continue;
    }
    CHECK(a.rfind("kernel_", 0) == 0);
    (part == 0 ? built : part == 1 ? reused : test_only).insert(a.substr(7));
  }
  CHECK(!built.empty());
  const model::BoltDesc& d = model::bolt();
  const model::BoltDesc d12 = model::bolt_truncated(d, 12);

  std::set<std::string> bound;
  size_t lists = 0;
  for (const model::BoltDesc* desc : {&d, &d12})
    for (bool eager : {false, true})
      for (bool b16 : {false, true}) {
        for (const std::string& v : kb::prefill_variants(*desc, eager, b16)) bound.insert(v);
        ++lists;
      }
  for (const model::BoltDesc* desc : {&d, &d12})
    for (bool int8 : {false, true})
      for (const std::string& v : kb::prefill_head_variants(*desc, int8)) bound.insert(v);
  for (const std::string& v : bound) {
    CHECK(!v.empty());   // a linear no slab binary covers (layout 1 with a tail) - the descriptor's cells are all covered
    if (!built.count(v) && !reused.count(v)) {
      std::fprintf(stderr, "the Bolt prefill binds %s, which neither the CMake spec 25d block nor the reused list builds\n",
                   v.c_str());
      return 1;
    }
  }
  for (const std::string& b : built)
    if (!bound.count(b) && !test_only.count(b)) {
      std::fprintf(stderr, "%s is built (B70_BOLT_PREFILL_KERNELS) but nothing binds it\n", b.c_str());
      return 1;
    }
  for (const std::string& t : test_only) CHECK(!bound.count(t));   // built only for the kernel tests
  // The names carry their shapes: the real model's flash / fp32 chunk binds these.
  const std::vector<std::string> base = kb::prefill_variants(d, false, false);
  auto has = [&](const std::vector<std::string>& v, const char* n) {
    for (const std::string& x : v)
      if (x == n) return true;
    return false;
  };
  CHECK(has(base, "nb_pf_embed_gather_K4096_V131072"));
  CHECK(has(base, "nb_res_norm_M2048_K4096_W8_NF") && has(base, "nb_res_norm_M2048_K4096_W8_S1") &&
        has(base, "nb_res_norm_M2048_K4096_W8_S1_T1"));
  CHECK(has(base, "k2_pf_dequant_slab_K4096_N18560") && has(base, "pf_dequant_slab_K8192_N4096_L1"));
  CHECK(has(base, "nb_pf_mamba_L64_W16_R16") && has(base, "nb_pf_cb_L64_DPAS"));
  CHECK(has(base, "nb_pf_moe_I2560_L1024_T9_TM32") && has(base, "nb_pf_moe_gemm_K1024_N2560_RELU2") &&
        has(base, "nb_pf_moe_gemm_K2560_N1024"));
  CHECK(has(base, "mg_pf_flash_attn_Q32KV2_F") && has(base, "nb_route_M2048_T9") && has(base, "pf_gemm_T0"));
  CHECK(has(kb::prefill_head_variants(d, true), "nb_res_norm_M1_K4096_W8_S1_T1"));   // block 87 is MoE

  // 25c's decode names, unchanged (the union bolt_variant_names_test holds to the 25c block)
  std::set<std::string> dec;
  for (const model::BoltDesc* desc : {&d, &d12})
    for (bool int8 : {false, true})
      for (bool eager : {false, true})
        for (bool peer : {false, true})
          for (bool b16 : {false, true})
            for (int arm = 0; arm < 8; ++arm) {
              kb::Arms arms;
              arms.router = arm & 1;
              arms.route = arm & 2;
              arms.conv = arm & 4;
              for (const std::string& v : kb::decode_variants(*desc, int8, eager, peer, b16, arms)) dec.insert(v);
            }
  CHECK(dec.size() == 41);
  for (const char* n : {"nb_embed_gather_M1_K4096_V131072", "nb_res_norm_M1_K4096_W8_NF", "nb_res_norm_M1_K4096_W8_S1_T1",
                        "nb_res_norm_M1_K4096_W8_S1_R", "nb_conv_M1_C10240_S1", "nb_ssm_M1_H128_P64_N128_G8_S1_C_B16",
                        "nb_gnorm_M1_D8192_G1024_S1", "nb_router_M1_K4096", "nb_route_M1_T9",
                        "nb_moe_up_M1_I2304_L1024_T9_R", "nb_moe_down_M1_I1536_L1024_T9", "nb_sh_act_M1_L1024_SH5376_S1",
                        "nb_attn_prep_M1_KV2_S1", "mg_attn_M1_T64_Q32KV2_F", "mg_attn_eager_M1_T64_Q32KV2_F",
                        "nb_argmax_stage1_M1_V131072", "nb_argmax_stage2_V131072", "gemv_M1_K4096_N18560_S1_L0",
                        "pp_handoff"})
    CHECK(dec.count(n));
  std::printf("bolt_pf_variant_names_test OK: %zu prefill lists, %zu bound names, %zu binaries + %zu reused + %zu test-only; "
              "25c's 41 decode names unchanged\n", lists, bound.size(), built.size(), reused.size(), test_only.size());
  return 0;
}
