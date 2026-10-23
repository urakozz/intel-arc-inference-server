// Spec 26c Task 1: the binaries Bolt's verify lists bind (kernels::bolt::verify_variants - the names
// runtime/bolt/bolt_capture.cc's build_verify forms) at M = 1..8, for every head form, attention form, SSM state precision and
// MoE form (union / plain), on the real descriptor and the synthetic N 12, against the ones src/kernels/CMakeLists.txt's spec
// 26c block builds (passed as arguments: B70_BOLT_VERIFY_KERNELS, `kernel_<name>`; after `--` the binaries built outside the
// block - 25c's M = 1 decode names and its test-only M 4 lines, 26a's M 2..8 GEMVs / routers / int8 heads / argmax stage 1,
// 26b's bf16 head at M 7, Muse's mg_attn at M 2..8). Host only: a name a verify capture would ask for that no CMake line
// builds fails here on the Mac, not at capture on the box; and nothing in the block is dead but the counting build
// (bolt_moe_union_test's `_CNT`).
//
// Also: the verify lists' shape facts the capture relies on - every row's own binary (`_M<M>` in each name of the list at M,
// the argmax stage 2 M-free), the SSM's slot form at every M and never the in-place one, no fused arm, the union form only
// above M 1, and the plain form's names under B70_BOLT_VERIFY_MOE=plain.
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/bolt_kernels.h"
#include "model/bolt.h"

int main(int argc, char** argv) {
  namespace kb = kernels::bolt;
  std::set<std::string> built, reused;
  bool after = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--") {
      after = true;
      continue;
    }
    CHECK(a.rfind("kernel_", 0) == 0);
    (after ? reused : built).insert(a.substr(7));
  }
  CHECK(!built.empty());
  const model::BoltDesc& d = model::bolt();
  const model::BoltDesc d12 = model::bolt_truncated(d, 12);
  CHECK(kb::kSlotStride == 1048576u);   // the CMake block's SPEC_SLOT_STRIDE
  CHECK(kb::kVerifyRows == 8 && kb::kSpecSlots == 8);

  std::set<std::string> bound;
  size_t lists = 0;
  for (const model::BoltDesc* desc : {&d, &d12})
    for (unsigned M = 1; M <= kb::kVerifyRows; ++M)
      for (bool int8 : {false, true})
        for (bool eager : {false, true})
          for (bool b16 : {false, true})
            for (bool un : {false, true}) {
              const std::vector<std::string> v = kb::verify_variants(*desc, M, int8, eager, b16, un);
              for (const std::string& n : v) {
                bound.insert(n);
                // every name but argmax stage 2 carries this list's rows
                if (n != kb::argmax2_variant()) {
                  const std::string tag = "_M" + std::to_string(M) + "_";
                  if (n.find(tag) == std::string::npos) {
                    std::fprintf(stderr, "the M = %u verify list binds %s, not an M = %u binary\n", M, n.c_str(), M);
                    return 1;
                  }
                }
                CHECK(n.size() < 2 || n.substr(n.size() - 2) != "_R");                     // no fused arm `_R`
                CHECK(n.rfind("nb_ssm", 0) != 0 || n.find("_SL8") != std::string::npos);  // the slot form only
                CHECK(n.find("_C_") == std::string::npos && (n.size() < 2 || n.substr(n.size() - 2) != "_C"));
                if (n.find("_U_M") != std::string::npos) CHECK(un && M > 1);
              }
              ++lists;
            }
  for (const std::string& v : bound)
    if (!built.count(v) && !reused.count(v)) {
      std::fprintf(stderr, "a Bolt verify list binds %s, which neither the CMake spec 26c block nor the reused list builds\n",
                   v.c_str());
      return 1;
    }
  size_t test_only = 0;
  for (const std::string& b : built) {
    if (b.size() > 4 && b.substr(b.size() - 4) == "_CNT") {   // the counting build: bolt_moe_union_test only
      ++test_only;
      continue;
    }
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is built (B70_BOLT_VERIFY_KERNELS) but no verify list binds it\n", b.c_str());
      return 1;
    }
  }
  CHECK_EQ(test_only, size_t(2));
  // The names carry their shapes: M 8, flash, int8 head, fp32 state, union on the real model binds these.
  const std::vector<std::string> v8 = kb::verify_variants(d, 8, true, false, false, true);
  auto has = [&](const std::vector<std::string>& v, const std::string& n) {
    for (const std::string& x : v)
      if (x == n) return true;
    return false;
  };
  CHECK(has(v8, "nb_embed_gather_M8_K4096_V131072"));
  CHECK(has(v8, "gemv_i8w_M8_K4096_N131072"));
  CHECK(has(v8, "nb_argmax_stage1_M8_V131072") && has(v8, "nb_argmax_stage2_V131072"));
  CHECK(has(v8, "nb_ssm_M8_H128_P64_N128_G8_S1_SL8") && !has(v8, "nb_ssm_M8_H128_P64_N128_G8_S1"));
  CHECK(has(v8, "nb_moe_up_U_M8_I2560_L1024_T9") && has(v8, "nb_moe_down_U_M8_I1536_L1024_T9"));
  CHECK(has(v8, "nb_moe_comb_U_M8_L1024_T9") && !has(v8, "nb_moe_up_M8_I2560_L1024_T9"));
  CHECK(has(v8, "mg_attn_M8_T64_Q32KV2_F") && has(v8, "gemv_M8_K4096_N18560_S1_L0"));
  const std::vector<std::string> p8 = kb::verify_variants(d, 8, true, false, false, false);
  CHECK(has(p8, "nb_moe_up_M8_I2560_L1024_T9") && has(p8, "nb_moe_down_M8_I2560_L1024_T9"));
  CHECK(!has(p8, "nb_moe_comb_U_M8_L1024_T9"));
  // M = 1 (the drafter's plain token) is 25c's M = 1 list but the slot SSM: union never, the slot form always.
  const std::vector<std::string> v1 = kb::verify_variants(d, 1, true, false, false, true);
  CHECK(has(v1, "nb_ssm_M1_H128_P64_N128_G8_S1_SL8") && has(v1, "nb_moe_up_M1_I2560_L1024_T9"));
  CHECK(!has(v1, "nb_moe_up_U_M1_I2560_L1024_T9"));
  std::printf("bolt_verify_names_test OK: %zu lists, %zu bound names, %zu binaries (%zu test-only) + %zu reused\n", lists,
              bound.size(), built.size(), test_only, reused.size());
  return 0;
}
