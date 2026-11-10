// verify_attn_test - spec 28c's capture rule (src/runtime/verify_attn.h), host only:
//   1. M = 1 -> V2 under every override, at every depth, both KV forms;
//   2. heads other than Qwen3.8's 24 / 4 (Ornith 16 / 2) -> V2 under every override;
//   3. the test / probe override (CaptureOptions::force_attn - no environment variable): Auto -> the rule, V2 -> V2,
//      Dpas -> Dpas where a table row exists (V2 where none);
//   4. each table row's threshold: V2 at dpas_from - 1, Dpas at dpas_from (never: V2 at every depth);
//   5. verify_attn_depth_free: true exactly when dpas_from is 0 or never; verify_attn_deep_from (the engines' second
//      list) is dpas_from exactly where it is finite and > 1, never elsewhere and at Ornith's heads;
//   6. every (M, KV form) the rule can return Dpas for has its binary on src/kernels/CMakeLists.txt's B70_VERIFY_DPAS
//      lines (the built names passed as arguments by tests/CMakeLists.txt), and an (M, KV form) without a row
//      returns V2 (M = 5..8 int8, M = 9);
//   7. spec 17d: a tensor-parallel card's 12 / 2 heads follow the same table at every row and override, and the
//      MTP verify rows (M = 2..4, both forms) have their per-card binary built (the 17d block).
#include <cstdio>
#include <set>
#include <string>

#include "check.h"
#include "kernels/kernels.h"
#include "runtime/gemv_kind.h"
#include "runtime/verify_attn.h"

using runtime::VerifyAttn;
using runtime::VerifyAttnForce;

int main(int argc, char** argv) {
  std::set<std::string> built;
  for (int i = 1; i < argc; ++i) built.insert(argv[i]);
  const VerifyAttnForce ovs[] = {VerifyAttnForce::Auto, VerifyAttnForce::V2, VerifyAttnForce::Dpas};
  const uint32_t depths[] = {1, 512, 4096, 32768, 130816};
  // 1, 2
  for (VerifyAttnForce ov : ovs)
    for (uint32_t keys : depths)
      for (bool kv8 : {false, true}) {
        CHECK(runtime::verify_attn_kind(1, keys, kv8, 24, 4, ov) == VerifyAttn::V2);
        for (uint32_t M = 1; M <= 8; ++M) CHECK(runtime::verify_attn_kind(M, keys, kv8, 16, 2, ov) == VerifyAttn::V2);
      }
  // 3: the options' default is the rule
  CHECK(runtime::CaptureOptions{}.force_attn == VerifyAttnForce::Auto);
  // 4, 5, 6
  size_t rows = 0;
  for (const runtime::VerifyAttnRow& r : runtime::kVerifyAttnTable) {
    ++rows;
    CHECK(r.m >= 2 && r.m <= 8);
    CHECK(!r.kv8 || r.m <= 4);
    const std::string name = kernels::attn_verify_dpas_variant(r.m, 32, r.kv8);
    CHECK(built.count(name) == 1);
    CHECK(runtime::verify_attn_kind(r.m, 4096, r.kv8, 24, 4, VerifyAttnForce::Dpas) == VerifyAttn::Dpas);
    CHECK(runtime::verify_attn_kind(r.m, 4096, r.kv8, 24, 4, VerifyAttnForce::V2) == VerifyAttn::V2);
    if (r.dpas_from == runtime::kDpasNever) {
      for (uint32_t keys : depths)
        CHECK(runtime::verify_attn_kind(r.m, keys, r.kv8, 24, 4, VerifyAttnForce::Auto) == VerifyAttn::V2);
      CHECK(runtime::verify_attn_depth_free(r.m, r.kv8));
    } else {
      CHECK(runtime::verify_attn_kind(r.m, r.dpas_from, r.kv8, 24, 4, VerifyAttnForce::Auto) == VerifyAttn::Dpas);
      CHECK(runtime::verify_attn_kind(r.m, 131072, r.kv8, 24, 4, VerifyAttnForce::Auto) == VerifyAttn::Dpas);
      if (r.dpas_from > 0)
        CHECK(runtime::verify_attn_kind(r.m, r.dpas_from - 1, r.kv8, 24, 4, VerifyAttnForce::Auto) == VerifyAttn::V2);
      CHECK(runtime::verify_attn_depth_free(r.m, r.kv8) == (r.dpas_from <= 1));
    }
    const bool two = r.dpas_from != runtime::kDpasNever && r.dpas_from > 1;
    CHECK(runtime::verify_attn_deep_from(r.m, r.kv8, 24, 4) == (two ? r.dpas_from : runtime::kDpasNever));
    CHECK(runtime::verify_attn_deep_from(r.m, r.kv8, 16, 2) == runtime::kDpasNever);
    // 7: the per-card heads, row for row the one-card rule
    for (VerifyAttnForce ov : ovs)
      for (uint32_t keys : depths)
        CHECK(runtime::verify_attn_kind(r.m, keys, r.kv8, 12, 2, ov) == runtime::verify_attn_kind(r.m, keys, r.kv8, 24, 4, ov));
    CHECK(runtime::verify_attn_deep_from(r.m, r.kv8, 12, 2) == runtime::verify_attn_deep_from(r.m, r.kv8, 24, 4));
    if (r.m <= 4) CHECK(built.count(kernels::attn_verify_dpas_variant(r.m, 32, r.kv8, 12, 2)) == 1);
    std::printf("row M=%u %s: dpas_from %s\n", r.m, r.kv8 ? "int8" : "bf16",
                r.dpas_from == runtime::kDpasNever ? "never" : std::to_string(r.dpas_from).c_str());
  }
  CHECK(rows == 10);   // bf16 at M = 2..8, int8 at M = 2..4
  for (uint32_t M : {5u, 6u, 7u, 8u}) {
    CHECK(runtime::verify_attn_row(M, true) == nullptr);
    CHECK(runtime::verify_attn_kind(M, 32768, true, 24, 4, VerifyAttnForce::Dpas) == VerifyAttn::V2);
  }
  CHECK(runtime::verify_attn_kind(9, 32768, false, 24, 4, VerifyAttnForce::Dpas) == VerifyAttn::V2);
  CHECK(runtime::verify_attn_kind(16, 32768, false, 24, 4, VerifyAttnForce::Dpas) == VerifyAttn::V2);
  std::printf("verify_attn_test: PASS (%zu table rows, %zu built names)\n", rows, built.size());
  return 0;
}
