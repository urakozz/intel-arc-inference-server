// Spec 24c: the binaries an LFM2.5-24B-A2B decode list binds (kernels::lfm2moe::decode_variants - the names
// runtime/lfm2moe/lfm2moe_capture.cc forms) for every head form, attention form, one / two cards and fused arm, on
// the real descriptor and the synthetic N 4, against the ones src/kernels/CMakeLists.txt's spec 24c block builds
// (passed as arguments: tests/CMakeLists.txt's B70_LFM2MOE_DECODE_KERNELS, `kernel_<name>`; after `--` the reused
// binaries built outside the block). Host only: a name the capture would ask for that no CMake line builds fails here
// on the Mac, not at capture on the box; and nothing in the list is dead.
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/lfm2moe_kernels.h"
#include "model/lfm2moe.h"

int main(int argc, char** argv) {
  namespace kl = kernels::lfm2moe;
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
  const model::Lfm2MoeDesc& d = model::lfm2moe();
  CHECK(d.hidden == kl::kHidden && d.qkv_n() == kl::kQkvN && 3 * d.hidden == kl::kInProjN && d.head_dim == kl::kHd &&
        d.q_heads == kl::kQHeads && d.kv_heads == kl::kKvHeads && d.experts == kl::kExperts && d.top_k == kl::kTopK &&
        d.moe_inter == kl::kInter && d.dense_inter == kl::kDenseInter && d.vocab == kl::kVocab &&
        d.vocab_used == kl::kVocabUsed);
  model::Lfm2MoeDesc d4 = d;
  d4.layers = 4;
  d4.kinds.assign(d.kinds.begin(), d.kinds.begin() + 4);

  std::set<std::string> bound;
  size_t lists = 0;
  for (const model::Lfm2MoeDesc* desc : {&d, static_cast<const model::Lfm2MoeDesc*>(&d4)})
    for (bool int8 : {false, true})
      for (bool eager : {false, true})
        for (bool two : {false, true})
          for (int arm = 0; arm < 1; ++arm) {   // the as-built list (Task 5 adds the arms)
            kl::Arms arms;
            arms.route = arm & 1;
            arms.router = arm & 2;
            arms.conv = arm & 4;
            for (const std::string& v : kl::decode_variants(*desc, int8, eager, two, arms)) bound.insert(v);
            ++lists;
          }
  for (const std::string& v : bound)
    if (!built.count(v) && !reused.count(v)) {
      std::fprintf(stderr, "the LFM2 decode list binds %s, which the CMake spec 24c block does not build\n", v.c_str());
      return 1;
    }
  for (const std::string& b : built)
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is built (B70_LFM2MOE_DECODE_KERNELS) but nothing binds it\n", b.c_str());
      return 1;
    }
  // The names carry their shapes: one card, flash, int8 head, no arm on the real model binds these (spot checks).
  const std::vector<std::string> base = kl::decode_variants(d, true, false, false);
  auto has = [&](const char* n) {
    for (const std::string& v : base)
      if (v == n) return true;
    return false;
  };
  CHECK(has("lfm_embed_gather_M1_K2048_V128000"));
  CHECK(has("lfm_argmax_stage1_M1_V128000_U125017"));
  CHECK(has("gemv_i8w_M1_K2048_N128000"));
  CHECK(has("lfm_route_M1_E64_T4"));
  CHECK(has("lfm_moe_M1_E64_T4_D2048_I1536"));
  CHECK(has("lfm_attn_M1_T32_Q32KV8"));
  CHECK(has("lfm_res_norm_M1_K2048_W8_NF"));
  std::printf("lfm2moe_variant_names_test OK: %zu lists, %zu bound names, %zu binaries + %zu reused\n", lists, bound.size(),
              built.size(), reused.size());
  return 0;
}
