// Spec 25c: the binaries a Nemotron 3.5 Bolt decode list binds (kernels::bolt::decode_variants - the names
// runtime/bolt/bolt_capture.cc forms) for every head form, attention form, one / two cards (copy, peer), fused arm and SSM
// state precision, on the real descriptor and the synthetic N 12, against the ones src/kernels/CMakeLists.txt's spec 25c
// block builds (passed as arguments: tests/CMakeLists.txt's B70_BOLT_DECODE_KERNELS, `kernel_<name>`; after `--` the
// reused binaries built outside the block). Host only: a name the capture would ask for that no CMake line builds fails
// here on the Mac, not at capture on the box; and nothing in the list is dead.
//
// Decision 15's check: the KV layout spec 23c's mg_attn reads ([pos][kv][128] K and V, kv-head j's row at (pos x 2 + j) x
// 128 - src/kernels/bolt/mg_attn.cl load_k / load_v) is the one nb_attn_prep writes (row x 256 + c, c = j x 128 + d -
// src/kernels/bolt/nb_prep.cl), and its q input is fp32 [M][32][128]: the attention binaries are spec 23c's, by name.
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
  CHECK(d.hidden == kb::kHidden && d.m_heads == kb::kMHeads && d.m_head_dim == kb::kMHeadDim && d.m_groups == kb::kMGroups &&
        d.m_state == kb::kMState && d.m_conv == kb::kMConv && d.m_inproj == kb::kInProjN && d.m_conv_dim == kb::kXbc &&
        d.m_inner == kb::kInner && d.q_heads == kb::kQHeads && d.kv_heads == kb::kKvHeads && d.head_dim == kb::kHd &&
        d.latent == kb::kLatent && d.shared_inter == kb::kShared && d.top_k == kb::kTopK && d.vocab == kb::kVocab &&
        d.qkv_n() == kb::kQkvN && d.max_experts() <= kb::kMaxExperts);
  // Decision 15: nb_attn_prep's K / V element (pos, j, d) at (pos x kv_heads + j) x 128 + d - mg_attn's addressing.
  CHECK(d.kv_n() == kb::kKvHeads * kb::kHd);
  const model::BoltDesc d12 = model::bolt_truncated(d, 12);

  std::set<std::string> bound;
  size_t lists = 0;
  for (const model::BoltDesc* desc : {&d, &d12})
    for (bool int8 : {false, true})
      for (bool eager : {false, true})
        for (bool peer : {false, true})
          for (bool b16 : {false, true})
            for (int arm = 0; arm < 8; ++arm) {   // the as-built list and every combination of the fused arms
              kb::Arms arms;
              arms.router = arm & 1;
              arms.route = arm & 2;
              arms.conv = arm & 4;
              for (const std::string& v : kb::decode_variants(*desc, int8, eager, peer, b16, arms)) bound.insert(v);
              ++lists;
            }
  for (const std::string& v : bound)
    if (!built.count(v) && !reused.count(v)) {
      std::fprintf(stderr, "the Bolt decode list binds %s, which the CMake spec 25c block does not build\n", v.c_str());
      return 1;
    }
  for (const std::string& b : built)
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is built (B70_BOLT_DECODE_KERNELS) but nothing binds it\n", b.c_str());
      return 1;
    }
  // The names carry their shapes: one card, flash, int8 head, no arm, fp32 state on the real model binds these.
  const std::vector<std::string> base = kb::decode_variants(d, true, false, false, false);
  auto has = [&](const char* n) {
    for (const std::string& v : base)
      if (v == n) return true;
    return false;
  };
  CHECK(has("nb_embed_gather_M1_K4096_V131072"));
  CHECK(has("nb_argmax_stage1_M1_V131072"));
  CHECK(has("gemv_i8w_M1_K4096_N131072"));
  CHECK(has("nb_res_norm_M1_K4096_W8_NF"));        // block 0 over the gathered embedding
  CHECK(has("nb_res_norm_M1_K4096_W8_S1_T1"));     // after a MoE block: fc2 and shared_down
  CHECK(has("nb_route_M1_T9"));
  CHECK(has("nb_moe_up_M1_I2560_L1024_T9") && has("nb_moe_up_M1_I1536_L1024_T9"));
  CHECK(has("nb_ssm_M1_H128_P64_N128_G8_S1"));
  CHECK(has("mg_attn_M1_T64_Q32KV2_F"));
  CHECK(has("gemv_M1_K4096_N18560_S1_L0"));
  std::printf("bolt_variant_names_test OK: %zu lists, %zu bound names, %zu binaries + %zu reused\n", lists, bound.size(),
              built.size(), reused.size());
  return 0;
}
