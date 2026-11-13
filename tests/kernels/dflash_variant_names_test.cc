// Spec 19b: every binary the DFlash lists bind (kernels::dflash::draft_variants, every drafter form x head form x
// draft vocabulary, both versions) is built, and the Spec 19b CMake block builds nothing they do not bind. Also:
// kernels::dflash::kQwen38 - the shapes the names spell - is z-lab/Qwen3.8-27B-DFlash2's config beside Qwen3.8.
// Host only (names, no device).
//
//   dflash_variant_names_test <config.json> kernel_<new>... -- kernel_<reused>...
#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "common/json.h"
#include "kernels/dflash_kernels.h"
#include "model/dflash.h"
#include "model/model_desc.h"
#include "runtime/gemv_kind.h"

int main(int argc, char** argv) {
  namespace kd = kernels::dflash;
  const runtime::CaptureOptions defaults;
  CHECK_EQ(defaults.dflash_attn_heads, 1u);
  CHECK_EQ(defaults.dflash_attn_rows, 4u);
  CHECK(!defaults.dflash_attn_parallel_exp);
  CHECK(defaults.dflash_attn_dpas);
  const auto default_names = kd::draft_variants(true, 1u, true, 0u, kernels::LinearKind::SmallM, kernels::LinearKind::SmallM);
  CHECK(std::find(default_names.begin(), default_names.end(), kd::attn_variant(kd::kQwen38, 1, 4)) != default_names.end());
  CHECK(argc > 2);
  std::ifstream f(argv[1]);
  std::stringstream ss;
  ss << f.rdbuf();
  const model::DFlashDesc d = model::dflash_desc_from_config(common::json::parse(ss.str()));
  const kd::Shape& s = kd::kQwen38;
  CHECK(d.hidden == s.H && d.inter == s.I && d.q_heads == s.NH && d.kv_heads == s.NKV && d.head_dim == s.HD);
  CHECK(d.max_window() == s.win && s.ring == s.win && d.conv_group == s.group && d.conv_taps == s.ctaps);
  CHECK(d.sel_rank == s.rank && d.vocab == s.vocab && model::qwen38().vocab_used == s.used);
  CHECK(d.sel_top_k == kd::kTopK && d.block == kd::kM && d.taps.size() == 5);
  CHECK(s.ring % kd::kAttnPb == 0 && kd::kTopK * kd::kQs == kd::kWg && s.rank % kd::kQs == 0);

  std::set<std::string> fresh, reused;
  bool after = false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--") {
      after = true;
      continue;
    }
    CHECK(a.rfind("kernel_", 0) == 0);
    (after ? reused : fresh).insert(a.substr(7));
  }
  std::set<std::string> bound;
  for (const auto& shape : {std::array<unsigned, 3>{5120, 6144, 4}, {4096, 5120, 4}, {5120, 34816, 8},
                            {17408, 5120, 4}, {25600, 5120, 8}, {5120, 2048, 8}, {5120, 1280, 16}, {5120, 256, 16}}) {
    const auto [K, N, S] = shape;
    CHECK_EQ(kd::int4_slices(K, N), S);
    const auto* t = kernels::small_m_tuning(K, N, S, 1);
    CHECK(t && kernels::gemm_sm_out_s(*t) == S);
    CHECK_EQ(kernels::gemm_sm_built_m(*t), std::vector<unsigned>{8});
    CHECK_EQ(kd::linear_variant(2u, 8, K, N), kernels::gemm_sm_variant(8, *t));
  }
  for (bool v2 : {true, false})
    for (unsigned fmt : {0u, 1u, 2u})
      for (bool int8_head : {true, false})
        for (unsigned dv : {0u, 32768u, 65536u, 131072u})
          for (kernels::LinearKind k : {kernels::LinearKind::SmallM, kernels::LinearKind::Gemv})   // spec 28d + the override
            for (unsigned heads : {0u, 1u, 2u})
              for (unsigned rows : {0u, 1u, 2u, 4u}) {
                if (rows && heads != 1) continue;
                for (bool parallel : {false, true}) {
                  if (parallel && (heads != 1 || rows != 4)) continue;
                  for (const auto& v : kd::draft_variants(v2, fmt, int8_head, dv, k, k, heads, rows, false, parallel))
                    bound.insert(v);
                }
              }
  const auto* candidate = kernels::small_m_tuning(5120, 248320, 1, 1);
  CHECK(candidate && kernels::gemm_sm_built_m(*candidate) == std::vector<unsigned>{7});
  for (bool v2 : {false, true})
    for (unsigned fmt : {0u, 1u, 2u})
      for (auto k : {kernels::LinearKind::SmallM, kernels::LinearKind::Gemv}) {
        const auto names = kd::draft_variants(v2, fmt, true, 0, k, k, 0, 0, true);
        CHECK(std::find(names.begin(), names.end(), kd::candidate_topk_variant()) != names.end());
        CHECK(std::find(names.begin(), names.end(), kd::rerank_variant()) != names.end());
        CHECK(std::find(names.begin(), names.end(), kd::topk_variant(248320, 248077, false)) == names.end());
        bound.insert(names.begin(), names.end());
      }
  bound.insert(kd::proj_wide_variant());   // DFlashDrafter::project_wide
  {   // spec 19 v3d Task 6: the DPAS part beside the H1 / B4 reduce
    for (unsigned ks : {1u, 2u, 4u}) {
      const auto names = kd::draft_variants(true, 1u, true, 0u, kernels::LinearKind::SmallM, kernels::LinearKind::SmallM,
                                            1, 4, true, false, true, ks);
      CHECK(std::find(names.begin(), names.end(), kd::attn_dpas_variant(kd::kQwen38, ks)) != names.end());
      bound.insert(names.begin(), names.end());
    }
  }
  int bad = 0;
  for (const std::string& v : bound)
    if (!fresh.count(v) && !reused.count(v)) {
      std::fprintf(stderr, "the DFlash lists bind %s, which no CMake block builds\n", v.c_str());
      ++bad;
    }
  for (const std::string& b : fresh)
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is built by the Spec 19b block but nothing binds it\n", b.c_str());
      ++bad;
    }
  if (bad) return 1;
  std::printf("dflash_variant_names_test OK: %zu bound names (%zu built by the 19b block, %zu reused)\n", bound.size(),
              fresh.size(), reused.size());
  return 0;
}
