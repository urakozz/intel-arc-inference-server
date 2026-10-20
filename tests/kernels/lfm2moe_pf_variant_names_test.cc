// Spec 24d: the binaries an LFM2.5-24B-A2B prefill chunk binds (kernels::lfm2moe::prefill_variants - the names
// runtime/lfm2moe/lfm2moe_prefill.cc forms) for both attention forms and both in_proj layouts, on the real descriptor
// and the synthetic N 4, against the ones src/kernels/CMakeLists.txt's spec 24d block builds (passed as arguments:
// tests/CMakeLists.txt's B70_LFM2MOE_PREFILL_KERNELS, `kernel_<name>`; after `--` the reused binaries built outside the
// block: pf_gemm_T0, pf_gemm_T0_SILU). Host only: a name the walk would ask for that no CMake line builds fails here on
// the Mac, not at prepare_prefill on the box; and nothing in the list is dead. The head after the last chunk binds
// decode's binaries (lfm2moe_variant_names_test holds those).
#include <cstdio>
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
  model::Lfm2MoeDesc d4 = d;
  d4.layers = 4;
  d4.kinds.assign(d.kinds.begin(), d.kinds.begin() + 4);
  std::set<std::string> bound;
  for (const model::Lfm2MoeDesc* desc : {&d, static_cast<const model::Lfm2MoeDesc*>(&d4)})
    for (bool eager : {false, true})
      for (bool tiles : {false, true})
        for (const std::string& v : kl::prefill_variants(*desc, eager, tiles)) bound.insert(v);
  for (const std::string& v : bound)
    if (!built.count(v) && !reused.count(v)) {
      std::fprintf(stderr, "the LFM2 prefill walk binds %s, which the CMake spec 24d block does not build\n", v.c_str());
      return 1;
    }
  for (const std::string& b : built)
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is built (B70_LFM2MOE_PREFILL_KERNELS) but nothing binds it\n", b.c_str());
      return 1;
    }
  // The slabs follow each linear's layout (24b review I1), not an assumed layout 1.
  const std::vector<std::string> base = kl::prefill_variants(d, false, false);
  auto has = [&](const std::string& n) {
    for (const std::string& v : base)
      if (v == n) return true;
    return false;
  };
  for (uint32_t l : {0u, 2u})
    for (const model::LfmLinear& ln : d.linears(l))
      if (ln.kind == model::WeightKind::Int4) CHECK(has(kl::pf_slab_variant(ln.shape)));
  CHECK(has("pf_dequant_slab_K2048_N3072_L0"));    // q||k||v: layout 0 (the descriptor's)
  CHECK(has("pf_dequant_slab_K2048_N23552_L0"));   // the dense gate||up: layout 0
  CHECK(has("pf_dequant_slab_K2048_N6144_L1"));    // in_proj: layout 1
  CHECK(has("lfm_pf_conv_K2048_C2048"));
  CHECK(has("lfm_pf_moe_E64_T4_D2048_I1536_TM32"));
  CHECK(has("lfm_pf_attn_Q32KV8_HPW4_RPW16_KT64"));
  CHECK(has("lfm_res_norm_M2048_K2048_W8_S1"));
  CHECK(has("pf_moe_gemm_K2048_N3072_SILU"));
  CHECK(has("pf_moe_gemm_K1536_N2048"));
  std::printf("lfm2moe_pf_variant_names_test OK: %zu bound names, %zu binaries + %zu reused\n", bound.size(),
              built.size(), reused.size());
  return 0;
}
