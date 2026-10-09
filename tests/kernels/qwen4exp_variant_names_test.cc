// Spec 21c Task 2: the binaries a Qwen3.8-Flash-Next decode list binds (kernels::qwen4exp::decode_variants - the
// names runtime/qwen4exp/qwen4exp_capture.cc forms) for every dense arm (ours int4 / Intel's bf16), head form,
// attention form, PLE scale form and card count, against the ones src/kernels/CMakeLists.txt's spec 21c block
// builds (passed as arguments: tests/CMakeLists.txt's B70_Q4EXP_DECODE_KERNELS, `kernel_<name>`; the reused
// binaries other blocks build - Qwen3.8's gdn_step_M1 / argmax, K2's and Kolibri's shared shapes - are in the list
// too). Host only: a name the capture would ask for that no CMake line builds fails here on the Mac, not at capture
// on the box; and nothing in the list is dead.
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/qwen4exp_kernels.h"
#include "model/qwen4exp.h"

int main(int argc, char** argv) {
  namespace kq = kernels::qwen4exp;
  std::set<std::string> built;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    CHECK(a.rfind("kernel_", 0) == 0);
    built.insert(a.substr(7));
  }
  CHECK(!built.empty());
  // The descriptor's shapes and PROVISIONAL split-K cells are the names' (qkvz S1, q||gate||k||v S2, out / o S4).
  const model::Qwen4ExpDesc& d = model::qwen4exp();
  CHECK(d.hidden == kq::kHidden && d.hc_n() == kq::kHcN && d.hc_low == kq::kHcLow && d.hc_down_rows() == kq::kHcDownN);
  CHECK(d.q_heads == kq::kQHeads && d.kv_heads == kq::kKvHeads && d.head_dim == kq::kHd && d.qkvg_n() == kq::kQkvgN);
  CHECK(d.idx_n() == kq::kIdxN && d.idx_compress == kq::kBlock && d.block_topk() == kq::kTopBlocks);
  CHECK(d.max_visible() + 1 == kq::kListMax && kq::kListRow >= kq::kListMax + 1 && kq::kListRow * 4 % 64 == 0);
  CHECK(d.experts == kq::kExperts && d.top_k == kq::kTopK && d.moe_inter == kq::kInter && d.router_n() == kq::kRouterN);
  CHECK(d.ple_heads == kq::kPleHeads && d.ple_dim == kq::kPleDim && d.ple_kv_n() == kq::kPleKvN && d.ple_eos == kq::kPleEos);
  CHECK(d.vocab == kq::kVocab && d.vocab_used == kq::kVocabUsed && d.qkvz_n() == kq::kQkvzN);
  CHECK(d.linear(model::Q4LinearId::GdnQkvz).shape.S == 1 && d.qkvg_s == 2 && d.gdn_out_s == 4 && d.o_s == 4);
  std::set<std::string> bound;
  model::Qwen4ExpDesc intel = d;
  intel.forms = {model::Q4Form::Bf16, model::Q4Form::Bf16, model::Q4Form::Bf16, 128};
  for (const model::Qwen4ExpDesc* f : {&d, static_cast<const model::Qwen4ExpDesc*>(&intel)})
    for (bool int8 : {false, true})
      for (bool eager : {false, true})
        for (bool two : {false, true})
          for (bool ple_bf16 : {false, true})
            for (const std::string& v : kq::decode_variants(*f, int8, eager, two, ple_bf16)) bound.insert(v);
  for (const std::string& v : bound)
    if (!built.count(v)) {
      std::fprintf(stderr, "the Qwen3.8-Flash-Next decode list binds %s, which no CMake line builds\n", v.c_str());
      return 1;
    }
  for (const std::string& b : built)
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is in B70_Q4EXP_DECODE_KERNELS but nothing binds it\n", b.c_str());
      return 1;
    }
  // The names carry their shapes (a host / device disagreement names a binary that does not exist).
  CHECK(kq::hc_combine_norm_variant(1, kq::HcSrc::Slices, 4, true) == "q4_hc_combine_norm_M1_S4");
  CHECK(kq::hc_combine_norm_variant(1, kq::HcSrc::Y, 0, false) == "q4_hc_combine_norm_M1_Y_NN");
  CHECK(kq::qsa_attn_variant(1, false) == "q4_qsa_attn_M1_T32" && kq::qsa_attn_variant(1, true) == "q4_qsa_attn_eager_M1");
  CHECK(kq::route_variant(1) == "q4_route_M1_E512_T10_N528_L256");
  CHECK(kq::hc_down_variant(true) == "gemv_bf16_M1_K10240_N336_C16_S16");
  CHECK(kq::attn_prep_q4_variant(1, 1) == "attn_prep_M1_Q24KV2_S1");
  std::printf("qwen4exp_variant_names_test OK: %zu bound names, %zu binaries\n", bound.size(), built.size());
  return 0;
}
