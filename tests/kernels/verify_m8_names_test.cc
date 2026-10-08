// Spec 19a (plan 19a Task 4): every binary the two probes of the verify cost at M = 5..8
// bind - probe_mtp_steps' Qwen3.8 verify lists at M = 5..8 (runtime::build_verify over an
// 8-slot MtpBuffers, int8 lm_head, bf16 KV, attention v2) and probe_draft_cost's DFlash2
// GEMVs (tools/probe/draft_cost_shapes.h) - formed as runtime/capture.cc and the probe form
// them, against the binaries CMake builds (passed as arguments, `kernel_<name>`). Host
// only, no device: a name a list would ask for that no CMake line builds fails here on the
// Mac instead of at capture on the box.
//
// usage: verify_m8_names_test <the B70_VERIFY_M8 block's kernel_*...> -- <reused kernel_*>
// The first list must be exactly bound (nothing in the block is dead); the union must cover
// every bound name.
#include <cstdio>
#include <set>
#include <string>

#include "check.h"
#include "draft_cost_shapes.h"
#include "kernels/kernels.h"
#include "model/model_desc.h"
#include "model/qwen35.h"

namespace {

// runtime::DecodeScratch::kAttnV2Blocks / kNormGroups, as the names carry them; the
// probe's MtpBuffers slots (runtime::MtpDims::kMaxSlots).
constexpr unsigned kT = 32, kG = 20, kSlots8 = 8;

std::string bf16(unsigned M, unsigned K, unsigned N) {
  return kernels::gemv_bf16_variant(M, K, N, kernels::gemv_bf16_tiling(N));
}

// capture.cc's Mode::Verify walk on Qwen3.8 at M rows: the decode list (dense layers, the
// checkpoint's bf16 a||b, the int8 head) with the 8-slot gdn_step, then head_kv_fill.
void verify_names(const model::ModelDesc& d, unsigned M, std::set<std::string>& out) {
  using model::LinearId;
  const unsigned H = d.hidden, k = d.gdn_k_heads, v = d.gdn_v_heads, q = d.fa_q_heads,
                 kv = d.fa_kv_heads;
  out.insert(kernels::embed_gather_variant(M, H));
  out.insert(kernels::prep_res_fold_variant(M, H, 0, kG));   // layer 0's input norm
  for (LinearId id : {LinearId::OutProj, LinearId::OProj, LinearId::Down})
    out.insert(kernels::prep_res_fold_variant(M, H, d.shape(id).S, kG));
  out.insert(kernels::prep_res_fold_variant(M, H, d.ffn_fold_s(), kG));   // the final norm
  out.insert(kernels::prep_norm_finish_variant(M, H, kG, kG));
  for (LinearId id : {LinearId::QkvZ, LinearId::OutProj, LinearId::Qkv, LinearId::OProj,
                      LinearId::GateUp, LinearId::Down}) {
    const model::GemvShape& s = d.shape(id);
    out.insert(kernels::gemv_variant(M, s.K, s.N, s.S, s.layout));
  }
  out.insert(bf16(M, H, d.shape(LinearId::AB).N));
  out.insert(kernels::prep_silu_mul_variant(M, d.intermediate));
  out.insert(kernels::gdn_step_slots_variant(M, d.gdn_layers, k, v, kSlots8));
  out.insert(kernels::prep_gated_head_variant(M, k, v));
  out.insert(kernels::attn_prep_variant(M, q, kv));
  out.insert(kernels::attn_v2_variant(M, kT, q, kv));
  out.insert(kernels::gemv_i8w_variant(M, H, model::Qwen35::kVocab));
  out.insert(kernels::argmax_stage1_variant(M, d.vocab_used));
  out.insert(kernels::argmax_stage2_variant());
  // head_kv_fill (head_front)
  out.insert(kernels::prep_norm_finish_strided_variant(M, H, kG, kG, 2 * H));
  out.insert(bf16(M, 2 * H, H));                                          // fc
  out.insert(kernels::prep_res_fold_zero_variant(M, H, kG));
  out.insert(bf16(M, H, d.fa_qkv_n()));                                   // q||k||v
  out.insert(kernels::attn_prep_s1_variant(M, q, kv));
}

// probe_draft_cost: every linear in the three formats at M = kRows, the int8 head at kHeadRows.
void draft_names(std::set<std::string>& out) {
  using namespace draft_cost;
  for (const Linear& L : kLinears) {
    out.insert(kernels::gemv_i8w_variant(kRows, L.K, L.N));
    out.insert(bf16(kRows, L.K, L.N));
    out.insert(kernels::gemv_variant(kRows, L.K, L.N, L.int4_s, 0));
  }
  for (unsigned V : kHeadVocab) out.insert(kernels::gemv_i8w_variant(kHeadRows, kHidden, V));
}

}  // namespace

int main(int argc, char** argv) {
  std::set<std::string> block, all;
  bool other = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--") {
      other = true;
      continue;
    }
    CHECK(a.rfind("kernel_", 0) == 0);
    (other ? all : block).insert(a.substr(7));
  }
  CHECK(!block.empty());
  all.insert(block.begin(), block.end());

  const model::ModelDesc& d = model::qwen38();
  std::set<std::string> bound;
  for (unsigned M = 5; M <= 8; ++M) verify_names(d, M, bound);
  draft_names(bound);

  int bad = 0;
  for (const std::string& n : bound)
    if (!all.count(n)) {
      std::fprintf(stderr, "the M = 5..8 probes bind %s, which no CMake line given here builds\n",
                   n.c_str());
      ++bad;
    }
  for (const std::string& n : block)
    if (!bound.count(n)) {
      std::fprintf(stderr, "%s is built by the B70_VERIFY_M8 block but neither probe binds it\n",
                   n.c_str());
      ++bad;
    }
  CHECK_EQ(bad, 0);
  // A few names spelled out, so a change in a helper is seen here and not only as a
  // missing file on the box.
  CHECK_EQ(kernels::gdn_step_slots_variant(8, 48, 16, 48, 8), std::string("gdn_step_slots_M8_N8"));
  CHECK_EQ(kernels::gdn_step_slots_variant(4, 48), std::string("gdn_step_slots_M4"));
  CHECK_EQ(kernels::gdn_step_slots_variant(2, 30, 16, 32, 8),
           std::string("gdn_step_slots_M2_N8_G30_GK16V32"));
  CHECK_EQ(kernels::attn_v2_variant(8, kT), std::string("attn_v2_M8_T32"));
  CHECK_EQ(bf16(8, 5120, 256), std::string("gemv_bf16_M8_K5120_N256_C16_S16"));
  CHECK_EQ(bf16(8, 5120, 2048), std::string("gemv_bf16_M8_K5120_N2048"));
  CHECK_EQ(kernels::gemv_i8w_variant(7, 5120, 32768), std::string("gemv_i8w_M7_K5120_N32768"));
  std::printf("verify_m8_names_test OK: %zu names bound (verify M = 5..8 + the draft-cost "
              "probe), %zu in the block, all built\n",
              bound.size(), block.size());
  return 0;
}
