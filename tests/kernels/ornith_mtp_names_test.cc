// Spec 15e: every binary Ornith's MTP lists bind - the verify lists at M = 1..4, the draft
// lists (full head and the three draft vocabularies, both lm_head forms), and the prefill's
// head KV fill (runtime/prefill/step.cc step_mtp_kv) - formed exactly as runtime/capture.cc
// and step.cc form them, against the binaries CMake builds (passed as arguments,
// `kernel_<name>`). Host only, no device: a name a list would ask for that no CMake line
// builds fails here on the Mac instead of at capture on the GPU machine.
//
// usage: ornith_mtp_names_test <15e block's kernel_*...> -- <every other kernel_* it needs>
// The first list must be exactly bound (nothing in the 15e block is dead); the union must
// cover every bound name.
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "model/model_desc.h"

namespace {

// runtime::DecodeScratch::kAttnV2Blocks / kNormGroups (runtime/buffer_sizes.h), spelled as
// the variant names carry them.
constexpr unsigned kT = 32, kG = 20;
constexpr unsigned kDraftVocab[] = {32768, 65536, 131072};   // loader::kDraftVocabSizes

std::string bf16(unsigned M, unsigned K, unsigned N) {
  return kernels::gemv_bf16_variant(M, K, N, kernels::gemv_bf16_tiling(N));
}

// The decode list at M rows with the verify list's SPEC_SLOTS gdn_step (capture.cc's walk
// for Mode::Verify), then head_kv_fill; `int8` picks the lm_head form.
void verify_names(const model::ModelDesc& d, unsigned M, bool int8, std::set<std::string>& out) {
  const unsigned H = d.hidden, k = d.gdn_k_heads, v = d.gdn_v_heads, q = d.fa_q_heads,
                 kv = d.fa_kv_heads;
  const model::MoeDesc& md = d.moe;
  out.insert(kernels::embed_gather_variant(M, H));
  out.insert(kernels::prep_res_fold_variant(M, H, 0, kG));                // layer 0, after MoE
  out.insert(kernels::prep_res_fold_variant(M, H, d.shape(model::LinearId::OutProj).S, kG));
  out.insert(kernels::prep_res_fold_variant(M, H, d.shape(model::LinearId::OProj).S, kG));
  out.insert(kernels::prep_norm_finish_variant(M, H, kG, kG));
  for (model::LinearId id : {model::LinearId::QkvZ, model::LinearId::OutProj, model::LinearId::Qkv,
                             model::LinearId::OProj}) {
    const model::GemvShape& s = d.shape(id);
    out.insert(kernels::gemv_variant(M, s.K, s.N, s.S, s.layout));   // forced (CaptureOptions) / M = 1
    if (M >= 2) out.insert(kernels::int4_linear_variant(kernels::LinearKind::SmallM, M, s.K, s.N, s.S, s.layout));
  }
  // a||b in either of the checkpoint's forms (spec 15 §13, ModelDesc::ab): bf16 for a
  // checkpoint that keeps in_proj_a / in_proj_b bf16, gemv.cl for the int4 checkpoint's.
  out.insert(bf16(M, H, d.shape(model::LinearId::AB).N));
  const model::GemvShape& ab4 = d.ab(model::WeightKind::Int4).shape;
  out.insert(kernels::gemv_variant(M, ab4.K, ab4.N, ab4.S, ab4.layout));
  if (M >= 2) out.insert(kernels::int4_linear_variant(kernels::LinearKind::SmallM, M, ab4.K, ab4.N, ab4.S, ab4.layout));
  out.insert(kernels::gdn_step_slots_variant(M, d.gdn_layers, k, v));
  out.insert(kernels::gdn_step_defer_variant(M, d.gdn_layers, k, v));   // PERF-51: one slot's lists, the flush (M 1)
  out.insert(kernels::prep_gated_head_variant(M, k, v));
  out.insert(kernels::attn_prep_variant(M, q, kv));
  out.insert(kernels::attn_v2_variant(M, kT, q, kv));
  out.insert(bf16(M, H, md.router_n()));
  out.insert(kernels::moe_variant(M, md.experts, md.top_k, H, md.expert_intermediate));
  out.insert(int8 ? kernels::gemv_i8w_variant(M, H, model::Qwen35::kVocab)
                  : bf16(M, H, model::Qwen35::kVocab));
  if (int8 && M >= 2) out.insert(kernels::int8_linear_variant(kernels::LinearKind::SmallM, M, H, model::Qwen35::kVocab));
  out.insert(kernels::argmax_stage1_variant(M, d.vocab_used));
  out.insert(kernels::argmax_stage2_variant());
  // head_kv_fill (head_front)
  out.insert(kernels::prep_res_fold_pair_variant(M, H, kG));             // PERF-84: both pre-fc norms
  out.insert(kernels::prep_norm_finish_pair_variant(M, H, kG, kG));
  out.insert(bf16(M, 2 * H, H));                                          // fc
  out.insert(kernels::prep_res_fold_zero_variant(M, H, kG));
  out.insert(bf16(M, H, d.fa_qkv_n()));                                   // q||k||v
  out.insert(kernels::attn_prep_s1_variant(M, q, kv));
}

// PERF-56: the wide lookup verify lists at M = 16 / 32 / 64 rows (one card, bf16 KV, the int8 head): the M-row list as
// verify_names' but gemm_sm only, the deferred GDN step (past 16 rows its chain links and the engine's ring copy), the
// DPAS attention in row chunks with attn_v2's reduce, and the ROWG builds of every bf16 GEMV (a||b, the router, the
// head's fc and q||k||v); the argmax pair writes the wide ids (Control::wide_out / deep_out).
void wide_names(const model::ModelDesc& d, unsigned M, std::set<std::string>& out) {
  const unsigned H = d.hidden, k = d.gdn_k_heads, v = d.gdn_v_heads, q = d.fa_q_heads, kv = d.fa_kv_heads;
  const model::MoeDesc& md = d.moe;
  const auto rowg = [](unsigned K, unsigned N) {
    return kernels::gemv_bf16_variant(8, K, N, kernels::gemv_bf16_tiling(N), true);
  };
  out.insert(kernels::embed_gather_variant(M, H));
  out.insert(kernels::prep_res_fold_variant(M, H, 0, kG));
  out.insert(kernels::prep_res_fold_variant(M, H, d.shape(model::LinearId::OutProj).S, kG));
  out.insert(kernels::prep_res_fold_variant(M, H, d.shape(model::LinearId::OProj).S, kG));
  out.insert(kernels::prep_norm_finish_variant(M, H, kG, kG));
  for (model::LinearId id : {model::LinearId::QkvZ, model::LinearId::OutProj, model::LinearId::Qkv,
                             model::LinearId::OProj}) {
    const model::GemvShape& s = d.shape(id);
    out.insert(kernels::int4_linear_variant(kernels::LinearKind::SmallM, M, s.K, s.N, s.S, s.layout));
  }
  out.insert(rowg(H, d.shape(model::LinearId::AB).N));
  const model::GemvShape& ab4 = d.ab(model::WeightKind::Int4).shape;
  out.insert(kernels::int4_linear_variant(kernels::LinearKind::SmallM, M, ab4.K, ab4.N, ab4.S, ab4.layout));
  if (M == 16) {
    out.insert(kernels::gdn_step_defer_variant(16, d.gdn_layers, k, v));
  } else {
    for (unsigned r0 = 0; r0 < M; r0 += 16) out.insert(kernels::gdn_step_chain_variant(r0, d.gdn_layers, k, v));
    out.insert(kernels::gdn_ring_commit_variant(d.gdn_conv_dim()));
  }
  out.insert(kernels::prep_gated_head_variant(M, k, v));
  out.insert(kernels::attn_prep_variant(M, q, kv));
  out.insert(kernels::attn_v2_variant(M, kT, q, kv));
  out.insert(kernels::attn_verify_dpas_variant(M, kT, false, q, kv));
  out.insert(rowg(H, md.router_n()));
  out.insert(kernels::moe_variant(M, md.experts, md.top_k, H, md.expert_intermediate));
  out.insert(kernels::int8_linear_variant(kernels::LinearKind::SmallM, M, H, model::Qwen35::kVocab));
  out.insert(kernels::argmax_stage1_variant(M, d.vocab_used));
  out.insert(kernels::argmax_stage2_variant(M));
  out.insert(kernels::prep_res_fold_pair_variant(M, H, kG));             // PERF-84: both pre-fc norms
  out.insert(kernels::prep_norm_finish_pair_variant(M, H, kG, kG));
  out.insert(rowg(2 * H, H));                                             // fc
  out.insert(kernels::prep_res_fold_zero_variant(M, H, kG));
  out.insert(rowg(H, d.fa_qkv_n()));                                      // q||k||v
  out.insert(kernels::attn_prep_s1_variant(M, q, kv));
}

// The draft list at M = 1 (capture.cc draft()): head_front, the attention pair, o, the
// post norm (SP1), the MoE block on the head's weights, the final norm (SP0), the head.
void draft_names(const model::ModelDesc& d, bool int8, unsigned nv, std::set<std::string>& out) {
  const unsigned H = d.hidden;
  for (const std::string& n :   // head_front
       {kernels::embed_gather_variant(1, H), kernels::prep_res_fold_pair_variant(1, H, kG),
        kernels::prep_norm_finish_pair_variant(1, H, kG, kG), bf16(1, 2 * H, H),
        kernels::prep_res_fold_zero_variant(1, H, kG), kernels::prep_norm_finish_variant(1, H, kG, kG),
        bf16(1, H, d.fa_qkv_n()), kernels::attn_prep_s1_variant(1, d.fa_q_heads, d.fa_kv_heads)})
    out.insert(n);
  out.insert(kernels::attn_v2_variant(1, kT, d.fa_q_heads, d.fa_kv_heads));
  out.insert(bf16(1, d.fa_value_dim(), H));                               // o
  out.insert(kernels::prep_res_fold_variant(1, H, 1, kG));
  out.insert(bf16(1, H, d.moe.router_n()));
  out.insert(kernels::moe_variant(1, d.moe.experts, d.moe.top_k, H, d.moe.expert_intermediate));
  if (nv == 0) {
    out.insert(int8 ? kernels::gemv_i8w_variant(1, H, model::Qwen35::kVocab)
                    : bf16(1, H, model::Qwen35::kVocab));
    out.insert(kernels::argmax_stage1_variant(1, d.vocab_used));
    out.insert(kernels::argmax_stage2_variant());
  } else {
    out.insert(int8 ? kernels::gemv_i8w_variant(1, H, nv) : bf16(1, H, nv));
    out.insert(kernels::dv_argmax_variant(nv));
  }
}

// step_mtp_kv (the prefill's head KV fill) beyond the main chunk's binaries.
void prefill_names(const model::ModelDesc& d, std::set<std::string>& out) {
  const unsigned H = d.hidden;
  out.insert(kernels::pf_res_norm_variant(H, d.ffn_fold_s() == 0 ? 0 : 1, kG));   // final norm (PERF-85: one launch)
  out.insert(kernels::pf_mtp_front_variant(H, kG));                       // the gather + both pre-fc norms
  out.insert(kernels::pf_res_norm_zero_variant(H, kG));                   // input_layernorm
  out.insert(kernels::pf_bf16_slab_variant(2 * H));                       // fc
  out.insert(kernels::pf_bf16_slab_variant(H));                           // q||k||v
  out.insert(kernels::kv_only(kernels::pf_attn_prep_q16_variant(d.fa_q_heads, d.fa_kv_heads)));
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

  const model::ModelDesc& o = model::ornith();
  std::set<std::string> bound;
  for (bool int8 : {false, true}) {
    for (unsigned M = 1; M <= 4; ++M) verify_names(o, M, int8, bound);
    draft_names(o, int8, 0, bound);
    for (unsigned nv : kDraftVocab) draft_names(o, int8, nv, bound);
  }
  prefill_names(o, bound);
  // PERF-56: where CMake built the wide lists (B70_VERIFY_M8)
  const bool wide =
      block.count(kernels::moe_variant(16, o.moe.experts, o.moe.top_k, o.hidden, o.moe.expert_intermediate)) != 0;
  if (wide)
    for (unsigned M : {16u, 32u, 64u}) wide_names(o, M, bound);

  int bad = 0;
  for (const std::string& n : bound)
    if (!all.count(n)) {
      std::fprintf(stderr, "Ornith's MTP lists bind %s, which no CMake line given here builds\n",
                   n.c_str());
      ++bad;
    }
  for (const std::string& n : block)
    if (!bound.count(n)) {
      std::fprintf(stderr, "%s is built by the spec 15e block but no Ornith MTP list binds it\n",
                   n.c_str());
      ++bad;
    }
  CHECK_EQ(bad, 0);
  // A few names spelled out, so a change in a helper is seen here and not only as a
  // missing binary on the GPU machine.
  CHECK(bound.count("gdn_step_slots_M1_G30_GK16V32") && bound.count("gdn_step_slots_M4_G30_GK16V32"));
  CHECK(bound.count("gdn_step_defer_M1_G30_GK16V32") && bound.count("gdn_step_defer_M4_G30_GK16V32"));
  CHECK(bound.count("moe_M4_E256_T8_D2048_I512") && bound.count("attn_prep_M2_Q16KV2_S1"));
  CHECK(bound.count("prep_norm_finish_M3_K2048_G20_W20_X4096_P2") &&
        bound.count("prep_res_fold_M1_K2048_SP1_G20_Z") && bound.count("gemv_bf16_M1_K4096_N2048"));
  CHECK(bound.count("pf_res_norm_K2048_SP1_G20_Z") && bound.count("pf_bf16_slab_K4096"));
  // Spec 15 §13: the int4 checkpoint's a||b and Qwen3.8's argmax (vocab_used 248077).
  CHECK(bound.count("gemv_M4_K2048_N128_S1_L1") && bound.count("gemv_bf16_M4_K2048_N128_C16_S16"));
  CHECK(bound.count("argmax_stage1_M3") && !bound.count("argmax_stage1_M3_V248070"));
  if (wide) {   // PERF-56
    CHECK(bound.count("attn_verify_dpas_M16_T32_Q16KV2") && bound.count("gdn_step_defer_M16_G30_GK16V32_R48"));
    CHECK(bound.count("gemm_sm_wide_M32_K4096_N2048_S4_L0_mg_sw_pf2") &&
          bound.count("gemm_sm_i8w_M64_K2048_N248320_g256"));
    CHECK(bound.count("gdn_ring_commit_C8192") && bound.count("gemv_bf16_M8_K2048_N272_C16_S16_RG"));
  }
  std::printf("ornith_mtp_names_test OK: %zu names bound by Ornith's MTP lists, %zu built by the "
              "15e block, %zu given\n",
              bound.size(), block.size(), all.size());
  return 0;
}
