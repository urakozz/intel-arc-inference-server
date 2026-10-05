// Spec 15b: the kernel variant names the runtime builds from a model descriptor.
// Host-only, no device and no compiled kernels: at Qwen3.8's and Agnes's shapes
// every name is exactly the one main built before spec 15b (so no binary is
// renamed - the kernel command-line diff covers the build side), and Ornith's
// shapes add the suffixes of kernels/shape_suffix.h.
#include <cstdio>
#include <string>
#include "check.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "model/model_desc.h"

namespace {

using model::ModelDesc;

// Every name the runtime builds that depends on a model shape (capture.cc, prefill/*.cc),
// formed exactly as the runtime forms it.
struct Names {
  std::string embed, gdn, gdn_slots, gated, attn_prep, attn_prep_s1, attn_dec, attn_red, attn_v2;
  std::string pf_embed, pf_ab, pf_gated, pf_conv, pf_wy, pf_scan, pf_prep_q16, pf_attn, pf_flash;
  std::string silu, res_fold, norm_finish, gemv_qkvz, gemv_ab, argmax;
};

Names names(const ModelDesc& d) {
  const unsigned k = d.gdn_k_heads, v = d.gdn_v_heads, q = d.fa_q_heads, kv = d.fa_kv_heads;
  const model::GemvShape& qz = d.shape(model::LinearId::QkvZ);
  const model::GemvShape& ab = d.shape(model::LinearId::AB);
  return {kernels::embed_gather_variant(1, d.hidden),
          kernels::gdn_step_variant(1, k, v),
          kernels::gdn_step_slots_variant(2, d.gdn_layers, k, v),
          kernels::prep_gated_head_variant(1, k, v),
          kernels::attn_prep_variant(1, q, kv),
          kernels::attn_prep_s1_variant(3, q, kv),
          kernels::attn_decode_variant(1, 16384, 64, q, kv),
          kernels::attn_reduce_variant(1, 16384, 64, q, kv),
          kernels::attn_v2_variant(1, 32, q, kv),
          kernels::pf_embed_gather_variant(d.hidden),
          kernels::pf_ab_proj_variant(d.hidden),
          kernels::pf_gated_head_variant(k, v),
          kernels::pf_gdn_conv_variant(k, v),
          kernels::pf_gdn_wy_variant(k, v),
          kernels::pf_gdn_scan_variant(k, v),
          kernels::pf_attn_prep_q16_variant(q, kv),
          kernels::pf_attn_variant(q, kv),
          kernels::pf_flash_attn_variant(q, kv),
          kernels::prep_silu_mul_variant(1, d.intermediate),
          kernels::prep_res_fold_variant(1, d.hidden, 4, 20),
          kernels::prep_norm_finish_variant(1, d.hidden, 20, 20),
          kernels::gemv_variant(1, qz.K, qz.N, qz.S, qz.layout),
          kernels::gemv_bf16_variant(1, ab.K, ab.N, kernels::gemv_bf16_tiling(ab.N)),
          kernels::argmax_stage1_variant(1, d.vocab_used)};
}

// main's names (c104247), spelled out: the Qwen3.8 binaries every list binds.
void check_main_names(const Names& n) {
  CHECK_EQ(n.embed, std::string("embed_gather_M1"));
  CHECK_EQ(n.gdn, std::string("gdn_step_M1"));
  CHECK_EQ(n.gated, std::string("prep_gated_head_M1"));
  CHECK_EQ(n.attn_prep, std::string("attn_prep_M1"));
  CHECK_EQ(n.attn_prep_s1, std::string("attn_prep_M3_S1"));
  CHECK_EQ(n.attn_dec, std::string("attn_decode_M1_L16384_B64"));
  CHECK_EQ(n.attn_red, std::string("attn_reduce_M1_L16384_B64"));
  CHECK_EQ(n.attn_v2, std::string("attn_v2_M1_T32"));
  CHECK_EQ(n.pf_embed, std::string("pf_embed_gather"));
  CHECK_EQ(n.pf_ab, std::string("pf_ab_proj"));
  CHECK_EQ(n.pf_gated, std::string("pf_gated_head"));
  CHECK_EQ(n.pf_conv, std::string("pf_gdn_conv"));
  CHECK_EQ(n.pf_wy, std::string("pf_gdn_wy"));
  CHECK_EQ(n.pf_scan, std::string("pf_gdn_scan"));
  CHECK_EQ(n.pf_prep_q16, std::string("pf_attn_prep_q16"));
  CHECK_EQ(n.pf_attn, std::string("pf_attn"));
  CHECK_EQ(n.pf_flash, std::string("pf_flash_attn"));
  CHECK_EQ(n.res_fold, std::string("prep_res_fold_M1_K5120_SP4_G20"));
  CHECK_EQ(n.norm_finish, std::string("prep_norm_finish_M1_K5120_G20_W20"));
  CHECK_EQ(n.gemv_qkvz, std::string("gemv_M1_K5120_N16384_S1_L1"));
  CHECK_EQ(n.gemv_ab, std::string("gemv_bf16_M1_K5120_N128_C16_S16"));
}

}  // namespace

int main() {
  const Names q = names(model::qwen38());
  check_main_names(q);
  CHECK_EQ(q.gdn_slots, std::string("gdn_step_slots_M2"));
  CHECK_EQ(q.silu, std::string("prep_silu_mul_M1"));
  CHECK_EQ(q.argmax, std::string("argmax_stage1_M1"));

  // Agnes: Qwen3.8's shapes, so the same names but for spec 14's three suffixes.
  const Names a = names(model::agnes());
  check_main_names(a);
  CHECK_EQ(a.gdn_slots, std::string("gdn_step_slots_M2_G54"));
  CHECK_EQ(a.silu, std::string("prep_silu_mul_M1_I19456"));
  CHECK_EQ(a.argmax, std::string("argmax_stage1_M1_V248089"));

  // Ornith: every shape-baking kernel names its own binary (built from spec 15c).
  const Names o = names(model::ornith());
  CHECK_EQ(o.embed, std::string("embed_gather_M1_D2048"));
  CHECK_EQ(o.gdn, std::string("gdn_step_M1_GK16V32"));
  CHECK_EQ(o.gdn_slots, std::string("gdn_step_slots_M2_G30_GK16V32"));
  CHECK_EQ(o.gated, std::string("prep_gated_head_M1_GK16V32"));
  CHECK_EQ(o.attn_prep, std::string("attn_prep_M1_Q16KV2"));
  CHECK_EQ(o.attn_prep_s1, std::string("attn_prep_M3_Q16KV2_S1"));
  CHECK_EQ(o.attn_dec, std::string("attn_decode_M1_L16384_B64_Q16KV2"));
  CHECK_EQ(o.attn_red, std::string("attn_reduce_M1_L16384_B64_Q16KV2"));
  CHECK_EQ(o.attn_v2, std::string("attn_v2_M1_T32_Q16KV2"));
  CHECK_EQ(o.pf_embed, std::string("pf_embed_gather_D2048"));
  CHECK_EQ(o.pf_ab, std::string("pf_ab_proj_D2048"));
  CHECK_EQ(o.pf_gated, std::string("pf_gated_head_GK16V32"));
  CHECK_EQ(o.pf_conv, std::string("pf_gdn_conv_GK16V32"));
  CHECK_EQ(o.pf_wy, std::string("pf_gdn_wy_GK16V32"));
  CHECK_EQ(o.pf_scan, std::string("pf_gdn_scan_GK16V32"));
  CHECK_EQ(o.pf_prep_q16, std::string("pf_attn_prep_q16_Q16KV2"));
  CHECK_EQ(o.pf_attn, std::string("pf_attn_Q16KV2"));
  CHECK_EQ(o.pf_flash, std::string("pf_flash_attn_Q16KV2"));
  CHECK_EQ(o.silu, std::string("prep_silu_mul_M1_I512"));
  CHECK_EQ(o.res_fold, std::string("prep_res_fold_M1_K2048_SP4_G20"));
  CHECK_EQ(o.norm_finish, std::string("prep_norm_finish_M1_K2048_G20_W20"));
  CHECK_EQ(o.gemv_qkvz, std::string("gemv_M1_K2048_N12288_S1_L1"));
  CHECK_EQ(o.gemv_ab, std::string("gemv_bf16_M1_K2048_N128_C16_S16"));
  CHECK_EQ(o.argmax, std::string("argmax_stage1_M1_V248070"));
  // Spec 15c: the MoE block's binaries (src/kernels/CMakeLists.txt's Ornith block) and the
  // geometry capture.cc binds them with: the router || shared-gate GEMV at 272 columns,
  // moe_gate_up over 9 slots x 16 work-groups of 256, moe_down over 128 n-tiles of 288.
  const model::MoeDesc& om = model::ornith().moe;
  CHECK_EQ(kernels::moe_variant(1, om.experts, om.top_k, model::ornith().hidden,
                                om.expert_intermediate),
           std::string("moe_M1_E256_T8_D2048_I512"));
  CHECK_EQ(kernels::gemv_bf16_variant(1, 2048, om.router_n(), kernels::gemv_bf16_tiling(om.router_n())),
           std::string("gemv_bf16_M1_K2048_N272_C16_S16"));
  CHECK_EQ(kernels::moe_gate_up_wg(), 256u);
  CHECK_EQ(kernels::moe_down_wg(om.top_k), 288u);
  CHECK_EQ(kernels::moe_gate_up_groups(om.top_k, om.expert_intermediate), 144u);
  CHECK_EQ(kernels::gemv_variant(1, 4096, 2048, 4, 0), std::string("gemv_M1_K4096_N2048_S4_L0"));
  CHECK_EQ(kernels::prep_res_fold_variant(1, 2048, 0, 20), std::string("prep_res_fold_M1_K2048_SP0_G20"));

  // Spec 8 §11: the draft-vocabulary binaries (src/kernels/CMakeLists.txt, B70_MTP), the
  // names capture.cc's draft list binds at each compiled |V'|.
  CHECK_EQ(kernels::gemv_i8w_variant(1, 5120, 32768), std::string("gemv_i8w_M1_K5120_N32768"));
  CHECK_EQ(kernels::gemv_i8w_variant(1, 5120, 131072), std::string("gemv_i8w_M1_K5120_N131072"));
  CHECK_EQ(kernels::dv_argmax_variant(32768), std::string("dv_argmax_N32768"));
  CHECK_EQ(kernels::dv_argmax_variant(65536), std::string("dv_argmax_N65536"));
  CHECK_EQ(kernels::dv_argmax_variant(131072), std::string("dv_argmax_N131072"));

  std::puts("variant_names_test OK");
  return 0;
}
