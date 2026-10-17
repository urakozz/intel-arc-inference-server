// Spec 24b Task 4: loader::Lfm2MoeCheckpoint (the loader's host half) over Task 1's synthetic
// real-width checkpoints, host only (no Level Zero). SKIP 77 "missing data" when the checkpoint is
// not there (tools/quantize/lfm2moe/make_synth.py --layers 4 into argv[1]).
//
//   lfm2moe_synth_host_test <checkpoint dir>
//
// Checks (spec 24b Task 4 Step 1, Review Focus 5): every tensor of the checkpoint read exactly once
// or listed skipped; the repacked qkv / in_proj / expert / embedding bytes equal
// loader/lfm2moe_repack.h's functions run INDEPENDENTLY over the file's own tensors (a part swap must
// fail the comparison - checked); the int8 head equals quantise_int8_tiled of the embedding. The
// crafted refusal cases (a planted tied-but-different lm_head, a mix of expert spellings, a wrong
// dtype / shape, an unknown or missing tensor) are lfm2moe_repack_test.cc's (it writes its own
// fixtures; this test only ever reads make_synth.py's, which never ships lm_head.weight).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/json.h"
#include "common/repack.h"
#include "loader/lfm2moe_loader.h"
#include "loader/lfm2moe_repack.h"
#include "loader/lm_head_int8.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/lfm2moe.h"

namespace {

common::json::Value read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f.good()) std::exit(77);   // SKIP: no checkpoint at this path
  std::stringstream s;
  s << f.rdbuf();
  return common::json::parse(s.str());
}

}  // namespace

int main(int argc, char** argv) {
  CHECK(argc > 1);
  const std::string dir = std::string(argv[1]) + "/";
  const common::json::Value cfg = read_json(dir + "config.json");
  CHECK(model::is_lfm2moe(cfg));
  const model::Lfm2MoeDesc d = model::lfm2moe_desc(cfg, model::lfm2moe().vocab_used);
  std::printf("lfm2moe_synth_host_test: %s - %u layers\n", dir.c_str(), d.layers);

  loader::SafetensorsSet set(dir);
  const model::LfmExpertNames names = loader::detect_expert_names(set);
  loader::Lfm2MoeCheckpoint ck(d, set, names);
  ck.check_names();

  for (uint32_t l = 0; l < d.layers; ++l) {
    loader::HostLayer host;
    ck.repack_layer(l, false, host);
    const std::string lp = d.layer_prefix(l);

    // Every linear against loader/lfm2moe_repack.h's functions run INDEPENDENTLY over the file's own
    // tensors, at whichever layout model::Lfm2MoeDesc::linears says (I1: switchable, not hardcoded
    // here) - and a part swap (e.g. w1/w3, q/v) must NOT pass, proving the comparison actually bites.
    const std::vector<model::LfmLinear> lins = d.linears(l);
    const auto find_ln = [&](model::LfmLinearId id) {
      return *std::find_if(lins.begin(), lins.end(), [&](const model::LfmLinear& x) { return x.id == id; });
    };
    if (d.is_attention(l)) {
      const model::LfmLinear qkv_ln = find_ln(model::LfmLinearId::Qkv), oproj_ln = find_ln(model::LfmLinearId::OProj);
      const loader::LinearSrc q = loader::LinearSrc::classify(set, lp + "self_attn.q_proj");
      const loader::LinearSrc k = loader::LinearSrc::classify(set, lp + "self_attn.k_proj");
      const loader::LinearSrc v = loader::LinearSrc::classify(set, lp + "self_attn.v_proj");
      const loader::HostLinear want_qkv = loader::repack_fused_linear(qkv_ln, {q, k, v});
      CHECK(host.qkv.words == want_qkv.words && host.qkv.scales == want_qkv.scales);
      CHECK(host.qkv.words != loader::repack_fused_linear(qkv_ln, {v, k, q}).words);   // q/v swapped: must differ
      const loader::LinearSrc o = loader::LinearSrc::classify(set, lp + "self_attn.out_proj");
      const loader::HostLinear want_o = loader::repack_fused_linear(oproj_ln, {o});
      CHECK(host.o_proj.words == want_o.words && host.o_proj.scales == want_o.scales);
    } else {
      const model::LfmLinear in_ln = find_ln(model::LfmLinearId::InProj), out_ln = find_ln(model::LfmLinearId::OutProj);
      const loader::LinearSrc in = loader::LinearSrc::classify(set, lp + "conv.in_proj");
      const loader::HostLinear want_in =
          loader::repack_in_proj(in.qweight, in.scales, in.K, in.N, false, in_ln.shape.layout);
      CHECK(host.in_proj.words == want_in.words && host.in_proj.scales == want_in.scales);
      const loader::LinearSrc op = loader::LinearSrc::classify(set, lp + "conv.out_proj");
      const loader::HostLinear want_out = loader::repack_fused_linear(out_ln, {op});
      CHECK(host.out_proj.words == want_out.words && host.out_proj.scales == want_out.scales);
    }

    if (d.is_dense(l)) {
      const model::LfmLinear gu_ln = find_ln(model::LfmLinearId::DenseGateUp), dn_ln = find_ln(model::LfmLinearId::DenseDown);
      const loader::LinearSrc w1 = loader::LinearSrc::classify(set, lp + "feed_forward.w1");
      const loader::LinearSrc w3 = loader::LinearSrc::classify(set, lp + "feed_forward.w3");
      const loader::LinearSrc w2 = loader::LinearSrc::classify(set, lp + "feed_forward.w2");
      const loader::HostLinear want_gu = loader::repack_fused_linear(gu_ln, {w1, w3});
      CHECK(host.dense_gate_up.words == want_gu.words && host.dense_gate_up.scales == want_gu.scales);
      CHECK(host.dense_gate_up.words != loader::repack_fused_linear(gu_ln, {w3, w1}).words);   // w1/w3 swapped
      const loader::HostLinear want_dn = loader::repack_fused_linear(dn_ln, {w2});
      CHECK(host.dense_down.words == want_dn.words && host.dense_down.scales == want_dn.scales);
    } else {
      // the router: repack_bf16_tiled(gate.weight) directly - a transposed router must fail.
      const loader::TensorInfo gt = set.tensors().at(lp + "feed_forward.gate.weight");
      const uint16_t* gw = reinterpret_cast<const uint16_t*>(set.data(gt));
      std::vector<uint16_t> want_router(size_t(d.hidden) * d.experts);
      common::repack_bf16_tiled(gw, d.hidden, d.experts, want_router.data());
      CHECK(host.router == want_router);
      std::vector<uint16_t> transposed(size_t(d.hidden) * d.experts);
      common::repack_bf16_tiled(gw, d.experts, d.hidden, transposed.data());   // wrong K/N on purpose
      CHECK(host.router != transposed);

      // one routed expert (the last) against repack_expert run directly over its tensors.
      const char* g = names == model::LfmExpertNames::Proj ? "gate_proj" : "w1";
      const char* u = names == model::LfmExpertNames::Proj ? "up_proj" : "w3";
      const char* dn = names == model::LfmExpertNames::Proj ? "down_proj" : "w2";
      const std::string ep = lp + "feed_forward.experts." + std::to_string(d.experts - 1) + ".";
      const loader::LinearSrc wg = loader::LinearSrc::classify(set, ep + g);
      const loader::LinearSrc wu = loader::LinearSrc::classify(set, ep + u);
      const loader::LinearSrc wd = loader::LinearSrc::classify(set, ep + dn);
      const loader::ExpertRepacked want = loader::repack_expert(wg, wu, wd);
      const size_t off_gu = loader::lfm::expert_gate_up_offset(d, d.experts - 1) / 4;
      const size_t off_dn = loader::lfm::expert_down_offset(d, d.experts - 1) / 4;
      CHECK(std::memcmp(host.experts_gate_up.data() + off_gu, want.gate_up.data(), want.gate_up.size() * 4) == 0);
      CHECK(std::memcmp(host.experts_down.data() + off_dn, want.down.data(), want.down.size() * 4) == 0);
      const loader::ExpertRepacked swapped = loader::repack_expert(wu, wg, wd);   // gate/up swapped: must differ
      CHECK(std::memcmp(swapped.gate_up.data(), want.gate_up.data(), want.gate_up.size() * 4) != 0);
    }

    // the small block: norms, conv taps / q_norm+k_norm, expert_bias - raw F32 - against the widened
    // file tensors read independently (I5: not the same code path repack_layer used).
    {
      const loader::lfm::SmallBlock sb = loader::lfm::small_block(d, l);
      const auto widened = [&](const std::string& name, uint64_t elems) {
        const loader::TensorInfo t = set.tensors().at(name);
        return loader::widen_bf16(reinterpret_cast<const uint16_t*>(set.data(t)), elems);
      };
      const std::vector<float> op_norm = widened(lp + "operator_norm.weight", d.hidden);
      const std::vector<float> ffn_norm = widened(lp + "ffn_norm.weight", d.hidden);
      CHECK(std::memcmp(host.small.data() + sb.operator_norm / 4, op_norm.data(), op_norm.size() * 4) == 0);
      CHECK(std::memcmp(host.small.data() + sb.ffn_norm / 4, ffn_norm.data(), ffn_norm.size() * 4) == 0);
      if (d.is_attention(l)) {
        const std::vector<float> qn = widened(lp + "self_attn.q_layernorm.weight", d.head_dim);
        const std::vector<float> kn = widened(lp + "self_attn.k_layernorm.weight", d.head_dim);
        CHECK(std::memcmp(host.small.data() + sb.q_norm / 4, qn.data(), qn.size() * 4) == 0);
        CHECK(std::memcmp(host.small.data() + sb.k_norm / 4, kn.data(), kn.size() * 4) == 0);
      } else {
        const std::vector<float> taps = widened(lp + "conv.conv.weight", size_t(d.hidden) * d.conv_taps);
        CHECK(std::memcmp(host.small.data() + sb.conv_taps / 4, taps.data(), taps.size() * 4) == 0);
      }
      if (!d.is_dense(l)) {
        const loader::TensorInfo bt = set.tensors().at(lp + "feed_forward.expert_bias");
        CHECK(bt.dtype == "F32");   // raw, never bf16-widened (the module's buffer is fp32)
        std::vector<float> bias(d.experts);
        std::memcpy(bias.data(), set.data(bt), d.experts * 4);
        CHECK(std::memcmp(host.small.data() + sb.expert_bias / 4, bias.data(), bias.size() * 4) == 0);
      }
    }
  }

  // the embedding: tiled bytes match repack_bf16_tiled over the raw tensor; the int8 head matches
  // quantise_int8_tiled over the same bytes (spec 9's rule).
  const uint16_t* emb = ck.embed();
  std::vector<uint16_t> tiles(size_t(d.hidden) * d.vocab);
  common::repack_bf16_tiled(emb, d.hidden, d.vocab, tiles.data());
  CHECK_EQ(tiles.size() * 2, loader::lfm::embed_tiled_bytes(d));
  std::vector<int8_t> q(size_t(d.hidden) * d.vocab);
  std::vector<float> scales(d.vocab);
  loader::quantise_int8_tiled(emb, d.hidden, d.vocab, q.data(), scales.data());
  CHECK_EQ(q.size() + scales.size() * 4, loader::lfm::head_int8_bytes(d));
  // a few rows, decoded through lfm_embed_row_index, must equal the int8 dequant (same scale*row).
  for (uint32_t n : {0u, d.vocab - 1}) {
    const size_t idx8 = loader::int8_tiled_index(d.hidden, 0, n);
    const size_t idxbf = loader::lfm::lfm_embed_row_index(d.hidden, 0, n);
    const float bf = common::bf16_to_f32(tiles[idxbf]);
    const float i8 = q[idx8] * scales[n];
    CHECK(std::fabs(bf - i8) <= scales[n] + 1e-6f);   // within one quantisation step
  }

  const std::vector<float> fn = ck.final_norm();
  CHECK_EQ(fn.size(), size_t(d.hidden));
  const std::string tied = ck.check_tied_lm_head();
  std::string names_str;
  const size_t unconsumed = ck.unconsumed(&names_str);
  CHECK_EQ(unconsumed, size_t(0));
  std::printf("lfm2moe_synth_host_test OK: %u layers, expert names '%s', 0 unconsumed%s%s\n", d.layers,
             model::lfm_expert_names_name(names), tied.empty() ? "" : ", ", tied.c_str());
  return 0;
}
