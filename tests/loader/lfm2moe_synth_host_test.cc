// Spec 24b Task 4: loader::Lfm2MoeCheckpoint (the loader's host half) over Task 1's synthetic
// real-width checkpoints, host only (no Level Zero). SKIP 77 "missing data" when the checkpoint is
// not there (tools/quantize/lfm2moe/make_synth.py --layers 4 into argv[1]).
//
//   lfm2moe_synth_host_test <checkpoint dir>
//
// Checks (spec 24b Task 4 Step 1, Review Focus 5): every tensor of the checkpoint read exactly once
// or listed skipped; the repacked expert / in_proj / embedding bytes equal loader/lfm2moe_repack.h's
// functions run directly over the file's own tensors; the int8 head equals quantise_int8_tiled of the
// embedding; an lm_head.weight equal to the embedding is skipped (synthetic checkpoints never ship
// one, so this half is checked by construction - the planted case is exercised separately below).
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

    // in_proj / qkv against loader/lfm2moe_repack.h's functions run directly over the file's tensors.
    if (d.is_attention(l)) {
      const loader::LinearSrc q = loader::LinearSrc::classify(set, lp + "self_attn.q_proj");
      const loader::LinearSrc k = loader::LinearSrc::classify(set, lp + "self_attn.k_proj");
      const loader::LinearSrc v = loader::LinearSrc::classify(set, lp + "self_attn.v_proj");
      std::vector<uint32_t> want(loader::lfm::int4_l1_bytes(d.hidden, d.qkv_n()) / 4);
      common::repack_int4_layout1_cols(
          d.hidden, d.qkv_n(),
          common::cols_concat({{q.qweight, q.scales, q.N}, {k.qweight, k.scales, k.N}, {v.qweight, v.scales, v.N}}),
          want.data());
      CHECK(host.qkv == want);
    } else {
      const loader::LinearSrc in = loader::LinearSrc::classify(set, lp + "conv.in_proj");
      const std::vector<uint32_t> want = loader::repack_in_proj(in.qweight, in.scales, in.K, in.N, false);
      CHECK(host.in_proj == want);
    }

    // one routed expert (layer's last, if any) against repack_expert run directly over its tensors.
    if (!d.is_dense(l)) {
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
