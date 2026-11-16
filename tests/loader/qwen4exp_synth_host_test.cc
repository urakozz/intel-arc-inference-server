// Spec 21b: the loader's host half over make_synth.py's synthetic checkpoints - the Python writer and the C++
// reader held to each other on real files, no device (host only; SKIP 77 without the data).
//   qwen4exp_synth_host_test <oracle-out-q4exp-synth>      (<dir>/{ours,intel}/ckpt and ckpt-ple-int8)
// For each form: config.json -> QuantConfig::parse and model::qwen4exp_desc at the checkpoint's 4 layers; the
// forms from the names (ours: int4 dense / shared, g64; Intel's: bf16, g128) = the config's group_size;
// assert_quant_invariants; every name both ways with the MTP head; every layer and the head repacked, each
// layer's bytes = loader::q4_layer_bytes; the PLE I64 tensors = the formula for the synthetic base; the PLE
// int8 file read per head and held to the descriptor (check_q4_ple); nothing unconsumed; the plan of the
// 4-layer model with the head fits one 32.53 GB card. Plan 21v2b: every layer's two tensor-parallel halves
// (loader::q4_tp_layer) joined back equal the one-card repack bitwise - the column cuts' columns each placed once, the
// K cuts and the experts' halves concatenated - and each half's bytes = q4_layer_bytes(q4_tp_card).
#include <array>
#include <cstring>
#include <vector>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "check.h"
#include "common/json.h"
#include "loader/qwen4exp_layout.h"
#include "loader/qwen4exp_ple.h"
#include "loader/qwen4exp_repack.h"
#include "loader/qwen4exp_tp_split.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/qwen4exp.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

namespace {

using loader::Q4HostWeight;

// A column cut joined back: card r's column j is one-card column cols[r][j]; every one-card column placed once.
void join_cols(const Q4HostWeight& full, const Q4HostWeight (&half)[2], const std::vector<uint32_t> (&cols)[2]) {
  const uint32_t K = full.shape.K, N = full.shape.N;
  std::vector<uint8_t> seen(N, 0);
  for (uint32_t r = 0; r < 2; ++r) {
    const Q4HostWeight& h = half[r];
    const uint32_t n = h.shape.N;
    for (uint32_t j = 0; j < cols[r].size(); ++j) {
      const uint32_t c = cols[r][j];
      CHECK(c < N && !seen[c]);
      seen[c] = 1;
      if (full.kind == model::WeightKind::Int4) {
        for (uint32_t w = 0; w < K / 8; ++w) CHECK_EQ(h.words[size_t(w) * n + j], full.words[size_t(w) * N + c]);
        for (uint32_t g = 0; g < K / 64; ++g) CHECK_EQ(h.scales[size_t(g) * n + j], full.scales[size_t(g) * N + c]);
      } else {
        const std::vector<uint32_t> one{c};
        const std::vector<uint16_t> a = loader::q4_tiles_cols(full.tiles, K, N, one, 16);
        const std::vector<uint32_t> mine{j};
        CHECK(a == loader::q4_tiles_cols(h.tiles, K, n, mine, 16));
      }
    }
  }
}
// A K cut joined back: card 0's rows then card 1's are the one-card rows (int4: the words' and scales' arrays
// concatenated; bf16: each n-tile's k-octets).
void join_krows(const Q4HostWeight& full, const Q4HostWeight (&half)[2]) {
  if (full.kind == model::WeightKind::Int4) {
    std::vector<uint32_t> w = half[0].words;
    w.insert(w.end(), half[1].words.begin(), half[1].words.end());
    std::vector<uint16_t> s = half[0].scales;
    s.insert(s.end(), half[1].scales.begin(), half[1].scales.end());
    CHECK(w == full.words && s == full.scales);
    return;
  }
  const uint32_t N = full.shape.N, k = half[0].shape.K;
  for (uint32_t r = 0; r < 2; ++r) CHECK(half[r].tiles == loader::q4_tiles_krows(full.tiles, full.shape.K, N, r * k, k));
}
// An expert's layout-1 halves joined back: gate||up card 0's block then card 1's; down per n-tile card 0's K groups then
// card 1's.
void join_l1(const uint32_t* gu, const uint32_t* dn, const uint32_t* const (&gu_h)[2], const uint32_t* const (&dn_h)[2],
             uint32_t H, uint32_t I) {
  const size_t gu_half = loader::q4_layout1_bytes(H, I) / 4;
  for (uint32_t r = 0; r < 2; ++r) CHECK(std::memcmp(gu + r * gu_half, gu_h[r], gu_half * 4) == 0);
  const size_t run = size_t(I / 128) * loader::kQ4TileU32;   // one n-tile's half of the K groups
  for (uint32_t t = 0; t < H / 16; ++t)
    for (uint32_t r = 0; r < 2; ++r) CHECK(std::memcmp(dn + (2 * t + r) * run, dn_h[r] + t * run, run * 4) == 0);
}

void check_tp_halves(const model::Qwen4ExpDesc& d, uint32_t l, const loader::Q4HostLayer& full) {
  using model::Q4LinearId;
  const loader::Q4HostLayer h[2] = {loader::q4_tp_layer(d, l, full, 0), loader::q4_tp_layer(d, l, full, 1)};
  for (uint32_t r = 0; r < 2; ++r) {
    const model::Qwen4ExpDesc c = model::q4_tp_card(d, r);
    CHECK_EQ(h[r].bytes().total(), loader::q4_layer_bytes(c, l).total());
    CHECK(h[r].hc_attn == full.hc_attn && h[r].hc_mlp == full.hc_mlp && h[r].ple == full.ple);
    CHECK(h[r].router.tiles == full.router.tiles && h[r].qsa_idx.tiles == full.qsa_idx.tiles);
    CHECK(h[r].qsa_small == full.qsa_small);
  }
  if (d.is_qsa(l)) {
    const std::vector<uint32_t> c[2] = {loader::q4_tp_columns(d, Q4LinearId::QsaQkvg, 0),
                                        loader::q4_tp_columns(d, Q4LinearId::QsaQkvg, 1)};
    join_cols(full.qsa_qkvg, {h[0].qsa_qkvg, h[1].qsa_qkvg}, c);
    join_krows(full.qsa_o, {h[0].qsa_o, h[1].qsa_o});
  } else {
    const std::vector<uint32_t> c[2] = {loader::q4_tp_columns(d, Q4LinearId::GdnQkvz, 0),
                                        loader::q4_tp_columns(d, Q4LinearId::GdnQkvz, 1)};
    join_cols(full.gdn_qkvz, {h[0].gdn_qkvz, h[1].gdn_qkvz}, c);
    const std::vector<uint32_t> ab[2] = {loader::q4_tp_columns(d, Q4LinearId::GdnAb, 0),
                                         loader::q4_tp_columns(d, Q4LinearId::GdnAb, 1)};
    join_cols(full.gdn_ab, {h[0].gdn_ab, h[1].gdn_ab}, ab);
    join_krows(full.gdn_out, {h[0].gdn_out, h[1].gdn_out});
    // the small block: each card's conv channels' taps, its v-heads' two rows, the gated norm whole
    const loader::SmallLayout s1 = loader::q4_gdn_small(d), sc = loader::q4_gdn_small(model::q4_tp_card(d, 0));
    std::vector<uint8_t> seen(d.conv_rows(), 0);
    for (uint32_t r = 0; r < 2; ++r) {
      const std::vector<uint32_t> ch = loader::q4_tp_conv_channels(d, r);
      for (uint32_t i = 0; i < ch.size(); ++i) {
        CHECK(!seen[ch[i]]);
        seen[ch[i]] = 1;
        CHECK(std::memcmp(h[r].gdn_small.data() + sc.gdn_off_conv + i * 16, full.gdn_small.data() + ch[i] * 16, 16) == 0);
      }
      const size_t hv = size_t(d.gdn_v_heads / 2) * 4;
      CHECK(std::memcmp(h[r].gdn_small.data() + sc.gdn_off_nega, full.gdn_small.data() + s1.gdn_off_nega + r * hv, hv) == 0);
      CHECK(std::memcmp(h[r].gdn_small.data() + sc.gdn_off_dtbias, full.gdn_small.data() + s1.gdn_off_dtbias + r * hv,
                        hv) == 0);
      CHECK(std::memcmp(h[r].gdn_small.data() + sc.gdn_off_gated_norm, full.gdn_small.data() + s1.gdn_off_gated_norm,
                        size_t(d.gdn_head) * 2) == 0);
    }
  }
  const size_t gu1 = loader::q4_gate_up_block_bytes(d) / 4, dn1 = loader::q4_down_block_bytes(d) / 4;
  for (uint32_t e = 0; e < d.experts; ++e)
    join_l1(&full.gate_up[e * gu1], &full.down[e * dn1], {&h[0].gate_up[e * gu1 / 2], &h[1].gate_up[e * gu1 / 2]},
            {&h[0].down[e * dn1 / 2], &h[1].down[e * dn1 / 2]}, d.hidden, d.moe_inter);
  const size_t sg1 = full.shared_gate_up, sgc = h[0].shared_gate_up;
  CHECK_EQ(2 * sgc, sg1);
  if (d.forms.shared == model::Q4Form::Int4) {
    const auto u = [](const loader::Q4HostLayer& L, size_t off) { return reinterpret_cast<const uint32_t*>(L.shared.data() + off); };
    join_l1(u(full, 0), u(full, sg1), {u(h[0], 0), u(h[1], 0)}, {u(h[0], sgc), u(h[1], sgc)}, d.hidden, d.shared_inter);
  } else {
    for (uint32_t r = 0; r < 2; ++r) CHECK(std::memcmp(h[r].shared.data(), full.shared.data() + r * sgc, sgc) == 0);
    std::vector<uint16_t> dn((full.shared.size() - sg1) / 2);
    std::memcpy(dn.data(), full.shared.data() + sg1, dn.size() * 2);
    for (uint32_t r = 0; r < 2; ++r) {
      const std::vector<uint16_t> want = loader::q4_tiles_krows(dn, d.shared_inter, d.hidden, r * d.shared_inter / 2,
                                                                d.shared_inter / 2);
      CHECK(std::memcmp(h[r].shared.data() + sgc, want.data(), want.size() * 2) == 0);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  namespace fs = std::filesystem;
  const std::string root = argc > 1 ? argv[1] : "oracle-out-q4exp-synth";
  if (!fs::exists(fs::path(root) / "ours" / "ckpt" / "config.json") ||
      !fs::exists(fs::path(root) / "intel" / "ckpt" / "config.json")) {
    std::printf("SKIP: no synthetic checkpoints under %s (make_synth.py; box: r31.synth)\n", root.c_str());
    return 77;
  }
  for (const char* form : {"ours", "intel"}) {
    const std::string snap = root + "/" + form + "/ckpt/", ple = root + "/" + form + "/ckpt-ple-int8/";
    std::ifstream f(snap + "config.json");
    std::stringstream s;
    s << f.rdbuf();
    const common::json::Value cfg = common::json::parse(s.str());
    const loader::QuantConfig qc = loader::check_qwen4exp_checkpoint_config(cfg);
    loader::SafetensorsSet set(snap);
    const model::Qwen4ExpDesc pre = model::qwen4exp_desc(cfg, model::Q4Forms{});
    const model::Qwen4ExpDesc d = model::qwen4exp_desc(cfg, loader::q4_forms(set, pre));
    const bool ours = std::string(form) == "ours";
    CHECK_EQ(d.layers, 4u);
    CHECK(d.forms.dense == (ours ? model::Q4Form::Int4 : model::Q4Form::Bf16));
    CHECK(d.forms.shared == d.forms.dense && d.forms.mtp_experts == model::Q4Form::Bf16);
    CHECK_EQ(d.forms.expert_group, ours ? 64u : 128u);
    CHECK_EQ(qc.group_size, d.forms.expert_group);
    const loader::QuantScan scan = loader::assert_quant_invariants(set);
    loader::check_quant_scan(qc, scan);
    loader::Q4Checkpoint ck(d, set);
    ck.check_names(true);
    loader::Q4HostLayer h;
    for (uint32_t l = 0; l < d.layers; ++l) {
      ck.repack_layer(l, h);
      CHECK_EQ(h.bytes().total(), loader::q4_layer_bytes(d, l).total());
      check_tp_halves(d, l, h);
    }
    ck.repack_mtp(h);
    CHECK_EQ(h.bytes().total(), loader::q4_mtp_layer_bytes(d).total());
    (void)ck.embed();
    (void)ck.lm_head();
    CHECK_EQ(ck.final_mixer().size(), loader::q4_final_mixer_bytes(d));
    CHECK_EQ(ck.mtp_fc().size(), loader::q4_mtp_fc_offsets(d).total);
    CHECK_EQ(ck.mtp_mixer().size(), loader::q4_final_mixer_bytes(d));
    const loader::Q4PleConstants c = ck.ple_constants();
    CHECK(c.multipliers == loader::q4_ple_multipliers(d.vocab, d.ngram, 0, d.ple_seed));
    CHECK(c.sizes == loader::q4_ple_primes(d.ple_base, d.ple_heads, 0));
    CHECK(c.offsets == loader::q4_ple_offsets(c.sizes));
    const size_t shards = ck.skip_ple_shards();
    CHECK_EQ(shards, size_t(128));
    std::string names;
    const size_t left = ck.unconsumed(&names);
    if (left) std::fprintf(stderr, "unconsumed: %s\n", names.c_str());
    CHECK_EQ(left, size_t(0));
    const loader::Q4PleHost table(ple);
    loader::check_q4_ple(d, table);
    CHECK_EQ(table.bytes(), loader::q4_ple_table_bytes(c.sizes, d.ple_dim, table.scale()));
    const std::array<size_t, runtime::kPpDevices> caps = {32530000000ull, 32530000000ull};
    runtime::qwen4exp::require_fits(runtime::qwen4exp::plan(d, model::Q4Placement::one(d), 32768, true, true), caps,
                                    1500000000ull);
    std::printf("  %s: %zu tensors, forms dense %s / shared %s / experts g%u, 4 layers + the MTP head repacked "
                "(bytes = the layout), 128 PLE shards skipped by design, the int8 file (%s scales, %zu B) = the "
                "descriptor, 0 unconsumed; both tensor-parallel halves of every layer join back bitwise\n",
                form, set.tensors().size(), model::q4_form_name(d.forms.dense), model::q4_form_name(d.forms.shared),
                d.forms.expert_group, loader::q4_ple_scale_name(table.scale()), table.bytes());
  }
  std::printf("qwen4exp_synth_host_test OK\n");
  return 0;
}
