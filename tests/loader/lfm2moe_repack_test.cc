// Spec 24b Task 3: loader/lfm2moe_layout.h's byte formulas and loader/lfm2moe_repack.h's host-only
// repack functions, host only (no Level Zero, no checkpoint on disk except the tiny fixture this
// test writes for detect_expert_names).
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "loader/lfm2moe_layout.h"
#include "loader/lfm2moe_loader.h"
#include "loader/lfm2moe_repack.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/lfm2moe.h"

namespace {
namespace fs = std::filesystem;
using model::Lfm2MoeDesc;
using model::LfmExpertNames;

struct Tensor {
  std::string dtype;
  std::vector<uint64_t> shape;
  std::vector<uint8_t> bytes;
};
using Tensors = std::map<std::string, Tensor>;

void write_checkpoint(const fs::path& dir, const Tensors& ts) {
  fs::create_directories(dir);
  std::string hdr = "{";
  uint64_t off = 0;
  bool first = true;
  for (const auto& [name, t] : ts) {
    hdr += (first ? "" : ",");
    first = false;
    hdr += "\"" + name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
    for (size_t i = 0; i < t.shape.size(); ++i) hdr += (i ? "," : "") + std::to_string(t.shape[i]);
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + t.bytes.size()) + "]}";
    off += t.bytes.size();
    off = (off + 7) / 8 * 8;
  }
  hdr += "}";
  while ((8 + hdr.size()) % 8 != 0) hdr += ' ';
  std::ofstream f(dir / "model-00001-of-00001.safetensors", std::ios::binary);
  const uint64_t hlen = hdr.size();
  f.write(reinterpret_cast<const char*>(&hlen), 8);
  f.write(hdr.data(), std::streamsize(hdr.size()));
  for (const auto& [name, t] : ts) {
    (void)name;
    f.write(reinterpret_cast<const char*>(t.bytes.data()), std::streamsize(t.bytes.size()));
    uint64_t pad = (t.bytes.size() + 7) / 8 * 8 - t.bytes.size();
    const char zeros[8] = {};
    f.write(zeros, std::streamsize(pad));
  }
  std::ofstream ix(dir / "model.safetensors.index.json");
  ix << "{\"metadata\":{},\"weight_map\":{";
  bool f2 = true;
  for (const auto& [name, t] : ts) {
    (void)t;
    ix << (f2 ? "" : ",") << "\"" << name << "\":\"model-00001-of-00001.safetensors\"";
    f2 = false;
  }
  ix << "}}";
}

Tensor i32(std::vector<uint64_t> shape, uint32_t fill) {
  Tensor t;
  t.dtype = "I32";
  t.shape = std::move(shape);
  uint64_t n = 1;
  for (uint64_t s : t.shape) n *= s;
  t.bytes.resize(n * 4);
  uint32_t* p = reinterpret_cast<uint32_t*>(t.bytes.data());
  for (uint64_t i = 0; i < n; ++i) p[i] = fill;
  return t;
}
Tensor f16(std::vector<uint64_t> shape, uint16_t fill) {
  Tensor t;
  t.dtype = "F16";
  t.shape = std::move(shape);
  uint64_t n = 1;
  for (uint64_t s : t.shape) n *= s;
  t.bytes.resize(n * 2);
  uint16_t* p = reinterpret_cast<uint16_t*>(t.bytes.data());
  for (uint64_t i = 0; i < n; ++i) p[i] = fill;
  return t;
}

Tensor bf16_t(std::vector<uint64_t> shape) {
  Tensor t;
  t.dtype = "BF16";
  t.shape = std::move(shape);
  uint64_t n = 1;
  for (uint64_t s : t.shape) n *= s;
  t.bytes.assign(n * 2, 0);
  return t;
}
Tensor f32_t(std::vector<uint64_t> shape, float fill = 0.0f) {
  Tensor t;
  t.dtype = "F32";
  t.shape = std::move(shape);
  uint64_t n = 1;
  for (uint64_t s : t.shape) n *= s;
  t.bytes.resize(n * 4);
  float* p = reinterpret_cast<float*>(t.bytes.data());
  for (uint64_t i = 0; i < n; ++i) p[i] = fill;
  return t;
}
void int4_set(Tensors& ts, const std::string& base, uint32_t K, uint32_t N) {
  ts[base + ".qweight"] = i32({K / 8, N}, 0);
  ts[base + ".qzeros"] = i32({K / 64, N / 8}, 0x77777777u);
  ts[base + ".scales"] = f16({K / 64, N}, 0x3C00);   // 1.0
}

// A SMALL but fully valid LFM2.5-24B-A2B-shaped checkpoint (a test copy of the descriptor, as
// kolibri1_repack_test.cc's - the widths are what the layouts depend on, not the published 40 / 64 /
// 2048): every tensor lfm_expected_names(d, names) lists, correctly shaped and typed, so
// check_names() passes and a full repack walk leaves 0 unconsumed - the baseline I5's crafted
// corruptions below each break exactly one way.
model::Lfm2MoeDesc tiny_desc() {
  model::Lfm2MoeDesc d = model::lfm2moe();
  d.layers = 3;
  d.hidden = 64;
  d.q_heads = 1;
  d.kv_heads = 1;
  d.head_dim = 64;
  d.dense_layers = 1;
  d.dense_inter = 64;
  // 16, not 2: common::repack_bf16_tiled (the router) addresses output columns in whole 16-wide
  // tiles - valid for the real model (64 experts, already a multiple of 16; no padding needed,
  // spec 24 "24b as built"'s note), but a smaller count here would read past the output buffer.
  d.experts = 16;
  d.top_k = 1;
  d.moe_inter = 64;
  d.vocab = 64;
  d.kinds = {model::LfmKind::Conv, model::LfmKind::Attention, model::LfmKind::Conv};   // dense-conv, attn+MoE, conv+MoE
  return d;
}

Tensors build_full_checkpoint(const Lfm2MoeDesc& d, LfmExpertNames names) {
  Tensors ts;
  ts[d.prefix + "embed_tokens.weight"] = bf16_t({d.vocab, d.hidden});
  ts[d.prefix + "embedding_norm.weight"] = bf16_t({d.hidden});
  const char* const kW[3] = {"w1", "w3", "w2"};
  const char* const kProj[3] = {"gate_proj", "up_proj", "down_proj"};
  const char* const* ex = names == LfmExpertNames::Proj ? kProj : kW;
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string lp = d.layer_prefix(l);
    ts[lp + "operator_norm.weight"] = bf16_t({d.hidden});
    ts[lp + "ffn_norm.weight"] = bf16_t({d.hidden});
    if (d.is_attention(l)) {
      ts[lp + "self_attn.q_layernorm.weight"] = bf16_t({d.head_dim});
      ts[lp + "self_attn.k_layernorm.weight"] = bf16_t({d.head_dim});
      int4_set(ts, lp + "self_attn.q_proj", d.hidden, d.q_n());
      int4_set(ts, lp + "self_attn.k_proj", d.hidden, d.kv_n());
      int4_set(ts, lp + "self_attn.v_proj", d.hidden, d.kv_n());
      int4_set(ts, lp + "self_attn.out_proj", d.q_n(), d.hidden);
    } else {
      ts[lp + "conv.conv.weight"] = bf16_t({d.hidden, 1, d.conv_taps});
      int4_set(ts, lp + "conv.in_proj", d.hidden, 3 * d.hidden);
      int4_set(ts, lp + "conv.out_proj", d.hidden, d.hidden);
    }
    if (d.is_dense(l)) {
      int4_set(ts, lp + "feed_forward.w1", d.hidden, d.dense_inter);
      int4_set(ts, lp + "feed_forward.w3", d.hidden, d.dense_inter);
      int4_set(ts, lp + "feed_forward.w2", d.dense_inter, d.hidden);
    } else {
      ts[lp + "feed_forward.gate.weight"] = bf16_t({d.experts, d.hidden});
      ts[lp + "feed_forward.expert_bias"] = f32_t({d.experts});
      for (uint32_t e = 0; e < d.experts; ++e) {
        const std::string ep = lp + "feed_forward.experts." + std::to_string(e) + ".";
        int4_set(ts, ep + ex[0], d.hidden, d.moe_inter);
        int4_set(ts, ep + ex[1], d.hidden, d.moe_inter);
        int4_set(ts, ep + ex[2], d.moe_inter, d.hidden);
      }
    }
  }
  return ts;
}

// Walks every layer (repack_layer), embed(), final_norm() - the full consumption a real load does.
void full_walk(loader::Lfm2MoeCheckpoint& ck, const Lfm2MoeDesc& d) {
  loader::HostLayer host;
  for (uint32_t l = 0; l < d.layers; ++l) ck.repack_layer(l, false, host);
  ck.embed();
  ck.final_norm();
}

}  // namespace

int main() {
  const Lfm2MoeDesc& d = model::lfm2moe();

  // --- the byte formulas (asserted, not a constant typed from the plan) ------------------------
  {
    CHECK_EQ(loader::lfm::expert_gate_up_bytes(d), loader::lfm::int4_l1_bytes(d.hidden, 2 * d.moe_inter));
    CHECK_EQ(loader::lfm::expert_down_bytes(d), loader::lfm::int4_l1_bytes(d.moe_inter, d.hidden));
    CHECK_EQ(loader::lfm::moe_layer_bytes(d),
            size_t(d.experts) * (loader::lfm::expert_gate_up_bytes(d) + loader::lfm::expert_down_bytes(d)));
    CHECK_EQ(loader::lfm::expert_gate_up_offset(d, 5), size_t(5) * loader::lfm::expert_gate_up_bytes(d));
    CHECK_EQ(loader::lfm::expert_down_offset(d, 5), size_t(5) * loader::lfm::expert_down_bytes(d));
    CHECK_EQ(loader::lfm::router_bytes(d), size_t(d.hidden) * d.experts * 2);
    CHECK_EQ(loader::lfm::embed_tiled_bytes(d), size_t(d.hidden) * d.vocab * 2);
    CHECK_EQ(loader::lfm::head_int8_bytes(d), size_t(d.hidden) * d.vocab + size_t(d.vocab) * 4);
    CHECK_EQ(loader::lfm::rope_table_bytes(d, 131072), size_t(131072) * 2 * (d.head_dim / 2) * 4);
    CHECK_EQ(d.rope_table_bytes(131072), loader::lfm::rope_table_bytes(d, 131072));
    CHECK_EQ(loader::lfm::kv_bytes_per_position(d), size_t(d.attn_layers()) * d.kv_heads * d.head_dim * 2 * 2);
    CHECK_EQ(loader::lfm::conv_ring_bytes(d), size_t(d.conv_layers()) * loader::lfm::kConvRing * d.hidden * 2);
    // the printed figures, cross-checked against spec 24 §3's (derived independently there)
    std::printf("  expert_gate_up %zu, expert_down %zu, moe_layer %zu, router %zu, embed %zu, head_int8 %zu\n",
               loader::lfm::expert_gate_up_bytes(d), loader::lfm::expert_down_bytes(d),
               loader::lfm::moe_layer_bytes(d), loader::lfm::router_bytes(d), loader::lfm::embed_tiled_bytes(d),
               loader::lfm::head_int8_bytes(d));
    CHECK_EQ(loader::lfm::expert_gate_up_bytes(d), size_t(3342336));
    CHECK_EQ(loader::lfm::expert_down_bytes(d), size_t(1671168));
    CHECK_EQ(loader::lfm::moe_layer_bytes(d), size_t(320864256));
    CHECK_EQ(loader::lfm::router_bytes(d), size_t(262144));
    CHECK_EQ(loader::lfm::embed_tiled_bytes(d), size_t(524288000));
    CHECK_EQ(loader::lfm::head_int8_bytes(d), size_t(262656000));
    CHECK_EQ(loader::lfm::kv_bytes_per_position(d), size_t(20480));
    CHECK_EQ(loader::lfm::conv_ring_bytes(d), size_t(1966080));
    // weight_bytes is its own sum of the formulas above, not a second hardcoded figure.
    size_t want = loader::lfm::embed_tiled_bytes(d) + size_t(d.hidden) * 4 + loader::lfm::head_int8_bytes(d);
    for (uint32_t l = 0; l < d.layers; ++l) want += loader::lfm::layer_weight_bytes(d, l);
    CHECK_EQ(loader::lfm::weight_bytes(d, true), want);
    std::printf("  weight_bytes(int8 head) %zu B (%.3f GB) - spec 24 §3's 13.39 GB\n",
               loader::lfm::weight_bytes(d, true), loader::lfm::weight_bytes(d, true) / 1e9);
    CHECK(loader::lfm::weight_bytes(d, true) > size_t(13.3e9) && loader::lfm::weight_bytes(d, true) < size_t(13.5e9));
  }

  // --- small_block: disjoint, 64-byte aligned ---------------------------------------------------
  for (uint32_t l : {0u, 1u, 2u, 3u, 39u}) {
    const loader::lfm::SmallBlock b = loader::lfm::small_block(d, l);
    CHECK_EQ(b.operator_norm % 64, size_t(0));
    CHECK_EQ(b.ffn_norm % 64, size_t(0));
    CHECK(b.ffn_norm >= b.operator_norm + size_t(d.hidden) * 4);
    if (d.is_attention(l)) {
      CHECK_EQ(b.conv_taps, size_t(0));   // unused for an attention layer
      CHECK(b.q_norm % 64 == 0 && b.k_norm % 64 == 0 && b.k_norm >= b.q_norm + size_t(d.head_dim) * 4);
    } else {
      CHECK(b.q_norm == 0 && b.k_norm == 0);
      CHECK_EQ(b.conv_taps % 64, size_t(0));
    }
    if (!d.is_dense(l)) CHECK_EQ(b.expert_bias % 64, size_t(0));
    CHECK_EQ(b.bytes % 64, size_t(0));
  }

  // --- repack_expert: Review Focus 1 - gate tile t then up tile t, both spellings ----------------
  {
    const uint32_t H = 64, I = 64;   // small, but K % 64 == 0 and N % 16 == 0 everywhere
    std::vector<uint32_t> qw1(H / 8 * I, 0x11111111u), qw3(H / 8 * I, 0x22222222u), qw2(I / 8 * H, 0x33333333u);
    std::vector<uint16_t> sc1(H / 64 * I, 0x3C00), sc3(H / 64 * I, 0xBC00), sc2(I / 64 * H, 0x3C00);
    loader::LinearSrc w1, w3, w2;
    w1.kind = loader::WKind::Int4;
    w1.K = H;
    w1.N = I;
    w1.qweight = qw1.data();
    w1.scales = sc1.data();
    w3 = w1;
    w3.qweight = qw3.data();
    w3.scales = sc3.data();
    w2.kind = loader::WKind::Int4;
    w2.K = I;
    w2.N = H;
    w2.qweight = qw2.data();
    w2.scales = sc2.data();
    const loader::ExpertRepacked r = loader::repack_expert(w1, w3, w2);
    CHECK_EQ(r.gate_up.size(), loader::lfm::int4_l1_bytes(H, 2 * I) / 4);
    const uint32_t G = H / 64, NT = 2 * I / 16;   // 1, 8
    for (uint32_t nt = 0; nt < NT; ++nt) {
      const uint32_t expect = (nt % 2 == 0) ? 0x11111111u : 0x22222222u;   // even tiles gate, odd up
      const uint16_t expect_sc = (nt % 2 == 0) ? 0x3C00 : 0xBC00;
      for (uint32_t g = 0; g < G; ++g) {
        const uint32_t* tile = r.gate_up.data() + (size_t(nt) * G + g) * 136;
        for (uint32_t w = 0; w < 128; ++w) CHECK_EQ(tile[w], expect);
        for (uint32_t w = 0; w < 8; ++w) CHECK_EQ(tile[128 + w] & 0xFFFF, uint32_t(expect_sc));
      }
    }
    for (uint32_t w = 0; w < r.down.size(); ++w)
      if (w % 136 < 128) CHECK_EQ(r.down[w], 0x33333333u);
    std::printf("  repack_expert: gate tile t then up tile t (Review Focus 1), down verbatim\n");
  }

  // --- detect_expert_names: both spellings, and a mix refused -------------------------------------
  {
    const fs::path tmp = fs::temp_directory_path() / ("lfm2moe_repack_test_" + std::to_string(::getpid()));
    for (const char* spell : {"w1", "gate_proj"}) {
      Tensors ts;
      ts["model.layers.2.feed_forward.experts.0." + std::string(spell) + ".qweight"] = i32({8, 64}, 0);
      ts["model.layers.2.feed_forward.experts.0." + std::string(spell) + ".scales"] = f16({1, 64}, 0);
      write_checkpoint(tmp, ts);
      loader::SafetensorsSet set(tmp.string() + "/");
      const LfmExpertNames got = loader::detect_expert_names(set);
      CHECK(got == (std::string(spell) == "w1" ? LfmExpertNames::W : LfmExpertNames::Proj));
      fs::remove_all(tmp);
    }
    {
      Tensors ts;
      ts["model.layers.2.feed_forward.experts.0.w1.qweight"] = i32({8, 64}, 0);
      ts["model.layers.2.feed_forward.experts.0.w1.scales"] = f16({1, 64}, 0);
      ts["model.layers.2.feed_forward.experts.1.gate_proj.qweight"] = i32({8, 64}, 0);
      ts["model.layers.2.feed_forward.experts.1.gate_proj.scales"] = f16({1, 64}, 0);
      write_checkpoint(tmp, ts);
      loader::SafetensorsSet set(tmp.string() + "/");
      bool threw = false;
      try {
        (void)loader::detect_expert_names(set);
      } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("mixes") != std::string::npos;
      }
      CHECK(threw);
      fs::remove_all(tmp);
    }
    std::printf("  detect_expert_names: both spellings read, a mix refused\n");
  }

  // --- repack_in_proj: the plain form, the conv-tile permutation, its inverse word for word -------
  {
    const uint32_t H = 64, K = 64, N = 3 * H;   // B | C | x, 192 columns, 12 n_tiles
    std::vector<uint32_t> qw(K / 8 * N);
    std::vector<uint16_t> sc(K / 64 * N);
    std::mt19937 rng(7);
    for (auto& w : qw) w = rng();
    for (auto& s : sc) s = uint16_t(rng());
    const loader::HostLinear flat = loader::repack_in_proj(qw.data(), sc.data(), K, N, false);
    const loader::HostLinear tiled = loader::repack_in_proj(qw.data(), sc.data(), K, N, true);
    CHECK(flat.scales.empty() && tiled.scales.empty());   // layout 1 (the default)
    CHECK_EQ(flat.words.size(), tiled.words.size());
    const uint32_t G = K / 64, TILE_W = 136;
    // true_tile[3t + part] == flat_tile[part x (H / 16) + t] (the arm-c permutation, spec 24 §4.1).
    for (uint32_t t = 0; t < H / 16; ++t)
      for (uint32_t part = 0; part < 3; ++part) {
        const uint32_t true_nt = 3 * t + part, flat_nt = part * (H / 16) + t;
        for (uint32_t g = 0; g < G; ++g) {
          const uint32_t* a = tiled.words.data() + (size_t(true_nt) * G + g) * TILE_W;
          const uint32_t* b = flat.words.data() + (size_t(flat_nt) * G + g) * TILE_W;
          CHECK(std::memcmp(a, b, TILE_W * 4) == 0);
        }
      }
    // I1: "a column map valid in both" - the same permutation at layout 0 (words + separate
    // scales), independently reconstructed via repack_int4_layout0_cols and compared.
    {
      const loader::HostLinear flat0 = loader::repack_in_proj(qw.data(), sc.data(), K, N, false, 0);
      const loader::HostLinear tiled0 = loader::repack_in_proj(qw.data(), sc.data(), K, N, true, 0);
      CHECK_EQ(flat0.words.size(), size_t(K / 8) * N);
      CHECK_EQ(flat0.scales.size(), size_t(K / 64) * N);
      std::vector<uint32_t> want_words(size_t(K / 8) * N);
      std::vector<uint16_t> want_scales(size_t(K / 64) * N);
      common::repack_int4_layout0_cols(K, N, common::cols_concat({{qw.data(), sc.data(), N}}), want_words.data(),
                                       want_scales.data());
      CHECK(flat0.words == want_words && flat0.scales == want_scales);
      // the conv-tile permutation, independently permuted column by column, at layout 0 too.
      std::vector<common::ColSource> perm_cols(N);
      uint32_t idx = 0;
      for (uint32_t t = 0; t < H / 16; ++t)
        for (uint32_t part = 0; part < 3; ++part)
          for (uint32_t l = 0; l < 16; ++l) perm_cols[idx++] = {qw.data(), sc.data(), part * H + t * 16 + l, N};
      std::vector<uint32_t> want_tiled_words(size_t(K / 8) * N);
      std::vector<uint16_t> want_tiled_scales(size_t(K / 64) * N);
      common::repack_int4_layout0_cols(K, N, perm_cols, want_tiled_words.data(), want_tiled_scales.data());
      CHECK(tiled0.words == want_tiled_words && tiled0.scales == want_tiled_scales);
    }
    std::printf("  repack_in_proj: conv_tiles permutes whole n_tiles; its inverse gives the plain "
               "layout back word for word\n");
  }

  // --- lfm_embed_row_index: against the row-major tensor (Review Focus 3) ------------------------
  {
    const uint32_t K = d.hidden, N = d.vocab;   // 2048, 128000 (the real embedding's shape)
    std::vector<uint16_t> full(size_t(N) * K, 0);
    std::mt19937 rng2(11);
    std::set<uint32_t> ns = {0, 124894, 127999};
    while (ns.size() < 64) ns.insert(rng2() % N);
    for (uint32_t n : ns)
      for (uint32_t k = 0; k < K; ++k) full[size_t(n) * K + k] = uint16_t(rng2());
    std::vector<uint16_t> tiled(size_t(K) * N);
    common::repack_bf16_tiled(full.data(), K, N, tiled.data());
    for (uint32_t n : ns)
      for (uint32_t k = 0; k < K; ++k)
        CHECK_EQ(tiled[loader::lfm::lfm_embed_row_index(K, k, n)], full[size_t(n) * K + k]);
    std::printf("  lfm_embed_row_index: every k of 64 ids (incl. 0, 124894, 127999) against the row-major "
               "tensor through the tiles\n");
  }

  // --- I5: crafted-checkpoint refusal cases (check_names, take()'s dtype refusal, the tied head) ----
  {
    const fs::path tmp = fs::temp_directory_path() / ("lfm2moe_repack_crafted_" + std::to_string(::getpid()));
    const Lfm2MoeDesc td = tiny_desc();

    // the baseline: valid, check_names passes, a full walk leaves 0 unconsumed.
    {
      Tensors ts = build_full_checkpoint(td, LfmExpertNames::W);
      write_checkpoint(tmp, ts);
      loader::SafetensorsSet set(tmp.string() + "/");
      loader::Lfm2MoeCheckpoint ck(td, set, LfmExpertNames::W);
      ck.check_names();   // must not throw
      full_walk(ck, td);
      CHECK_EQ(ck.unconsumed(), size_t(0));
      fs::remove_all(tmp);
    }
    // an unknown extra tensor is refused by name.
    {
      Tensors ts = build_full_checkpoint(td, LfmExpertNames::W);
      ts["model.layers.0.something_else.weight"] = bf16_t({4});
      write_checkpoint(tmp, ts);
      loader::SafetensorsSet set(tmp.string() + "/");
      loader::Lfm2MoeCheckpoint ck(td, set, LfmExpertNames::W);
      bool threw = false;
      try {
        ck.check_names();
      } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("something_else") != std::string::npos;
      }
      CHECK(threw);
      fs::remove_all(tmp);
    }
    // a missing required tensor is refused by name.
    {
      Tensors ts = build_full_checkpoint(td, LfmExpertNames::W);
      const std::string victim = td.layer_prefix(2) + "feed_forward.experts.1.w2.qweight";
      CHECK(ts.erase(victim) == 1);
      write_checkpoint(tmp, ts);
      loader::SafetensorsSet set(tmp.string() + "/");
      loader::Lfm2MoeCheckpoint ck(td, set, LfmExpertNames::W);
      bool threw = false;
      try {
        ck.check_names();
      } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find(victim) != std::string::npos;
      }
      CHECK(threw);
      fs::remove_all(tmp);
    }
    // expert_bias shipped as BF16 (not F32) is refused by take()'s dtype check, at repack time.
    {
      Tensors ts = build_full_checkpoint(td, LfmExpertNames::W);
      const std::string name = td.layer_prefix(1) + "feed_forward.expert_bias";
      ts[name] = bf16_t({td.experts});   // check_names only checks names; the dtype throw is take()'s
      write_checkpoint(tmp, ts);
      loader::SafetensorsSet set(tmp.string() + "/");
      loader::Lfm2MoeCheckpoint ck(td, set, LfmExpertNames::W);
      ck.check_names();
      bool threw = false;
      try {
        full_walk(ck, td);
      } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find(name) != std::string::npos && std::string(e.what()).find("F32") != std::string::npos;
      }
      CHECK(threw);
      fs::remove_all(tmp);
    }
    // the tied head: lm_head.weight present and bitwise == embed_tokens is skipped, not refused.
    {
      Tensors ts = build_full_checkpoint(td, LfmExpertNames::W);
      ts["lm_head.weight"] = ts[td.prefix + "embed_tokens.weight"];   // bitwise copy
      write_checkpoint(tmp, ts);
      loader::SafetensorsSet set(tmp.string() + "/");
      loader::Lfm2MoeCheckpoint ck(td, set, LfmExpertNames::W);
      ck.check_names();
      ck.embed();
      const std::string note = ck.check_tied_lm_head();   // must not throw
      CHECK(!note.empty());
      full_walk(ck, td);
      CHECK_EQ(ck.unconsumed(), size_t(0));   // lm_head.weight is consumed too (M1/I5)
      fs::remove_all(tmp);
    }
    // the tied head: lm_head.weight present and DIFFERENT is refused naming both.
    {
      Tensors ts = build_full_checkpoint(td, LfmExpertNames::W);
      Tensor lm = ts[td.prefix + "embed_tokens.weight"];
      reinterpret_cast<uint16_t*>(lm.bytes.data())[0] ^= 0xFFFF;   // one bit flip: no longer equal
      ts["lm_head.weight"] = lm;
      write_checkpoint(tmp, ts);
      loader::SafetensorsSet set(tmp.string() + "/");
      loader::Lfm2MoeCheckpoint ck(td, set, LfmExpertNames::W);
      ck.check_names();
      ck.embed();
      bool threw = false;
      try {
        (void)ck.check_tied_lm_head();
      } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        threw = msg.find("lm_head") != std::string::npos && msg.find("embed_tokens") != std::string::npos;
      }
      CHECK(threw);
      fs::remove_all(tmp);
    }
    std::printf("  crafted refusals: an unknown tensor, a missing one, expert_bias as BF16, a tied "
               "lm_head (skipped), a different lm_head (refused naming both)\n");
  }

  std::printf("lfm2moe_repack_test: ok\n");
  return 0;
}
