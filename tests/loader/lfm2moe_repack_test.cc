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
    CHECK_EQ(loader::lfm::rope_table_bytes(131072), size_t(131072) * 2 * 32 * 4);
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
    const std::vector<uint32_t> flat = loader::repack_in_proj(qw.data(), sc.data(), K, N, false);
    const std::vector<uint32_t> tiled = loader::repack_in_proj(qw.data(), sc.data(), K, N, true);
    CHECK_EQ(flat.size(), tiled.size());
    const uint32_t G = K / 64, TILE_W = 136;
    // true_tile[3t + part] == flat_tile[part x (H / 16) + t] (the arm-c permutation, spec 24 §4.1).
    for (uint32_t t = 0; t < H / 16; ++t)
      for (uint32_t part = 0; part < 3; ++part) {
        const uint32_t true_nt = 3 * t + part, flat_nt = part * (H / 16) + t;
        for (uint32_t g = 0; g < G; ++g) {
          const uint32_t* a = tiled.data() + (size_t(true_nt) * G + g) * TILE_W;
          const uint32_t* b = flat.data() + (size_t(flat_nt) * G + g) * TILE_W;
          CHECK(std::memcmp(a, b, TILE_W * 4) == 0);
        }
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

  std::printf("lfm2moe_repack_test: ok\n");
  return 0;
}
