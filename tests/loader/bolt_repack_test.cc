// Spec 25b Task 4: the Bolt layouts and the host repack, over crafted checkpoints (no device, no weights).
//   * Review Focus 1: a linear read from the g128 GPTQ spelling and the same weights packed g64 with each g128 scale written
//     to both g64 rows give BYTE-IDENTICAL device buffers (both layouts, fused, the expert block)
//   * Review Focus 3: half the scales negative survive the repack bitwise and the tile decodes to (q - 8) x scale
//   * the 64 B alignment of every size and offset, the expert address formula, the bytes of spec 25 §3 re-derived from the
//     formulas (printed; no total typed), the trunk alias, exp_torch
// Host only. No argument.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/exp_torch.h"
#include "common/kv8.h"
#include "common/repack.h"
#include "loader/bolt_layout.h"
#include "loader/bolt_repack.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/bolt.h"

namespace {

void write_st(const std::string& path, const std::string& header, const std::vector<uint8_t>& data) {
  std::string h = header;
  while ((8 + h.size()) % 8 != 0) h.push_back(' ');
  std::ofstream f(path, std::ios::binary);
  uint64_t n = h.size();
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(h.data(), std::streamsize(h.size()));
  f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
}

// One checkpoint directory of GPTQ linears: each entry (name, K, N), words random, g128 scales signed (random sign bit).
struct Lin {
  std::string name;
  uint32_t K, N;
  std::vector<uint32_t> qweight;
  std::vector<uint16_t> s128;   // [K/128][N]
};

Lin make_lin(const std::string& name, uint32_t K, uint32_t N, uint32_t seed) {
  Lin l{name, K, N, {}, {}};
  std::mt19937 rng(seed);
  l.qweight.resize(size_t(K / 8) * N);
  for (auto& w : l.qweight) w = rng();
  l.s128.resize(size_t(K / 128) * N);
  std::uniform_real_distribution<float> mag(0.002f, 0.06f);
  for (auto& s : l.s128) {
    float v = mag(rng);
    if (rng() & 1) v = -v;                                 // about half negative (AutoRound's signed scale)
    s = common::kv8::f32_to_f16_rne(v);
  }
  // a few f16 subnormals, positive and negative
  if (l.s128.size() > 4) {
    l.s128[1] = 0x00A8;
    l.s128[2] = 0x80A8;
  }
  return l;
}

// Writes the checkpoint directory `dir` holding `lins` in the g128 GPTQ spelling (with g_idx and qzeros) or as the exact g64
// expansion (each g128 scale row written twice, no g_idx).
void write_dir(const std::string& dir, const std::vector<Lin>& lins, bool g128) {
  std::vector<uint8_t> bytes;
  std::string hdr = "{", ix = R"({"weight_map":{)";
  auto put = [&](const void* p, size_t n) -> std::pair<size_t, size_t> {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    const size_t at = bytes.size();
    bytes.insert(bytes.end(), b, b + n);
    return {at, bytes.size()};
  };
  auto entry = [&](const std::string& name, const char* dt, const std::string& shape, std::pair<size_t, size_t> off) {
    if (hdr.size() > 1) hdr += ",";
    hdr += "\"" + name + "\":{\"dtype\":\"" + dt + "\",\"shape\":" + shape + ",\"data_offsets\":[" +
           std::to_string(off.first) + "," + std::to_string(off.second) + "]}";
    if (ix.back() != '{') ix += ",";
    ix += "\"" + name + "\":\"w.safetensors\"";
  };
  for (const Lin& l : lins) {
    const uint32_t G = g128 ? l.K / 128 : l.K / 64;
    entry(l.name + ".qweight", "I32", "[" + std::to_string(l.K / 8) + "," + std::to_string(l.N) + "]",
          put(l.qweight.data(), l.qweight.size() * 4));
    std::vector<uint16_t> sc;
    if (g128) {
      sc = l.s128;
    } else {
      sc.resize(size_t(G) * l.N);
      for (uint32_t j = 0; j < l.K / 128; ++j)
        for (uint32_t n = 0; n < l.N; ++n) sc[size_t(2 * j) * l.N + n] = sc[size_t(2 * j + 1) * l.N + n] = l.s128[size_t(j) * l.N + n];
    }
    entry(l.name + ".scales", "F16", "[" + std::to_string(G) + "," + std::to_string(l.N) + "]", put(sc.data(), sc.size() * 2));
    const std::vector<uint32_t> qz(size_t(G) * (l.N / 8), 0x77777777u);
    entry(l.name + ".qzeros", "I32", "[" + std::to_string(G) + "," + std::to_string(l.N / 8) + "]", put(qz.data(), qz.size() * 4));
    if (g128) {
      std::vector<int32_t> gi(l.K);
      for (uint32_t k = 0; k < l.K; ++k) gi[k] = int32_t(k / 128);
      entry(l.name + ".g_idx", "I32", "[" + std::to_string(l.K) + "]", put(gi.data(), gi.size() * 4));
    }
  }
  hdr += "}";
  ix += "}}";
  write_st(dir + "/w.safetensors", hdr, bytes);
  std::ofstream(dir + "/model.safetensors.index.json") << ix;
}

// Layout 1: the (k, n) weight's nibble and its scale out of a repacked tile buffer.
int tile_q(const std::vector<uint32_t>& w, uint32_t K, uint32_t k, uint32_t n) {
  const uint32_t G = K / 64, nt = n / 16, g = k / 64, kk = k % 64;
  return int((w[(size_t(nt) * G + g) * 136 + (kk / 8) * 16 + n % 16] >> (4 * (kk % 8))) & 0xFu);
}
uint16_t tile_s(const std::vector<uint32_t>& w, uint32_t K, uint32_t k, uint32_t n) {
  const uint32_t G = K / 64, nt = n / 16, g = k / 64;
  const uint32_t word = w[(size_t(nt) * G + g) * 136 + 128 + (n % 16) / 2];
  return uint16_t((n % 2) ? word >> 16 : word & 0xFFFFu);
}

bool same(const void* a, const void* b, size_t n) { return std::memcmp(a, b, n) == 0; }

}  // namespace

int main() {
  using loader::LinearSrc;
  // per process: two box validation streams (one per card) may run this test at once
  const std::string base = "/tmp/b70_bolt_repack_test." + std::to_string(getpid());
  CHECK(system(("mkdir -p " + base + "/g128 " + base + "/g64 " + base + "/g128e " + base + "/g64e").c_str()) == 0);

  // ---------------------------------------------------------------------------------------------------------------
  // Review Focus 1 + 3: one set of weights, two spellings, one device layout; the signs survive.
  // ---------------------------------------------------------------------------------------------------------------
  const uint32_t Kq = 256;   // the fused q||k||v stand-in: three parts K=256, N 64 + 16 + 16 (a g128 K)
  std::vector<Lin> lins = {make_lin("q", Kq, 64, 1),  make_lin("k", Kq, 16, 2),    make_lin("v", Kq, 16, 3),
                           make_lin("up", 256, 96, 4), make_lin("down", 384, 256, 5), make_lin("one", 512, 128, 6)};
  write_dir(base + "/g128", lins, true);
  write_dir(base + "/g64", lins, false);
  loader::SafetensorsSet s128(base + "/g128/"), s64(base + "/g64/");
  const loader::QuantScan sc128 = loader::assert_quant_invariants(s128), sc64 = loader::assert_quant_invariants(s64);
  CHECK_EQ(sc128.g_idx_tensors, size_t(6));   // g_idx = i // 128 for every linear of the g128 spelling
  CHECK_EQ(sc64.g_idx_tensors, size_t(0));
  size_t neg = 0, tot = 0;
  for (const Lin& l : lins)
    for (uint16_t s : l.s128) {
      neg += (s & 0x8000u) != 0;
      ++tot;
    }
  CHECK(neg > tot / 3 && neg < 2 * tot / 3);   // about half negative
  std::printf("bolt_repack_test: %zu of %zu g128 scales negative (%.1f %%)\n", neg, tot, 100.0 * double(neg) / double(tot));

  std::vector<LinearSrc> a, b;
  for (const Lin& l : lins) {
    a.push_back(LinearSrc::classify(s128, l.name));
    b.push_back(LinearSrc::classify(s64, l.name));
    CHECK_EQ(a.back().group, 128u);
    CHECK_EQ(b.back().group, 64u);
    CHECK(same(a.back().scales, b.back().scales, size_t(l.K / 64) * l.N * 2));   // the expansion IS the g64 spelling
  }
  // every layout, fused (Concat) and single
  for (uint32_t layout : {0u, 1u}) {
    model::BoltLinear qkv{model::BoltLinearId::Qkv, {Kq, 96, 1, layout}, model::WeightKind::Int4, model::Fuse::Concat,
                          {"q", "k", "v"}, {64, 16, 16}};
    const loader::BoltHostLinear x = loader::repack_bolt_linear(qkv, {a[0], a[1], a[2]});
    const loader::BoltHostLinear y = loader::repack_bolt_linear(qkv, {b[0], b[1], b[2]});
    CHECK_EQ(x.words.size(), y.words.size());
    CHECK(same(x.words.data(), y.words.data(), x.words.size() * 4));
    CHECK_EQ(x.scales.size(), y.scales.size());
    CHECK(same(x.scales.data(), y.scales.data(), x.scales.size() * 2));
    CHECK_EQ(x.scales.empty(), layout == 1u);   // inline at layout 1
    for (size_t p = 0; p < lins.size(); ++p) {
      model::BoltLinear one{model::BoltLinearId::InProj, {lins[p].K, lins[p].N, 1, layout}, model::WeightKind::Int4,
                            model::Fuse::Single, {lins[p].name}, {lins[p].N}};
      const loader::BoltHostLinear s = loader::repack_bolt_linear(one, {a[p]}), t = loader::repack_bolt_linear(one, {b[p]});
      CHECK(same(s.words.data(), t.words.data(), s.words.size() * 4));
      CHECK(same(s.scales.data(), t.scales.data(), s.scales.size() * 2));
      // sizes are loader::bolt's formulas
      CHECK_EQ(s.words.size() * 4 + s.scales.size() * 2, loader::bolt::int4_bytes({lins[p].K, lins[p].N, 1, layout}));
    }
    // Review Focus 3 at layout 1: every (k, n) of the fused linear decodes to (q - 8) x the g128 checkpoint's own signed
    // scale, and the scale word is the checkpoint's bit for bit (sign included)
    if (layout == 1) {
      const std::vector<uint32_t> cols_off = {0, 64, 80};
      for (uint32_t part = 0; part < 3; ++part)
        for (uint32_t k = 0; k < Kq; ++k)
          for (uint32_t n = 0; n < lins[part].N; ++n) {
            const uint32_t fn = cols_off[part] + n;
            const int q = int((lins[part].qweight[size_t(k / 8) * lins[part].N + n] >> (4 * (k % 8))) & 0xFu);
            CHECK_EQ(tile_q(x.words, Kq, k, fn), q);
            const uint16_t s = lins[part].s128[size_t(k / 128) * lins[part].N + n];
            CHECK_EQ(tile_s(x.words, Kq, k, fn), s);
            CHECK_EQ(common::f16_to_f32(tile_s(x.words, Kq, k, fn)) * float(q - 8), common::f16_to_f32(s) * float(q - 8));
          }
    }
  }
  // the routed expert block: up (K latent, N I) then down (K I, N latent), both layout 1 at id x block
  {
    const uint32_t latent = 256, I = 128;   // g128-clean widths
    std::vector<Lin> ex;
    for (uint32_t e = 0; e < 3; ++e) {
      ex.push_back(make_lin("e" + std::to_string(e) + ".up", latent, I, 40 + e));
      ex.push_back(make_lin("e" + std::to_string(e) + ".down", I, latent, 50 + e));
    }
    write_dir(base + "/g128e", ex, true);
    write_dir(base + "/g64e", ex, false);
    loader::SafetensorsSet e128(base + "/g128e/"), e64(base + "/g64e/");
    const size_t block = loader::bolt::expert_block_bytes(latent, I);
    CHECK_EQ(block % 64, size_t(0));
    std::vector<uint32_t> buf128(3 * block / 4, 0xFFFFFFFFu), buf64(3 * block / 4, 0xFFFFFFFFu);
    for (uint32_t e = 0; e < 3; ++e) {
      const std::string p = "e" + std::to_string(e);
      loader::repack_bolt_expert(LinearSrc::classify(e128, p + ".up"), LinearSrc::classify(e128, p + ".down"),
                                 buf128.data() + loader::bolt::expert_up_offset(latent, I, e) / 4);
      loader::repack_bolt_expert(LinearSrc::classify(e64, p + ".up"), LinearSrc::classify(e64, p + ".down"),
                                 buf64.data() + loader::bolt::expert_up_offset(latent, I, e) / 4);
    }
    CHECK(same(buf128.data(), buf64.data(), buf128.size() * 4));
    // expert e's up/down tiles sit at the formula's offsets
    for (uint32_t e = 0; e < 3; ++e) {
      const std::vector<uint32_t> up(buf128.begin() + loader::bolt::expert_up_offset(latent, I, e) / 4,
                                     buf128.begin() + loader::bolt::expert_dn_offset(latent, I, e) / 4);
      const std::vector<uint32_t> dn(buf128.begin() + loader::bolt::expert_dn_offset(latent, I, e) / 4,
                                     buf128.begin() + (loader::bolt::expert_dn_offset(latent, I, e) +
                                                       loader::bolt::expert_dn_bytes(latent, I)) / 4);
      CHECK_EQ(up.size() * 4, loader::bolt::expert_up_bytes(latent, I));
      CHECK_EQ(dn.size() * 4, loader::bolt::expert_dn_bytes(latent, I));
      CHECK_EQ(loader::bolt::expert_up_offset(latent, I, e) % 64, size_t(0));
      CHECK_EQ(loader::bolt::expert_dn_offset(latent, I, e) % 64, size_t(0));
      const Lin& U = ex[2 * e];
      const Lin& D = ex[2 * e + 1];
      for (uint32_t k = 0; k < latent; k += 5)
        for (uint32_t n = 0; n < I; n += 3) {
          CHECK_EQ(tile_q(up, latent, k, n), int((U.qweight[size_t(k / 8) * I + n] >> (4 * (k % 8))) & 0xFu));
          CHECK_EQ(tile_s(up, latent, k, n), U.s128[size_t(k / 128) * I + n]);
        }
      for (uint32_t k = 0; k < I; k += 3)
        for (uint32_t n = 0; n < latent; n += 5) {
          CHECK_EQ(tile_q(dn, I, k, n), int((D.qweight[size_t(k / 8) * latent + n] >> (4 * (k % 8))) & 0xFu));
          CHECK_EQ(tile_s(dn, I, k, n), D.s128[size_t(k / 128) * latent + n]);
        }
    }
    // up/down that do not agree are refused
    bool threw = false;
    try {
      loader::repack_bolt_expert(LinearSrc::classify(e128, "e0.up"), LinearSrc::classify(e128, "e0.up"), buf128.data());
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
  }

  // ---------------------------------------------------------------------------------------------------------------
  // The router tiles, the trunk alias, the vision prefixes
  // ---------------------------------------------------------------------------------------------------------------
  {
    const uint32_t H = 128, E = 32;
    std::vector<uint16_t> w(size_t(E) * H);
    for (size_t i = 0; i < w.size(); ++i) w[i] = uint16_t(i * 7 + 3);
    const std::vector<uint16_t> t = loader::repack_bolt_router(w.data(), H, E);
    for (uint32_t n = 0; n < E; ++n)
      for (uint32_t k = 0; k < H; k += 3) CHECK_EQ(t[loader::bolt::embed_row_index(H, k, n)], w[size_t(n) * H + k]);
    bool threw = false;
    try { loader::repack_bolt_router(w.data(), H, 24); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
  }
  {
    auto trunk_of = [&](const std::vector<std::string>& names) -> std::string {
      std::string hdr = "{", ix = R"({"weight_map":{)";
      size_t off = 0;
      for (const std::string& n : names) {
        if (hdr.size() > 1) hdr += ",";
        hdr += "\"" + n + "\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + 8) + "]}";
        if (ix.back() != '{') ix += ",";
        ix += "\"" + n + "\":\"t.safetensors\"";
        off += 8;
      }
      write_st(base + "/t.safetensors", hdr + "}", std::vector<uint8_t>(off, 0));
      std::ofstream(base + "/model.safetensors.index.json") << ix << "}}";
      return loader::bolt_trunk_prefix(loader::SafetensorsSet(base + "/"));
    };
    CHECK(trunk_of({"language_model.model.embeddings.weight", "vision_model.x"}) == "language_model.model.");
    CHECK(trunk_of({"language_model.backbone.embeddings.weight"}) == "language_model.backbone.");   // NVIDIA's spelling
    for (const std::vector<std::string>& bad : {std::vector<std::string>{"language_model.model.embeddings.weight",
                                                                          "language_model.backbone.embeddings.weight"},
                                                 std::vector<std::string>{"model.embeddings.weight"}}) {
      bool threw = false;
      try { trunk_of(bad); } catch (const std::runtime_error& e) { threw = std::string(e.what()).find("load_bolt") != std::string::npos; }
      CHECK(threw);
    }
    CHECK(loader::bolt_is_vision("vision_model.encoder.layers.0.weight"));
    CHECK(loader::bolt_is_vision("vision_projector.mlp1.0.weight"));
    CHECK(loader::bolt_is_vision("mlp1.3.weight"));
    CHECK(!loader::bolt_is_vision("language_model.model.layers.0.mixer.in_proj.qweight"));
    CHECK(!loader::bolt_is_vision("language_model.lm_head.weight"));
  }

  // ---------------------------------------------------------------------------------------------------------------
  // exp_torch: deterministic, monotone, within a few ulp of libm over every bf16 value of A_log's range
  // ---------------------------------------------------------------------------------------------------------------
  {
    CHECK_EQ(common::exp_torch(0.0f), 1.0f);
    size_t n = 0, max_ulp = 0;
    for (uint32_t bits = 0; bits < 0x10000u; ++bits) {
      const float x = common::bf16_to_f32(uint16_t(bits));
      if (!(x >= -7.0f && x <= 9.0f)) continue;
      const float y = common::exp_torch(x), r = std::exp(x);
      uint32_t yb, rb;
      std::memcpy(&yb, &y, 4);
      std::memcpy(&rb, &r, 4);
      max_ulp = std::max<size_t>(max_ulp, yb > rb ? yb - rb : rb - yb);
      ++n;
    }
    std::printf("bolt_repack_test: exp_torch vs libm expf over %zu bf16 values in [-7, 9]: max %zu ulp\n", n, max_ulp);
    CHECK(max_ulp <= 2);
    const std::vector<float> A = loader::bolt_neg_exp({-6.81f, 0.0f, 8.81f});
    CHECK(A[0] < 0 && A[1] == -1.0f && A[2] < -6000.0f);   // A down to -6,700 (spec 25 §13)
  }

  // ---------------------------------------------------------------------------------------------------------------
  // The real descriptor: alignment of every size and offset; the bytes of spec 25 §3 re-derived from the formulas
  // ---------------------------------------------------------------------------------------------------------------
  {
    const model::BoltDesc& d = model::bolt();
    namespace bl = loader::bolt;
    size_t routed = 0, mamba = 0, attn = 0, moe_dense = 0, routers = 0, small = 0, n_exp = 0;
    unsigned long long int4_weights = 0, routed_params = 0;
    for (uint32_t l = 0; l < d.layers; ++l) {
      const bl::SmallBlock sb = bl::small_block(d, l);
      for (size_t off : {sb.norm, sb.conv_w, sb.conv_b, sb.A, sb.D, sb.dt_bias, sb.gnorm, sb.e_bias, sb.bytes}) CHECK_EQ(off % bl::kAlign, size_t(0));
      CHECK_EQ(sb.bytes / 4 * 4, sb.bytes);
      small += sb.bytes;
      if (d.is_mamba(l)) {
        CHECK_EQ(sb.bytes - bl::norm_bytes(d), bl::mamba_small_bytes(d));
        // the order norm, conv_w, conv_b, A, D, dt_bias, gnorm packs without a gap
        CHECK(sb.conv_w == bl::norm_bytes(d) && sb.conv_b == sb.conv_w + size_t(10240) * 4 * 4 && sb.A == sb.conv_b + 10240 * 4 &&
              sb.D == sb.A + 128 * 4 && sb.dt_bias == sb.D + 128 * 4 && sb.gnorm == sb.dt_bias + 128 * 4 &&
              sb.bytes == sb.gnorm + 8192 * 4);
      }
      for (const model::BoltLinear& ln : d.linears(l)) {
        if (ln.kind != model::WeightKind::Int4) {
          CHECK_EQ(bl::router_bytes(ln.shape.K, ln.shape.N) % bl::kAlign, size_t(0));
          routers += bl::router_bytes(ln.shape.K, ln.shape.N);
          continue;
        }
        if (ln.id == model::BoltLinearId::ExpertUp || ln.id == model::BoltLinearId::ExpertDown) continue;
        const size_t by = bl::int4_bytes(ln.shape);
        CHECK_EQ(by % bl::kAlign, size_t(0));
        CHECK_EQ(by, size_t(ln.shape.K) * ln.shape.N * 17 / 32);   // 4.25 bits a weight, exactly
        int4_weights += (unsigned long long)ln.shape.K * ln.shape.N;
        (d.is_mamba(l) ? mamba : d.is_attention(l) ? attn : moe_dense) += by;
      }
      if (d.is_moe(l)) {
        const uint32_t E = d.experts[l], I = d.inter[l];
        CHECK_EQ(bl::expert_up_bytes(d.latent, I), size_t(I) * 544);
        CHECK_EQ(bl::expert_dn_bytes(d.latent, I), size_t(I) * 544);
        CHECK_EQ(bl::expert_block_bytes(d.latent, I) % bl::kAlign, size_t(0));
        CHECK_EQ(bl::layer_expert_bytes(d.latent, E, I) % bl::kAlign, size_t(0));
        CHECK_EQ(bl::expert_up_offset(d.latent, I, E - 1) + bl::expert_block_bytes(d.latent, I), bl::layer_expert_bytes(d.latent, E, I));
        CHECK_EQ(bl::expert_dn_offset(d.latent, I, 7) - bl::expert_up_offset(d.latent, I, 7), bl::expert_up_bytes(d.latent, I));
        CHECK_EQ(bl::expert_up_offset(d.latent, I, 8) - bl::expert_up_offset(d.latent, I, 7), bl::expert_block_bytes(d.latent, I));
        routed += bl::layer_expert_bytes(d.latent, E, I);
        n_exp += E;
        routed_params += (unsigned long long)E * 2 * d.latent * I;
        int4_weights += (unsigned long long)E * 2 * d.latent * I;
      }
      CHECK_EQ(bl::layer_weight_bytes(d, l) % bl::kAlign, size_t(0));
    }
    CHECK_EQ(n_exp, size_t(13824));
    CHECK_EQ(routed, size_t(routed_params / 32 * 17));   // independent: params x 17/32 B, from E x 2 x latent x I
    // the checkpoint's qweights hold the int4 weights = the linears' parameter count (25a's header scan: 72,909,586,432)
    CHECK_EQ(int4_weights, 72909586432ull);
    // each category from its own closed formula (not from the layout functions above)
    CHECK_EQ(mamba, size_t(40) * (4096ull * 18560 + 8192ull * 4096) * 17 / 32);
    CHECK_EQ(attn, size_t(8) * (4096ull * 4608 + 4096ull * 4096) * 17 / 32);
    CHECK_EQ(moe_dense, size_t(40) * (4096ull * 6400 + 1024ull * 4096 + 5376ull * 4096) * 17 / 32);
    // the small blocks: 88 norms + 40 Mamba extras (taps [10240][4], bias, A / D / dt_bias, gated norm) + 4 B a routed expert
    CHECK_EQ(small, size_t(88) * 4096 * 4 +
                        size_t(40) * (size_t(10240) * 4 * 4 + 10240 * 4 + 3 * 128 * 4 + 8192 * 4) + n_exp * 4);
    const size_t embed = size_t(4096) * 131072 * 2, head8 = 4096ull * 131072 + 131072ull * 4, fnorm = 4096 * 4;
    CHECK_EQ(bl::embed_tiled_bytes(d), embed);
    CHECK_EQ(bl::head_int8_bytes(d), head8);
    const size_t total = routed + mamba + attn + moe_dense + routers + small + embed + head8 + fnorm;
    CHECK_EQ(bl::weight_bytes(d, true), total);   // the sum of the independently derived categories
    CHECK_EQ(bl::weight_bytes(d, false), total - head8 + size_t(4096) * 131072 * 2);
    size_t per_layer = embed + head8 + fnorm;
    for (uint32_t l = 0; l < 88; ++l) per_layer += bl::layer_weight_bytes(d, l);
    CHECK_EQ(per_layer, total);
    // per placement: the two devices partition the model
    for (uint32_t split : {1u, 44u, 48u, 87u}) {
      const model::BoltPlacement p = model::BoltPlacement::two(d, split);
      model::validate(p, d);
      CHECK_EQ(bl::device_weight_bytes(d, p, 0, true) + bl::device_weight_bytes(d, p, 1, true), total);
      CHECK_EQ(bl::device_weight_bytes(d, model::BoltPlacement::one(d), 0, true), total);
    }
    std::printf("bolt_repack_test: bytes on the cards (derived from loader/bolt_layout.h, int8 head)\n"
                "  routed experts      %14zu B  (%zu experts)\n  Mamba in/out_proj   %14zu B\n  attention q/k/v/o   %14zu B\n"
                "  MoE dense linears   %14zu B  (fc1||shared_up, fc2, shared_down)\n  routers (bf16)      %14zu B\n"
                "  small blocks (fp32) %14zu B\n  embeddings (bf16)   %14zu B\n  lm_head (int8)      %14zu B\n"
                "  norm_f              %14zu B\n  total               %14zu B  (%.3f GB)\n",
                routed, n_exp, mamba, attn, moe_dense, routers, small, embed, head8, fnorm, total, double(total) / 1e9);
    const model::BoltPlacement p48 = model::BoltPlacement::two(d, 48);
    std::printf("  at split 48: device 0 %zu B, device 1 %zu B\n", bl::device_weight_bytes(d, p48, 0, true),
                bl::device_weight_bytes(d, p48, 1, true));
    // state: derived, per position / per sequence
    std::printf("  kv %zu B a position; SSM state %zu B (fp32, 40 layers, one slot); conv rings %zu B\n",
                bl::kv_bytes_per_position(d), bl::ssm_state_bytes(d, d.mamba_layers()), bl::conv_ring_bytes(d, d.mamba_layers()));
    CHECK_EQ(bl::kv_bytes_per_position(d), size_t(8192));
    CHECK_EQ(bl::ssm_state_bytes(d, 40), size_t(167772160));
  }

  std::puts("bolt_repack_test OK");
  return 0;
}
