// Spec 15c: the MoE layer repack (loader/moe.h) on a synthetic checkpoint, host-only.
//
// The test WRITES its fixture - a one-layer safetensors checkpoint in the per-expert
// GPTQ naming loader/moe_layout.h assumes (AutoRound's export), at a small MoE shape
// (16 experts, top-4, hidden 128, expert 64) - into a temporary directory, reads it
// back through loader::SafetensorsSet and LinearSrc::classify exactly as the loader
// does, repacks, and decodes every device element back:
//
//   1. the router || shared-gate tile: rows 0..E-1 the router, row E the gate, zero
//      rows to router_n, at gemv_bf16's tile index;
//   2. EVERY expert block (shared last) of gate||up and down: each nibble and scale of
//      the layout-1 tiles equals its source column - gate||up interleaved in 16-column
//      blocks - in the block of THAT expert. Every expert's weights are distinct, so a
//      wrong block stride or a swapped expert fails here (spec 15c Review Focus 2);
//   3. the per-expert FUSED form (`experts.E.gate_up_proj`, gate = first half) repacks
//      to the same bytes as the separate form of the same weights;
//   4. g128 scales on one expert are expanded onto g64 (LinearSrc::classify's rule);
//   5. the refusals: a missing expert, an int4 router, mixed per-expert forms, a
//      shape that is not the descriptor's;
//   6. moe_layer_bytes at Ornith's real shape (the planner's and the loader's figure).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "loader/moe.h"
#include "loader/moe_layout.h"
#include "loader/safetensors.h"
#include "model/model_desc.h"

namespace {

namespace fs = std::filesystem;

// --- a minimal safetensors writer (one file + its index) -------------------------
struct Tensor {
  std::string dtype;
  std::vector<uint64_t> shape;
  std::vector<uint8_t> bytes;
};

void write_checkpoint(const fs::path& dir, const std::map<std::string, Tensor>& ts) {
  fs::create_directories(dir);
  std::string hdr = "{";
  uint64_t off = 0;
  bool first = true;
  for (const auto& [name, t] : ts) {
    hdr += (first ? "" : ",");
    first = false;
    hdr += "\"" + name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
    for (size_t i = 0; i < t.shape.size(); ++i)
      hdr += (i ? "," : "") + std::to_string(t.shape[i]);
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + t.bytes.size()) +
           "]}";
    off += t.bytes.size();
    off = (off + 7) / 8 * 8;   // every tensor 8-aligned (the loader casts to u32/u16)
  }
  hdr += "}";
  while ((8 + hdr.size()) % 8 != 0) hdr += ' ';   // data section 8-aligned
  std::ofstream f(dir / "model-00001-of-00001.safetensors", std::ios::binary);
  const uint64_t hlen = hdr.size();
  f.write(reinterpret_cast<const char*>(&hlen), 8);
  f.write(hdr.data(), std::streamsize(hdr.size()));
  uint64_t pos = 0;
  for (const auto& [name, t] : ts) {
    (void)name;
    f.write(reinterpret_cast<const char*>(t.bytes.data()), std::streamsize(t.bytes.size()));
    pos += t.bytes.size();
    const uint64_t pad = (pos + 7) / 8 * 8 - pos;
    const char zeros[8] = {};
    f.write(zeros, std::streamsize(pad));
    pos += pad;
  }
  std::ofstream ix(dir / "model.safetensors.index.json");
  ix << "{\"metadata\":{},\"weight_map\":{";
  first = true;
  for (const auto& [name, t] : ts) {
    (void)t;
    ix << (first ? "" : ",") << "\"" << name << "\":\"model-00001-of-00001.safetensors\"";
    first = false;
  }
  ix << "}}";
}

template <class T>
std::vector<uint8_t> as_bytes(const std::vector<T>& v) {
  std::vector<uint8_t> b(v.size() * sizeof(T));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

// A packed GPTQ linear of K x N with distinct random content (and qzeros 0x77777777).
struct Packed {
  uint32_t K = 0, N = 0, group = 64;
  std::vector<uint32_t> qweight;   // [K/8][N]
  std::vector<uint16_t> scales;    // [K/group][N]
  uint32_t nibble(uint32_t k, uint32_t n) const { return (qweight[size_t(k / 8) * N + n] >> (4 * (k % 8))) & 15u; }
  uint16_t scale_g64(uint32_t k, uint32_t n) const { return scales[size_t(k / group) * N + n]; }
};
Packed random_packed(uint32_t K, uint32_t N, std::mt19937& rng, uint32_t group = 64) {
  Packed p;
  p.K = K;
  p.N = N;
  p.group = group;
  p.qweight.resize(size_t(K / 8) * N);
  for (uint32_t& w : p.qweight) w = rng();
  p.scales.resize(size_t(K / group) * N);
  std::uniform_real_distribution<float> sd(0.01f, 0.09f);
  for (uint16_t& s : p.scales) s = common::f32_to_f16(sd(rng));
  return p;
}
void add_packed(std::map<std::string, Tensor>& ts, const std::string& prefix, const Packed& p) {
  ts[prefix + ".qweight"] = {"I32", {p.K / 8, p.N}, as_bytes(p.qweight)};
  ts[prefix + ".scales"] = {"F16", {p.K / p.group, p.N}, as_bytes(p.scales)};
  std::vector<uint32_t> qz(size_t(p.K / p.group) * (p.N / 8), 0x77777777u);
  ts[prefix + ".qzeros"] = {"I32", {p.K / p.group, p.N / 8}, as_bytes(qz)};
}
// --- the test model: Ornith's descriptor at a small MoE shape ---------------------
model::ModelDesc small_moe() {
  model::ModelDesc d = model::ornith();
  d.hidden = 128;
  d.moe = {16, 4, 64, 64, true};
  d.intermediate = 64;
  return d;
}

struct Layer {   // the source weights of one layer, to check the repack against
  std::vector<uint16_t> router, gate;   // bf16 [E][H], [1][H]
  std::vector<Packed> g, u, dn;         // per block (E routed + shared last)
};

const std::string kLp = "model.language_model.layers.0.";

// The fixture: separate gate/up per expert (`fused` = false) or per-expert gate_up
// (`fused` = true) of the SAME weights. `g128_expert` (if < E) ships g128 scales.
Layer make_fixture(const model::ModelDesc& d, const fs::path& dir, bool fused, uint32_t g128_expert,
                   const std::function<void(std::map<std::string, Tensor>&)>& edit = {}) {
  const uint32_t E = d.moe.experts, H = d.hidden, I = d.moe.expert_intermediate;
  std::mt19937 rng(20261005);
  Layer L;
  L.router.resize(size_t(E) * H);
  L.gate.resize(H);
  std::uniform_real_distribution<float> wd(-0.5f, 0.5f);
  for (uint16_t& w : L.router) w = common::f32_to_bf16(wd(rng));
  for (uint16_t& w : L.gate) w = common::f32_to_bf16(wd(rng));
  std::map<std::string, Tensor> ts;
  ts[kLp + "mlp.gate.weight"] = {"BF16", {E, H}, as_bytes(L.router)};
  ts[kLp + "mlp.shared_expert_gate.weight"] = {"BF16", {1, H}, as_bytes(L.gate)};
  for (uint32_t b = 0; b <= E; ++b) {
    const uint32_t grp = b == g128_expert ? 128 : 64;
    L.g.push_back(random_packed(H, I, rng, grp));
    L.u.push_back(random_packed(H, I, rng, grp));
    L.dn.push_back(random_packed(I, H, rng, 64));   // K = 64: g64 only
    const std::string p = b == E ? kLp + "mlp.shared_expert." : kLp + "mlp.experts." + std::to_string(b) + ".";
    if (fused && b < E) {
      Packed gu;   // [K/8][2I]: gate columns first, then up
      gu.K = H;
      gu.N = 2 * I;
      gu.group = grp;
      gu.qweight.resize(size_t(H / 8) * 2 * I);
      gu.scales.resize(size_t(H / grp) * 2 * I);
      for (uint32_t r = 0; r < H / 8; ++r)
        for (uint32_t j = 0; j < I; ++j) {
          gu.qweight[size_t(r) * 2 * I + j] = L.g[b].qweight[size_t(r) * I + j];
          gu.qweight[size_t(r) * 2 * I + I + j] = L.u[b].qweight[size_t(r) * I + j];
        }
      for (uint32_t r = 0; r < H / grp; ++r)
        for (uint32_t j = 0; j < I; ++j) {
          gu.scales[size_t(r) * 2 * I + j] = L.g[b].scales[size_t(r) * I + j];
          gu.scales[size_t(r) * 2 * I + I + j] = L.u[b].scales[size_t(r) * I + j];
        }
      add_packed(ts, p + "gate_up_proj", gu);
    } else {
      add_packed(ts, p + "gate_proj", L.g[b]);
      add_packed(ts, p + "up_proj", L.u[b]);
    }
    add_packed(ts, p + "down_proj", L.dn[b]);
  }
  if (edit) edit(ts);
  write_checkpoint(dir, ts);
  return L;
}

// The loader's binding of MoeSource (loader.cc load_moe_layer), minus the consumed set.
loader::MoeSource source_for(const loader::SafetensorsSet& set, std::vector<std::string>* asked) {
  loader::MoeSource src;
  src.has = [&set](const std::string& part) {
    return set.tensors().count(kLp + part + ".qweight") != 0 ||
           set.tensors().count(kLp + part + ".weight") != 0;
  };
  src.linear = [&set, asked](const std::string& part) {
    if (asked) asked->push_back(part);
    return loader::LinearSrc::classify(set, kLp + part);
  };
  return src;
}

// Decoders of the device layouts (common/repack.h), independent of the repack code.
uint16_t bf16_tile_at(const std::vector<uint16_t>& t, uint32_t K, uint32_t n, uint32_t k) {
  return t[((size_t(n / 16) * (K / 8) + k / 8) * 8 + k % 8) * 16 + n % 16];
}
struct Nib {
  uint32_t q;
  uint16_t scale;
};
Nib layout1_at(const uint32_t* blk, uint32_t K, uint32_t n, uint32_t k) {
  const uint32_t* tile = blk + (size_t(n / 16) * (K / 64) + k / 64) * loader::kInt4TileU32;
  const uint32_t word = tile[((k % 64) / 8) * 16 + n % 16];
  const uint32_t sw = tile[128 + (n % 16) / 2];
  return {(word >> (4 * (k % 8))) & 15u, uint16_t(n % 2 == 0 ? sw & 0xFFFFu : sw >> 16)};
}

void check_layer(const model::ModelDesc& d, const loader::MoeHost& h, const Layer& L) {
  const uint32_t E = d.moe.experts, H = d.hidden, I = d.moe.expert_intermediate;
  const loader::MoeLayerBytes b = loader::moe_layer_bytes(d);
  CHECK_EQ(h.router.size() * 2, b.router);
  CHECK_EQ(h.gate_up.size() * 4, b.gate_up());
  CHECK_EQ(h.down.size() * 4, b.down());
  // 1. the router || gate rows, zero padding past them.
  for (uint32_t n = 0; n < d.moe.router_n(); ++n)
    for (uint32_t k = 0; k < H; ++k) {
      const uint16_t want = n < E ? L.router[size_t(n) * H + k] : n == E ? L.gate[k] : uint16_t(0);
      CHECK_EQ(bf16_tile_at(h.router, H, n, k), want);
    }
  // 2. every block: gate||up interleaved in 16-column blocks, then down.
  for (uint32_t blk = 0; blk <= E; ++blk) {
    const uint32_t* gu = h.gate_up.data() + blk * (b.gate_up_block / 4);
    for (uint32_t c = 0; c < 2 * I; ++c) {
      const bool is_up = (c % 32) >= 16;
      const uint32_t col = (c / 32) * 16 + c % 16;
      const Packed& src = is_up ? L.u[blk] : L.g[blk];
      for (uint32_t k = 0; k < H; ++k) {
        const Nib got = layout1_at(gu, H, c, k);
        if (got.q != src.nibble(k, col) || got.scale != src.scale_g64(k, col)) {
          std::fprintf(stderr, "gate||up block %u column %u k %u: got q %u s %04x, want q %u s %04x\n",
                       blk, c, k, got.q, got.scale, src.nibble(k, col), src.scale_g64(k, col));
          std::exit(1);
        }
      }
    }
    const uint32_t* dn = h.down.data() + blk * (b.down_block / 4);
    for (uint32_t n = 0; n < H; ++n)
      for (uint32_t k = 0; k < I; ++k) {
        const Nib got = layout1_at(dn, I, n, k);
        if (got.q != L.dn[blk].nibble(k, n) || got.scale != L.dn[blk].scale_g64(k, n)) {
          std::fprintf(stderr, "down block %u column %u k %u mismatch\n", blk, n, k);
          std::exit(1);
        }
      }
  }
}

template <class F>
bool throws_with(F&& f, const std::string& needle) {
  try {
    f();
  } catch (const std::runtime_error& e) {
    if (std::string(e.what()).find(needle) != std::string::npos) return true;
    std::fprintf(stderr, "threw, but without '%s': %s\n", needle.c_str(), e.what());
    return false;
  }
  return false;
}

void remove_fixture(const fs::path& dir) {
  fs::remove(dir / "model-00001-of-00001.safetensors");
  fs::remove(dir / "model.safetensors.index.json");
  fs::remove(dir);
}

}  // namespace

int main() {
  const model::ModelDesc d = small_moe();
  const uint32_t E = d.moe.experts;
  const fs::path root = fs::temp_directory_path() / ("ornith_repack_test_" + std::to_string(getpid()));
  const fs::path sep = root / "separate", fus = root / "fused", g128 = root / "g128";

  // 1 + 2: the separate per-expert form (AutoRound's export).
  loader::MoeHost hs;
  {
    const Layer L = make_fixture(d, sep, /*fused=*/false, /*g128_expert=*/E + 1);
    loader::SafetensorsSet set(sep.string() + "/");
    std::vector<std::string> asked;
    loader::repack_moe_layer(d, source_for(set, &asked), hs, "layer 0");
    CHECK(!hs.fused_gate_up);
    // router, gate, then (gate, up, down) for each of E + 1 blocks.
    CHECK_EQ(asked.size(), size_t(2 + 3 * (E + 1)));
    CHECK_EQ(asked[0], std::string("mlp.gate"));
    CHECK_EQ(asked[1], std::string("mlp.shared_expert_gate"));
    CHECK_EQ(asked.back(), std::string("mlp.shared_expert.down_proj"));
    check_layer(d, hs, L);
    // The checkpoint bytes the layer consumed: (E + 1) x 3 linears of 128 x 64 int4
    // (16 KiB nibbles... per linear: K/8 x N x 4 + K/64 x N x 2) + the bf16 rows.
    const size_t lin = size_t(d.hidden / 8) * 64 * 4 + size_t(d.hidden / 64) * 64 * 2;
    CHECK_EQ(hs.int4_src_bytes, size_t(E + 1) * 3 * lin);
    CHECK_EQ(hs.bf16_src_bytes, size_t(E + 1) * d.hidden * 2);
  }
  // 3: the per-expert fused form of the same weights repacks to the same bytes.
  {
    const Layer L = make_fixture(d, fus, /*fused=*/true, E + 1);
    loader::SafetensorsSet set(fus.string() + "/");
    loader::MoeHost hf;
    loader::repack_moe_layer(d, source_for(set, nullptr), hf, "layer 0");
    CHECK(hf.fused_gate_up);
    check_layer(d, hf, L);
    CHECK(hf.gate_up == hs.gate_up && hf.down == hs.down && hf.router == hs.router);
  }
  // 4: g128 scales on expert 5 are expanded onto the g64 tiles.
  {
    const Layer L = make_fixture(d, g128, false, /*g128_expert=*/5);
    loader::SafetensorsSet set(g128.string() + "/");
    loader::MoeHost hg;
    loader::repack_moe_layer(d, source_for(set, nullptr), hg, "layer 0");
    check_layer(d, hg, L);   // Packed::scale_g64 reads the g128 row of each g64 group
  }
  remove_fixture(sep);
  remove_fixture(fus);
  remove_fixture(g128);

  // 5: the refusals, each on its own fixture.
  const fs::path bad = root / "bad";
  auto refuses = [&](const std::function<void(std::map<std::string, Tensor>&)>& edit,
                     const std::string& needle) {
    make_fixture(d, bad, false, E + 1, edit);
    bool ok = false;
    {
      loader::SafetensorsSet set(bad.string() + "/");
      loader::MoeHost h;
      ok = throws_with([&] { loader::repack_moe_layer(d, source_for(set, nullptr), h, "layer 0"); },
                       needle);
    }
    remove_fixture(bad);
    return ok;
  };
  // a missing expert tensor
  CHECK(refuses([](std::map<std::string, Tensor>& ts) {
          ts.erase(kLp + "mlp.experts.7.up_proj.qweight");
          ts.erase(kLp + "mlp.experts.7.up_proj.scales");
          ts.erase(kLp + "mlp.experts.7.up_proj.qzeros");
        },
        "mlp.experts.7.up_proj"));
  // an int4 router
  CHECK(refuses([&](std::map<std::string, Tensor>& ts) {
          ts.erase(kLp + "mlp.gate.weight");
          std::mt19937 rng(1);
          add_packed(ts, kLp + "mlp.gate", random_packed(d.hidden, E, rng));
        },
        "is int4"));
  // mixed per-expert forms: expert 3 fused while expert 0 is separate
  CHECK(refuses([&](std::map<std::string, Tensor>& ts) {
          std::mt19937 rng(2);
          add_packed(ts, kLp + "mlp.experts.3.gate_up_proj", random_packed(d.hidden, 128, rng));
        },
        "one per-expert form"));
  // an expert of the wrong width
  CHECK(refuses([&](std::map<std::string, Tensor>& ts) {
          std::mt19937 rng(3);
          add_packed(ts, kLp + "mlp.experts.9.down_proj", random_packed(64, 64, rng));
        },
        "the descriptor says"));
  fs::remove(root);

  // 6: Ornith's real figures (loader/moe_layout.h's derivation, spec 15 §2).
  const model::ModelDesc& o = model::ornith();
  const loader::MoeLayerBytes ob = loader::moe_layer_bytes(o);
  CHECK_EQ(ob.router, size_t(1114112));          // 272 x 2048 x 2
  CHECK_EQ(ob.gate_up_block, size_t(1114112));   // 64 n-tiles x 32 groups x 544
  CHECK_EQ(ob.down_block, size_t(557056));       // 128 x 8 x 544
  CHECK_EQ(ob.blocks, uint32_t(257));
  CHECK_EQ(ob.total(), size_t(430604288));
  CHECK_EQ(loader::moe_bytes(o), size_t(17224171520ull));
  CHECK_EQ(ob.per_token(8), size_t(1114112) + size_t(9) * (1114112 + 557056));   // 16.15 MB
  CHECK_EQ(loader::moe_bytes(model::qwen38()), size_t(0));
  CHECK_EQ(loader::moe_layer_bytes(model::agnes()).total(), size_t(0));
  CHECK_EQ(loader::moe_expert_part(12, "down_proj"), std::string("mlp.experts.12.down_proj"));

  std::puts("ornith_repack_test OK");
  return 0;
}
