// Spec 18b Task 1: the K2-Horizon loader's host half (loader/k2_repack.h) on a synthetic
// checkpoint in K2's naming, host only.
//
// The test WRITES its fixture - a three-layer K2 checkpoint (layer 0 dense, layers 1-2
// MoVA + MoE) at a small shape (hidden 128 in 2 norm groups, 2 q / 1 kv heads of 64,
// 8 experts + 1 shared, 16 value experts, vocab 64), every tensor named exactly as
// urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ names it, GPTQ-packed with
// qzeros and g_idx - into a temporary directory, reads it back through
// loader::SafetensorsSet exactly as load_k2 does, repacks every layer and decodes the
// device bytes back:
//
//   1. the fixture passes assert_quant_invariants and K2Checkpoint::check_names;
//   2. the fused attention rows (q || k || gate || v, and v_router in v's place on the
//      MoVA layers) and the dense gate||up (interleave16) at their column offsets;
//   3. the MoE router: rows 0..7 the checkpoint's, rows 8..15 zero (Review Focus 4: the
//      padding the top-k never reads), at gemv_bf16's tile index;
//   4. EVERY expert block of the three groups - value, gate||up (interleave16) and down,
//      the shared expert as the last block - decoded nibble by nibble and scale by scale
//      against ITS expert's tensors: a wrong stride or a swapped expert fails here;
//   5. the norms widened PLAIN (no 1 + w: the first norm trap), the MoE bias (bf16) and
//      the MoVA bias (F16) widened into the route block at their offsets, zero padding;
//   6. nothing unconsumed after a full walk;
//   7. refusals BY NAME: an unexpected tensor (a q_norm, an mtp.* tensor, an extra
//      expert), a missing one, a wrong expert shape, an int4 lm_head;
//   8. the checkpoint's real quantization_config (argv[1], tests/model/k2/config.json)
//      parses as int4 g64 sym desc_act false with its dynamic exclusions.
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/json.h"
#include "loader/k2_layout.h"
#include "loader/k2_repack.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/k2_horizon.h"

namespace {

namespace fs = std::filesystem;
using model::K2Desc;

// --- a minimal safetensors writer (one shard + its index) -------------------------------
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
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + t.bytes.size()) +
           "]}";
    off += t.bytes.size();
    off = (off + 7) / 8 * 8;
  }
  hdr += "}";
  while ((8 + hdr.size()) % 8 != 0) hdr += ' ';
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
template <class T>
const T* view(const Tensors& ts, const std::string& n) {
  return reinterpret_cast<const T*>(ts.at(n).bytes.data());
}

std::mt19937 rng(20261005);

// A GPTQ int4 g64 linear K x N: random words, distinct finite scales, qzeros 0x77777777,
// g_idx the identity k / 64 (what assert_quant_invariants demands).
void add_int4(Tensors& ts, const std::string& prefix, uint32_t K, uint32_t N) {
  std::vector<uint32_t> qw(size_t(K / 8) * N);
  for (uint32_t& w : qw) w = rng();
  std::vector<uint16_t> sc(size_t(K / 64) * N);
  std::uniform_real_distribution<float> sd(0.01f, 0.09f);
  for (uint16_t& s : sc) s = common::f32_to_f16(sd(rng));
  std::vector<uint32_t> qz(size_t(K / 64) * (N / 8), 0x77777777u);
  std::vector<int32_t> gi(K);
  for (uint32_t k = 0; k < K; ++k) gi[k] = int32_t(k / 64);
  ts[prefix + ".qweight"] = {"I32", {K / 8, N}, as_bytes(qw)};
  ts[prefix + ".scales"] = {"F16", {K / 64, N}, as_bytes(sc)};
  ts[prefix + ".qzeros"] = {"I32", {K / 64, N / 8}, as_bytes(qz)};
  ts[prefix + ".g_idx"] = {"I32", {K}, as_bytes(gi)};
}
void add_bf16(Tensors& ts, const std::string& name, std::vector<uint64_t> shape) {
  size_t n = 1;
  for (uint64_t s : shape) n *= size_t(s);
  std::vector<uint16_t> v(n);
  std::uniform_real_distribution<float> d(-2.f, 2.f);
  for (uint16_t& x : v) x = common::f32_to_bf16(d(rng));
  ts[name] = {"BF16", std::move(shape), as_bytes(v)};
}
void add_f16(Tensors& ts, const std::string& name, uint64_t n) {
  std::vector<uint16_t> v(n);
  std::uniform_real_distribution<float> d(-0.5f, 0.5f);
  for (uint16_t& x : v) x = common::f32_to_f16(d(rng));
  ts[name] = {"F16", {n}, as_bytes(v)};
}

// The small K2 the fixture is written at.
K2Desc tiny() {
  K2Desc d = model::k2();
  d.name = "k2-tiny";
  d.layers = 3;
  d.dense_layers = 1;
  d.hidden = 128;
  d.norm_groups = 2;
  d.q_heads = 2;
  d.kv_heads = 1;
  d.head_dim = 64;
  d.dense_inter = 128;
  d.moe_inter = 64;
  d.experts = 8;
  d.top_k = 2;
  d.value_experts = 16;
  d.value_top_k = 2;
  d.vocab = 64;
  d.vocab_used = 64;
  d.attn_s = d.oproj_s = d.gate_up_s = d.down_s = 1;
  return d;
}

// Every tensor of the K2 checkpoint at `d`'s shape, spelled by hand from the index (NOT
// from K2Checkpoint::expected_names, which is under test).
Tensors fixture(const K2Desc& d) {
  Tensors ts;
  const uint32_t H = d.hidden;
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    add_bf16(ts, p + "input_layernorm.weight", {H});
    add_bf16(ts, p + "post_attention_layernorm.weight", {H});
    add_int4(ts, p + "self_attn.q_proj", H, d.q_n());
    add_int4(ts, p + "self_attn.k_proj", H, d.kv_n());
    add_int4(ts, p + "self_attn.gate_proj", H, d.q_n());
    add_int4(ts, p + "self_attn.o_proj", d.q_n(), H);
    if (l < d.dense_layers) {
      add_int4(ts, p + "self_attn.v_proj", H, d.kv_n());
      add_int4(ts, p + "mlp.gate_proj", H, d.dense_inter);
      add_int4(ts, p + "mlp.up_proj", H, d.dense_inter);
      add_int4(ts, p + "mlp.down_proj", d.dense_inter, H);
      continue;
    }
    add_int4(ts, p + "self_attn.v_router", H, d.value_experts);
    add_f16(ts, p + "self_attn.v_router.bias", d.value_experts);
    for (uint32_t e = 0; e < d.value_experts; ++e)
      add_int4(ts, p + "self_attn.v_experts." + std::to_string(e), H, d.kv_n());
    add_bf16(ts, p + "mlp.gate.weight", {d.experts, H});
    add_bf16(ts, p + "mlp.gate.bias", {d.experts});
    for (uint32_t e = 0; e < d.experts; ++e) {
      const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
      add_int4(ts, ep + "gate_proj", H, d.moe_inter);
      add_int4(ts, ep + "up_proj", H, d.moe_inter);
      add_int4(ts, ep + "down_proj", d.moe_inter, H);
    }
    add_int4(ts, p + "mlp.shared_experts.gate_proj", H, d.moe_inter);
    add_int4(ts, p + "mlp.shared_experts.up_proj", H, d.moe_inter);
    add_int4(ts, p + "mlp.shared_experts.down_proj", d.moe_inter, H);
  }
  add_bf16(ts, "model.embed_tokens.weight", {d.vocab, H});
  add_bf16(ts, "model.norm.weight", {H});
  add_bf16(ts, "lm_head.weight", {d.vocab, H});
  return ts;
}

fs::path tmpdir(const char* tag) {
  return fs::temp_directory_path() /
         ("k2_repack_test_" + std::to_string(::getpid()) + "_" + tag);
}

// A layout-0 fused linear's (word, scale) at fused column n against part column j.
void check_l0_col(const loader::K2HostLinear& h, uint32_t K, uint32_t N, uint32_t n,
                  const Tensors& ts, const std::string& part, uint32_t part_n, uint32_t j) {
  const uint32_t* qw = view<uint32_t>(ts, part + ".qweight");
  const uint16_t* sc = view<uint16_t>(ts, part + ".scales");
  for (uint32_t r : {0u, K / 8 - 1}) CHECK_EQ(h.words[size_t(r) * N + n], qw[size_t(r) * part_n + j]);
  for (uint32_t g : {0u, K / 64 - 1}) CHECK_EQ(h.scales[size_t(g) * N + n], sc[size_t(g) * part_n + j]);
}

// Decode one layout-1 block (common::repack_int4_layout1) at column n, k-group g against
// a part's column j: every nibble word and the scale.
void check_l1_col(const uint32_t* blk, uint32_t K, uint32_t n, const Tensors& ts,
                  const std::string& part, uint32_t part_n, uint32_t j) {
  const uint32_t G = K / 64;
  const uint32_t* qw = view<uint32_t>(ts, part + ".qweight");
  const uint16_t* sc = view<uint16_t>(ts, part + ".scales");
  for (uint32_t g = 0; g < G; ++g) {
    const uint32_t* tile = blk + (size_t(n / 16) * G + g) * loader::kK2TileU32;
    for (uint32_t w = 0; w < 8; ++w) CHECK_EQ(tile[w * 16 + n % 16], qw[size_t(g * 8 + w) * part_n + j]);
    const uint32_t sw = tile[128 + (n % 16) / 2];
    CHECK_EQ(uint16_t(n % 2 == 0 ? sw & 0xFFFFu : sw >> 16), sc[size_t(g) * part_n + j]);
  }
}

// The refusal of a modified fixture: check_names (or a repack) throws naming `needle`.
void refused(const K2Desc& d, const std::function<void(Tensors&)>& edit, const char* needle,
             const char* tag) {
  Tensors ts = fixture(d);
  edit(ts);
  const fs::path dir = tmpdir(tag);
  write_checkpoint(dir, ts);
  bool threw = false;
  try {
    loader::SafetensorsSet set(dir.string() + "/");
    loader::K2Checkpoint ck(d, set);
    ck.check_names();
    loader::K2HostLayer h;
    for (uint32_t l = 0; l < d.layers; ++l) ck.repack_layer(l, h);
    (void)ck.lm_head();
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find(needle) != std::string::npos;
    if (!threw) std::fprintf(stderr, "%s: refused for another reason: %s\n", tag, e.what());
  }
  fs::remove_all(dir);
  if (!threw) std::fprintf(stderr, "%s: NOT refused (wanted a message naming '%s')\n", tag, needle);
  CHECK(threw);
  std::printf("  refused by name: %s\n", tag);
}

}  // namespace

int main(int argc, char** argv) {
  CHECK(argc > 1);
  const K2Desc d = tiny();
  const uint32_t H = d.hidden;
  const Tensors ts = fixture(d);
  const fs::path dir = tmpdir("ok");
  write_checkpoint(dir, ts);
  loader::SafetensorsSet set(dir.string() + "/");

  // 1. the fixture is a well-formed GPTQ checkpoint, and K2's names are all accounted for.
  const loader::QuantScan scan = loader::assert_quant_invariants(set);
  CHECK(scan.g_idx_tensors > 0);
  loader::K2Checkpoint ck(d, set);
  ck.check_names();
  CHECK_EQ(ck.expected_names(false).size(), ts.size());

  loader::K2HostLayer h;
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    ck.repack_layer(l, h);
    const bool dense = l < d.dense_layers;

    // 2. the fused attention row, both forms, at the descriptor's column map.
    const loader::K2HostLinear& at = h.linears[0];
    const uint32_t AN = dense ? d.attn_dense_n() : d.attn_sparse_n();
    CHECK_EQ(at.words.size(), size_t(H / 8) * AN);
    check_l0_col(at, H, AN, 0, ts, p + "self_attn.q_proj", d.q_n(), 0);
    check_l0_col(at, H, AN, d.k_off() + 5, ts, p + "self_attn.k_proj", d.kv_n(), 5);
    check_l0_col(at, H, AN, d.gate_off() + 7, ts, p + "self_attn.gate_proj", d.q_n(), 7);
    if (dense)
      check_l0_col(at, H, AN, d.v_off() + d.kv_n() - 1, ts, p + "self_attn.v_proj", d.kv_n(),
                   d.kv_n() - 1);
    else
      check_l0_col(at, H, AN, d.v_off() + 3, ts, p + "self_attn.v_router", d.value_experts, 3);
    check_l0_col(h.linears[1], d.q_n(), H, 9, ts, p + "self_attn.o_proj", H, 9);
    if (dense) {
      CHECK_EQ(h.linears.size(), size_t(4));
      // interleave16: fused 0..15 gate 0..15, fused 16..31 up 0..15, fused 32 gate 16.
      check_l0_col(h.linears[2], H, 2 * d.dense_inter, 0, ts, p + "mlp.gate_proj", d.dense_inter, 0);
      check_l0_col(h.linears[2], H, 2 * d.dense_inter, 16, ts, p + "mlp.up_proj", d.dense_inter, 0);
      check_l0_col(h.linears[2], H, 2 * d.dense_inter, 32, ts, p + "mlp.gate_proj", d.dense_inter, 16);
      check_l0_col(h.linears[3], d.dense_inter, H, H - 1, ts, p + "mlp.down_proj", H, H - 1);
      CHECK(h.value.empty() && h.gate_up.empty() && h.router.empty() && h.route.empty());
    } else {
      CHECK_EQ(h.linears.size(), size_t(2));
    }

    // 5. the norms: plain w, widened verbatim.
    const uint16_t* in_w = view<uint16_t>(ts, p + "input_layernorm.weight");
    const uint16_t* post_w = view<uint16_t>(ts, p + "post_attention_layernorm.weight");
    for (uint32_t k = 0; k < H; ++k) {
      CHECK(h.norms[k] == common::bf16_to_f32(in_w[k]));
      CHECK(h.norms[H + k] == common::bf16_to_f32(post_w[k]));
    }
    if (dense) continue;
    const uint16_t* mb = view<uint16_t>(ts, p + "mlp.gate.bias");
    const uint16_t* vb = view<uint16_t>(ts, p + "self_attn.v_router.bias");
    CHECK_EQ(h.route.size(), size_t(d.route_bytes() / 4));
    for (uint32_t e = 0; e < d.router_n(); ++e)
      CHECK(h.route[e] == (e < d.experts ? common::bf16_to_f32(mb[e]) : 0.0f));
    for (uint32_t e = 0; e < d.value_experts; ++e)
      CHECK(h.route[d.route_off_mova() / 4 + e] == common::f16_to_f32(vb[e]));

    // 3. the router: rows < experts the checkpoint's, the padding zero, gemv_bf16's tiles.
    const uint16_t* gw = view<uint16_t>(ts, p + "mlp.gate.weight");
    CHECK_EQ(h.router.size(), size_t(d.router_n()) * H);
    for (uint32_t n = 0; n < d.router_n(); ++n)
      for (uint32_t k = 0; k < H; ++k) {
        const uint16_t got = h.router[((size_t(n / 16) * (H / 8) + k / 8) * 8 + k % 8) * 16 + n % 16];
        CHECK_EQ(got, n < d.experts ? gw[size_t(n) * H + k] : uint16_t(0));
      }

    // 4. every block of every group against ITS expert.
    const size_t vw = loader::k2_layout1_bytes(H, d.kv_n()) / 4;
    const size_t gw4 = loader::k2_layout1_bytes(H, 2 * d.moe_inter) / 4;
    const size_t dw = loader::k2_layout1_bytes(d.moe_inter, H) / 4;
    CHECK_EQ(h.value.size(), vw * d.value_experts);
    CHECK_EQ(h.gate_up.size(), gw4 * d.moe_blocks());
    CHECK_EQ(h.down.size(), dw * d.moe_blocks());
    for (uint32_t e = 0; e < d.value_experts; ++e)
      for (uint32_t n : {0u, 17u, d.kv_n() - 1})
        check_l1_col(h.value.data() + e * vw, H, n, ts, p + "self_attn.v_experts." + std::to_string(e),
                     d.kv_n(), n);
    for (uint32_t b = 0; b < d.moe_blocks(); ++b) {
      const bool shared = b == d.shared_block();
      const std::string ep = shared ? p + "mlp.shared_experts." : p + "mlp.experts." + std::to_string(b) + ".";
      for (uint32_t i : {0u, 15u, 16u, d.moe_inter - 1}) {
        const uint32_t gc = (i / 16) * 32 + i % 16;   // gate column of intermediate i
        check_l1_col(h.gate_up.data() + b * gw4, H, gc, ts, ep + "gate_proj", d.moe_inter, i);
        check_l1_col(h.gate_up.data() + b * gw4, H, gc + 16, ts, ep + "up_proj", d.moe_inter, i);
      }
      for (uint32_t n : {0u, 31u, H - 1})
        check_l1_col(h.down.data() + b * dw, d.moe_inter, n, ts, ep + "down_proj", H, n);
    }
  }
  CHECK(ck.embed() == reinterpret_cast<const uint16_t*>(
                          set.data(set.tensors().at("model.embed_tokens.weight"))));
  const std::vector<float> fn = ck.final_norm();
  CHECK(fn[3] == common::bf16_to_f32(view<uint16_t>(ts, "model.norm.weight")[3]));
  CHECK(ck.lm_head() != nullptr);
  // 6. every tensor read once (qzeros / g_idx excepted).
  std::string names;
  CHECK_EQ(ck.unconsumed(&names), size_t(0));
  fs::remove_all(dir);
  std::printf("repack: %u layers, every linear / router / expert block / small tensor decoded "
              "back to its source; 0 unconsumed\n", d.layers);

  // 7. refusals, by name.
  refused(d, [](Tensors& t) { add_bf16(t, "model.layers.1.self_attn.q_norm.weight", {64}); },
          "model.layers.1.self_attn.q_norm.weight", "an unexpected q_norm");
  refused(d, [](Tensors& t) { add_bf16(t, "mtp.fc.weight", {128, 256}); }, "mtp.fc.weight",
          "an mtp head");
  refused(d, [&](Tensors& t) { add_int4(t, "model.layers.2.mlp.experts.8.down_proj", 64, 128); },
          "model.layers.2.mlp.experts.8.down_proj", "an extra expert");
  refused(d, [](Tensors& t) { t.erase("model.layers.2.mlp.experts.3.up_proj.qweight"); },
          "model.layers.2.mlp.experts.3.up_proj.qweight", "a missing expert");
  refused(d, [](Tensors& t) { t.erase("model.layers.1.self_attn.v_router.bias"); },
          "model.layers.1.self_attn.v_router.bias", "a missing MoVA bias");
  refused(d, [&](Tensors& t) { add_int4(t, "model.layers.1.self_attn.v_experts.5", 128, 32); },
          "model.layers.1.self_attn.v_experts.5", "a value expert of the wrong width");
  refused(d, [&](Tensors& t) {
            t.erase("lm_head.weight");
            add_int4(t, "lm_head", 128, 64);
          },
          "lm_head", "an int4 lm_head");

  // 8. the checkpoint's own quantization_config.
  {
    std::ifstream f(argv[1]);
    CHECK(f.good());
    std::stringstream s;
    s << f.rdbuf();
    const common::json::Value cfg = common::json::parse(s.str());
    const loader::QuantConfig qc = loader::check_k2_checkpoint_config(model::k2(), cfg);
    CHECK_EQ(qc.bits, 4u);
    CHECK_EQ(qc.group_size, 64u);
    CHECK(qc.sym && !qc.desc_act && qc.desc_act_declared);
    CHECK_EQ(qc.dynamic_rule_count, size_t(2));   // the fixture's two of the checkpoint's 45
    CHECK(qc.quant_method == "gptq");
  }
  std::puts("k2_repack_test OK");
  return 0;
}
