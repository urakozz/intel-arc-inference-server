// Spec 21b Task 3: the Qwen3.8-Flash-Next loader's host half (loader/qwen4exp_repack.h) on synthetic
// checkpoints in both forms, host only.
//
// The test WRITES its fixtures: 4-layer checkpoints (layers 0..2 GDN - layer 1 the PLE layer - and the QSA
// layer 3) at the REAL widths (hidden 2560 x 4 HC streams, GDN 16 / 48 heads, QSA 24 / 2 x 256, the indexer,
// experts of 640, the PLE projections) but 8 experts and a 64-row vocabulary (a test copy of the descriptor:
// the widths are what the layouts depend on), every tensor named as the export names it - ours (int4 g64
// dense / shared / experts, with the MTP head: bf16 experts) and Intel's (int4 g128 experts, bf16 dense /
// shared) - into a temporary directory, read back through loader::SafetensorsSet exactly as load_qwen4exp
// does, every layer repacked and decoded back:
//
//   1. q4_forms per group; a mixed group, a g64 expert among g128, a bf16 or fused routed expert are each
//      refused naming the first odd tensor (a names-only checkpoint);
//   2. q4_expected_names == the file's names both ways (the tower and the PLE shards skipped by design);
//      check_names refuses a missing and an extra tensor, each by name;
//   3. EVERY expert block of every layer (Review Focus 1): its words are its source's, and its g64 scales
//      dequantise every weight to (q - 8) x scale of the source - g64, and g128 through the exact expansion
//      (scales 2j and 2j + 1 = the source's j); block e at q4_gate_up_offset(d, e) / q4_down_offset(d, e);
//   4. gate||up interleave16 (Review Focus 2): an expert whose up is gate's sign flip decodes to +gate at
//      fused column (j / 16) x 32 + j % 16 and -gate 16 columns on;
//   5. the router: rows 0..7 mlp.gate, row 8 the shared gate, rows 9..15 zero;
//   6. the GDN small block at make_small_layout's offsets (conv widened, -exp(A_log), dt_bias, the gated
//      norm's plain w); every (1 + w) norm baked bit for bit (float(1) + float(w));
//   7. the dense rows: int4 layout 0 q||gate||k||v (k at 12288, v at 12800), qkv||z; bf16 tiles in Intel's;
//   8. the MTP head's bf16 experts equal rtn_int4_g64 of the source, block for block;
//   9. skip_layers_from(2) consumes layers 2..3, skip_prefix the tower and the head; 0 unconsumed;
//  10. the derived bytes: every host layer = q4_layer_bytes field by field; the real descriptor's formulas
//      (experts a layer, HC a layer, the final mixer, the int8 head, a GDN / QSA layer in Intel's forms,
//      the MTP head) printed.
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "loader/qwen4exp_layout.h"
#include "loader/qwen4exp_repack.h"
#include "loader/quant.h"
#include "loader/rtn.h"
#include "loader/safetensors.h"
#include "model/qwen4exp.h"

namespace {

namespace fs = std::filesystem;
using model::Q4Form;
using model::Q4Forms;
using model::Qwen4ExpDesc;

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
const T* view(const Tensors& ts, const std::string& n) {
  auto it = ts.find(n);
  if (it == ts.end()) {
    std::fprintf(stderr, "fixture has no %s\n", n.c_str());
    std::exit(1);
  }
  return reinterpret_cast<const T*>(it->second.bytes.data());
}

// A fast deterministic generator (the fixture is ~0.6 GB of values).
struct Rng {
  uint64_t s = 0x9E3779B97F4A7C15ull;
  uint32_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return uint32_t(s >> 16);
  }
  float uni(float a) { return (float(next() & 0xFFFFFF) / float(0xFFFFFF) * 2.0f - 1.0f) * a; }
};
Rng rng;

void add_bf16(Tensors& ts, const std::string& name, std::vector<uint64_t> shape, float amp = 1.0f) {
  size_t n = 1;
  for (uint64_t s : shape) n *= size_t(s);
  Tensor t{"BF16", std::move(shape), std::vector<uint8_t>(n * 2)};
  uint16_t* v = reinterpret_cast<uint16_t*>(t.bytes.data());
  for (size_t i = 0; i < n; ++i) v[i] = common::f32_to_bf16(rng.uni(amp));
  ts[name] = std::move(t);
}
// An AutoRound int4 linear [out N][in K] at group g: random words, distinct finite scales, qzeros.
void add_int4(Tensors& ts, const std::string& prefix, uint32_t K, uint32_t N, uint32_t g) {
  Tensor qw{"I32", {K / 8, N}, std::vector<uint8_t>(size_t(K / 8) * N * 4)};
  uint32_t* w = reinterpret_cast<uint32_t*>(qw.bytes.data());
  for (size_t i = 0; i < size_t(K / 8) * N; ++i) w[i] = rng.next() ^ (rng.next() << 16);
  Tensor sc{"F16", {K / g, N}, std::vector<uint8_t>(size_t(K / g) * N * 2)};
  uint16_t* s = reinterpret_cast<uint16_t*>(sc.bytes.data());
  for (size_t i = 0; i < size_t(K / g) * N; ++i) s[i] = common::f32_to_f16(0.01f + 0.04f * (rng.uni(1.0f) + 1.0f));
  Tensor qz{"I32", {K / g, N / 8}, std::vector<uint8_t>(size_t(K / g) * (N / 8) * 4)};
  uint32_t* z = reinterpret_cast<uint32_t*>(qz.bytes.data());
  for (size_t i = 0; i < size_t(K / g) * (N / 8); ++i) z[i] = 0x77777777u;
  ts[prefix + ".qweight"] = std::move(qw);
  ts[prefix + ".scales"] = std::move(sc);
  ts[prefix + ".qzeros"] = std::move(qz);
}
void add_i64(Tensors& ts, const std::string& name, const std::vector<int64_t>& v) {
  Tensor t{"I64", {v.size()}, std::vector<uint8_t>(v.size() * 8)};
  std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
  ts[name] = std::move(t);
}
void lin(Tensors& ts, const std::string& base, uint32_t K, uint32_t N, bool int4, uint32_t g = 64) {
  if (int4)
    add_int4(ts, base, K, N, g);
  else
    add_bf16(ts, base + ".weight", {N, K}, 1.0f / std::sqrt(float(K)));
}
void hc(Tensors& ts, const Qwen4ExpDesc& d, const std::string& base, bool inject) {
  add_bf16(ts, base + ".hc_norm.weight", {d.hc_n()}, 0.1f);
  add_bf16(ts, base + ".input_mix_weight_down.weight", {d.hc_low, d.hc_n()}, 0.01f);
  add_bf16(ts, base + ".input_mix_weight_up.weight", {d.hc_n(), d.hc_low}, 0.05f);
  if (inject) add_bf16(ts, base + ".block_inject_weight.weight", {d.hc, d.hc_n()}, 0.01f);
}
void moe(Tensors& ts, const Qwen4ExpDesc& d, const std::string& b, bool shared_int4, bool experts_int4, uint32_t g) {
  const uint32_t H = d.hidden, I = d.moe_inter, SI = d.shared_inter;
  add_bf16(ts, b + "mlp.gate.weight", {d.experts, H}, 0.06f);
  add_bf16(ts, b + "mlp.shared_expert_gate.weight", {1, H}, 0.02f);
  lin(ts, b + "mlp.shared_expert.gate_proj", H, SI, shared_int4);
  lin(ts, b + "mlp.shared_expert.up_proj", H, SI, shared_int4);
  lin(ts, b + "mlp.shared_expert.down_proj", SI, H, shared_int4);
  for (uint32_t e = 0; e < d.experts; ++e) {
    const std::string ep = b + "mlp.experts." + std::to_string(e) + ".";
    lin(ts, ep + "gate_proj", H, I, experts_int4, g);
    lin(ts, ep + "up_proj", H, I, experts_int4, g);
    lin(ts, ep + "down_proj", I, H, experts_int4, g);
  }
}
void qsa(Tensors& ts, const Qwen4ExpDesc& d, const std::string& b, bool dense_int4) {
  const uint32_t H = d.hidden;
  add_bf16(ts, b + "self_attn.q_norm.weight", {d.head_dim}, 0.1f);
  add_bf16(ts, b + "self_attn.k_norm.weight", {d.head_dim}, 0.1f);
  add_bf16(ts, b + "self_attn.indexer.index_qk_proj.weight", {d.idx_n(), H}, 0.02f);
  add_bf16(ts, b + "self_attn.indexer.q_layernorm.weight", {d.idx_dim}, 0.1f);
  add_bf16(ts, b + "self_attn.indexer.k_layernorm.weight", {d.idx_dim}, 0.1f);
  lin(ts, b + "self_attn.q_proj", H, 2 * d.q_n(), dense_int4);
  lin(ts, b + "self_attn.k_proj", H, d.kv_n(), dense_int4);
  lin(ts, b + "self_attn.v_proj", H, d.kv_n(), dense_int4);
  lin(ts, b + "self_attn.o_proj", d.q_n(), H, dense_int4);
}

Qwen4ExpDesc test_desc(const Q4Forms& f) {
  Qwen4ExpDesc d = model::qwen4exp();
  d.name = "qwen3.8-flash-next-test";
  d.layers = 4;
  d.experts = 8;
  d.top_k = 2;
  d.vocab = d.vocab_used = 64;
  d.ple_base = 1000;
  d.forms = f;
  return d;
}

// The fixture: every name q4_expected_names lists, plus the tower and two PLE shards (skipped by design).
Tensors fixture(const Qwen4ExpDesc& d, bool mtp) {
  Tensors ts;
  const bool dq = d.forms.dense == Q4Form::Int4, sq = d.forms.shared == Q4Form::Int4;
  const uint32_t H = d.hidden, g = d.forms.expert_group;
  add_bf16(ts, d.prefix + "embed_tokens.weight", {d.vocab, H});
  add_bf16(ts, "lm_head.weight", {d.vocab, H}, 0.02f);
  hc(ts, d, d.prefix + "hyper_connection_mixer", false);
  add_bf16(ts, "model.visual.patch_embed.proj.bias", {1152});
  add_bf16(ts, "model.visual.pos_embed.weight", {16, 1152});
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string b = Qwen4ExpDesc::layer_prefix(l);
    hc(ts, d, b + "attn_hyper_connection", true);
    hc(ts, d, b + "mlp_hyper_connection", true);
    moe(ts, d, b, sq, true, g);
    if (!d.is_qsa(l)) {
      add_bf16(ts, b + "linear_attn.A_log", {d.gdn_v_heads}, 2.0f);
      add_bf16(ts, b + "linear_attn.dt_bias", {d.gdn_v_heads});
      add_bf16(ts, b + "linear_attn.conv1d.weight", {d.conv_rows(), 1, d.conv_taps}, 0.5f);
      add_bf16(ts, b + "linear_attn.norm.weight", {d.gdn_head});
      add_bf16(ts, b + "linear_attn.in_proj_a.weight", {d.gdn_v_heads, H}, 0.02f);
      add_bf16(ts, b + "linear_attn.in_proj_b.weight", {d.gdn_v_heads, H}, 0.02f);
      lin(ts, b + "linear_attn.in_proj_qkv", H, d.conv_rows(), dq);
      lin(ts, b + "linear_attn.in_proj_z", H, d.gdn_z_n(), dq);
      lin(ts, b + "linear_attn.out_proj", d.gdn_z_n(), H, dq);
    } else {
      qsa(ts, d, b, dq);
    }
    if (l == d.ple_layer) {
      const std::string p = b + "ple.";
      add_bf16(ts, p + "conv1d.weight", {d.hc_n(), 1, d.ple_conv_taps}, 0.5f);
      add_bf16(ts, p + "key_proj.weight", {d.hc_n(), d.ple_e()}, 0.02f);
      add_bf16(ts, p + "value_proj.weight", {d.hidden, d.ple_e()}, 0.02f);
      add_bf16(ts, p + "norm_conv.weight", {d.hc_n()}, 0.1f);
      add_bf16(ts, p + "norm_key.weight", {d.hc_n()}, 0.1f);
      add_bf16(ts, p + "norm_query.weight", {d.hc_n()}, 0.1f);
      add_i64(ts, p + "ple_embedding.layer_multipliers", {11, 13, 17});
      std::vector<int64_t> sizes(16), offs(16);
      for (int h = 0; h < 16; ++h) sizes[h] = 1009 + h, offs[h] = h * 1100;
      add_i64(ts, p + "ple_embedding.ngram_heads_vocab_sizes", sizes);
      add_i64(ts, p + "ple_embedding.ngram_heads_offsets", offs);
      add_bf16(ts, p + "ple_embedding.ngram_embedding.shard_0.weight", {10, d.ple_dim});
      add_bf16(ts, p + "ple_embedding.ngram_embedding.shard_1.weight", {10, d.ple_dim});
    }
  }
  if (mtp) {
    add_bf16(ts, "mtp.fc_embedding.weight", {H, H}, 0.02f);
    add_bf16(ts, "mtp.fc_hidden.weight", {H, H}, 0.02f);
    add_bf16(ts, "mtp.pre_fc_norm_embedding.weight", {H}, 0.1f);
    add_bf16(ts, "mtp.pre_fc_norm_hidden.weight", {d.hc_n()}, 0.1f);
    hc(ts, d, "mtp.hyper_connection_mixer", false);
    const std::string b = "mtp.layers.0.";
    hc(ts, d, b + "attn_hyper_connection", true);
    hc(ts, d, b + "mlp_hyper_connection", true);
    moe(ts, d, b, false, false, 64);
    qsa(ts, d, b, false);
  }
  return ts;
}

// Layer 2 expert 5's up_proj := gate_proj's sign flip (Review Focus 2): nibbles kept in 1..15, up's nibble
// 16 - gate's, the same scales - so up dequantises to exactly -gate.
void plant_sign_flip(Tensors& ts) {
  const std::string b = Qwen4ExpDesc::layer_prefix(2) + "mlp.experts.5.";
  Tensor& g = ts[b + "gate_proj.qweight"];
  Tensor& u = ts[b + "up_proj.qweight"];
  uint32_t* gw = reinterpret_cast<uint32_t*>(g.bytes.data());
  uint32_t* uw = reinterpret_cast<uint32_t*>(u.bytes.data());
  for (size_t i = 0; i < g.bytes.size() / 4; ++i) {
    uint32_t a = 0, c = 0;
    for (int j = 0; j < 8; ++j) {
      uint32_t q = (gw[i] >> (4 * j)) & 0xF;
      if (q == 0) q = 1;
      a |= q << (4 * j);
      c |= (16 - q) << (4 * j);
    }
    gw[i] = a;
    uw[i] = c;
  }
  ts[b + "up_proj.scales"].bytes = ts[b + "gate_proj.scales"].bytes;
}

fs::path tmpdir(const std::string& tag) {
  return fs::temp_directory_path() / ("qwen4exp_repack_test_" + tag + "_" + std::to_string(::getpid()));
}

template <class F>
void throws_naming(const std::string& what, F fn) {
  bool ok = false;
  try {
    fn();
  } catch (const std::exception& e) {
    ok = std::string(e.what()).find(what) != std::string::npos;
    if (!ok) std::fprintf(stderr, "threw, but without '%s': %s\n", what.c_str(), e.what());
  }
  if (!ok) std::fprintf(stderr, "expected a refusal naming '%s'\n", what.c_str());
  CHECK(ok);
}

// A layout-1 block's word / scale at (k-octet r, column c) - common::repack_int4_layout1's layout.
uint32_t l1_word(const uint32_t* blk, uint32_t K, uint32_t r, uint32_t c) {
  return blk[(size_t(c / 16) * (K / 64) + r / 8) * 136 + (r % 8) * 16 + c % 16];
}
uint16_t l1_scale(const uint32_t* blk, uint32_t K, uint32_t g, uint32_t c) {
  const uint32_t w = blk[(size_t(c / 16) * (K / 64) + g) * 136 + 128 + (c % 16) / 2];
  return uint16_t(c % 2 == 0 ? w & 0xFFFFu : w >> 16);
}
// (q - 8) x scale of element (k, c) of a layout-1 block / of a GPTQ source at group g.
float l1_deq(const uint32_t* blk, uint32_t K, uint32_t k, uint32_t c) {
  const uint32_t q = (l1_word(blk, K, k / 8, c) >> (4 * (k % 8))) & 0xF;
  return (float(q) - 8.0f) * common::f16_to_f32(l1_scale(blk, K, k / 64, c));
}
float src_deq(const uint32_t* qw, const uint16_t* sc, uint32_t N, uint32_t g, uint32_t k, uint32_t c) {
  const uint32_t q = (qw[size_t(k / 8) * N + c] >> (4 * (k % 8))) & 0xF;
  return (float(q) - 8.0f) * common::f16_to_f32(sc[size_t(k / g) * N + c]);
}
uint32_t q4_nib(const uint32_t* blk, uint32_t K, uint32_t k, uint32_t c) {
  return (l1_word(blk, K, k / 8, c) >> (4 * (k % 8))) & 0xF;
}
// gemv_bf16's tile element (k, n) of a {K, N} tiled matrix.
uint16_t tiled(const uint16_t* t, uint32_t K, uint32_t k, uint32_t n) {
  return t[((size_t(n / 16) * (K / 8) + k / 8) * 8 + k % 8) * 16 + n % 16];
}
bool same_f32(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }
// (1 + w) baked: float(1) + float(w), bit for bit.
void check_one_plus_w(const uint8_t* dst, const uint16_t* w, uint32_t n) {
  const float* f = reinterpret_cast<const float*>(dst);
  for (uint32_t i = 0; i < n; ++i) CHECK(same_f32(f[i], 1.0f + common::bf16_to_f32(w[i])));
}
// One HC block against its source tensors.
void check_hc(const Qwen4ExpDesc& d, const std::vector<uint8_t>& blk, const Tensors& ts, const std::string& base, bool inject) {
  const loader::Q4HcOffsets o = loader::q4_hc_offsets(d, inject);
  CHECK_EQ(blk.size(), o.total);
  const uint16_t* dt = reinterpret_cast<const uint16_t*>(blk.data() + o.down);
  const uint16_t* ut = reinterpret_cast<const uint16_t*>(blk.data() + o.up);
  const uint16_t* dn = view<uint16_t>(ts, base + ".input_mix_weight_down.weight");
  const uint16_t* up = view<uint16_t>(ts, base + ".input_mix_weight_up.weight");
  const uint32_t HC = d.hc_n();
  for (uint32_t n = 0; n < o.down_n; ++n)
    for (uint32_t k = 0; k < HC; k += 97) {
      uint16_t want = 0;
      if (n < d.hc_low)
        want = dn[size_t(n) * HC + k];
      else if (inject && n < d.hc_low + d.hc)
        want = view<uint16_t>(ts, base + ".block_inject_weight.weight")[size_t(n - d.hc_low) * HC + k];
      CHECK(tiled(dt, HC, k, n) == want);
    }
  for (uint32_t n = 0; n < HC; n += 31)
    for (uint32_t k = 0; k < d.hc_low; ++k) CHECK(tiled(ut, d.hc_low, k, n) == up[size_t(n) * d.hc_low + k]);
  check_one_plus_w(blk.data() + o.norm, view<uint16_t>(ts, base + ".hc_norm.weight"), HC);
}

// Every routed expert's two blocks against its source (Review Focus 1: word for word, and every weight
// dequantised through the layout's g64 scales equal to the source's at its own group).
void check_experts(const Qwen4ExpDesc& d, const loader::Q4HostLayer& h, const Tensors& ts, const std::string& lp,
                   uint32_t g) {
  const uint32_t H = d.hidden, I = d.moe_inter;
  const size_t gub = loader::q4_gate_up_block_bytes(d) / 4, dnb = loader::q4_down_block_bytes(d) / 4;
  CHECK_EQ(h.gate_up.size(), gub * d.experts);
  CHECK_EQ(h.down.size(), dnb * d.experts);
  for (uint32_t e = 0; e < d.experts; ++e) {
    CHECK_EQ(loader::q4_gate_up_offset(d, e), size_t(e) * gub * 4);
    const uint32_t* gu = h.gate_up.data() + loader::q4_gate_up_offset(d, e) / 4;
    const uint32_t* dn = h.down.data() + loader::q4_down_offset(d, e) / 4;
    const std::string ep = lp + "mlp.experts." + std::to_string(e) + ".";
    const uint32_t* gq = view<uint32_t>(ts, ep + "gate_proj.qweight");
    const uint16_t* gs = view<uint16_t>(ts, ep + "gate_proj.scales");
    const uint32_t* uq = view<uint32_t>(ts, ep + "up_proj.qweight");
    const uint16_t* us = view<uint16_t>(ts, ep + "up_proj.scales");
    const uint32_t* dq = view<uint32_t>(ts, ep + "down_proj.qweight");
    const uint16_t* ds = view<uint16_t>(ts, ep + "down_proj.scales");
    for (uint32_t c = 0; c < 2 * I; ++c) {
      const bool is_up = (c % 32) >= 16;
      const uint32_t j = (c / 32) * 16 + c % 16;
      const uint32_t* q = is_up ? uq : gq;
      const uint16_t* s = is_up ? us : gs;
      for (uint32_t r = 0; r < H / 8; ++r) CHECK(l1_word(gu, H, r, c) == q[size_t(r) * I + j]);
      for (uint32_t gg = 0; gg < H / 64; ++gg) CHECK(l1_scale(gu, H, gg, c) == s[size_t(gg * 64 / g) * I + j]);
      if (c % 61 == 0)
        for (uint32_t k = 0; k < H; ++k) CHECK(same_f32(l1_deq(gu, H, k, c), src_deq(q, s, I, g, k, j)));
    }
    for (uint32_t c = 0; c < H; ++c) {
      for (uint32_t r = 0; r < I / 8; ++r) CHECK(l1_word(dn, I, r, c) == dq[size_t(r) * H + c]);
      for (uint32_t gg = 0; gg < I / 64; ++gg) CHECK(l1_scale(dn, I, gg, c) == ds[size_t(gg * 64 / g) * H + c]);
      if (c % 127 == 0)
        for (uint32_t k = 0; k < I; ++k) CHECK(same_f32(l1_deq(dn, I, k, c), src_deq(dq, ds, H, g, k, c)));
    }
  }
}

void check_router(const Qwen4ExpDesc& d, const loader::Q4HostLayer& h, const Tensors& ts, const std::string& lp) {
  const uint32_t H = d.hidden;
  CHECK_EQ(d.router_n(), 16u);
  CHECK(h.router.shape.K == H && h.router.shape.N == 16 && h.router.tiles.size() == size_t(16) * H);
  const uint16_t* gw = view<uint16_t>(ts, lp + "mlp.gate.weight");
  const uint16_t* sg = view<uint16_t>(ts, lp + "mlp.shared_expert_gate.weight");
  for (uint32_t n = 0; n < 16; ++n)
    for (uint32_t k = 0; k < H; k += 3) {
      const uint16_t want = n < d.experts ? gw[size_t(n) * H + k] : n == d.experts ? sg[k] : uint16_t(0);
      CHECK(tiled(h.router.tiles.data(), H, k, n) == want);
    }
}

void check_dense_int4(const loader::Q4HostWeight& w, const Tensors& ts, const std::string& lp,
                      const std::vector<std::pair<std::string, uint32_t>>& parts) {
  CHECK(w.kind == model::WeightKind::Int4 && w.tiles.empty());
  const uint32_t K = w.shape.K, N = w.shape.N;
  CHECK_EQ(w.words.size(), size_t(K / 8) * N);
  CHECK_EQ(w.scales.size(), size_t(K / 64) * N);
  uint32_t off = 0;
  for (const auto& [name, pn] : parts) {
    const uint32_t* q = view<uint32_t>(ts, lp + name + ".qweight");
    const uint16_t* s = view<uint16_t>(ts, lp + name + ".scales");
    std::set<uint32_t> cols = {0, 1, pn / 2, pn - 1};
    for (uint32_t c = 0; c < pn; c += 97) cols.insert(c);
    for (uint32_t c : cols) {
      for (uint32_t r = 0; r < K / 8; ++r) CHECK(w.words[size_t(r) * N + off + c] == q[size_t(r) * pn + c]);
      for (uint32_t g = 0; g < K / 64; ++g) CHECK(w.scales[size_t(g) * N + off + c] == s[size_t(g) * pn + c]);
    }
    off += pn;
  }
  CHECK_EQ(off, N);
}

void check_dense_bf16(const loader::Q4HostWeight& w, const Tensors& ts, const std::string& lp,
                      const std::vector<std::pair<std::string, uint32_t>>& parts) {
  CHECK(w.kind == model::WeightKind::Bf16 && w.words.empty());
  const uint32_t K = w.shape.K, N = w.shape.N;
  CHECK_EQ(w.tiles.size(), size_t(K) * N);
  uint32_t off = 0;
  for (const auto& [name, pn] : parts) {
    const uint16_t* src = view<uint16_t>(ts, lp + name + ".weight");
    for (uint32_t c = 0; c < pn; c += 13)
      for (uint32_t k = 0; k < K; k += 37) CHECK(tiled(w.tiles.data(), K, k, off + c) == src[size_t(c) * K + k]);
    off += pn;
  }
  for (uint32_t c = off; c < N; ++c)   // padding columns (a||b) are zero
    for (uint32_t k = 0; k < K; k += 37) CHECK(tiled(w.tiles.data(), K, k, c) == 0);
}

// Names-only checkpoints for q4_forms' refusals: every tensor 8 bytes, the experts' shapes real.
void check_forms_refusals() {
  const Qwen4ExpDesc d = test_desc({Q4Form::Int4, Q4Form::Int4, Q4Form::Bf16, 64});
  const auto names_ckpt = [&](const std::string& tag, const std::vector<std::string>& names,
                              const std::map<std::string, std::vector<uint64_t>>& shapes) {
    Tensors ts;
    for (const std::string& n : names) {
      Tensor t{n.size() > 8 && n.compare(n.size() - 7, 7, ".scales") == 0 ? "F16" : "I32", {2}, std::vector<uint8_t>(8)};
      if (n.size() > 7 && n.compare(n.size() - 7, 7, ".weight") == 0) t.dtype = "BF16";
      auto it = shapes.find(n);
      if (it != shapes.end()) t.shape = it->second;
      ts[n] = t;
    }
    const fs::path dir = tmpdir("names_" + tag);
    write_checkpoint(dir, ts);
    return dir;
  };
  std::vector<std::string> base = loader::q4_expected_names(d, d.forms, false);
  std::map<std::string, std::vector<uint64_t>> shapes;
  for (const std::string& n : base)
    if (n.find(".mlp.experts.") != std::string::npos) {
      const bool down = n.find("down_proj") != std::string::npos;
      const uint64_t K = down ? d.moe_inter : d.hidden, N = down ? d.hidden : d.moe_inter;
      if (n.find(".qweight") != std::string::npos) shapes[n] = {K / 8, N};
      if (n.find(".scales") != std::string::npos) shapes[n] = {K / 64, N};
    }
  {   // the base passes
    const fs::path dir = names_ckpt("ok", base, shapes);
    loader::SafetensorsSet set(dir.string() + "/");
    const Q4Forms f = loader::q4_forms(set, d);
    CHECK(f.dense == Q4Form::Int4 && f.shared == Q4Form::Int4 && f.mtp_experts == Q4Form::Bf16 && f.expert_group == 64);
    fs::remove_all(dir);
  }
  const auto variant = [&](const std::string& tag, const std::string& drop, const std::vector<std::string>& add,
                           const std::map<std::string, std::vector<uint64_t>>& sh2, const std::string& naming) {
    std::vector<std::string> v;
    for (const std::string& n : base)
      if (n.compare(0, drop.size(), drop) != 0) v.push_back(n);
    v.insert(v.end(), add.begin(), add.end());
    auto sh = shapes;
    for (const auto& [k, s] : sh2) sh[k] = s;
    const fs::path dir = names_ckpt(tag, v, sh);
    loader::SafetensorsSet set(dir.string() + "/");
    throws_naming(naming, [&] { loader::q4_forms(set, d); });
    fs::remove_all(dir);
  };
  const std::string l3o = Qwen4ExpDesc::layer_prefix(3) + "self_attn.o_proj";
  variant("mixed_dense", l3o + ".", {l3o + ".weight"}, {}, l3o + ".weight");
  const std::string l2s = Qwen4ExpDesc::layer_prefix(2) + "mlp.shared_expert.down_proj";
  variant("mixed_shared", l2s + ".", {l2s + ".weight"}, {}, l2s + ".weight");
  const std::string e = Qwen4ExpDesc::layer_prefix(1) + "mlp.experts.6.up_proj";
  variant("mixed_group", "@none", {}, {{e + ".scales", {d.hidden / 128, d.moe_inter}}}, e + ".scales");
  variant("bf16_expert", e + ".", {e + ".weight"}, {}, e + ".weight");
  const std::string fused = Qwen4ExpDesc::layer_prefix(2) + "mlp.experts.gate_up_proj";
  variant("fused", "@none", {fused}, {}, fused);
}

void check_form(const Q4Forms& want, bool mtp) {
  const Qwen4ExpDesc d = test_desc(want);
  Tensors ts = fixture(d, mtp);
  plant_sign_flip(ts);
  const std::string tag = want.dense == Q4Form::Int4 ? "ours" : "intel";
  const fs::path dir = tmpdir(tag);
  write_checkpoint(dir, ts);
  const uint32_t H = d.hidden, I = d.moe_inter;
  {
    loader::SafetensorsSet set(dir.string() + "/");
    (void)loader::assert_quant_invariants(set);
    // 1. the forms from the names
    const Q4Forms f = loader::q4_forms(set, d);
    CHECK(f.dense == want.dense && f.shared == want.shared && f.expert_group == want.expert_group);
    CHECK(f.mtp_experts == Q4Form::Bf16);
    // 2. the names both ways (the tower and the PLE shards skipped by design)
    const std::vector<std::string> exp = loader::q4_expected_names(d, f, mtp);
    const std::set<std::string> es(exp.begin(), exp.end());
    CHECK_EQ(es.size(), exp.size());
    size_t skipped = 0;
    for (const auto& [n, t] : ts) {
      (void)t;
      if (n.rfind("model.visual.", 0) == 0 || n.find("ngram_embedding.shard_") != std::string::npos) {
        ++skipped;
        continue;
      }
      if (!es.count(n)) std::fprintf(stderr, "not expected: %s\n", n.c_str());
      CHECK(es.count(n) == 1);
    }
    for (const std::string& n : exp) CHECK(ts.count(n) == 1);
    CHECK_EQ(skipped, size_t(4));
    loader::Q4Checkpoint ck(d, set);
    ck.check_names(mtp);
    if (mtp) ck.check_names(false);   // the head not loaded: mtp.* skipped by design
    {
      Qwen4ExpDesc more = d, fewer = d;
      more.experts = 9;
      fewer.experts = 7;
      throws_naming("mlp.experts.8.gate_proj", [&] { loader::Q4Checkpoint(more, set).check_names(mtp); });
      throws_naming("mlp.experts.7.", [&] { loader::Q4Checkpoint(fewer, set).check_names(mtp); });
    }

    // 3 - 7 and 10, layer by layer
    for (uint32_t l = 0; l < d.layers; ++l) {
      const std::string lp = Qwen4ExpDesc::layer_prefix(l);
      loader::Q4HostLayer h;
      ck.repack_layer(l, h);
      const loader::Q4LayerBytes hb = h.bytes(), want_b = loader::q4_layer_bytes(d, l);
      CHECK(hb.hc_attn == want_b.hc_attn && hb.hc_mlp == want_b.hc_mlp && hb.gdn_qkvz == want_b.gdn_qkvz &&
            hb.gdn_ab == want_b.gdn_ab && hb.gdn_out == want_b.gdn_out && hb.gdn_small == want_b.gdn_small &&
            hb.qsa_qkvg == want_b.qsa_qkvg && hb.qsa_idx == want_b.qsa_idx && hb.qsa_o == want_b.qsa_o &&
            hb.qsa_small == want_b.qsa_small && hb.router == want_b.router && hb.gate_up == want_b.gate_up &&
            hb.down == want_b.down && hb.shared_gate_up == want_b.shared_gate_up &&
            hb.shared_down == want_b.shared_down && hb.ple == want_b.ple);
      CHECK_EQ(hb.total(), want_b.total());
      check_hc(d, h.hc_attn, ts, lp + "attn_hyper_connection", true);
      check_hc(d, h.hc_mlp, ts, lp + "mlp_hyper_connection", true);
      check_experts(d, h, ts, lp, want.expert_group);
      check_router(d, h, ts, lp);
      // 4. the planted sign flip
      if (l == 2) {
        const uint32_t* gu = h.gate_up.data() + loader::q4_gate_up_offset(d, 5) / 4;
        for (uint32_t j = 0; j < I; j += 7)
          for (uint32_t k = 0; k < H; k += 11) {
            const uint32_t cg = (j / 16) * 32 + j % 16;
            const float a = l1_deq(gu, H, k, cg), b = l1_deq(gu, H, k, cg + 16);
            CHECK(a == -b && (a != 0.0f || q4_nib(gu, H, k, cg) == 8));
          }
      }
      // the shared expert
      if (want.shared == Q4Form::Int4) {
        const uint32_t* gu = reinterpret_cast<const uint32_t*>(h.shared.data());
        const uint32_t* q = view<uint32_t>(ts, lp + "mlp.shared_expert.up_proj.qweight");
        for (uint32_t r = 0; r < H / 8; r += 5) CHECK(l1_word(gu, H, r, 16) == q[size_t(r) * I + 0]);
        const uint32_t* dn = reinterpret_cast<const uint32_t*>(h.shared.data() + h.shared_gate_up);
        const uint32_t* dq = view<uint32_t>(ts, lp + "mlp.shared_expert.down_proj.qweight");
        for (uint32_t r = 0; r < I / 8; ++r) CHECK(l1_word(dn, I, r, H - 1) == dq[size_t(r) * H + H - 1]);
      } else {
        const uint16_t* t = reinterpret_cast<const uint16_t*>(h.shared.data());
        const uint16_t* u = view<uint16_t>(ts, lp + "mlp.shared_expert.up_proj.weight");
        for (uint32_t k = 0; k < H; k += 7) CHECK(tiled(t, H, k, 16 + 3) == u[size_t(3) * H + k]);
        const uint16_t* dt = reinterpret_cast<const uint16_t*>(h.shared.data() + h.shared_gate_up);
        const uint16_t* dn = view<uint16_t>(ts, lp + "mlp.shared_expert.down_proj.weight");
        for (uint32_t k = 0; k < I; k += 3) CHECK(tiled(dt, I, k, H - 1) == dn[size_t(H - 1) * I + k]);
      }
      if (!d.is_qsa(l)) {
        // 6. the GDN small block
        const loader::SmallLayout sl = loader::q4_gdn_small(d);
        CHECK_EQ(h.gdn_small.size(), sl.gdn_block_bytes);
        const float* conv = reinterpret_cast<const float*>(h.gdn_small.data() + sl.gdn_off_conv);
        const uint16_t* cw = view<uint16_t>(ts, lp + "linear_attn.conv1d.weight");
        for (uint32_t i = 0; i < d.conv_rows() * d.conv_taps; ++i) CHECK(same_f32(conv[i], common::bf16_to_f32(cw[i])));
        const float* nega = reinterpret_cast<const float*>(h.gdn_small.data() + sl.gdn_off_nega);
        const float* dtb = reinterpret_cast<const float*>(h.gdn_small.data() + sl.gdn_off_dtbias);
        const uint16_t* al = view<uint16_t>(ts, lp + "linear_attn.A_log");
        const uint16_t* db = view<uint16_t>(ts, lp + "linear_attn.dt_bias");
        for (uint32_t i = 0; i < d.gdn_v_heads; ++i) {
          CHECK(same_f32(nega[i], -std::exp(common::bf16_to_f32(al[i]))));
          CHECK(same_f32(dtb[i], common::bf16_to_f32(db[i])));
        }
        CHECK(std::memcmp(h.gdn_small.data() + sl.gdn_off_gated_norm, view<uint8_t>(ts, lp + "linear_attn.norm.weight"),
                          size_t(d.gdn_head) * 2) == 0);
        // 7. the dense rows
        const std::vector<std::pair<std::string, uint32_t>> qkvz = {{"linear_attn.in_proj_qkv", d.conv_rows()},
                                                                     {"linear_attn.in_proj_z", d.gdn_z_n()}};
        const std::vector<std::pair<std::string, uint32_t>> out = {{"linear_attn.out_proj", H}};
        if (want.dense == Q4Form::Int4) {
          check_dense_int4(h.gdn_qkvz, ts, lp, qkvz);
          check_dense_int4(h.gdn_out, ts, lp, out);
          CHECK(h.gdn_qkvz.shape.N == 16384 && h.gdn_qkvz.shape.S == 1);
        } else {
          check_dense_bf16(h.gdn_qkvz, ts, lp, qkvz);
          check_dense_bf16(h.gdn_out, ts, lp, out);
        }
        check_dense_bf16(h.gdn_ab, ts, lp, {{"linear_attn.in_proj_a", 48}, {"linear_attn.in_proj_b", 48}});
        CHECK(h.qsa_qkvg.bytes() == 0 && h.qsa_small.empty());
      } else {
        const loader::Q4QsaSmall s = loader::q4_qsa_small(d);
        CHECK_EQ(h.qsa_small.size(), s.total);
        check_one_plus_w(h.qsa_small.data() + s.q_norm, view<uint16_t>(ts, lp + "self_attn.q_norm.weight"), d.head_dim);
        check_one_plus_w(h.qsa_small.data() + s.k_norm, view<uint16_t>(ts, lp + "self_attn.k_norm.weight"), d.head_dim);
        check_one_plus_w(h.qsa_small.data() + s.idx_q_norm, view<uint16_t>(ts, lp + "self_attn.indexer.q_layernorm.weight"),
                         d.idx_dim);
        check_one_plus_w(h.qsa_small.data() + s.idx_k_norm, view<uint16_t>(ts, lp + "self_attn.indexer.k_layernorm.weight"),
                         d.idx_dim);
        const std::vector<std::pair<std::string, uint32_t>> qkvg = {
            {"self_attn.q_proj", 2 * d.q_n()}, {"self_attn.k_proj", d.kv_n()}, {"self_attn.v_proj", d.kv_n()}};
        if (want.dense == Q4Form::Int4) {
          check_dense_int4(h.qsa_qkvg, ts, lp, qkvg);   // k at 12288, v at 12800
          check_dense_int4(h.qsa_o, ts, lp, {{"self_attn.o_proj", H}});
          CHECK(h.qsa_qkvg.shape.S == d.qkvg_s && h.qsa_o.shape.S == d.o_s);
        } else {
          check_dense_bf16(h.qsa_qkvg, ts, lp, qkvg);
          check_dense_bf16(h.qsa_o, ts, lp, {{"self_attn.o_proj", H}});
        }
        check_dense_bf16(h.qsa_idx, ts, lp, {{"self_attn.indexer.index_qk_proj", d.idx_n()}});
        CHECK(h.gdn_qkvz.bytes() == 0 && h.gdn_small.empty());
      }
      if (l == d.ple_layer) {
        const loader::Q4PleOffsets o = loader::q4_ple_offsets(d);
        CHECK_EQ(h.ple.size(), o.total);
        const uint16_t* kt = reinterpret_cast<const uint16_t*>(h.ple.data() + o.kv);
        const uint16_t* kw = view<uint16_t>(ts, lp + "ple.key_proj.weight");
        const uint16_t* vw = view<uint16_t>(ts, lp + "ple.value_proj.weight");
        for (uint32_t k = 0; k < d.ple_e(); k += 41) {
          CHECK(tiled(kt, d.ple_e(), k, 10239) == kw[size_t(10239) * d.ple_e() + k]);
          CHECK(tiled(kt, d.ple_e(), k, 10240) == vw[k]);
        }
        check_one_plus_w(h.ple.data() + o.norm_key, view<uint16_t>(ts, lp + "ple.norm_key.weight"), d.hc_n());
        check_one_plus_w(h.ple.data() + o.norm_query, view<uint16_t>(ts, lp + "ple.norm_query.weight"), d.hc_n());
        check_one_plus_w(h.ple.data() + o.norm_conv, view<uint16_t>(ts, lp + "ple.norm_conv.weight"), d.hc_n());
        const float* pc = reinterpret_cast<const float*>(h.ple.data() + o.conv);
        const uint16_t* pw = view<uint16_t>(ts, lp + "ple.conv1d.weight");
        for (uint32_t i = 0; i < d.hc_n() * d.ple_conv_taps; ++i) CHECK(same_f32(pc[i], common::bf16_to_f32(pw[i])));
      } else {
        CHECK(h.ple.empty());
      }
    }
    // the top level
    CHECK(std::memcmp(ck.embed(), view<uint16_t>(ts, d.prefix + "embed_tokens.weight"), loader::q4_embed_bytes(d)) == 0);
    CHECK(std::memcmp(ck.lm_head(), view<uint16_t>(ts, "lm_head.weight"), size_t(d.vocab) * H * 2) == 0);
    check_hc(d, ck.final_mixer(), ts, d.prefix + "hyper_connection_mixer", false);
    const loader::Q4PleConstants pc = ck.ple_constants();
    CHECK(pc.multipliers[0] == 11 && pc.multipliers[2] == 17 && pc.sizes[15] == 1024 && pc.offsets[3] == 3300);
    CHECK_EQ(ck.skip_ple_shards(), size_t(2));
    CHECK_EQ(ck.skip_prefix("model.visual."), size_t(2));
    if (mtp) {
      // 8. the MTP head: bf16 experts -> rtn_int4_g64, block for block; the bf16 dense and shared arms
      loader::Q4HostLayer h;
      ck.repack_mtp(h);
      const Qwen4ExpDesc m = loader::q4_mtp_desc(d);
      const loader::Q4LayerBytes hb = h.bytes(), want_b = loader::q4_mtp_layer_bytes(d);
      CHECK_EQ(hb.total(), want_b.total());
      CHECK(h.qsa_qkvg.kind == model::WeightKind::Bf16 && hb.shared() == loader::q4_shared_gate_up_bytes(m) +
                                                                            loader::q4_shared_down_bytes(m));
      check_hc(d, h.hc_attn, ts, "mtp.layers.0.attn_hyper_connection", true);
      check_router(d, h, ts, "mtp.layers.0.");
      std::vector<uint32_t> qg(size_t(H / 8) * I), qu(qg.size()), qd(size_t(I / 8) * H);
      std::vector<uint16_t> sg(size_t(H / 64) * I), su(sg.size()), sd(size_t(I / 64) * H);
      std::vector<uint32_t> gu(loader::q4_gate_up_block_bytes(d) / 4), dn(loader::q4_down_block_bytes(d) / 4);
      for (uint32_t e = 0; e < d.experts; ++e) {
        const std::string ep = "mtp.layers.0.mlp.experts." + std::to_string(e) + ".";
        loader::rtn_int4_g64(view<uint16_t>(ts, ep + "gate_proj.weight"), H, I, qg.data(), sg.data());
        loader::rtn_int4_g64(view<uint16_t>(ts, ep + "up_proj.weight"), H, I, qu.data(), su.data());
        loader::rtn_int4_g64(view<uint16_t>(ts, ep + "down_proj.weight"), I, H, qd.data(), sd.data());
        common::repack_int4_layout1_cols(H, 2 * I, common::cols_interleave16({qg.data(), sg.data(), I}, {qu.data(), su.data(), I}),
                                         gu.data());
        common::repack_int4_layout1_cols(I, H, common::cols_concat({{qd.data(), sd.data(), H}}), dn.data());
        CHECK(std::memcmp(gu.data(), h.gate_up.data() + loader::q4_gate_up_offset(d, e) / 4, gu.size() * 4) == 0);
        CHECK(std::memcmp(dn.data(), h.down.data() + loader::q4_down_offset(d, e) / 4, dn.size() * 4) == 0);
      }
      const std::vector<uint8_t> fc = ck.mtp_fc();
      const loader::Q4MtpFcOffsets fo = loader::q4_mtp_fc_offsets(d);
      CHECK_EQ(fc.size(), fo.total);
      check_one_plus_w(fc.data() + fo.norm_hidden, view<uint16_t>(ts, "mtp.pre_fc_norm_hidden.weight"), d.hc_n());
      const uint16_t* fh = view<uint16_t>(ts, "mtp.fc_hidden.weight");
      for (uint32_t k = 0; k < H; k += 17)
        CHECK(tiled(reinterpret_cast<const uint16_t*>(fc.data() + fo.fc_hidden), H, k, 7) == fh[size_t(7) * H + k]);
      check_hc(d, ck.mtp_mixer(), ts, "mtp.hyper_connection_mixer", false);
      CHECK_EQ(loader::q4_mtp_bytes(d), want_b.total() + fo.total + loader::q4_final_mixer_bytes(d));
    }
    // 9. everything accounted for
    std::string names;
    const size_t left = ck.unconsumed(&names);
    if (left) std::fprintf(stderr, "unconsumed: %s\n", names.c_str());
    CHECK_EQ(left, size_t(0));
    {
      loader::Q4Checkpoint part(d, set);
      for (uint32_t l = 0; l < 2; ++l) {
        loader::Q4HostLayer h;
        part.repack_layer(l, h);
      }
      (void)part.embed();
      (void)part.lm_head();
      (void)part.final_mixer();
      (void)part.ple_constants();
      part.skip_ple_shards();
      part.skip_prefix("model.visual.");
      CHECK(part.unconsumed() > 0);
      const size_t skipped_layers = part.skip_layers_from(2);
      CHECK(skipped_layers > 0);
      if (mtp) CHECK(part.skip_prefix("mtp.") > 0);
      CHECK_EQ(part.unconsumed(), size_t(0));
    }
  }
  fs::remove_all(dir);
  std::printf("  %s form%s: forms read, names both ways, %u layers x %u expert blocks word for word (g%u), the "
              "router, the small blocks, every (1 + w), the dense rows%s; 0 unconsumed\n",
              tag.c_str(), mtp ? " + MTP" : "", d.layers, d.experts, want.expert_group,
              mtp ? ", the MTP head = rtn_int4_g64 of its bf16 experts" : "");
}

}  // namespace

int main() {
  check_forms_refusals();
  check_form({Q4Form::Int4, Q4Form::Int4, Q4Form::Bf16, 64}, true);     // ours (21q), with the MTP head
  check_form({Q4Form::Bf16, Q4Form::Bf16, Q4Form::Bf16, 128}, false);   // Intel's interim export

  // 10. the real descriptor's derived bytes (the formulas the planner and the loader share).
  const Qwen4ExpDesc& d = model::qwen4exp();
  Qwen4ExpDesc intel = d;
  intel.forms = {Q4Form::Bf16, Q4Form::Bf16, Q4Form::Bf16, 128};
  const loader::Q4LayerBytes g = loader::q4_layer_bytes(intel, 0), q = loader::q4_layer_bytes(intel, 3);
  const loader::Q4LayerBytes p = loader::q4_layer_bytes(intel, 1), go = loader::q4_layer_bytes(d, 0);
  CHECK_EQ(g.experts(), size_t(d.experts) * (loader::q4_layout1_bytes(2560, 1280) + loader::q4_layout1_bytes(640, 2560)));
  CHECK_EQ(g.hc_attn + g.hc_mlp, size_t(2) * (loader::q4_bf16_bytes(10240, 336) + loader::q4_bf16_bytes(320, 10240) +
                                              size_t(10240) * 4));
  CHECK_EQ(loader::q4_final_mixer_bytes(d), loader::q4_bf16_bytes(10240, 320) * 2 + size_t(10240) * 4);
  CHECK_EQ(loader::q4_lm_head_bytes(d, true), size_t(248320) * 2560 + size_t(248320) * 4);
  CHECK_EQ(g.router, loader::q4_bf16_bytes(2560, 528));
  CHECK_EQ(g.gdn_qkvz, loader::q4_bf16_bytes(2560, 16384));
  CHECK_EQ(go.gdn_qkvz, loader::q4_layout0_bytes(2560, 16384));
  CHECK_EQ(p.ple, loader::q4_bf16_bytes(2560, 12800) + size_t(3) * 10240 * 4 + size_t(10240) * 4 * 4);
  CHECK_EQ(p.total(), g.total() + p.ple);
  CHECK(g.shared() == loader::q4_bf16_bytes(2560, 1280) + loader::q4_bf16_bytes(640, 2560) &&
        go.shared() == loader::q4_layout1_bytes(2560, 1280) + loader::q4_layout1_bytes(640, 2560));
  const model::Q4Placement one = model::Q4Placement::one(d);
  size_t sum = 0;
  for (uint32_t l = 0; l < d.layers; ++l) sum += loader::q4_layer_bytes(intel, l).total();
  CHECK_EQ(loader::q4_device_weight_bytes(intel, one, 0, true, true),
           sum + loader::q4_embed_bytes(d) + loader::q4_final_mixer_bytes(d) + loader::q4_lm_head_bytes(d, true) +
               loader::q4_mtp_bytes(intel));
  const model::Q4Placement two = model::Q4Placement::two(d, 24);
  CHECK_EQ(loader::q4_device_weight_bytes(intel, two, 0, true, false) + loader::q4_device_weight_bytes(intel, two, 1, true, false),
           loader::q4_device_weight_bytes(intel, one, 0, true, false));
  CHECK(!loader::q4_device_has_qsa(d, model::Q4Placement::two(d, 2), 0, false) &&
        loader::q4_device_has_qsa(d, model::Q4Placement::two(d, 2), 1, false));
  const std::vector<float> rope = loader::q4_rope_table(d, 4);
  CHECK(rope.size() == size_t(4) * 2 * 32 && rope[0] == 1.0f && rope[32] == 0.0f);
  std::printf("qwen4exp_repack_test OK (derived, the real descriptor): experts %zu B a layer, HC %zu B a layer, "
              "final mixer %zu B, int8 head %zu B; Intel's forms: a GDN layer %zu B (%.3f GB), a QSA layer %zu B "
              "(%.3f GB), the PLE layer %zu B, the MTP head %zu B; ours: a GDN layer %zu B (%.3f GB); read per token "
              "(ours, a GDN layer) %zu B\n",
              g.experts(), g.hc_attn + g.hc_mlp, loader::q4_final_mixer_bytes(d), loader::q4_lm_head_bytes(d, true),
              g.total(), g.total() / 1e9, q.total(), q.total() / 1e9, p.total(), loader::q4_mtp_bytes(intel),
              go.total(), go.total() / 1e9, loader::q4_layer_read_per_token(d, 0));
  return 0;
}
