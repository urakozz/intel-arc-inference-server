// Spec 20c Task 3: the Kolibri-1 loader's host half (loader/kolibri1_repack.h) on a synthetic
// checkpoint in Kolibri's naming, host only.
//
// The test WRITES its fixture: a two-layer checkpoint (layer 0 sliding, layer 1 too - the
// pattern's first two) at the REAL widths (hidden 2560, 48 / 4 heads x 128, expert intermediate
// 512, shared 512) but 8 experts and a 64-row vocabulary (a test copy of the descriptor: the
// widths are what the layouts depend on), every tensor named as AutoRound's auto_round:auto_gptq
// export names it, in both attention arms - into a temporary directory, read back through
// loader::SafetensorsSet exactly as load_kolibri1 does, every layer repacked and decoded back:
//
//   1. kol_attn_form: Int4 / Bf16 from the names; a checkpoint whose layer 1 o_proj is the other
//      arm is refused naming model.layers.1.self_attn.o_proj.weight;
//   2. kol_expected_names == the file's names both ways; check_names refuses a missing and an
//      extra tensor, each by name;
//   3. EVERY expert block (gate||up interleave16 and down), word by word and scale by scale
//      against ITS expert's tensors, block e at e x kol_layer_bytes(d).gate_up / experts
//      (plan 15c's Review Focus 2: test expert slices individually);
//   4. the shared expert's bf16 tiles equal repack_bf16_tiled of the gate||up interleave;
//   5. the router: rows 0..7 the checkpoint's, rows 8..15 (of router_n 16) zero;
//   6. expert_bias widened exactly, zero past the experts; the norms widened verbatim (plain w);
//   7. the attention rows: int4 q||k||v / o_proj layout 0, bf16 tiles in the other arm;
//   8. nothing unconsumed after a full walk;
//   9. the derived per-layer and per-device bytes of the REAL descriptor (plan 20c Task 3).
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "loader/kolibri1_layout.h"
#include "loader/kolibri1_repack.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/kolibri1.h"

namespace {

namespace fs = std::filesystem;
using model::KolAttnForm;
using model::Kolibri1Desc;

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
std::vector<uint8_t> as_bytes(const std::vector<T>& v) {
  std::vector<uint8_t> b(v.size() * sizeof(T));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}
template <class T>
const T* view(const Tensors& ts, const std::string& n) {
  return reinterpret_cast<const T*>(ts.at(n).bytes.data());
}

std::mt19937 rng(20261006);

// An AutoRound int4 g64 linear [out N][in K]: random words, distinct finite scales, qzeros.
void add_int4(Tensors& ts, const std::string& prefix, uint32_t K, uint32_t N) {
  std::vector<uint32_t> qw(size_t(K / 8) * N);
  for (uint32_t& w : qw) w = rng();
  std::vector<uint16_t> sc(size_t(K / 64) * N);
  std::uniform_real_distribution<float> sd(0.01f, 0.09f);
  for (uint16_t& s : sc) s = common::f32_to_f16(sd(rng));
  std::vector<uint32_t> qz(size_t(K / 64) * (N / 8), 0x77777777u);
  ts[prefix + ".qweight"] = {"I32", {K / 8, N}, as_bytes(qw)};
  ts[prefix + ".scales"] = {"F16", {K / 64, N}, as_bytes(sc)};
  ts[prefix + ".qzeros"] = {"I32", {K / 64, N / 8}, as_bytes(qz)};
}
void add_bf16(Tensors& ts, const std::string& name, std::vector<uint64_t> shape) {
  size_t n = 1;
  for (uint64_t s : shape) n *= size_t(s);
  std::vector<uint16_t> v(n);
  std::uniform_real_distribution<float> d(-2.f, 2.f);
  for (uint16_t& x : v) x = common::f32_to_bf16(d(rng));
  ts[name] = {"BF16", std::move(shape), as_bytes(v)};
}

Kolibri1Desc test_desc(KolAttnForm a) {
  Kolibri1Desc d = model::kolibri1();
  d.name = "kolibri-1-test";
  d.layers = 2;
  d.experts = 8;
  d.vocab = d.vocab_used = 64;
  d.attn = a;
  return d;
}

Tensors fixture(const Kolibri1Desc& d) {
  Tensors ts;
  const uint32_t H = d.hidden, I = d.moe_inter, SI = d.shared_inter, hd = d.head_dim;
  add_bf16(ts, "model.embed_tokens.weight", {d.vocab, H});
  add_bf16(ts, "lm_head.weight", {d.vocab, H});
  add_bf16(ts, "model.norm.weight", {H});
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string p = Kolibri1Desc::layer_prefix(l);
    for (const char* n : {"input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm"})
      add_bf16(ts, p + n + ".weight", {H});
    add_bf16(ts, p + "self_attn.q_norm.weight", {hd});
    add_bf16(ts, p + "self_attn.k_norm.weight", {hd});
    const std::pair<const char*, std::pair<uint32_t, uint32_t>> attn[] = {
        {"q_proj", {H, d.q_n()}}, {"k_proj", {H, d.kv_n()}}, {"v_proj", {H, d.kv_n()}}, {"o_proj", {d.q_n(), H}}};
    for (const auto& [n, kn] : attn) {
      if (d.attn == KolAttnForm::Int4)
        add_int4(ts, p + "self_attn." + n, kn.first, kn.second);
      else
        add_bf16(ts, p + "self_attn." + n + ".weight", {kn.second, kn.first});
    }
    add_bf16(ts, p + "mlp.gate.weight", {d.experts, H});
    add_bf16(ts, p + "moe.router.expert_bias", {d.experts});
    add_bf16(ts, p + "mlp.shared_experts.gate_proj.weight", {SI, H});
    add_bf16(ts, p + "mlp.shared_experts.up_proj.weight", {SI, H});
    add_bf16(ts, p + "mlp.shared_experts.down_proj.weight", {H, SI});
    for (uint32_t e = 0; e < d.experts; ++e) {
      const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
      add_int4(ts, ep + "gate_proj", H, I);
      add_int4(ts, ep + "up_proj", H, I);
      add_int4(ts, ep + "down_proj", I, H);
    }
  }
  return ts;
}

fs::path tmpdir(const std::string& tag) {
  return fs::temp_directory_path() / ("kolibri1_repack_test_" + tag + "_" + std::to_string(::getpid()));
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
// gemv_bf16's tile element (k, n) of a {K, N} tiled matrix.
uint16_t tiled(const std::vector<uint16_t>& t, uint32_t K, uint32_t k, uint32_t n) {
  return t[((size_t(n / 16) * (K / 8) + k / 8) * 8 + k % 8) * 16 + n % 16];
}

void check_arm(KolAttnForm arm) {
  const Kolibri1Desc d = test_desc(arm);
  const Tensors ts = fixture(d);
  const fs::path dir = tmpdir(model::kol_attn_form_name(arm));
  write_checkpoint(dir, ts);
  const uint32_t H = d.hidden, I = d.moe_inter, SI = d.shared_inter;
  {
    loader::SafetensorsSet set(dir.string() + "/");
    (void)loader::assert_quant_invariants(set);
    CHECK(loader::kol_attn_form(set, d) == arm);
    // 2. the names both ways
    const std::vector<std::string> exp = loader::kol_expected_names(d, arm);
    CHECK_EQ(exp.size(), ts.size());
    for (const std::string& n : exp) CHECK(ts.count(n) == 1);
    loader::KolCheckpoint ck(d, set);
    ck.check_names();

    for (uint32_t l = 0; l < d.layers; ++l) {
      const std::string p = Kolibri1Desc::layer_prefix(l);
      loader::KolHostLayer h;
      ck.repack_layer(l, h);
      // 7. attention
      if (arm == KolAttnForm::Int4) {
        CHECK_EQ(h.qkv_words.size(), size_t(H / 8) * d.qkv_n());
        const uint32_t N = d.qkv_n();
        const std::pair<const char*, uint32_t> parts[] = {{"q_proj", 0}, {"k_proj", d.q_n()}, {"v_proj", d.q_n() + d.kv_n()}};
        for (const auto& [n, off] : parts) {
          const uint32_t pn = std::string(n) == "q_proj" ? d.q_n() : d.kv_n();
          const uint32_t* qw = view<uint32_t>(ts, p + "self_attn." + n + ".qweight");
          const uint16_t* sc = view<uint16_t>(ts, p + "self_attn." + n + ".scales");
          for (uint32_t r = 0; r < H / 8; r += 37)
            for (uint32_t c = 0; c < pn; ++c) CHECK(h.qkv_words[size_t(r) * N + off + c] == qw[size_t(r) * pn + c]);
          for (uint32_t g = 0; g < H / 64; ++g)
            for (uint32_t c = 0; c < pn; c += 3) CHECK(h.qkv_scales[size_t(g) * N + off + c] == sc[size_t(g) * pn + c]);
        }
        CHECK(std::memcmp(h.oproj_words.data(), view<uint32_t>(ts, p + "self_attn.o_proj.qweight"),
                          h.oproj_words.size() * 4) == 0);
        CHECK(h.qkv_bf16.empty() && h.oproj_bf16.empty());
      } else {
        CHECK(h.qkv_words.empty() && h.qkv_bf16.size() == size_t(H) * d.qkv_n());
        const uint16_t* k = view<uint16_t>(ts, p + "self_attn.k_proj.weight");
        const uint16_t* o = view<uint16_t>(ts, p + "self_attn.o_proj.weight");
        for (uint32_t n = 0; n < d.kv_n(); n += 7)
          for (uint32_t kk = 0; kk < H; kk += 13)
            CHECK(tiled(h.qkv_bf16, H, kk, d.q_n() + n) == k[size_t(n) * H + kk]);
        for (uint32_t n = 0; n < H; n += 11)
          for (uint32_t kk = 0; kk < d.q_n(); kk += 17) CHECK(tiled(h.oproj_bf16, d.q_n(), kk, n) == o[size_t(n) * d.q_n() + kk]);
      }
      // 6. norms (plain w) and the bias
      const char* norms[] = {"input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm"};
      const uint32_t offs[] = {d.norm_off_input(), d.norm_off_post_attn(), d.norm_off_post_attention(), d.norm_off_post_ffn()};
      CHECK_EQ(h.norms.size(), size_t(d.norm_floats()));
      for (int i = 0; i < 4; ++i) {
        const uint16_t* w = view<uint16_t>(ts, p + norms[i] + ".weight");
        for (uint32_t k = 0; k < H; ++k) CHECK(h.norms[offs[i] + k] == common::bf16_to_f32(w[k]));
      }
      const uint16_t* qn = view<uint16_t>(ts, p + "self_attn.q_norm.weight");
      const uint16_t* kn = view<uint16_t>(ts, p + "self_attn.k_norm.weight");
      for (uint32_t i = 0; i < d.head_dim; ++i) {
        CHECK(h.norms[d.norm_off_q() + i] == common::bf16_to_f32(qn[i]));
        CHECK(h.norms[d.norm_off_k() + i] == common::bf16_to_f32(kn[i]));
      }
      const uint16_t* eb = view<uint16_t>(ts, p + "moe.router.expert_bias");
      CHECK_EQ(h.bias.size(), size_t(d.router_n()));
      CHECK_EQ(d.router_n(), 16u);
      for (uint32_t e = 0; e < d.router_n(); ++e) CHECK(h.bias[e] == (e < d.experts ? common::bf16_to_f32(eb[e]) : 0.0f));
      // 5. the router rows
      const uint16_t* gw = view<uint16_t>(ts, p + "mlp.gate.weight");
      for (uint32_t e = 0; e < d.router_n(); ++e)
        for (uint32_t k = 0; k < H; k += 5)
          CHECK(tiled(h.router, H, k, e) == (e < d.experts ? gw[size_t(e) * H + k] : uint16_t(0)));
      // 4. the shared expert
      {
        const uint16_t* g = view<uint16_t>(ts, p + "mlp.shared_experts.gate_proj.weight");
        const uint16_t* u = view<uint16_t>(ts, p + "mlp.shared_experts.up_proj.weight");
        const uint16_t* dn = view<uint16_t>(ts, p + "mlp.shared_experts.down_proj.weight");
        std::vector<uint16_t> rows(size_t(2) * SI * H);
        for (uint32_t c = 0; c < 2 * SI; ++c) {
          const uint16_t* src = ((c % 32) < 16 ? g : u) + size_t((c / 32) * 16 + c % 16) * H;
          std::memcpy(rows.data() + size_t(c) * H, src, size_t(H) * 2);
        }
        std::vector<uint16_t> want(rows.size());
        common::repack_bf16_tiled(rows.data(), H, 2 * SI, want.data());
        CHECK(want == h.shared_gate_up);
        std::vector<uint16_t> wd(size_t(SI) * H);
        common::repack_bf16_tiled(dn, SI, H, wd.data());
        CHECK(wd == h.shared_down);
      }
      // 3. every expert block, individually
      const loader::KolLayerBytes lb = loader::kol_layer_bytes(d);
      CHECK_EQ(h.gate_up.size() * 4, lb.gate_up);
      CHECK_EQ(h.down.size() * 4, lb.down);
      const size_t gub = lb.gate_up / d.experts / 4, dnb = lb.down / d.experts / 4;
      CHECK_EQ(gub * 4, loader::kol_gate_up_block_bytes(d));
      for (uint32_t e = 0; e < d.experts; ++e) {
        const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
        const uint32_t* blk = h.gate_up.data() + size_t(e) * gub;
        const uint32_t* gq = view<uint32_t>(ts, ep + "gate_proj.qweight");
        const uint32_t* uq = view<uint32_t>(ts, ep + "up_proj.qweight");
        const uint16_t* gs = view<uint16_t>(ts, ep + "gate_proj.scales");
        const uint16_t* us = view<uint16_t>(ts, ep + "up_proj.scales");
        for (uint32_t c = 0; c < 2 * I; ++c) {
          const bool gate = (c % 32) < 16;
          const uint32_t sc = (c / 32) * 16 + c % 16;   // the source column (gate tile t, then up tile t)
          for (uint32_t r = 0; r < H / 8; ++r) CHECK(l1_word(blk, H, r, c) == (gate ? gq : uq)[size_t(r) * I + sc]);
          for (uint32_t g = 0; g < H / 64; ++g) CHECK(l1_scale(blk, H, g, c) == (gate ? gs : us)[size_t(g) * I + sc]);
        }
        const uint32_t* dblk = h.down.data() + size_t(e) * dnb;
        const uint32_t* dq = view<uint32_t>(ts, ep + "down_proj.qweight");
        const uint16_t* ds = view<uint16_t>(ts, ep + "down_proj.scales");
        for (uint32_t c = 0; c < H; ++c) {
          for (uint32_t r = 0; r < I / 8; ++r) CHECK(l1_word(dblk, I, r, c) == dq[size_t(r) * H + c]);
          for (uint32_t g = 0; g < I / 64; ++g) CHECK(l1_scale(dblk, I, g, c) == ds[size_t(g) * H + c]);
        }
        // the dequantised value of one element, (q - 8) x scale, for the record
        const uint32_t q = (gq[0] & 0xFu);
        const float w = float(int(q) - 8) * common::f16_to_f32(gs[0]);
        const float got = float(int(l1_word(blk, H, 0, 0) & 0xFu) - 8) * common::f16_to_f32(l1_scale(blk, H, 0, 0));
        CHECK(w == got);
      }
    }
    (void)ck.embed();
    (void)ck.final_norm();
    (void)ck.lm_head();
    std::string names;
    const size_t left = ck.unconsumed(&names);
    if (left) std::fprintf(stderr, "unconsumed: %s\n", names.c_str());
    CHECK_EQ(left, size_t(0));
  }

  // 2. refusals by name: a missing tensor and an extra one
  {
    Tensors miss = ts;
    const std::string gone = Kolibri1Desc::layer_prefix(1) + "mlp.experts.3.down_proj.scales";
    miss.erase(gone);
    const fs::path d2 = tmpdir(std::string("miss_") + model::kol_attn_form_name(arm));
    write_checkpoint(d2, miss);
    loader::SafetensorsSet set(d2.string() + "/");
    loader::KolCheckpoint ck(d, set);
    throws_naming(gone, [&] { ck.check_names(); });
    fs::remove_all(d2);
  }
  {
    Tensors extra = ts;
    const std::string more = Kolibri1Desc::layer_prefix(0) + "mlp.shared_expert_gate.weight";
    add_bf16(extra, more, {1, H});
    const fs::path d2 = tmpdir(std::string("extra_") + model::kol_attn_form_name(arm));
    write_checkpoint(d2, extra);
    loader::SafetensorsSet set(d2.string() + "/");
    loader::KolCheckpoint ck(d, set);
    throws_naming(more, [&] { ck.check_names(); });
    fs::remove_all(d2);
  }
  fs::remove_all(dir);
}

}  // namespace

int main() {
  // 9. the derived bytes of the real model (plan 20c Task 3 Step 3)
  {
    const Kolibri1Desc& d = model::kolibri1();
    const loader::KolLayerBytes b = loader::kol_layer_bytes(d);
    CHECK_EQ(b.total(), size_t(830794752));
    CHECK_EQ(b.experts(), size_t(802160640));
    CHECK_EQ(b.shared_gate_up + b.shared_down, size_t(7864320));
    CHECK_EQ(b.attention(), size_t(18104320));
    CHECK_EQ(b.router, size_t(2621440));
    CHECK_EQ(b.bias + b.norms, size_t(44032));
    Kolibri1Desc bf = d;
    bf.attn = KolAttnForm::Bf16;
    CHECK_EQ(loader::kol_layer_bytes(bf).total(), size_t(880847872));
    CHECK_EQ(loader::kol_embed_bytes(d), size_t(655360000));
    CHECK_EQ(loader::kol_lm_head_bytes(d, false), size_t(655360000));
    CHECK_EQ(loader::kol_lm_head_bytes(d, true), size_t(328192000));
    const model::KolPlacement p = model::KolPlacement::two(d, 25);
    CHECK_EQ(loader::kol_device_weight_bytes(d, p, 0, true), size_t(21425228800ull));
    CHECK_EQ(loader::kol_device_weight_bytes(d, p, 1, true), size_t(21098071040ull));
    CHECK(loader::kol_device_has_sliding(d, p, 0) && loader::kol_device_has_sliding(d, p, 1));
  }
  check_arm(KolAttnForm::Int4);
  check_arm(KolAttnForm::Bf16);
  // 1. a checkpoint mixing the arms: layer 1's o_proj bf16 in an int4 checkpoint
  {
    const Kolibri1Desc d = test_desc(KolAttnForm::Int4);
    Tensors ts = fixture(d);
    const std::string p = Kolibri1Desc::layer_prefix(1) + "self_attn.o_proj";
    for (const char* s : {".qweight", ".scales", ".qzeros"}) ts.erase(p + s);
    add_bf16(ts, p + ".weight", {d.hidden, d.q_n()});
    const fs::path dir = tmpdir("mixed");
    write_checkpoint(dir, ts);
    loader::SafetensorsSet set(dir.string() + "/");
    throws_naming("model.layers.1.self_attn.o_proj.weight", [&] { (void)loader::kol_attn_form(set, d); });
    fs::remove_all(dir);
  }
  std::puts("kolibri1_repack_test OK");
  return 0;
}
