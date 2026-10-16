#pragma once
// Spec 19b: tests/loader/dflash_ref_fixture.safetensors (tools/oracle/dflash_fixture.py) read into the host twin's
// structures (tests/kernels/dflash_ref.h) - shared by dflash_ref_test (the twin vs dflash_ref.py) and
// tools/mac/clrun/dflash_run.cc (the kernels at the fixture's tiny shapes on the Mac's GPU). Spec 23f: also
// tests/loader/muse_dflash_fixture.safetensors (dflash_fixture.py --muse) through muse_weights().
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/json.h"
#include "common/repack.h"
#include "kernels/dflash_ref.h"
#include "loader/lm_head_int8.h"
#include "loader/rtn.h"
#include "loader/safetensors.h"
#include "model/dflash.h"

namespace dffix {

struct Fixture {
  loader::MappedFile file;
  std::map<std::string, loader::TensorInfo> t;
  common::json::Value meta;
  size_t data0 = 0;
  explicit Fixture(const std::string& path) : file(path) {
    uint64_t hlen = 0;
    std::memcpy(&hlen, file.data(), 8);
    data0 = 8 + hlen;
    const common::json::Value h =
        common::json::parse(std::string_view(reinterpret_cast<const char*>(file.data()) + 8, hlen));
    meta = h.at("__metadata__");
    for (auto& [n, ti] : loader::SafetensorsSet::parse_header(file.data(), file.size())) t[n] = ti;
  }
  bool has(const std::string& n) const { return t.count(n) != 0; }
  const uint8_t* raw(const std::string& n) const {
    auto it = t.find(n);
    if (it == t.end()) throw std::runtime_error("fixture: no tensor " + n);
    return file.data() + data0 + it->second.begin;
  }
  size_t count(const std::string& n) const {
    const loader::TensorInfo& ti = t.at(n);
    size_t c = 1;
    for (uint64_t s : ti.shape) c *= s;
    return c;
  }
  std::vector<uint16_t> bf16(const std::string& n) const {
    std::vector<uint16_t> v(count(n));
    std::memcpy(v.data(), raw(n), v.size() * 2);
    return v;
  }
  std::vector<float> f32(const std::string& n) const {
    const loader::TensorInfo& ti = t.at(n);
    std::vector<float> v(count(n));
    if (ti.dtype == "F32") {
      std::memcpy(v.data(), raw(n), v.size() * 4);
    } else {
      const std::vector<uint16_t> b = bf16(n);
      for (size_t i = 0; i < v.size(); ++i) v[i] = dfref::bf(b[i]);
    }
    return v;
  }
  std::vector<int32_t> i32(const std::string& n) const {
    std::vector<int32_t> v(count(n));
    std::memcpy(v.data(), raw(n), v.size() * 4);
    return v;
  }
  std::string meta_str(const std::string& k) const { return meta.at(k).str(); }
};

// How a fixture linear becomes a dfref::Lin.
// Int4: spec 23f's int4 g64 RTN (loader::rtn_int4_g64) in gemv.cl's arithmetic; Int4Ref: the same RTN dequantised to
// fp32 rows (Reference numerics over the quantised weights - dflash_accept.int4g64_simulate's form).
enum class Form { Reference, Bf16, Int8, Int4, Int4Ref };

inline dfref::Lin make_lin(const std::vector<uint16_t>& rows, uint32_t K, uint32_t N, Form f, uint32_t ksplit = 1) {
  dfref::Lin l;
  l.K = K;
  l.N = N;
  l.ksplit = ksplit;
  if (f == Form::Reference) {
    l.f32.resize(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) l.f32[i] = dfref::bf(rows[i]);
  } else if (f == Form::Bf16) {
    l.bf16.resize(rows.size());
    common::repack_bf16_tiled(rows.data(), K, N, l.bf16.data());
  } else if (f == Form::Int4 || f == Form::Int4Ref) {
    l.q4.resize(size_t(K / 8) * N);
    l.s4.resize(size_t(K / 64) * N);
    loader::rtn_int4_g64(rows.data(), K, N, l.q4.data(), l.s4.data());
    if (f == Form::Int4Ref) {
      l.f32.resize(size_t(N) * K);
      for (uint32_t n = 0; n < N; ++n)
        for (uint32_t k = 0; k < K; ++k)
          l.f32[size_t(n) * K + k] = float(int((l.q4[size_t(k / 8) * N + n] >> (4 * (k % 8))) & 0xFu) - 8) *
                                     common::f16_to_f32(l.s4[size_t(k / 64) * N + n]);
      l.q4.clear();
      l.s4.clear();
    }
  } else {
    l.i8.resize(rows.size());
    l.scales.resize(N);
    loader::quantise_int8_tiled(rows.data(), K, N, l.i8.data(), l.scales.data(), 1);
  }
  return l;
}

inline std::vector<uint16_t> cat(const Fixture& fx, const std::vector<std::string>& names) {
  std::vector<uint16_t> out;
  for (const std::string& n : names) {
    const std::vector<uint16_t> v = fx.bf16(n);
    out.insert(out.end(), v.begin(), v.end());
  }
  return out;
}

// The fixture's drafter (version 2 or 1) as dfref::Weights in `f` (the head in `head_f`); the ring geometry: ring =
// the window, `pb` offsets per context block.
inline dfref::Weights weights(const Fixture& fx, uint32_t version, Form f, Form head_f, uint32_t pb) {
  const model::DFlashDesc d =
      model::dflash_desc_from_config(common::json::parse(fx.meta_str(version == 2 ? "config_v2" : "config_v1")));
  dfref::Weights w;
  dfref::Dims& D = w.d;
  D.version = d.version;
  D.layers = d.layers;
  D.H = d.hidden;
  D.NH = d.q_heads;
  D.NKV = d.kv_heads;
  D.HD = d.head_dim;
  D.I = d.inter;
  D.V = d.vocab;
  D.used = uint32_t(std::stoul(fx.meta_str("vocab_used")));
  D.M = d.block;
  D.ring = d.max_window();
  D.win = d.max_window();
  D.pb = pb;
  D.taps = uint32_t(d.taps.size());
  D.th = uint32_t(fx.t.at("fc.weight").shape[1]) / D.taps;
  D.group = d.conv_group;
  D.ctaps = d.conv_taps;
  D.rank = d.sel_rank;
  D.topk = d.v2() ? d.sel_top_k : 16;
  D.qs = 16;
  D.eps = d.eps;
  D.in_scale = d.in_scale;
  D.out_mult = d.out_mult;
  D.softcap = d.softcap;
  D.scale = float(1.0 / std::sqrt(double(d.head_dim)));
  D.theta = d.theta;
  D.causal = d.causal[0];
  const uint32_t H = D.H;
  w.fc = make_lin(fx.bf16("fc.weight"), D.fck(), H, f);
  w.hidden_norm = fx.f32("hidden_norm.weight");
  w.final_norm = fx.f32("norm.weight");
  for (uint32_t l = 0; l < D.layers; ++l) {
    const std::string L = "layers." + std::to_string(l) + ".";
    dfref::Layer y;
    y.qkv = make_lin(cat(fx, {L + "self_attn.q_proj.weight", L + "self_attn.k_proj.weight", L + "self_attn.v_proj.weight"}),
                     H, D.qkvn(), f);
    y.o = make_lin(fx.bf16(L + "self_attn.o_proj.weight"), D.qn(), H, f);
    y.gate_up = make_lin(cat(fx, {L + "mlp.gate_proj.weight", L + "mlp.up_proj.weight"}), H, 2 * D.I, f);
    y.down = make_lin(fx.bf16(L + "mlp.down_proj.weight"), D.I, H, f);
    y.in_norm = fx.f32(L + "input_layernorm.weight");
    y.post_norm = fx.f32(L + "post_attention_layernorm.weight");
    y.q_norm = fx.f32(L + "self_attn.q_norm.weight");
    y.k_norm = fx.f32(L + "self_attn.k_norm.weight");
    if (D.version == 2) {
      y.akp = make_lin(fx.bf16(L + "attention_conv.kernel_projection.weight"), H, D.kpn(), f);
      y.mkp = make_lin(fx.bf16(L + "mlp_conv.kernel_projection.weight"), H, D.kpn(), f);
      for (const char* cv : {"attention_conv", "mlp_conv"}) {
        const std::vector<float> b = fx.f32(L + cv + ".base_kernel");
        y.conv_base.insert(y.conv_base.end(), b.begin(), b.end());
      }
    }
    w.layers.push_back(std::move(y));
  }
  if (D.version == 2) {
    w.sel = make_lin(fx.bf16("candidate_selector.hidden_projection.weight"), H, D.rank, f);
    w.pred = fx.bf16("candidate_selector.predecessor_codebook");
    w.succ = fx.bf16("candidate_selector.successor_codebook");
  }
  w.head = make_lin(fx.bf16("target.head"), H, D.V, head_f);
  w.embed = fx.bf16("target.embed");
  w.mask_row.resize(H);
  for (uint32_t i = 0; i < H; ++i) w.mask_row[i] = dfref::bf(w.embed[size_t(d.mask_id) * H + i]);
  return w;
}

// Spec 23f: the Muse fixture's assistant (config_muse) as dfref::Weights in `f` (the head in `head_f`). The ring
// geometry is the engine's rule at the fixture's window: df_attn's WIN = window + 1 (|q - k| <= window), the ring the
// smallest multiple of `pb` holding WIN slots (2112 at Muse's 2048 and PB 64).
inline dfref::Weights muse_weights(const Fixture& fx, Form f, Form head_f, uint32_t pb) {
  const model::DFlashDesc d = model::dflash_desc_from_config(common::json::parse(fx.meta_str("config_muse")));
  if (!d.muse()) throw std::runtime_error("fixture: config_muse is not a Muse-form drafter");
  dfref::Weights w;
  dfref::Dims& D = w.d;
  D.version = 1;
  D.muse = true;
  D.layers = d.layers;
  D.H = d.hidden;
  D.NH = d.q_heads;
  D.NKV = d.kv_heads;
  D.HD = d.head_dim;
  D.I = d.inter;
  D.V = uint32_t(fx.t.at("target.head").shape[0]);
  D.used = uint32_t(std::stoul(fx.meta_str("vocab_used")));
  D.M = d.block;
  D.win = d.max_window() + 1;
  D.ring = (D.win + pb - 1) / pb * pb;
  D.pb = pb;
  D.taps = uint32_t(d.taps.size());
  D.th = uint32_t(fx.t.at("fc.weight").shape[1]) / D.taps;
  D.topk = 16;
  D.eps = d.eps;
  D.in_scale = 1.0f;
  D.out_mult = 1.0f;
  D.softcap = 0.0f;
  D.scale = float(1.0 / std::sqrt(double(d.head_dim)));
  D.theta = d.theta;
  D.causal = false;
  const uint32_t H = D.H;
  w.fc = make_lin(fx.bf16("fc.weight"), D.fck(), H, f);
  w.hidden_norm = fx.f32("hidden_norm.weight");
  w.final_norm = fx.f32("norm.weight");
  for (uint32_t l = 0; l < D.layers; ++l) {
    const std::string L = "layers." + std::to_string(l) + ".";
    dfref::Layer y;
    y.qkv = make_lin(cat(fx, {L + "self_attn.q_proj.weight", L + "self_attn.k_proj.weight", L + "self_attn.v_proj.weight"}),
                     H, D.qkvn(), f);
    y.o = make_lin(fx.bf16(L + "self_attn.o_proj.weight"), D.qn(), H, f);
    y.gate_up = make_lin(cat(fx, {L + "mlp.gate_proj.weight", L + "mlp.up_proj.weight"}), H, 2 * D.I, f);
    y.down = make_lin(fx.bf16(L + "mlp.down_proj.weight"), D.I, H, f);
    y.in_norm = fx.f32(L + "input_layernorm.weight");
    y.post_norm = fx.f32(L + "post_attention_layernorm.weight");
    y.q_norm = fx.f32(L + "self_attn.q_norm.weight");
    y.k_norm = fx.f32(L + "self_attn.k_norm.weight");
    w.layers.push_back(std::move(y));
  }
  w.head = make_lin(fx.bf16("target.head"), H, D.V, head_f);
  w.embed = fx.bf16("target.embed");
  w.mask_row.resize(H);
  for (uint32_t i = 0; i < H; ++i) w.mask_row[i] = dfref::bf(w.embed[size_t(d.mask_id) * H + i]);
  return w;
}

struct Case {
  uint32_t p, anchor, K;
  float T;
  uint64_t seed;
};
inline std::vector<Case> cases(const Fixture& fx) {
  std::vector<Case> out;
  const common::json::Value all = common::json::parse(fx.meta_str("cases"));
  for (const common::json::Value& c : all.arr())
    out.push_back({uint32_t(c.at("p").num()), uint32_t(c.at("anchor").num()), uint32_t(c.at("K").num()),
                   float(c.at("T").num()), uint64_t(c.at("seed").num())});
  return out;
}

// The ring holding positions [0, p) (the last `ring` of them), projected one position at a time.
inline dfref::Ring ring_at(const Fixture& fx, const dfref::Weights& w, dfref::Numerics nm,
                           const std::vector<float>& rope, uint32_t p) {
  dfref::Ring r = dfref::make_ring(w.d);
  const std::vector<uint16_t> taps = fx.bf16("taps");
  for (uint32_t pos = 0; pos < p; ++pos)
    dfref::project(w, nm, &taps[size_t(pos) * w.d.fck()], 1, pos, rope, r);
  return r;
}

}  // namespace dffix
