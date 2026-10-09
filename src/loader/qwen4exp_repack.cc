#include "loader/qwen4exp_repack.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "common/bf16.h"
#include "common/repack.h"
#include "loader/rtn.h"

namespace loader {
namespace {

using model::Q4Form;
using model::Q4LinearId;
using model::Qwen4ExpDesc;

bool ends_with(const std::string& s, const char* suf) {
  const size_t n = std::strlen(suf);
  return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}
bool starts_with(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }

const char* const kGdnDense[] = {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"};
const char* const kQsaDense[] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"};
const char* const kMlp[] = {"gate_proj", "up_proj", "down_proj"};
const char* const kPleShards = "ple.ple_embedding.ngram_embedding.";

void lin(std::vector<std::string>& v, const std::string& base, bool int4) {
  if (int4) {
    v.push_back(base + ".qweight");
    v.push_back(base + ".qzeros");
    v.push_back(base + ".scales");
  } else {
    v.push_back(base + ".weight");
  }
}
void hc_names(std::vector<std::string>& v, const std::string& base, bool inject) {
  v.push_back(base + ".hc_norm.weight");
  v.push_back(base + ".input_mix_weight_down.weight");
  v.push_back(base + ".input_mix_weight_up.weight");
  if (inject) v.push_back(base + ".block_inject_weight.weight");
}
void moe_names(std::vector<std::string>& v, const Qwen4ExpDesc& d, const std::string& b, bool experts_int4,
               bool shared_int4) {
  v.push_back(b + "mlp.gate.weight");
  v.push_back(b + "mlp.shared_expert_gate.weight");
  for (const char* m : kMlp) lin(v, b + "mlp.shared_expert." + m, shared_int4);
  for (uint32_t e = 0; e < d.experts; ++e)
    for (const char* m : kMlp) lin(v, b + "mlp.experts." + std::to_string(e) + "." + m, experts_int4);
}
void qsa_names(std::vector<std::string>& v, const std::string& b, bool dense_int4) {
  for (const char* n : {"self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.indexer.index_qk_proj.weight",
                        "self_attn.indexer.q_layernorm.weight", "self_attn.indexer.k_layernorm.weight"})
    v.push_back(b + n);
  for (const char* n : kQsaDense) lin(v, b + n, dense_int4);
}

size_t int4_src_bytes(const LinearSrc& s) { return size_t(s.K / 8) * s.N * 4 + size_t(s.K / s.group) * s.N * 2; }

// gate||up interleaved in 16-row blocks of an [N][K] row-major pair (cols_interleave16's column order, as
// rows), then gemv_bf16's tiles (kolibri1_repack.cc's helper).
void tiled_interleave16(const uint16_t* gate, const uint16_t* up, uint32_t I, uint32_t K, uint16_t* out) {
  std::vector<uint16_t> rows(size_t(2) * I * K);
  for (uint32_t r = 0; r < 2 * I; ++r) {
    const uint32_t src = (r / 32) * 16 + r % 16;
    const uint16_t* w = (r % 32 < 16 ? gate : up) + size_t(src) * K;
    std::memcpy(rows.data() + size_t(r) * K, w, size_t(K) * 2);
  }
  common::repack_bf16_tiled(rows.data(), K, 2 * I, out);
}

// The q4_forms check of one name group: `base` + the chosen form's suffix must not coexist with the other.
void agree(const SafetensorsSet& st, const std::string& base, Q4Form f, const char* group, const char* first) {
  const auto& ts = st.tensors();
  const std::string other = base + (f == Q4Form::Int4 ? ".weight" : ".qweight");
  const std::string mine = base + (f == Q4Form::Int4 ? ".qweight" : ".weight");
  if (ts.count(other) && !ts.count(mine))
    throw std::runtime_error("load_qwen4exp: '" + other + "' mixes the " + group + " forms - " + first + " is " +
                             model::q4_form_name(f) + ", and every member of the group in every layer must agree "
                             "(one checkpoint, one form per group: spec 21b)");
}

}  // namespace

model::Q4Forms q4_forms(const SafetensorsSet& st, const Qwen4ExpDesc& d) {
  const auto& ts = st.tensors();
  const auto form_of = [&](const std::string& base, const char* what) {
    if (ts.count(base + ".qweight")) return Q4Form::Int4;
    if (ts.count(base + ".weight")) return Q4Form::Bf16;
    throw std::runtime_error("load_qwen4exp: the checkpoint has neither " + base + ".qweight nor " + base +
                             ".weight - no " + what + " to read");
  };
  model::Q4Forms f;
  const std::string d0 = Qwen4ExpDesc::layer_prefix(0) + "linear_attn.in_proj_qkv";
  const std::string s0 = Qwen4ExpDesc::layer_prefix(0) + "mlp.shared_expert.gate_proj";
  f.dense = form_of(d0, "dense projection");
  f.shared = form_of(s0, "shared expert");
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string lp = Qwen4ExpDesc::layer_prefix(l);
    if (d.is_qsa(l))
      for (const char* n : kQsaDense) agree(st, lp + n, f.dense, "dense projection", d0.c_str());
    else
      for (const char* n : kGdnDense) agree(st, lp + n, f.dense, "dense projection", d0.c_str());
    for (const char* m : kMlp) agree(st, lp + "mlp.shared_expert." + m, f.shared, "shared expert", s0.c_str());
  }
  // The main model's routed experts: int4 only, one group size.
  const std::string e0 = Qwen4ExpDesc::layer_prefix(0) + "mlp.experts.0.gate_proj";
  f.expert_group = 0;
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string lp = Qwen4ExpDesc::layer_prefix(l);
    if (ts.count(lp + "mlp.experts.gate_up_proj"))
      throw std::runtime_error("load_qwen4exp: '" + lp + "mlp.experts.gate_up_proj' is the bf16 original's fused "
                               "experts - the original is the reference's input, never the engine's (one format: "
                               "AutoRound int4 g64, or Intel's g128 expanded exactly; spec 21 §5)");
    for (uint32_t e = 0; e < d.experts; ++e)
      for (const char* m : kMlp) {
        const std::string b = lp + "mlp.experts." + std::to_string(e) + "." + m;
        if (ts.count(b + ".weight"))
          throw std::runtime_error("load_qwen4exp: '" + b + ".weight' is a bf16 routed expert - the main model's "
                                   "routed experts are int4 (g64, or Intel's g128 expanded at load); a bf16 one is "
                                   "the reference's input, never the engine's");
        const auto qw = ts.find(b + ".qweight");
        const auto sc = ts.find(b + ".scales");
        if (qw == ts.end() || sc == ts.end() || qw->second.shape.size() != 2 || sc->second.shape.empty()) continue;
        const uint64_t K = qw->second.shape[0] * 8, rows = sc->second.shape[0];
        const uint32_t g = rows != 0 && K % rows == 0 ? uint32_t(K / rows) : 0;
        if (f.expert_group == 0) {
          if (g != 64 && g != 128)
            throw std::runtime_error("load_qwen4exp: '" + b + ".scales' has " + std::to_string(rows) + " rows for K = " +
                                     std::to_string(K) + " - neither g64 nor g128");
          f.expert_group = g;
        } else if (g != f.expert_group) {
          throw std::runtime_error("load_qwen4exp: '" + b + ".scales' is g" + std::to_string(g) + ", but " + e0 +
                                   " is g" + std::to_string(f.expert_group) +
                                   " - every routed expert of the checkpoint must share one group size");
        }
      }
  }
  if (f.expert_group == 0) f.expert_group = 64;   // no expert shipped: check_names names the first missing one
  // The MTP head's experts (Bf16 when the checkpoint has no head).
  const std::string m0 = "mtp.layers.0.mlp.experts.0.gate_proj";
  f.mtp_experts = ts.count(m0 + ".qweight") ? Q4Form::Int4 : Q4Form::Bf16;
  if (ts.count(m0 + ".qweight") || ts.count(m0 + ".weight"))
    for (uint32_t e = 0; e < d.experts; ++e)
      for (const char* m : kMlp)
        agree(st, "mtp.layers.0.mlp.experts." + std::to_string(e) + "." + m, f.mtp_experts, "MTP expert", m0.c_str());
  return f;
}

std::vector<std::string> q4_expected_names(const Qwen4ExpDesc& d, const model::Q4Forms& f, bool mtp) {
  const std::string P = d.prefix;
  std::vector<std::string> v = {P + "embed_tokens.weight", "lm_head.weight"};
  hc_names(v, P + "hyper_connection_mixer", false);
  const bool dq = f.dense == Q4Form::Int4, sq = f.shared == Q4Form::Int4;
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string b = Qwen4ExpDesc::layer_prefix(l);
    hc_names(v, b + "attn_hyper_connection", true);
    hc_names(v, b + "mlp_hyper_connection", true);
    moe_names(v, d, b, true, sq);
    if (!d.is_qsa(l)) {
      for (const char* n : {"linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.conv1d.weight",
                            "linear_attn.norm.weight", "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"})
        v.push_back(b + n);
      for (const char* n : kGdnDense) lin(v, b + n, dq);
    } else {
      qsa_names(v, b, dq);
    }
    if (l == d.ple_layer) {
      const std::string p = b + "ple.";
      for (const char* n : {"conv1d.weight", "key_proj.weight", "value_proj.weight", "norm_conv.weight",
                            "norm_key.weight", "norm_query.weight", "ple_embedding.layer_multipliers",
                            "ple_embedding.ngram_heads_vocab_sizes", "ple_embedding.ngram_heads_offsets"})
        v.push_back(p + n);
    }
  }
  if (mtp) {
    const std::string b = "mtp.layers.0.";
    for (const char* n : {"mtp.fc_embedding.weight", "mtp.fc_hidden.weight", "mtp.pre_fc_norm_embedding.weight",
                          "mtp.pre_fc_norm_hidden.weight"})
      v.push_back(n);
    hc_names(v, "mtp.hyper_connection_mixer", false);
    hc_names(v, b + "attn_hyper_connection", true);
    hc_names(v, b + "mlp_hyper_connection", true);
    moe_names(v, d, b, f.mtp_experts == Q4Form::Int4, false);
    qsa_names(v, b, false);
  }
  return v;
}

std::vector<float> q4_rope_table(const Qwen4ExpDesc& d, uint32_t max_len) {
  const uint32_t half = d.rope_dims / 2;
  const float base = float(d.rope_theta);   // 1e7 is exact in fp32
  std::vector<float> inv(half);
  for (uint32_t i = 0; i < half; ++i) inv[i] = 1.0f / std::pow(base, float(2 * i) / float(d.rope_dims));
  std::vector<float> t(size_t(max_len) * 2 * half);
  for (uint32_t p = 0; p < max_len; ++p)
    for (uint32_t i = 0; i < half; ++i) {
      const float ang = float(p) * inv[i];
      t[(size_t(p) * 2 + 0) * half + i] = common::bf16_to_f32(common::f32_to_bf16(float(std::cos(double(ang)))));
      t[(size_t(p) * 2 + 1) * half + i] = common::bf16_to_f32(common::f32_to_bf16(float(std::sin(double(ang)))));
    }
  return t;
}

Q4LayerBytes Q4HostLayer::bytes() const {
  Q4LayerBytes b;
  b.hc_attn = hc_attn.size();
  b.hc_mlp = hc_mlp.size();
  b.gdn_qkvz = gdn_qkvz.bytes();
  b.gdn_ab = gdn_ab.bytes();
  b.gdn_out = gdn_out.bytes();
  b.gdn_small = gdn_small.size();
  b.qsa_qkvg = qsa_qkvg.bytes();
  b.qsa_idx = qsa_idx.bytes();
  b.qsa_o = qsa_o.bytes();
  b.qsa_small = qsa_small.size();
  b.router = router.bytes();
  b.gate_up = gate_up.size() * 4;
  b.down = down.size() * 4;
  b.shared_gate_up = shared_gate_up;   // one allocation, split where q4_shared_gate_up_bytes says
  b.shared_down = shared.size() - shared_gate_up;
  b.ple = ple.size();
  return b;
}

Q4Checkpoint::Q4Checkpoint(const Qwen4ExpDesc& d, const SafetensorsSet& set) : d_(d), set_(set) {}

void Q4Checkpoint::check_names(bool mtp) const {
  const std::vector<std::string> exp = q4_expected_names(d_, d_.forms, mtp);
  const std::set<std::string> all(exp.begin(), exp.end());
  const std::string shards = Qwen4ExpDesc::layer_prefix(d_.ple_layer) + kPleShards;
  std::string bad;
  size_t n_bad = 0;
  for (const auto& [name, info] : set_.tensors()) {
    (void)info;
    if (all.count(name)) continue;
    if (starts_with(name, "model.visual.") || starts_with(name, shards) || (!mtp && starts_with(name, "mtp."))) continue;
    if (ends_with(name, ".g_idx") && all.count(name.substr(0, name.size() - 6) + ".qweight")) continue;
    if (n_bad++ < 5) bad += (bad.empty() ? "" : ", ") + name;
  }
  if (n_bad != 0)
    throw std::runtime_error("load_qwen4exp: the checkpoint has " + std::to_string(n_bad) + " tensor(s) " + d_.name +
                             " (model/qwen4exp.h; dense " + model::q4_form_name(d_.forms.dense) + ", shared " +
                             model::q4_form_name(d_.forms.shared) + ", experts g" +
                             std::to_string(d_.forms.expert_group) + ") does not describe: " + bad +
                             (n_bad > 5 ? ", ..." : "") +
                             " - refusing to guess what they are (a quantised router or HC, an extra expert or layer, "
                             "another packing are all outside spec 21's one format)");
  for (const std::string& name : exp) {
    if (ends_with(name, ".qzeros")) continue;   // never read; assert_quant_invariants checks it when shipped
    if (!set_.tensors().count(name))
      throw std::runtime_error("load_qwen4exp: the checkpoint has no tensor '" + name + "', which " + d_.name + " needs" +
                               (starts_with(name, "mtp.") ? " for its MTP head (load without --mtp)" : ""));
  }
}

LinearSrc Q4Checkpoint::int4(const std::string& prefix, uint32_t K, uint32_t N, uint32_t group) {
  LinearSrc s = LinearSrc::classify(set_, prefix);
  if (s.kind != WKind::Int4 || s.packing != Int4Packing::Gptq)
    throw std::runtime_error("load_qwen4exp: '" + prefix + "' is not an int4 GPTQ linear; " + d_.name +
                             " reads it as int4 (auto_round:auto_gptq)");
  if (s.K != K || s.N != N)
    throw std::runtime_error("load_qwen4exp: '" + prefix + "' is K=" + std::to_string(s.K) + " N=" + std::to_string(s.N) +
                             ", the descriptor says K=" + std::to_string(K) + " N=" + std::to_string(N));
  if (s.group != group)
    throw std::runtime_error("load_qwen4exp: '" + prefix + ".scales' is g" + std::to_string(s.group) + ", expected g" +
                             std::to_string(group) +
                             " (int4 g64 everywhere but Intel's routed experts, g128 expanded exactly at load)");
  for (const std::string& suffix : s.suffixes()) consumed_.insert(prefix + suffix);
  return s;
}

LinearSrc Q4Checkpoint::bf16(const std::string& prefix, uint32_t K, uint32_t N) {
  LinearSrc s = LinearSrc::classify(set_, prefix);
  if (s.kind != WKind::Bf16)
    throw std::runtime_error("load_qwen4exp: '" + prefix + "' is int4; " + d_.name + " keeps it bf16 (spec 21 §5)");
  if (s.K != K || s.N != N)
    throw std::runtime_error("load_qwen4exp: '" + prefix + ".weight' is [" + std::to_string(s.N) + "][" +
                             std::to_string(s.K) + "], the descriptor says [" + std::to_string(N) + "][" +
                             std::to_string(K) + "]");
  consumed_.insert(prefix + ".weight");
  return s;
}

TensorInfo Q4Checkpoint::take(const std::string& name, uint64_t elems, const char* dtype) {
  auto it = set_.tensors().find(name);
  if (it == set_.tensors().end()) throw std::runtime_error("load_qwen4exp: no tensor '" + name + "'");
  const TensorInfo t = it->second;   // a copy (loader.cc's -Wdangling-reference note)
  if (t.dtype != dtype)
    throw std::runtime_error("load_qwen4exp: '" + name + "' is " + t.dtype + ", expected " + dtype + " (spec 21 §5)");
  uint64_t n = 1;
  for (uint64_t x : t.shape) n *= x;
  if (n != elems)
    throw std::runtime_error("load_qwen4exp: '" + name + "' has " + std::to_string(n) + " elements, expected " +
                             std::to_string(elems));
  consumed_.insert(name);
  return t;
}

void Q4Checkpoint::widen(const TensorInfo& t, float* dst) const {
  const uint8_t* p = set_.data(t);
  check_align(p, 2, "a bf16 tensor");
  const uint16_t* w = reinterpret_cast<const uint16_t*>(p);
  const size_t n = set_.bytes(t) / 2;
  for (size_t i = 0; i < n; ++i) dst[i] = common::bf16_to_f32(w[i]);
}

// Qwen's (1 + w) RMSNorm weight baked to fp32 at load: float(1) + float(w), stored fp32 - the reference
// computes the whole product in fp32 (loader.cc's SmallBake::OnePlusWFp32, ruling 2026-08-25).
void Q4Checkpoint::one_plus_w(const std::string& name, uint32_t n, uint8_t* dst) {
  const TensorInfo t = take(name, n);
  const uint8_t* p = set_.data(t);
  check_align(p, 2, name);
  const uint16_t* w = reinterpret_cast<const uint16_t*>(p);
  float* f = reinterpret_cast<float*>(dst);
  for (uint32_t i = 0; i < n; ++i) f[i] = 1.0f + common::bf16_to_f32(w[i]);
}

void Q4Checkpoint::dense(const Qwen4ExpDesc& d, const std::string& lp, Q4LinearId id, Q4HostWeight& out,
                         Q4HostLayer& L) {
  const model::Q4Linear fl = d.linear(id);
  const model::GemvShape& sh = fl.shape;
  // Each part's N: the descriptor's columns split as the checkpoint ships them.
  std::vector<uint32_t> part_n;
  switch (id) {
    case Q4LinearId::GdnQkvz: part_n = {d.conv_rows(), d.gdn_z_n()}; break;
    case Q4LinearId::GdnAb: part_n = {d.gdn_v_heads, d.gdn_v_heads}; break;
    case Q4LinearId::QsaQkvg: part_n = {2 * d.q_n(), d.kv_n(), d.kv_n()}; break;
    case Q4LinearId::GdnOut:
    case Q4LinearId::QsaO: part_n = {d.hidden}; break;
    case Q4LinearId::QsaIdx: part_n = {d.idx_n()}; break;
    default: throw std::logic_error("Q4Checkpoint::dense: not a per-layer dense row");
  }
  out.clear();
  out.shape = sh;
  out.kind = fl.kind;
  if (fl.kind == model::WeightKind::Int4) {
    std::vector<LinearSrc> srcs;
    std::vector<common::Part> parts;
    srcs.reserve(fl.parts.size());
    for (size_t i = 0; i < fl.parts.size(); ++i) {
      srcs.push_back(int4(lp + fl.parts[i], sh.K, part_n[i], 64));
      parts.push_back({srcs.back().qweight, srcs.back().scales, srcs.back().N});
      L.src_int4_bytes += int4_src_bytes(srcs.back());
    }
    out.words.resize(size_t(sh.K / 8) * sh.N);
    out.scales.resize(size_t(sh.K / 64) * sh.N);
    common::repack_int4_layout0_cols(sh.K, sh.N, common::cols_concat(parts), out.words.data(), out.scales.data());
    return;
  }
  // bf16: the parts' [N_i][K] rows stacked in column order, zero rows to the padded N (a||b), then tiles.
  std::vector<uint16_t> rows(size_t(sh.N) * sh.K, 0);
  size_t at = 0;
  for (size_t i = 0; i < fl.parts.size(); ++i) {
    const LinearSrc s = bf16(lp + fl.parts[i], sh.K, part_n[i]);
    std::memcpy(rows.data() + at, s.weight, size_t(s.N) * s.K * 2);
    at += size_t(s.N) * s.K;
    L.src_bf16_bytes += size_t(s.N) * s.K * 2;
  }
  out.tiles.resize(rows.size());
  common::repack_bf16_tiled(rows.data(), sh.K, sh.N, out.tiles.data());
}

void Q4Checkpoint::hc(const std::string& base, bool inject, std::vector<uint8_t>& out, Q4HostLayer& L) {
  const Q4HcOffsets o = q4_hc_offsets(d_, inject);
  const uint32_t HC = d_.hc_n();
  out.assign(o.total, 0);
  std::vector<uint16_t> rows(size_t(o.down_n) * HC, 0);
  const LinearSrc dn = bf16(base + ".input_mix_weight_down", HC, d_.hc_low);
  std::memcpy(rows.data(), dn.weight, size_t(d_.hc_low) * HC * 2);
  L.src_bf16_bytes += size_t(d_.hc_low) * HC * 2;
  if (inject) {
    const LinearSrc inj = bf16(base + ".block_inject_weight", HC, d_.hc);
    std::memcpy(rows.data() + size_t(d_.hc_low) * HC, inj.weight, size_t(d_.hc) * HC * 2);
    L.src_bf16_bytes += size_t(d_.hc) * HC * 2;
  }
  common::repack_bf16_tiled(rows.data(), HC, o.down_n, reinterpret_cast<uint16_t*>(out.data() + o.down));
  const LinearSrc up = bf16(base + ".input_mix_weight_up", d_.hc_low, HC);
  common::repack_bf16_tiled(up.weight, d_.hc_low, HC, reinterpret_cast<uint16_t*>(out.data() + o.up));
  L.src_bf16_bytes += size_t(d_.hc_low) * HC * 2;
  one_plus_w(base + ".hc_norm.weight", HC, out.data() + o.norm);
  L.src_bf16_bytes += size_t(HC) * 2;
}

void Q4Checkpoint::moe(const Qwen4ExpDesc& d, const std::string& lp, bool mtp, Q4HostLayer& L) {
  const uint32_t H = d.hidden, I = d.moe_inter, SI = d.shared_inter, E = d.experts, RN = d.router_n();
  // The router: rows 0..E-1 mlp.gate, row E shared_expert_gate, zero rows to router_n; bf16 tiles.
  {
    const LinearSrc r = bf16(lp + "mlp.gate", H, E);
    const LinearSrc g = bf16(lp + "mlp.shared_expert_gate", H, 1);
    std::vector<uint16_t> rows(size_t(RN) * H, 0);
    std::memcpy(rows.data(), r.weight, size_t(E) * H * 2);
    std::memcpy(rows.data() + size_t(E) * H, g.weight, size_t(H) * 2);
    L.router.clear();
    L.router.shape = {H, RN, 1, 0};
    L.router.kind = model::WeightKind::Bf16;
    L.router.tiles.resize(rows.size());
    common::repack_bf16_tiled(rows.data(), H, RN, L.router.tiles.data());
    L.src_bf16_bytes += size_t(E + 1) * H * 2;
  }
  // The shared expert: one allocation, gate||up then down.
  {
    const std::string sp = lp + "mlp.shared_expert.";
    const size_t gu = q4_shared_gate_up_bytes(d), dnb = q4_shared_down_bytes(d);
    L.shared.assign(gu + dnb, 0);
    L.shared_gate_up = gu;
    if (d.forms.shared == Q4Form::Int4) {
      const LinearSrc g = int4(sp + "gate_proj", H, SI, 64), u = int4(sp + "up_proj", H, SI, 64);
      const LinearSrc dn = int4(sp + "down_proj", SI, H, 64);
      common::repack_int4_layout1_cols(H, 2 * SI,
                                       common::cols_interleave16({g.qweight, g.scales, g.N}, {u.qweight, u.scales, u.N}),
                                       reinterpret_cast<uint32_t*>(L.shared.data()));
      common::repack_int4_layout1_cols(SI, H, common::cols_concat({{dn.qweight, dn.scales, dn.N}}),
                                       reinterpret_cast<uint32_t*>(L.shared.data() + gu));
      L.src_int4_bytes += int4_src_bytes(g) + int4_src_bytes(u) + int4_src_bytes(dn);
    } else {
      const LinearSrc g = bf16(sp + "gate_proj", H, SI), u = bf16(sp + "up_proj", H, SI);
      const LinearSrc dn = bf16(sp + "down_proj", SI, H);
      tiled_interleave16(g.weight, u.weight, SI, H, reinterpret_cast<uint16_t*>(L.shared.data()));
      common::repack_bf16_tiled(dn.weight, SI, H, reinterpret_cast<uint16_t*>(L.shared.data() + gu));
      L.src_bf16_bytes += size_t(3) * SI * H * 2;
    }
  }
  // The routed experts: layout-1 blocks, block e at e x block (gate||up interleave16). The MTP head's bf16
  // experts are RTN-quantised to int4 g64 first (spec 15e's rule).
  const size_t gu_words = q4_gate_up_block_bytes(d) / 4, dn_words = q4_down_block_bytes(d) / 4;
  L.gate_up.assign(gu_words * E, 0);
  L.down.assign(dn_words * E, 0);
  const bool rtn = mtp && d.forms.mtp_experts == Q4Form::Bf16;
  std::vector<uint32_t> qg, qu, qd;
  std::vector<uint16_t> sg, su, sd;
  if (rtn) {
    qg.resize(size_t(H / 8) * I);
    qu.resize(qg.size());
    sg.resize(size_t(H / 64) * I);
    su.resize(sg.size());
    qd.resize(size_t(I / 8) * H);
    sd.resize(size_t(I / 64) * H);
  }
  const uint32_t group = mtp ? 64 : d.forms.expert_group;
  for (uint32_t e = 0; e < E; ++e) {
    const std::string ep = lp + "mlp.experts." + std::to_string(e) + ".";
    common::Part pg{}, pu{}, pd{};
    if (rtn) {
      const LinearSrc g = bf16(ep + "gate_proj", H, I), u = bf16(ep + "up_proj", H, I);
      const LinearSrc dn = bf16(ep + "down_proj", I, H);
      rtn_int4_g64(g.weight, H, I, qg.data(), sg.data());
      rtn_int4_g64(u.weight, H, I, qu.data(), su.data());
      rtn_int4_g64(dn.weight, I, H, qd.data(), sd.data());
      pg = {qg.data(), sg.data(), I};
      pu = {qu.data(), su.data(), I};
      pd = {qd.data(), sd.data(), H};
      L.src_bf16_bytes += size_t(3) * I * H * 2;
      common::repack_int4_layout1_cols(H, 2 * I, common::cols_interleave16(pg, pu), L.gate_up.data() + size_t(e) * gu_words);
      common::repack_int4_layout1_cols(I, H, common::cols_concat({pd}), L.down.data() + size_t(e) * dn_words);
      continue;
    }
    const LinearSrc g = int4(ep + "gate_proj", H, I, group), u = int4(ep + "up_proj", H, I, group);
    const LinearSrc dn = int4(ep + "down_proj", I, H, group);
    L.src_int4_bytes += int4_src_bytes(g) + int4_src_bytes(u) + int4_src_bytes(dn);
    common::repack_int4_layout1_cols(H, 2 * I,
                                     common::cols_interleave16({g.qweight, g.scales, g.N}, {u.qweight, u.scales, u.N}),
                                     L.gate_up.data() + size_t(e) * gu_words);
    common::repack_int4_layout1_cols(I, H, common::cols_concat({{dn.qweight, dn.scales, dn.N}}),
                                     L.down.data() + size_t(e) * dn_words);
  }
}

void Q4Checkpoint::qsa(const Qwen4ExpDesc& d, const std::string& lp, Q4HostLayer& L) {
  dense(d, lp, Q4LinearId::QsaQkvg, L.qsa_qkvg, L);
  dense(d, lp, Q4LinearId::QsaIdx, L.qsa_idx, L);
  dense(d, lp, Q4LinearId::QsaO, L.qsa_o, L);
  const Q4QsaSmall s = q4_qsa_small(d);
  L.qsa_small.assign(s.total, 0);
  one_plus_w(lp + "self_attn.q_norm.weight", d.head_dim, L.qsa_small.data() + s.q_norm);
  one_plus_w(lp + "self_attn.k_norm.weight", d.head_dim, L.qsa_small.data() + s.k_norm);
  one_plus_w(lp + "self_attn.indexer.q_layernorm.weight", d.idx_dim, L.qsa_small.data() + s.idx_q_norm);
  one_plus_w(lp + "self_attn.indexer.k_layernorm.weight", d.idx_dim, L.qsa_small.data() + s.idx_k_norm);
  L.src_bf16_bytes += size_t(2 * d.head_dim + 2 * d.idx_dim) * 2;
}

void Q4Checkpoint::repack_layer(uint32_t layer, Q4HostLayer& out) {
  if (layer >= d_.layers)
    throw std::out_of_range("load_qwen4exp: layer " + std::to_string(layer) + " of " + std::to_string(d_.layers));
  const std::string lp = Qwen4ExpDesc::layer_prefix(layer);
  out = Q4HostLayer{};
  hc(lp + "attn_hyper_connection", true, out.hc_attn, out);
  hc(lp + "mlp_hyper_connection", true, out.hc_mlp, out);
  if (d_.is_qsa(layer)) {
    qsa(d_, lp, out);
  } else {
    dense(d_, lp, Q4LinearId::GdnQkvz, out.gdn_qkvz, out);
    dense(d_, lp, Q4LinearId::GdnAb, out.gdn_ab, out);
    dense(d_, lp, Q4LinearId::GdnOut, out.gdn_out, out);
    // The GDN small block (Qwen3.8's make_small_layout at this family's widths): conv taps widened, the decay
    // -exp(A_log) hoisted, dt_bias widened, the gated norm's plain w verbatim (bf16).
    const SmallLayout sl = q4_gdn_small(d_);
    out.gdn_small.assign(sl.gdn_block_bytes, 0);
    widen(take(lp + "linear_attn.conv1d.weight", uint64_t(d_.conv_rows()) * d_.conv_taps),
          reinterpret_cast<float*>(out.gdn_small.data() + sl.gdn_off_conv));
    {
      const TensorInfo a = take(lp + "linear_attn.A_log", d_.gdn_v_heads);
      std::vector<float> f(d_.gdn_v_heads);
      widen(a, f.data());
      float* dst = reinterpret_cast<float*>(out.gdn_small.data() + sl.gdn_off_nega);
      for (uint32_t i = 0; i < d_.gdn_v_heads; ++i) dst[i] = -std::exp(f[i]);
    }
    widen(take(lp + "linear_attn.dt_bias", d_.gdn_v_heads),
          reinterpret_cast<float*>(out.gdn_small.data() + sl.gdn_off_dtbias));
    {
      const TensorInfo n = take(lp + "linear_attn.norm.weight", d_.gdn_head);
      std::memcpy(out.gdn_small.data() + sl.gdn_off_gated_norm, set_.data(n), size_t(d_.gdn_head) * 2);
    }
    out.src_bf16_bytes += (size_t(d_.conv_rows()) * d_.conv_taps + 2 * d_.gdn_v_heads + d_.gdn_head) * 2;
  }
  moe(d_, lp, false, out);
  if (layer == d_.ple_layer) {
    const std::string pp = lp + "ple.";
    const Q4PleOffsets o = q4_ple_offsets(d_);
    out.ple.assign(o.total, 0);
    {
      std::vector<uint16_t> rows(size_t(d_.ple_kv_n()) * d_.ple_e());
      const LinearSrc k = bf16(pp + "key_proj", d_.ple_e(), d_.hc_n());
      const LinearSrc v = bf16(pp + "value_proj", d_.ple_e(), d_.hidden);
      std::memcpy(rows.data(), k.weight, size_t(d_.hc_n()) * d_.ple_e() * 2);
      std::memcpy(rows.data() + size_t(d_.hc_n()) * d_.ple_e(), v.weight, size_t(d_.hidden) * d_.ple_e() * 2);
      common::repack_bf16_tiled(rows.data(), d_.ple_e(), d_.ple_kv_n(), reinterpret_cast<uint16_t*>(out.ple.data() + o.kv));
      out.src_bf16_bytes += rows.size() * 2;
    }
    one_plus_w(pp + "norm_key.weight", d_.hc_n(), out.ple.data() + o.norm_key);
    one_plus_w(pp + "norm_query.weight", d_.hc_n(), out.ple.data() + o.norm_query);
    one_plus_w(pp + "norm_conv.weight", d_.hc_n(), out.ple.data() + o.norm_conv);
    widen(take(pp + "conv1d.weight", uint64_t(d_.hc_n()) * d_.ple_conv_taps),
          reinterpret_cast<float*>(out.ple.data() + o.conv));
    out.src_bf16_bytes += size_t(d_.hc_n()) * (3 + d_.ple_conv_taps) * 2;
  }
}

void Q4Checkpoint::repack_mtp(Q4HostLayer& out) {
  const Qwen4ExpDesc m = q4_mtp_desc(d_);
  const std::string lp = "mtp.layers.0.";
  out = Q4HostLayer{};
  hc(lp + "attn_hyper_connection", true, out.hc_attn, out);
  hc(lp + "mlp_hyper_connection", true, out.hc_mlp, out);
  qsa(m, lp, out);
  moe(m, lp, true, out);
}

const uint16_t* Q4Checkpoint::embed() {
  const TensorInfo t = take(d_.prefix + "embed_tokens.weight", uint64_t(d_.vocab) * d_.hidden);
  check_align(set_.data(t), 2, "embed_tokens");
  return reinterpret_cast<const uint16_t*>(set_.data(t));
}

const uint16_t* Q4Checkpoint::lm_head() { return bf16("lm_head", d_.hidden, d_.vocab).weight; }

std::vector<uint8_t> Q4Checkpoint::final_mixer() {
  std::vector<uint8_t> out;
  Q4HostLayer scratch;
  hc(d_.prefix + "hyper_connection_mixer", false, out, scratch);
  return out;
}

std::vector<uint8_t> Q4Checkpoint::mtp_mixer() {
  std::vector<uint8_t> out;
  Q4HostLayer scratch;
  hc("mtp.hyper_connection_mixer", false, out, scratch);
  return out;
}

std::vector<uint8_t> Q4Checkpoint::mtp_fc() {
  const Q4MtpFcOffsets o = q4_mtp_fc_offsets(d_);
  std::vector<uint8_t> out(o.total, 0);
  const LinearSrc e = bf16("mtp.fc_embedding", d_.hidden, d_.hidden);
  const LinearSrc h = bf16("mtp.fc_hidden", d_.hidden, d_.hidden);
  common::repack_bf16_tiled(e.weight, d_.hidden, d_.hidden, reinterpret_cast<uint16_t*>(out.data() + o.fc_embedding));
  common::repack_bf16_tiled(h.weight, d_.hidden, d_.hidden, reinterpret_cast<uint16_t*>(out.data() + o.fc_hidden));
  one_plus_w("mtp.pre_fc_norm_embedding.weight", d_.hidden, out.data() + o.norm_embedding);
  one_plus_w("mtp.pre_fc_norm_hidden.weight", d_.hc_n(), out.data() + o.norm_hidden);
  return out;
}

Q4PleConstants Q4Checkpoint::ple_constants() {
  const std::string p = Qwen4ExpDesc::layer_prefix(d_.ple_layer) + "ple.ple_embedding.";
  const auto i64 = [&](const char* n, uint64_t elems) {
    const TensorInfo t = take(p + n, elems, "I64");
    check_align(set_.data(t), 8, p + n);
    const int64_t* v = reinterpret_cast<const int64_t*>(set_.data(t));
    std::vector<uint64_t> out(elems);
    for (uint64_t i = 0; i < elems; ++i) {
      if (v[i] < 0) throw std::runtime_error("load_qwen4exp: '" + p + n + "'[" + std::to_string(i) + "] is negative");
      out[i] = uint64_t(v[i]);
    }
    return out;
  };
  Q4PleConstants c;
  const std::vector<uint64_t> m = i64("layer_multipliers", d_.ngram);
  for (uint32_t i = 0; i < 3 && i < d_.ngram; ++i) c.multipliers[i] = m[i];
  c.sizes = i64("ngram_heads_vocab_sizes", d_.ple_heads);
  c.offsets = i64("ngram_heads_offsets", d_.ple_heads);
  return c;
}

size_t Q4Checkpoint::skip_layers_from(uint32_t from) {
  size_t n = 0;
  const std::string p = d_.prefix + "layers.";
  for (const auto& [name, info] : set_.tensors()) {
    (void)info;
    if (!starts_with(name, p)) continue;
    const uint32_t l = uint32_t(std::stoul(name.substr(p.size())));
    if (l >= from && consumed_.insert(name).second) ++n;
  }
  return n;
}

size_t Q4Checkpoint::skip_prefix(const std::string& pre) {
  size_t n = 0;
  for (const auto& [name, info] : set_.tensors()) {
    (void)info;
    if (starts_with(name, pre) && consumed_.insert(name).second) ++n;
  }
  return n;
}

size_t Q4Checkpoint::skip_ple_shards() { return skip_prefix(Qwen4ExpDesc::layer_prefix(d_.ple_layer) + kPleShards); }

size_t Q4Checkpoint::unconsumed(std::string* names) const {
  size_t n = 0;
  for (const auto& [name, info] : set_.tensors()) {
    (void)info;
    if (consumed_.count(name) || ends_with(name, ".qzeros") || ends_with(name, ".g_idx")) continue;
    if (names && n < 5) *names += (names->empty() ? "" : ", ") + name;
    ++n;
  }
  return n;
}

QuantConfig check_qwen4exp_checkpoint_config(const common::json::Value& config) { return QuantConfig::parse(config); }

}  // namespace loader
