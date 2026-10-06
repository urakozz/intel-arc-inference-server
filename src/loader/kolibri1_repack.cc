#include "loader/kolibri1_repack.h"

#include <cstring>
#include <stdexcept>

#include "common/bf16.h"
#include "common/repack.h"
#include "loader/kolibri1_layout.h"

namespace loader {
namespace {

using model::Kolibri1Desc;
using model::KolAttnForm;

bool ends_with(const std::string& s, const char* suf) {
  const size_t n = std::strlen(suf);
  return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

const char* const kAttnParts[] = {"q_proj", "k_proj", "v_proj", "o_proj"};
const char* const kNorms[] = {"input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm"};
const char* const kMlp[] = {"gate_proj", "up_proj", "down_proj"};

// The int4 linear's names as AutoRound's auto_round:auto_gptq export writes them.
void add_int4(std::vector<std::string>& v, const std::string& base) {
  v.push_back(base + ".qweight");
  v.push_back(base + ".scales");
  v.push_back(base + ".qzeros");
}

size_t int4_src_bytes(const LinearSrc& s) { return size_t(s.K / 8) * s.N * 4 + size_t(s.K / s.group) * s.N * 2; }

// gate||up interleaved in 16-row blocks of an [N][K] row-major pair (cols_interleave16's column
// order, as rows): row r of the fused [2I][K] is gate[(r / 32) x 16 + r % 16] for r % 32 < 16,
// else up[...]. Then gemv_bf16's tiles.
void tiled_interleave16(const uint16_t* gate, const uint16_t* up, uint32_t I, uint32_t K, uint16_t* out) {
  std::vector<uint16_t> rows(size_t(2) * I * K);
  for (uint32_t r = 0; r < 2 * I; ++r) {
    const uint32_t src = (r / 32) * 16 + r % 16;
    const uint16_t* w = (r % 32 < 16 ? gate : up) + size_t(src) * K;
    std::memcpy(rows.data() + size_t(r) * K, w, size_t(K) * 2);
  }
  common::repack_bf16_tiled(rows.data(), K, 2 * I, out);
}

}  // namespace

KolAttnForm kol_attn_form(const SafetensorsSet& st, const Kolibri1Desc& d) {
  const auto& ts = st.tensors();
  const std::string l0 = Kolibri1Desc::layer_prefix(0) + "self_attn.q_proj";
  KolAttnForm a;
  if (ts.count(l0 + ".qweight"))
    a = KolAttnForm::Int4;
  else if (ts.count(l0 + ".weight"))
    a = KolAttnForm::Bf16;
  else
    throw std::runtime_error("load_kolibri1: the checkpoint has neither " + l0 + ".qweight nor " + l0 +
                             ".weight - no attention to read");
  for (uint32_t l = 0; l < d.layers; ++l)
    for (const char* p : kAttnParts) {
      const std::string base = Kolibri1Desc::layer_prefix(l) + "self_attn." + p;
      const std::string other = base + (a == KolAttnForm::Int4 ? ".weight" : ".qweight");
      if (ts.count(other))
        throw std::runtime_error("load_kolibri1: '" + other + "' mixes the attention arms - layer 0's q_proj is " +
                                 model::kol_attn_form_name(a) +
                                 ", and every attention projection of every layer must agree (spec 20 decision 2)");
    }
  return a;
}

std::vector<std::string> kol_expected_names(const Kolibri1Desc& d, KolAttnForm a) {
  std::vector<std::string> v = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"};
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string p = Kolibri1Desc::layer_prefix(l);
    for (const char* n : kNorms) v.push_back(p + n + ".weight");
    for (const char* n : kAttnParts) {
      if (a == KolAttnForm::Int4)
        add_int4(v, p + "self_attn." + n);
      else
        v.push_back(p + "self_attn." + n + ".weight");
    }
    v.push_back(p + "self_attn.q_norm.weight");
    v.push_back(p + "self_attn.k_norm.weight");
    v.push_back(p + "mlp.gate.weight");
    v.push_back(p + "moe.router.expert_bias");
    for (const char* n : kMlp) v.push_back(p + "mlp.shared_experts." + n + ".weight");
    for (uint32_t e = 0; e < d.experts; ++e)
      for (const char* n : kMlp) add_int4(v, p + "mlp.experts." + std::to_string(e) + "." + n);
  }
  return v;
}

KolCheckpoint::KolCheckpoint(const Kolibri1Desc& d, const SafetensorsSet& set) : d_(d), set_(set) {}

void KolCheckpoint::check_names() const {
  const std::vector<std::string> exp = kol_expected_names(d_, d_.attn);
  const std::set<std::string> all(exp.begin(), exp.end());
  std::string bad;
  size_t n_bad = 0;
  for (const auto& [name, info] : set_.tensors()) {
    (void)info;
    if (all.count(name)) continue;
    // `.g_idx` beside an expected int4 linear is allowed (assert_quant_invariants proves it the
    // identity); `.qzeros` is expected (AutoRound writes it).
    if (ends_with(name, ".g_idx") && all.count(name.substr(0, name.size() - 6) + ".qweight")) continue;
    if (n_bad++ < 5) bad += (bad.empty() ? "" : ", ") + name;
  }
  if (n_bad != 0)
    throw std::runtime_error("load_kolibri1: the checkpoint has " + std::to_string(n_bad) + " tensor(s) " + d_.name +
                             " (model/kolibri1.h, " + model::kol_attn_form_name(d_.attn) +
                             " attention) does not describe: " + bad + (n_bad > 5 ? ", ..." : "") +
                             " - refusing to guess what they are (a quantised lm_head or shared expert, an "
                             "extra expert or layer, another packing are all outside spec 20's one format)");
  for (const std::string& name : exp) {
    if (ends_with(name, ".qzeros")) continue;   // never read; assert_quant_invariants checks it when shipped
    if (!set_.tensors().count(name))
      throw std::runtime_error("load_kolibri1: the checkpoint has no tensor '" + name + "', which " + d_.name +
                               " needs");
  }
}

LinearSrc KolCheckpoint::int4(const std::string& prefix, uint32_t K, uint32_t N) {
  LinearSrc s = LinearSrc::classify(set_, prefix);
  if (s.kind != WKind::Int4 || s.packing != Int4Packing::Gptq)
    throw std::runtime_error("load_kolibri1: '" + prefix + "' is not an int4 GPTQ linear; " + d_.name +
                             " reads it as int4 g64 (auto_round:auto_gptq)");
  if (s.K != K || s.N != N)
    throw std::runtime_error("load_kolibri1: '" + prefix + "' is K=" + std::to_string(s.K) + " N=" +
                             std::to_string(s.N) + ", the descriptor says K=" + std::to_string(K) + " N=" +
                             std::to_string(N));
  for (const std::string& suffix : s.suffixes()) consumed_.insert(prefix + suffix);
  return s;
}

LinearSrc KolCheckpoint::bf16(const std::string& prefix, uint32_t K, uint32_t N) {
  LinearSrc s = LinearSrc::classify(set_, prefix);
  if (s.kind != WKind::Bf16)
    throw std::runtime_error("load_kolibri1: '" + prefix + "' is int4; " + d_.name +
                             " keeps it bf16 (spec 20 §3.1)");
  if (s.K != K || s.N != N)
    throw std::runtime_error("load_kolibri1: '" + prefix + ".weight' is [" + std::to_string(s.N) + "][" +
                             std::to_string(s.K) + "], the descriptor says [" + std::to_string(N) + "][" +
                             std::to_string(K) + "]");
  consumed_.insert(prefix + ".weight");
  return s;
}

TensorInfo KolCheckpoint::take(const std::string& name, uint64_t elems) {
  auto it = set_.tensors().find(name);
  if (it == set_.tensors().end()) throw std::runtime_error("load_kolibri1: no tensor '" + name + "'");
  const TensorInfo t = it->second;   // a copy (loader.cc's -Wdangling-reference note)
  if (t.dtype != "BF16")
    throw std::runtime_error("load_kolibri1: '" + name + "' is " + t.dtype + ", expected BF16 (spec 20 §3.1)");
  uint64_t n = 1;
  for (uint64_t x : t.shape) n *= x;
  if (n != elems)
    throw std::runtime_error("load_kolibri1: '" + name + "' has " + std::to_string(n) + " elements, expected " +
                             std::to_string(elems));
  consumed_.insert(name);
  return t;
}

void KolCheckpoint::widen(const TensorInfo& t, float* dst) const {
  const uint8_t* p = set_.data(t);
  check_align(p, 2, "a bf16 tensor");
  const uint16_t* w = reinterpret_cast<const uint16_t*>(p);
  const size_t n = set_.bytes(t) / 2;
  for (size_t i = 0; i < n; ++i) dst[i] = common::bf16_to_f32(w[i]);
}

void KolCheckpoint::attn_linear(const std::string& lp, model::KolLinearId id, std::vector<uint32_t>& words,
                                std::vector<uint16_t>& scales, std::vector<uint16_t>& tiles, KolHostLayer& out) {
  const model::KolLinear fl = d_.linear(id);
  const model::GemvShape& sh = fl.shape;
  const uint32_t q_n = d_.q_n(), kv_n = d_.kv_n();
  const auto part_n = [&](size_t i) {
    return id == model::KolLinearId::OProj ? d_.hidden : (i == 0 ? q_n : kv_n);
  };
  words.clear();
  scales.clear();
  tiles.clear();
  if (d_.attn == KolAttnForm::Int4) {
    std::vector<LinearSrc> srcs;
    std::vector<common::Part> parts;
    for (size_t i = 0; i < fl.parts.size(); ++i) {
      srcs.push_back(int4(lp + fl.parts[i], sh.K, uint32_t(part_n(i))));
      parts.push_back({srcs.back().qweight, srcs.back().scales, srcs.back().N});
      out.src_int4_bytes += int4_src_bytes(srcs.back());
    }
    words.resize(size_t(sh.K / 8) * sh.N);
    scales.resize(size_t(sh.K / 64) * sh.N);
    common::repack_int4_layout0_cols(sh.K, sh.N, common::cols_concat(parts), words.data(), scales.data());
    return;
  }
  // bf16 arm: the parts' [N_i][K] rows stacked (q, then k, then v), then gemv_bf16's tiles.
  std::vector<uint16_t> rows(size_t(sh.N) * sh.K);
  size_t at = 0;
  for (size_t i = 0; i < fl.parts.size(); ++i) {
    const LinearSrc s = bf16(lp + fl.parts[i], sh.K, uint32_t(part_n(i)));
    std::memcpy(rows.data() + at, s.weight, size_t(s.N) * s.K * 2);
    at += size_t(s.N) * s.K;
    out.src_bf16_bytes += size_t(s.N) * s.K * 2;
  }
  tiles.resize(rows.size());
  common::repack_bf16_tiled(rows.data(), sh.K, sh.N, tiles.data());
}

void KolCheckpoint::repack_layer(uint32_t layer, KolHostLayer& out) {
  if (layer >= d_.layers)
    throw std::out_of_range("load_kolibri1: layer " + std::to_string(layer) + " of " + std::to_string(d_.layers));
  const std::string lp = Kolibri1Desc::layer_prefix(layer);
  const uint32_t H = d_.hidden, I = d_.moe_inter, SI = d_.shared_inter, E = d_.experts, RN = d_.router_n();
  out.src_int4_bytes = out.src_bf16_bytes = 0;

  attn_linear(lp, model::KolLinearId::Qkv, out.qkv_words, out.qkv_scales, out.qkv_bf16, out);
  attn_linear(lp, model::KolLinearId::OProj, out.oproj_words, out.oproj_scales, out.oproj_bf16, out);

  // Norms widened verbatim (plain w: Kolibri multiplies by the stored weight, no 1 + w).
  out.norms.assign(d_.norm_floats(), 0.0f);
  const uint32_t offs[] = {d_.norm_off_input(), d_.norm_off_post_attn(), d_.norm_off_post_attention(),
                           d_.norm_off_post_ffn()};
  for (int i = 0; i < 4; ++i) {
    widen(take(lp + kNorms[i] + ".weight", H), out.norms.data() + offs[i]);
    out.src_bf16_bytes += size_t(H) * 2;
  }
  widen(take(lp + "self_attn.q_norm.weight", d_.head_dim), out.norms.data() + d_.norm_off_q());
  widen(take(lp + "self_attn.k_norm.weight", d_.head_dim), out.norms.data() + d_.norm_off_k());
  out.src_bf16_bytes += size_t(d_.head_dim) * 4;

  // The router: bf16 [experts][hidden], zero rows up to router_n, gemv_bf16's tiles.
  {
    const LinearSrc r = bf16(lp + "mlp.gate", H, E);
    std::vector<uint16_t> rows(size_t(RN) * H, 0);
    std::memcpy(rows.data(), r.weight, size_t(E) * H * 2);
    out.router.resize(rows.size());
    common::repack_bf16_tiled(rows.data(), H, RN, out.router.data());
    out.src_bf16_bytes += size_t(E) * H * 2;
  }
  // expert_bias, widened exactly, zero past the experts (selection only; never a weight).
  out.bias.assign(RN, 0.0f);
  widen(take(lp + "moe.router.expert_bias", E), out.bias.data());
  out.src_bf16_bytes += size_t(E) * 2;

  // The shared expert, bf16: gate||up interleave16 then tiles; down tiles.
  {
    const LinearSrc g = bf16(lp + "mlp.shared_experts.gate_proj", H, SI);
    const LinearSrc u = bf16(lp + "mlp.shared_experts.up_proj", H, SI);
    const LinearSrc dn = bf16(lp + "mlp.shared_experts.down_proj", SI, H);
    out.shared_gate_up.resize(size_t(2) * SI * H);
    tiled_interleave16(g.weight, u.weight, SI, H, out.shared_gate_up.data());
    out.shared_down.resize(size_t(SI) * H);
    common::repack_bf16_tiled(dn.weight, SI, H, out.shared_down.data());
    out.src_bf16_bytes += size_t(3) * SI * H * 2;
  }

  // The routed experts: layout-1 blocks, block e at e x stride (gate||up interleave16).
  const size_t gu_words = kol_gate_up_block_bytes(d_) / 4, dn_words = kol_down_block_bytes(d_) / 4;
  out.gate_up.resize(gu_words * E);
  out.down.resize(dn_words * E);
  for (uint32_t e = 0; e < E; ++e) {
    const std::string ep = lp + "mlp.experts." + std::to_string(e) + ".";
    const LinearSrc g = int4(ep + "gate_proj", H, I);
    const LinearSrc u = int4(ep + "up_proj", H, I);
    const LinearSrc dn = int4(ep + "down_proj", I, H);
    out.src_int4_bytes += int4_src_bytes(g) + int4_src_bytes(u) + int4_src_bytes(dn);
    common::repack_int4_layout1_cols(H, 2 * I,
                                     common::cols_interleave16({g.qweight, g.scales, g.N}, {u.qweight, u.scales, u.N}),
                                     out.gate_up.data() + size_t(e) * gu_words);
    common::repack_int4_layout1_cols(I, H, common::cols_concat({{dn.qweight, dn.scales, dn.N}}),
                                     out.down.data() + size_t(e) * dn_words);
  }
}

const uint16_t* KolCheckpoint::embed() {
  const TensorInfo t = take("model.embed_tokens.weight", uint64_t(d_.vocab) * d_.hidden);
  return reinterpret_cast<const uint16_t*>(set_.data(t));
}

std::vector<float> KolCheckpoint::final_norm() {
  std::vector<float> w(d_.hidden);
  widen(take("model.norm.weight", d_.hidden), w.data());
  return w;
}

const uint16_t* KolCheckpoint::lm_head() { return bf16("lm_head", d_.hidden, d_.vocab).weight; }

size_t KolCheckpoint::skip_layers_from(uint32_t from) {
  size_t n = 0;
  for (const auto& [name, info] : set_.tensors()) {
    (void)info;
    if (name.rfind("model.layers.", 0) != 0) continue;
    const uint32_t l = uint32_t(std::stoul(name.substr(13)));
    if (l >= from && consumed_.insert(name).second) ++n;
  }
  return n;
}

size_t KolCheckpoint::unconsumed(std::string* names) const {
  size_t n = 0;
  for (const auto& [name, info] : set_.tensors()) {
    (void)info;
    if (consumed_.count(name) || ends_with(name, ".qzeros") || ends_with(name, ".g_idx")) continue;
    if (names && n < 5) *names += (names->empty() ? "" : ", ") + name;
    ++n;
  }
  return n;
}

QuantConfig check_kolibri1_checkpoint_config(const common::json::Value& config) {
  return QuantConfig::parse(config);
}

}  // namespace loader
