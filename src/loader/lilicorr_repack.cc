#include "loader/lilicorr_repack.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "common/bf16.h"
#include "common/json.h"
#include "common/repack.h"
#include "loader/lm_head_int8.h"
#include "loader/nvfp4.h"
#include "loader/rtn.h"
#include "loader/snapshot.h"

// lilicorr_repack.cc - the host half of the LiLiCorr drafter loader (lilicorr_repack.h has the contract).
namespace loader {
namespace {

[[noreturn]] void refuse(const std::string& dir, const std::string& what) {
  throw std::runtime_error("LiLiCorr drafter " + dir + ": " + what);
}

std::string with_slash(std::string p) {
  if (!p.empty() && p.back() != '/') p += '/';
  return p;
}

common::json::Value read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream s;
  s << f.rdbuf();
  return common::json::parse(s.str());
}

bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}
std::string shape_str(const std::vector<uint64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) out += (i ? ", " : "") + std::to_string(s[i]);
  return out + "]";
}

// Every linear's base and its [N][K] (fc's K: `fc_k`).
std::map<std::string, std::pair<uint32_t, uint32_t>> linear_shapes(const model::DFlashDesc& d, uint32_t fc_k) {
  std::map<std::string, std::pair<uint32_t, uint32_t>> m;
  const uint32_t H = d.hidden;
  m["fc"] = {H, fc_k};
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string L = "layers." + std::to_string(l) + ".";
    m[L + "self_attn.q_proj"] = {d.q_n(), H};
    m[L + "self_attn.k_proj"] = {d.kv_n(), H};
    m[L + "self_attn.v_proj"] = {d.kv_n(), H};
    m[L + "self_attn.o_proj"] = {H, d.q_n()};
    m[L + "mlp.gate_proj"] = {d.inter, H};
    m[L + "mlp.up_proj"] = {d.inter, H};
    m[L + "mlp.down_proj"] = {H, d.inter};
    m[L + "attention_conv.kernel_projection"] = {d.kproj_n(), H};
    m[L + "mlp_conv.kernel_projection"] = {d.kproj_n(), H};
  }
  const std::string P = "lilicorr.";
  auto head = [&](const std::string& n, LiliHead w) { m[P + n] = {lili_head_shape(d, w).N, lili_head_shape(d, w).K}; };
  if (lili_has_token_proj(d)) head("token_proj", LiliHead::TokenProj);
  head("pass_hidden_proj", LiliHead::PassHidden);
  head("feature_mlp.up_proj", LiliHead::FeatUp);
  head("feature_mlp.down_proj", LiliHead::FeatDown);
  head("context_proj", LiliHead::ContextProj);
  for (uint32_t l = 0; l < d.lil_layers; ++l) {
    const std::string L = "layers." + std::to_string(l) + ".";
    head(L + "attn.out_proj", LiliHead::OutProj);
    head(L + "mlp.up_proj", LiliHead::MlpUp);
    head(L + "mlp.down_proj", LiliHead::MlpDown);
  }
  head("factor_input_proj", LiliHead::Factor);
  m[P + "out_head"] = {d.factor_dim, d.lil_hidden};
  m[P + "in_head"] = {d.factor_dim, d.lil_hidden};
  head("anchor_out_head", LiliHead::AnchorOut);
  return m;
}

// Every non-linear tensor and its shape (all BF16).
std::map<std::string, std::vector<uint64_t>> other_shapes(const model::DFlashDesc& d) {
  std::map<std::string, std::vector<uint64_t>> m;
  const uint64_t H = d.hidden, Hl = d.lil_hidden;
  m["hidden_norm.weight"] = {H};
  m["norm.weight"] = {H};
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string L = "layers." + std::to_string(l) + ".";
    m[L + "input_layernorm.weight"] = {H};
    m[L + "post_attention_layernorm.weight"] = {H};
    m[L + "self_attn.q_norm.weight"] = {d.head_dim};
    m[L + "self_attn.k_norm.weight"] = {d.head_dim};
    m[L + "self_attn.attention_sink_bias"] = {d.q_heads};
    m[L + "attention_conv.base_kernel"] = {2, d.conv_taps, H};
    m[L + "mlp_conv.base_kernel"] = {2, d.conv_taps, H};
  }
  const std::string P = "lilicorr.";
  for (const auto& [base, nk] : linear_shapes(d, 0))
    if (base.rfind(P, 0) == 0) m[base + ".bias"] = {nk.first};
  m[P + "feature_norm.weight"] = {5};
  m[P + "feature_norm.bias"] = {5};
  m[P + "slot_embedding"] = {1, 1, d.lil_slots(), 1, Hl};
  m[P + "rank_embedding"] = {1, 1, 1, d.cand_topk, Hl};
  m[P + "relative_slot_bias"] = {d.lil_heads, 2ull * d.block - 1};
  m[P + "same_slot_bias"] = {d.lil_heads};
  for (uint32_t l = 0; l < d.lil_layers; ++l) {
    const std::string L = P + "layers." + std::to_string(l) + ".";
    m[L + "attn_norm.weight"] = {Hl};
    m[L + "attn.in_proj_weight"] = {3 * Hl, Hl};
    m[L + "attn.in_proj_bias"] = {3 * Hl};
    m[L + "mlp_norm.weight"] = {Hl};
  }
  m[P + "output_norm.weight"] = {Hl};
  m[P + "anchor_norm.weight"] = {Hl};
  return m;
}

}  // namespace

std::vector<std::string> lilicorr_expected_names(const model::DFlashDesc& d) {
  std::vector<std::string> v;
  for (const auto& [base, nk] : linear_shapes(d, 0)) v.push_back(base + ".weight");
  for (const auto& [n, s] : other_shapes(d)) v.push_back(n);
  std::sort(v.begin(), v.end());
  return v;
}

bool lilicorr_is_linear(const model::DFlashDesc& d, const std::string& w) {
  return ends_with(w, ".weight") && linear_shapes(d, 0).count(w.substr(0, w.size() - 7)) != 0;
}

LiliCheckpoint::LiliCheckpoint(const std::string& snapshot_or_repo)
    : dir_(with_slash(resolve_snapshot(snapshot_or_repo))) {
  const common::json::Value cfg = read_json(dir_ + "config.json");
  d_ = model::dflash_desc_from_config(cfg);
  if (!d_.lilicorr()) refuse(dir_, "config.json is a " + d_.architecture + ", not a LiLiCorrDraftModel");
  const common::json::Value* qc = cfg.find("quantization_config");
  const bool has_qc = qc && !qc->is_null();
  open_and_check();
  if (!nvfp4_.empty() && !has_qc)
    refuse(dir_, "NVFP4 tensors (" + nvfp4_.front() + ") without a quantization_config in config.json");
}

void LiliCheckpoint::open_and_check() {
  for (const char* f : {"mask_embedding.pt", "mask_embedding.safetensors"})
    if (std::ifstream(dir_ + f).good())
      refuse(dir_, std::string("ships ") + f + ": the LiLiCorr mask row is the target's embedding row " +
                       std::to_string(d_.mask_id) + " - refusing to guess what the file would mean");
  set_ = std::make_unique<SafetensorsSet>(dir_, true);
  std::map<std::string, std::string> canon;
  for (const auto& [name, t] : set_->tensors()) {
    (void)t;
    const std::string c = dflash_canon(name);
    for (const char* own : {"embed_tokens.weight", "lm_head.weight"})
      if (c == own) refuse(dir_, "ships " + name + ": LiLiCorr shares the target's embedding and head (spec 26 §1)");
    if (c == "d2t" || c == "t2d" || c.find(".d2t") != std::string::npos || c.find(".t2d") != std::string::npos)
      refuse(dir_, "ships " + name + ": a drafter vocabulary of its own (d2t / t2d) is not supported");
    if (c.find("mask_hidden") != std::string::npos) refuse(dir_, "ships " + name + ": DFlash ships no mask_hidden");
    if (!canon.emplace(c, name).second) refuse(dir_, "two tensors canonicalise to " + c);
  }
  canon_.assign(canon.begin(), canon.end());
  auto info = [&](const std::string& c) -> const TensorInfo& { return set_->tensors().at(canon.at(c)); };

  // fc's K first (every other shape is the config's).
  if (!canon.count("fc.weight")) refuse(dir_, "missing tensor fc.weight");
  {
    const TensorInfo& t = info("fc.weight");
    if (t.shape.size() != 2) refuse(dir_, "fc.weight is not 2-D");
    fc_k_ = uint32_t(t.dtype == "U8" ? 2 * t.shape[1] : t.shape[1]);
    if (fc_k_ == 0 || fc_k_ % d_.taps.size()) refuse(dir_, "fc.weight must be [hidden, taps x target hidden]");
  }
  const auto lin = linear_shapes(d_, fc_k_);
  const auto other = other_shapes(d_);
  // NVFP4 parts: each belongs to a linear whose weight is U8.
  for (const auto& [c, name] : canon) {
    for (const char* sfx : {".weight_scale_2", ".weight_scale", ".input_scale"}) {
      if (!ends_with(c, sfx)) continue;
      const std::string base = c.substr(0, c.size() - std::strlen(sfx));
      if (!lin.count(base)) refuse(dir_, "unknown tensor " + name + " (no linear " + base + ")");
      if (!canon.count(base + ".weight") || info(base + ".weight").dtype != "U8")
        refuse(dir_, name + ": a quantised tensor outside a complete NVFP4 linear (" + base + ".weight is not U8)");
      break;
    }
  }
  // the names both ways
  for (const auto& [base, nk] : lin)
    if (!canon.count(base + ".weight")) refuse(dir_, "missing tensor " + base + ".weight");
  for (const auto& [n, s] : other)
    if (!canon.count(n)) refuse(dir_, "missing tensor " + n);
  for (const auto& [c, name] : canon) {
    bool known = other.count(c) != 0;
    if (!known && ends_with(c, ".weight")) known = lin.count(c.substr(0, c.size() - 7)) != 0;
    for (const char* sfx : {".weight_scale_2", ".weight_scale", ".input_scale"})
      known |= ends_with(c, sfx);
    if (!known) refuse(dir_, "unknown tensor " + name);
  }
  // dtypes and shapes
  for (const auto& [n, s] : other) {
    const TensorInfo& t = info(n);
    if (t.dtype != "BF16") refuse(dir_, canon.at(n) + " is " + t.dtype + ", not BF16");
    if (t.shape != s) refuse(dir_, canon.at(n) + " has shape " + shape_str(t.shape) + ", not the config's " + shape_str(s));
  }
  for (const auto& [base, nk] : lin) {
    const auto [N, K] = nk;
    const TensorInfo& w = info(base + ".weight");
    if (w.dtype == "U8") {
      if (K % 16) refuse(dir_, base + ": an NVFP4 linear's K must be a multiple of 16");
      if (w.shape != std::vector<uint64_t>{N, K / 2})
        refuse(dir_, base + ".weight (NVFP4) has shape " + shape_str(w.shape) + ", not [" + std::to_string(N) + ", " +
                         std::to_string(K / 2) + "]");
      for (const char* need : {".weight_scale", ".weight_scale_2"})
        if (!canon.count(base + need)) refuse(dir_, "missing tensor " + base + need + " (NVFP4 " + base + ")");
      const TensorInfo& s = info(base + ".weight_scale");
      if (s.dtype != "F8_E4M3" || s.shape != std::vector<uint64_t>{N, K / 16})
        refuse(dir_, base + ".weight_scale must be F8_E4M3 [" + std::to_string(N) + ", " + std::to_string(K / 16) + "]");
      const TensorInfo& g = info(base + ".weight_scale_2");
      if (g.dtype != "F32" || g.end - g.begin != 4)
        refuse(dir_, base + ".weight_scale_2 must be one F32 (the tensor's global scale)");
      nvfp4_.push_back(base);
    } else {
      if (w.dtype != "BF16") refuse(dir_, base + ".weight is " + w.dtype + ", not BF16 or NVFP4 (U8)");
      if (w.shape != std::vector<uint64_t>{N, K})
        refuse(dir_, base + ".weight has shape " + shape_str(w.shape) + ", not [" + std::to_string(N) + ", " +
                         std::to_string(K) + "]");
      for (const char* sfx : {".weight_scale", ".weight_scale_2", ".input_scale"})
        if (canon.count(base + sfx)) refuse(dir_, base + sfx + " beside a BF16 weight");
    }
  }
  std::sort(nvfp4_.begin(), nvfp4_.end());
}

bool LiliCheckpoint::is_nvfp4(const std::string& base) const {
  return std::binary_search(nvfp4_.begin(), nvfp4_.end(), base);
}

uint64_t LiliCheckpoint::backbone_nvfp4_rows() const {
  const auto lin = linear_shapes(d_, fc_k_);
  uint64_t rows = 0;
  for (const std::string& b : nvfp4_)
    if (b.rfind("lilicorr.", 0) != 0) rows += lin.at(b).first;
  return rows;
}

void LiliCheckpoint::check_target(const model::DFlashTargetShape& target) const {
  model::check_dflash_target(d_, target);
  if (fc_k_ != d_.fc_k(target.hidden))
    refuse(dir_, "fc.weight's K " + std::to_string(fc_k_) + " is not " + std::to_string(d_.taps.size()) +
                     " taps x the target's hidden " + std::to_string(target.hidden));
}

const uint8_t* LiliCheckpoint::raw(const std::string& c) const {
  auto it = std::lower_bound(canon_.begin(), canon_.end(), std::make_pair(c, std::string()));
  if (it == canon_.end() || it->first != c) refuse(dir_, "no tensor " + c);
  return set_->data(set_->tensors().at(it->second));
}

std::vector<float> LiliCheckpoint::widen(const std::string& c, size_t n) const {
  const uint16_t* p = reinterpret_cast<const uint16_t*>(raw(c));
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = common::bf16_to_f32(p[i]);
  return v;
}

LiliDecoded LiliCheckpoint::decoded(const std::string& base) const {
  const auto lin = linear_shapes(d_, fc_k_);
  LiliDecoded o;
  if (base.size() > 13 && ends_with(base, ".attn.in_proj")) {   // the lattice attention's in_proj_weight (BF16)
    o.N = 3 * d_.lil_hidden;
    o.K = d_.lil_hidden;
    const uint16_t* p = reinterpret_cast<const uint16_t*>(raw(base + "_weight"));
    o.wp.assign(p, p + size_t(o.N) * o.K);
    return o;
  }
  auto it = lin.find(base);
  if (it == lin.end()) refuse(dir_, "no linear " + base);
  o.N = it->second.first;
  o.K = it->second.second;
  if (is_nvfp4(base)) {
    o.nvfp4 = true;
    o.wp = decode_nvfp4(raw(base + ".weight"), raw(base + ".weight_scale"), o.K, o.N, base);
    std::memcpy(&o.g, raw(base + ".weight_scale_2"), 4);
    if (!std::isfinite(o.g)) refuse(dir_, base + ".weight_scale_2 is not finite");
  } else {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(raw(base + ".weight"));
    o.wp.assign(p, p + size_t(o.N) * o.K);
  }
  return o;
}

DFlashHostLinear LiliCheckpoint::make_linear(const std::vector<std::string>& bases, DraftFormat f,
                                             unsigned threads) const {
  if (f == DraftFormat::Bf16)
    refuse(dir_, "--draft-format bf16 is not a LiLiCorr format: exact is (W' bf16 + g, spec 26 §3)");
  DFlashHostLinear h;
  bool any_nvfp4 = false;
  std::vector<LiliDecoded> parts;
  for (const std::string& b : bases) {
    parts.push_back(decoded(b));
    any_nvfp4 |= parts.back().nvfp4;
    if (parts.back().K != parts.front().K) refuse(dir_, "the parts of " + bases[0] + " differ in K");
    if (parts.back().N % 16) refuse(dir_, b + "'s N is not a multiple of 16 (the GEMV tiles)");
  }
  const uint32_t K = parts.front().K;
  if (K % 16) refuse(dir_, bases[0] + "'s K is not a multiple of 16 (the GEMV tiles)");
  h.K = K;
  for (const LiliDecoded& p : parts) {
    const uint32_t N = p.N;
    h.N += N;
    if (f == DraftFormat::Exact) {
      h.kind = model::WeightKind::Bf16;
      std::vector<uint16_t> t(p.wp.size());
      common::repack_bf16_tiled(p.wp.data(), K, N, t.data());
      h.bf16.insert(h.bf16.end(), t.begin(), t.end());
      if (any_nvfp4) h.gs.insert(h.gs.end(), N, p.g);
    } else if (f == DraftFormat::Int8) {
      h.kind = model::WeightKind::Int8;
      std::vector<int8_t> q(p.wp.size());
      std::vector<float> s(N);
      quantise_int8_tiled_g(p.wp.data(), p.g, K, N, q.data(), s.data(), threads);
      h.i8.insert(h.i8.end(), q.begin(), q.end());
      h.scales.insert(h.scales.end(), s.begin(), s.end());
    } else {
      if (K % 64) refuse(dir_, "an int4 g64 linear's K is not a multiple of 64: " + bases[0]);
      h.kind = model::WeightKind::Int4;
      std::vector<uint32_t> qw(size_t(K / 8) * N), tiles(size_t(N / 16) * (K / 64) * 136);
      std::vector<uint16_t> sc(size_t(K / 64) * N);
      if (p.nvfp4)
        rtn_int4_g64_g(p.wp.data(), double(p.g), K, N, qw.data(), sc.data());
      else
        rtn_int4_g64(p.wp.data(), K, N, qw.data(), sc.data());
      common::repack_int4_layout1(qw.data(), sc.data(), K, N, tiles.data());
      h.int4.insert(h.int4.end(), tiles.begin(), tiles.end());
    }
  }
  return h;
}

DFlashHostLinear LiliCheckpoint::fc(DraftFormat f, unsigned threads) const { return make_linear({"fc"}, f, threads); }

DFlashHostLinear LiliCheckpoint::layer_linear(uint32_t l, DFlashLinear w, DraftFormat f, unsigned threads) const {
  if (l >= d_.layers) refuse(dir_, "no layer " + std::to_string(l));
  const std::string L = "layers." + std::to_string(l) + ".";
  switch (w) {
    case DFlashLinear::Qkv:
      return make_linear({L + "self_attn.q_proj", L + "self_attn.k_proj", L + "self_attn.v_proj"}, f, threads);
    case DFlashLinear::O: return make_linear({L + "self_attn.o_proj"}, f, threads);
    case DFlashLinear::GateUp: return make_linear({L + "mlp.gate_proj", L + "mlp.up_proj"}, f, threads);
    case DFlashLinear::Down: return make_linear({L + "mlp.down_proj"}, f, threads);
    case DFlashLinear::AttnKproj: return make_linear({L + "attention_conv.kernel_projection"}, f, threads);
    case DFlashLinear::MlpKproj: return make_linear({L + "mlp_conv.kernel_projection"}, f, threads);
  }
  refuse(dir_, "unknown linear");
}

std::vector<float> LiliCheckpoint::hidden_norm() const { return widen("hidden_norm.weight", d_.hidden); }
std::vector<float> LiliCheckpoint::final_norm() const { return widen("norm.weight", d_.hidden); }
std::vector<float> LiliCheckpoint::layer_norms(uint32_t l) const {
  const std::string L = "layers." + std::to_string(l) + ".";
  std::vector<float> v = widen(L + "input_layernorm.weight", d_.hidden);
  const std::vector<float> p = widen(L + "post_attention_layernorm.weight", d_.hidden);
  v.insert(v.end(), p.begin(), p.end());
  return v;
}
std::vector<float> LiliCheckpoint::qk_norms() const {
  std::vector<float> v;
  for (uint32_t l = 0; l < d_.layers; ++l)
    for (const char* n : {"q_norm.weight", "k_norm.weight"}) {
      const std::vector<float> x = widen("layers." + std::to_string(l) + ".self_attn." + n, d_.head_dim);
      v.insert(v.end(), x.begin(), x.end());
    }
  return v;
}
std::vector<float> LiliCheckpoint::conv_bases() const {
  std::vector<float> v;
  for (uint32_t l = 0; l < d_.layers; ++l)
    for (const char* cv : {"attention_conv", "mlp_conv"}) {
      const std::vector<float> x =
          widen("layers." + std::to_string(l) + "." + cv + ".base_kernel", size_t(2) * d_.conv_taps * d_.hidden);
      v.insert(v.end(), x.begin(), x.end());
    }
  return v;
}
std::vector<float> LiliCheckpoint::sinks() const {
  std::vector<float> v;
  for (uint32_t l = 0; l < d_.layers; ++l) {
    const std::vector<float> x = widen("layers." + std::to_string(l) + ".self_attn.attention_sink_bias", d_.q_heads);
    v.insert(v.end(), x.begin(), x.end());
  }
  return v;
}

LiliHostHead LiliCheckpoint::head_linear(LiliHead which, uint32_t layer) const {
  const std::string P = "lilicorr.", L = P + "layers." + std::to_string(layer) + ".";
  std::vector<std::string> bases;
  std::vector<std::string> biases;
  switch (which) {
    case LiliHead::TokenProj: bases = {P + "token_proj"}; break;
    case LiliHead::PassHidden: bases = {P + "pass_hidden_proj"}; break;
    case LiliHead::FeatUp: bases = {P + "feature_mlp.up_proj"}; break;
    case LiliHead::FeatDown: bases = {P + "feature_mlp.down_proj"}; break;
    case LiliHead::ContextProj: bases = {P + "context_proj"}; break;
    case LiliHead::InProj: bases = {L + "attn.in_proj"}; biases = {L + "attn.in_proj_bias"}; break;
    case LiliHead::OutProj: bases = {L + "attn.out_proj"}; break;
    case LiliHead::MlpUp: bases = {L + "mlp.up_proj"}; break;
    case LiliHead::MlpDown: bases = {L + "mlp.down_proj"}; break;
    case LiliHead::Factor: bases = {P + "factor_input_proj"}; break;
    case LiliHead::OutIn: bases = {P + "out_head", P + "in_head"}; break;
    case LiliHead::AnchorOut: bases = {P + "anchor_out_head"}; break;
  }
  if (which == LiliHead::TokenProj && !lili_has_token_proj(d_))
    refuse(dir_, "no token_proj: lilicorr_hidden_size equals hidden_size (the embedding rows are the tokens)");
  if ((which == LiliHead::InProj || which == LiliHead::OutProj || which == LiliHead::MlpUp ||
       which == LiliHead::MlpDown) && layer >= d_.lil_layers)
    refuse(dir_, "the LiLiCorr head has no layer " + std::to_string(layer));
  if (biases.empty())
    for (const std::string& b : bases) biases.push_back(b + ".bias");
  LiliHostHead h;
  for (size_t i = 0; i < bases.size(); ++i) {
    const LiliDecoded p = decoded(bases[i]);
    if (h.K && p.K != h.K) refuse(dir_, "the parts of " + bases[0] + " differ in K");
    h.K = p.K;
    const std::vector<float> b = widen(biases[i], p.N);
    h.gs.insert(h.gs.end(), p.N, p.g);
    h.bias.insert(h.bias.end(), b.begin(), b.end());
    h.N += p.N;
  }
  h.wt.assign(size_t(h.K) * h.N, 0);
  uint32_t n0 = 0;
  for (const std::string& b : bases) {   // W'^T: element (k, n) at k N + n
    const LiliDecoded p = decoded(b);
    for (uint32_t n = 0; n < p.N; ++n)
      for (uint32_t k = 0; k < p.K; ++k) h.wt[size_t(k) * h.N + n0 + n] = p.wp[size_t(n) * p.K + k];
    n0 += p.N;
  }
  const LiliShape want = lili_head_shape(d_, which);
  if (h.K != want.K || h.N != want.N) refuse(dir_, bases[0] + ": the head linear's shape is not the layout's");
  return h;
}

std::vector<float> LiliCheckpoint::head_small() const {
  const LiliHeadSmall s = lili_head_small(d_);
  const std::string P = "lilicorr.";
  const uint32_t Hl = d_.lil_hidden;
  std::vector<float> v(s.total, 0.0f);
  auto put = [&](size_t at, const std::string& n, size_t count) {
    const std::vector<float> x = widen(n, count);
    std::copy(x.begin(), x.end(), v.begin() + long(at));
  };
  put(s.fn_w, P + "feature_norm.weight", 5);
  put(s.fn_b, P + "feature_norm.bias", 5);
  put(s.slot, P + "slot_embedding", size_t(d_.lil_slots()) * Hl);
  put(s.rank, P + "rank_embedding", size_t(d_.cand_topk) * Hl);
  put(s.rel, P + "relative_slot_bias", size_t(d_.lil_heads) * (2 * d_.block - 1));
  put(s.same, P + "same_slot_bias", d_.lil_heads);
  for (uint32_t l = 0; l < d_.lil_layers; ++l) {
    const std::string L = P + "layers." + std::to_string(l) + ".";
    put(s.attn_norm(l, Hl), L + "attn_norm.weight", Hl);
    put(s.mlp_norm(l, Hl), L + "mlp_norm.weight", Hl);
  }
  put(s.out_norm, P + "output_norm.weight", Hl);
  put(s.anchor_norm, P + "anchor_norm.weight", Hl);
  return v;
}

}  // namespace loader
