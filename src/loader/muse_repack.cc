#include "loader/muse_repack.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "common/bf16.h"
#include "common/repack.h"
#include "loader/muse_layout.h"
#include "loader/snapshot.h"

namespace loader {
namespace {

using model::MuseGlimmerDesc;
using model::MuseLinearId;

const char* const kNorms[] = {"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                              "post_feedforward_layernorm"};

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }
bool ends_with(const std::string& s, const char* suf) {
  const size_t n = std::strlen(suf);
  return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

common::json::Value read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("load_muse_glimmer: cannot read " + path);
  std::stringstream s;
  s << f.rdbuf();
  return common::json::parse(s.str());
}

// The layer index of a `model.language_model.layers.<l>.` name, or -1.
long layer_of(const std::string& name) {
  static const std::string p = std::string(MuseGlimmerDesc::text_prefix()) + "layers.";
  if (name.rfind(p, 0) != 0) return -1;
  return std::stol(name.substr(p.size()));
}

}  // namespace

std::string resolve_muse_snapshot(const std::string& arg) {
  static const std::string a = "Vishva007/Muse-Glimmer-30B-W4A16-AutoRound", g = a + "-GPTQ";
  try {
    return resolve_snapshot(arg);
  } catch (const std::exception& e) {
    if (arg != a && arg != g) throw;
    const std::string other = arg == a ? g : a;
    try {
      return resolve_snapshot(other);
    } catch (const std::exception&) {
      throw std::runtime_error(std::string(e.what()) + " - nor its other spelling " + other +
                               " (spec 23 §3.1: either Vishva007 repo is the same export)");
    }
  }
}

const char* muse_quant_form_name(MuseQuantForm f) { return f == MuseQuantForm::Gptq ? "gptq" : "autoround"; }

bool muse_is_text(const std::string& name) {
  return starts_with(name, MuseGlimmerDesc::text_prefix()) || name == "lm_head.weight";
}
bool muse_is_vision(const std::string& name) {
  return starts_with(name, "model.vision_tower.") || starts_with(name, "model.vision_adapter.") ||
         starts_with(name, "model.vision_projection.");
}

std::vector<std::string> muse_expected_names(const MuseGlimmerDesc& d, MuseQuantForm f) {
  const std::string lm = MuseGlimmerDesc::text_prefix();
  std::vector<std::string> v = {lm + "embed_tokens.weight", lm + "norm.weight", "lm_head.weight"};
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string p = MuseGlimmerDesc::layer_prefix(l);
    for (const char* n : kNorms) v.push_back(p + n + ".weight");
    for (MuseLinearId id : d.layer_linears())
      for (const std::string& part : d.linear(id).parts) {
        v.push_back(p + part + ".qweight");
        v.push_back(p + part + ".scales");
        v.push_back(p + part + ".qzeros");
        if (f == MuseQuantForm::Gptq) v.push_back(p + part + ".g_idx");
      }
  }
  std::sort(v.begin(), v.end());
  return v;
}

std::vector<float> MuseHostLayer::norm_block() const {
  std::vector<float> b;
  b.reserve(in_norm.size() * 4);
  for (const std::vector<float>* n : {&in_norm, &post_attn, &pre_ff, &post_ff}) b.insert(b.end(), n->begin(), n->end());
  return b;
}

size_t MuseHostLayer::bytes() const {
  return qgkv.bytes() + oproj.bytes() + gate_up.bytes() + down.bytes() +
         (in_norm.size() + post_attn.size() + pre_ff.size() + post_ff.size()) * 4;
}

MuseCheckpoint::MuseCheckpoint(const std::string& snapshot, uint32_t layers_override) : snap_(snapshot) {
  if (!snap_.empty() && snap_.back() != '/') snap_ += '/';
  const common::json::Value cfg = read_json(snap_ + "config.json");
  // The structure at the checkpoint's own layer count (every name of every layer is accounted for), then the
  // development-mode truncation.
  d_ = model::check_muse_config(cfg);
  ckpt_layers_ = d_.layers;
  open(cfg);
  if (layers_override != 0) {
    d_ = model::check_muse_config(cfg, layers_override);
    for (const auto& [name, info] : set_->tensors()) {
      (void)info;
      if (layer_of(name) >= long(d_.layers) && consumed_.insert(name).second) ++skipped_;
    }
    text_ -= skipped_;
  }
}

MuseCheckpoint::MuseCheckpoint(const MuseGlimmerDesc& d, const std::string& snapshot) : snap_(snapshot), d_(d) {
  if (!snap_.empty() && snap_.back() != '/') snap_ += '/';
  if (d_.full.size() != d_.layers)
    throw std::invalid_argument("MuseCheckpoint: the descriptor's `full` has " + std::to_string(d_.full.size()) +
                                " entries for " + std::to_string(d_.layers) + " layers");
  ckpt_layers_ = d_.layers;
  open(read_json(snap_ + "config.json"));
}

void MuseCheckpoint::open(const common::json::Value& cfg) {
  qc_ = QuantConfig::parse(cfg);   // bits, group, sym, desc_act - refused by name
  set_ = std::make_unique<SafetensorsSet>(snap_);
  // The spelling from the names: any text `.g_idx` is the gptq export; check_names then holds EVERY linear to it.
  form_ = MuseQuantForm::AutoRound;
  for (const auto& [name, info] : set_->tensors()) {
    (void)info;
    if (muse_is_text(name) && ends_with(name, ".g_idx")) {
      form_ = MuseQuantForm::Gptq;
      break;
    }
  }
  check_names();
  // docs/13's data asserts over every tensor: qzeros 0x77777777, g_idx i // group, finite scales.
  scan_ = assert_quant_invariants(*set_);
  if (!qc_.desc_act_declared && scan_.g_idx_tensors != 0)
    throw std::runtime_error("load_muse_glimmer: config.json declares no desc_act (the auto-round spelling) but the "
                             "checkpoint ships " + std::to_string(scan_.g_idx_tensors) +
                             " g_idx tensors - an activation order nobody wrote down");
  check_quant_scan(qc_, scan_);
  if (scan_.ct_linears != 0)
    throw std::runtime_error("load_muse_glimmer: compressed-tensors linears - spec 23 serves the int4 g64 GPTQ export "
                             "only (either Vishva007 spelling, §3.1)");
}

void MuseCheckpoint::check_names() {
  const std::vector<std::string> exp = muse_expected_names(d_, form_);
  const std::set<std::string> want(exp.begin(), exp.end());
  std::string bad;
  size_t n_bad = 0;
  size_t text = 0, vision = 0;
  for (const auto& [name, info] : set_->tensors()) {
    (void)info;
    if (want.count(name)) {
      ++text;
      continue;
    }
    if (muse_is_vision(name)) {
      ++vision;
      continue;
    }
    if (n_bad++ < 5) bad += (bad.empty() ? "" : ", ") + name;
  }
  if (n_bad != 0)
    throw std::runtime_error("load_muse_glimmer: the checkpoint has " + std::to_string(n_bad) +
                             " tensor(s) Muse-Glimmer-30B (model/muse_glimmer.h, the " + muse_quant_form_name(form_) +
                             " spelling) does not describe: " + bad + (n_bad > 5 ? ", ..." : "") +
                             " - refusing to guess what they are (a bf16 or quantised head / linear, another packing, an "
                             "extra layer or norm are all outside spec 23's one format)");
  for (const std::string& name : exp)
    if (!set_->tensors().count(name))
      throw std::runtime_error("load_muse_glimmer: the checkpoint has no tensor '" + name + "', which the " +
                               muse_quant_form_name(form_) + " spelling of Muse-Glimmer-30B needs");
  text_ = text;
  vision_ = vision;
}

void MuseCheckpoint::linear(const std::string& lp, MuseLinearId id, MuseHostLinear& out, size_t& src) const {
  const model::MuseLinear ln = d_.linear(id);
  const model::GemvShape& sh = ln.shape;
  std::vector<LinearSrc> srcs;
  std::vector<common::Part> parts;
  srcs.reserve(ln.parts.size());
  for (size_t i = 0; i < ln.parts.size(); ++i) {
    const std::string base = lp + ln.parts[i];
    LinearSrc s = LinearSrc::classify(*set_, base);
    if (s.kind != WKind::Int4 || s.packing != Int4Packing::Gptq)
      throw std::runtime_error("load_muse_glimmer: '" + base + "' is not an int4 GPTQ linear");
    if (s.K != sh.K || s.N != ln.part_n[i])
      throw std::runtime_error("load_muse_glimmer: '" + base + "' is K=" + std::to_string(s.K) + " N=" +
                               std::to_string(s.N) + ", the descriptor says K=" + std::to_string(sh.K) + " N=" +
                               std::to_string(ln.part_n[i]));
    for (const char* sfx : {".qweight", ".scales", ".qzeros", ".g_idx"}) consumed_.insert(base + sfx);
    src += size_t(s.K / 8) * s.N * 4 + size_t(s.K / s.group) * s.N * 2;
    srcs.push_back(std::move(s));
    parts.push_back({srcs.back().qweight, srcs.back().scales, srcs.back().N});
  }
  const std::vector<common::ColSource> cols = common::cols_concat(parts);   // the descriptor's column order
  if (sh.layout == 0) {
    out.words.resize(size_t(sh.K / 8) * sh.N);
    out.scales.resize(size_t(sh.K / 64) * sh.N);
    common::repack_int4_layout0_cols(sh.K, sh.N, cols, out.words.data(), out.scales.data());
  } else {
    out.words.resize(int4_layout1_bytes(sh.K, sh.N) / 4);
    out.scales.clear();
    common::repack_int4_layout1_cols(sh.K, sh.N, cols, out.words.data());
  }
}

const uint16_t* MuseCheckpoint::bf16(const std::string& name, uint64_t elems) const {
  auto it = set_->tensors().find(name);
  if (it == set_->tensors().end()) throw std::runtime_error("load_muse_glimmer: no tensor '" + name + "'");
  const TensorInfo t = it->second;
  if (t.dtype != "BF16")
    throw std::runtime_error("load_muse_glimmer: '" + name + "' is " + t.dtype + ", expected BF16 (spec 23 §3.1)");
  uint64_t n = 1;
  for (uint64_t x : t.shape) n *= x;
  if (n != elems)
    throw std::runtime_error("load_muse_glimmer: '" + name + "' has " + std::to_string(n) + " elements, expected " +
                             std::to_string(elems));
  const uint8_t* p = set_->data(t);
  check_align(p, 2, name);
  consumed_.insert(name);
  return reinterpret_cast<const uint16_t*>(p);
}

void MuseCheckpoint::layer(uint32_t l, MuseHostLayer& out) const {
  if (l >= d_.layers)
    throw std::out_of_range("load_muse_glimmer: layer " + std::to_string(l) + " of " + std::to_string(d_.layers));
  const std::string lp = MuseGlimmerDesc::layer_prefix(l);
  out.src_bytes = 0;
  linear(lp, MuseLinearId::QGKV, out.qgkv, out.src_bytes);
  linear(lp, MuseLinearId::OProj, out.oproj, out.src_bytes);
  linear(lp, MuseLinearId::GateUp, out.gate_up, out.src_bytes);
  linear(lp, MuseLinearId::Down, out.down, out.src_bytes);
  std::vector<float>* dst[] = {&out.in_norm, &out.post_attn, &out.pre_ff, &out.post_ff};
  for (int i = 0; i < 4; ++i) {
    const uint16_t* w = bf16(lp + kNorms[i] + ".weight", d_.hidden);
    dst[i]->resize(d_.hidden);
    for (uint32_t k = 0; k < d_.hidden; ++k) (*dst[i])[k] = common::bf16_to_f32(w[k]);   // exact widening
    out.src_bytes += size_t(d_.hidden) * 2;
  }
}

MuseHostLayer MuseCheckpoint::layer(uint32_t l) const {
  MuseHostLayer h;
  layer(l, h);
  return h;
}

const uint16_t* MuseCheckpoint::embed() const {
  return bf16(std::string(MuseGlimmerDesc::text_prefix()) + "embed_tokens.weight", uint64_t(d_.vocab) * d_.hidden);
}

const uint16_t* MuseCheckpoint::lm_head() const { return bf16("lm_head.weight", uint64_t(d_.vocab) * d_.hidden); }

std::vector<float> MuseCheckpoint::final_norm() const {
  const uint16_t* w = bf16(std::string(MuseGlimmerDesc::text_prefix()) + "norm.weight", d_.hidden);
  std::vector<float> f(d_.hidden);
  for (uint32_t i = 0; i < d_.hidden; ++i) f[i] = common::bf16_to_f32(w[i]);
  return f;
}

std::vector<std::string> MuseCheckpoint::unconsumed() const {
  std::vector<std::string> v;
  for (const auto& [name, info] : set_->tensors()) {
    (void)info;
    if (consumed_.count(name) || muse_is_vision(name)) continue;
    v.push_back(name);
  }
  return v;
}

}  // namespace loader
