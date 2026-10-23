#include "loader/dflash_repack.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "common/bf16.h"
#include "common/json.h"
#include "common/repack.h"
#include "loader/lm_head_int8.h"
#include "loader/rtn.h"
#include "loader/snapshot.h"

// dflash_repack.cc - the host half of the DFlash drafter loader (dflash_repack.h has the contract).
namespace loader {
namespace {

[[noreturn]] void refuse(const std::string& dir, const std::string& what) {
  throw std::runtime_error("DFlash drafter " + dir + ": " + what);
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

bool exists(const std::string& p) { return bool(std::ifstream(p)); }

}  // namespace

const char* draft_format_name(DraftFormat f) {
  return f == DraftFormat::Int8 ? "int8" : f == DraftFormat::Int4G64 ? "int4" : f == DraftFormat::Exact ? "exact" : "bf16";
}
bool parse_draft_format(const std::string& s, DraftFormat& out) {
  if (s == "bf16") return out = DraftFormat::Bf16, true;
  if (s == "int8") return out = DraftFormat::Int8, true;
  if (s == "int4" || s == "int4g64") return out = DraftFormat::Int4G64, true;
  return false;
}
bool parse_lilicorr_draft_format(const std::string& s, DraftFormat& out) {
  if (s == "exact") return out = DraftFormat::Exact, true;
  return s != "bf16" && parse_draft_format(s, out);
}

std::string dflash_canon(const std::string& name) {
  std::string n = name;
  if (n.rfind("model.", 0) == 0) n = n.substr(6);
  for (const auto& [a, b] : {std::pair<const char*, const char*>{"midlayer.", "layers.0."},
                             {"encoder.output_norm_enc.", "hidden_norm."},
                             {"encoder.fc.", "fc."}}) {
    for (size_t at = n.find(a); at != std::string::npos; at = n.find(a, at + std::strlen(b)))
      n.replace(at, std::strlen(a), b);
  }
  return n;
}

std::vector<std::string> dflash_expected_names(const model::DFlashDesc& d) {
  std::vector<std::string> v = {"fc.weight", "hidden_norm.weight", "norm.weight"};
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string L = "layers." + std::to_string(l) + ".";
    for (const char* n : {"input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_proj.weight",
                          "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                          "self_attn.q_norm.weight", "self_attn.k_norm.weight", "mlp.gate_proj.weight",
                          "mlp.up_proj.weight", "mlp.down_proj.weight"})
      v.push_back(L + n);
    if (d.v2())
      for (const char* n : {"attention_conv.base_kernel", "attention_conv.kernel_projection.weight",
                            "mlp_conv.base_kernel", "mlp_conv.kernel_projection.weight"})
        v.push_back(L + n);
  }
  if (d.v2())
    for (const char* n : {"candidate_selector.hidden_projection.weight", "candidate_selector.predecessor_codebook",
                          "candidate_selector.successor_codebook"})
      v.push_back(n);
  std::sort(v.begin(), v.end());
  return v;
}

DFlashCheckpoint::DFlashCheckpoint(const std::string& snapshot_or_repo)
    : dir_(with_slash(resolve_snapshot(snapshot_or_repo))) {
  d_ = model::dflash_desc_from_config(read_json(dir_ + "config.json"));
  open_and_check();
}

DFlashCheckpoint::DFlashCheckpoint(const model::DFlashDesc& d, const std::string& snapshot_dir)
    : d_(d), dir_(with_slash(snapshot_dir)) {
  open_and_check();
}

void DFlashCheckpoint::open_and_check() {
  if (exists(dir_ + "mask_embedding.pt"))
    refuse(dir_, "ships mask_embedding.pt (a torch pickle, not read here): convert it with "
                 "tools/oracle/dflash_mask_embedding.py to mask_embedding.safetensors");
  set_ = std::make_unique<SafetensorsSet>(dir_, true);
  // Canonical names, refusing what a drafter must not ship.
  std::map<std::string, std::string> canon;
  for (const auto& [name, t] : set_->tensors()) {
    (void)t;
    const std::string c = dflash_canon(name);
    for (const char* own : {"embed_tokens.weight", "lm_head.weight"})
      if (c == own)
        refuse(dir_, "ships " + name + ": a DFlash drafter shares the target's embedding and head and must not own "
                     "a copy (spec 19 §2)");
    if (c == "d2t" || c == "t2d" || c.find(".d2t") != std::string::npos || c.find(".t2d") != std::string::npos)
      refuse(dir_, "ships " + name + ": a drafter vocabulary of its own (d2t / t2d) is not supported");
    if (c.find("mask_hidden") != std::string::npos)
      refuse(dir_, "ships " + name + ": DFlash ships no mask_hidden (qwen3_dflash.py:872)");
    if (!canon.emplace(c, name).second) refuse(dir_, "two tensors canonicalise to " + c);
  }
  const std::vector<std::string> want = dflash_expected_names(d_);
  for (const std::string& n : want)
    if (!canon.count(n)) refuse(dir_, "missing tensor " + n);
  for (const auto& [c, name] : canon)
    if (!std::binary_search(want.begin(), want.end(), c)) refuse(dir_, "unknown tensor " + name);
  canon_.assign(canon.begin(), canon.end());

  // dtypes and shapes.
  const uint32_t H = d_.hidden;
  const auto& fcs = set_->tensors().at(canon.at("fc.weight")).shape;
  if (fcs.size() != 2 || fcs[0] != H || fcs[1] == 0 || fcs[1] % d_.taps.size())
    refuse(dir_, "fc.weight must be [hidden, taps x target hidden]");
  fc_k_ = uint32_t(fcs[1]);
  std::map<std::string, std::vector<uint64_t>> shape = {{"fc.weight", {H, fc_k_}},
                                                        {"hidden_norm.weight", {H}},
                                                        {"norm.weight", {H}}};
  for (uint32_t l = 0; l < d_.layers; ++l) {
    const std::string L = "layers." + std::to_string(l) + ".";
    shape[L + "input_layernorm.weight"] = {H};
    shape[L + "post_attention_layernorm.weight"] = {H};
    shape[L + "self_attn.q_proj.weight"] = {d_.q_n(), H};
    shape[L + "self_attn.k_proj.weight"] = {d_.kv_n(), H};
    shape[L + "self_attn.v_proj.weight"] = {d_.kv_n(), H};
    shape[L + "self_attn.o_proj.weight"] = {H, d_.q_n()};
    shape[L + "self_attn.q_norm.weight"] = {d_.head_dim};
    shape[L + "self_attn.k_norm.weight"] = {d_.head_dim};
    shape[L + "mlp.gate_proj.weight"] = {d_.inter, H};
    shape[L + "mlp.up_proj.weight"] = {d_.inter, H};
    shape[L + "mlp.down_proj.weight"] = {H, d_.inter};
    if (d_.v2())
      for (const char* cv : {"attention_conv.", "mlp_conv."}) {
        shape[L + cv + "base_kernel"] = {2, d_.conv_taps, H};
        shape[L + cv + "kernel_projection.weight"] = {d_.kproj_n(), H};
      }
  }
  if (d_.v2()) {
    shape["candidate_selector.hidden_projection.weight"] = {d_.sel_rank, H};
    shape["candidate_selector.predecessor_codebook"] = {d_.vocab, d_.sel_rank};
    shape["candidate_selector.successor_codebook"] = {d_.vocab, d_.sel_rank};
  }
  for (const auto& [c, name] : canon) {
    const TensorInfo& t = set_->tensors().at(name);
    if (t.dtype != "BF16") refuse(dir_, name + " is " + t.dtype + ", not BF16 (our formats: the bf16 drafter)");
    if (t.shape != shape.at(c)) {
      std::string got;
      for (uint64_t s : t.shape) got += (got.empty() ? "" : ", ") + std::to_string(s);
      refuse(dir_, name + " has shape [" + got + "], not the config's");
    }
  }

  // Spec 23f: a Muse-Glimmer assistant ships no mask embedding (the mask row is the target's embedding row 201818,
  // G:1551 / 1633-1638): one that ships a file is refused rather than guessed at.
  if (d_.muse() && exists(dir_ + "mask_embedding.safetensors"))
    refuse(dir_, "ships mask_embedding.safetensors: a Muse-Glimmer assistant embeds its mask with the target's row " +
                     std::to_string(d_.mask_id) + " (transformers' DFlashTokenCandidateGenerator) - refusing to guess "
                     "what the file would mean");
  // The mask embedding (Review Focus 4).
  if (exists(dir_ + "mask_embedding.safetensors")) {
    mask_path_ = dir_ + "mask_embedding.safetensors";
    MappedFile f(mask_path_);
    uint64_t hlen = 0;
    if (f.size() < 8) refuse(dir_, "mask_embedding.safetensors is truncated");
    std::memcpy(&hlen, f.data(), 8);
    if (hlen > f.size() - 8) refuse(dir_, "mask_embedding.safetensors: header length exceeds the file");
    const common::json::Value h =
        common::json::parse(std::string_view(reinterpret_cast<const char*>(f.data()) + 8, hlen));
    if (const common::json::Value* m = h.find("__metadata__"))
      if (const common::json::Value* id = m->find("mask_token_id"))
        if (!id->is_string() || id->str() != std::to_string(d_.mask_id))
          refuse(dir_, "mask_embedding.safetensors' mask_token_id does not match dflash_config.mask_token_id " +
                           std::to_string(d_.mask_id));
    const auto entries = SafetensorsSet::parse_header(f.data(), f.size());
    if (entries.size() != 1 || entries[0].first != "embedding")
      refuse(dir_, "mask_embedding.safetensors must hold exactly one tensor, `embedding`");
    const TensorInfo& t = entries[0].second;
    uint64_t n = 1;
    for (uint64_t s : t.shape) n *= s;
    if (n != H || (t.dtype != "BF16" && t.dtype != "F32"))
      refuse(dir_, "mask_embedding.safetensors' embedding must be [hidden] (or [1, hidden]) in BF16 or F32");
    if (t.end - t.begin != n * (t.dtype == "F32" ? 4 : 2) || 8 + hlen + t.end > f.size())
      refuse(dir_, "mask_embedding.safetensors' embedding has inconsistent data_offsets");
  }
}

void DFlashCheckpoint::check_target(const model::ModelDesc& target, uint32_t target_vocab_rows) const {
  check_target(model::DFlashTargetShape{target.name, target.layers, target.hidden, target_vocab_rows});
}

void DFlashCheckpoint::check_target(const model::DFlashTargetShape& target) const {
  model::check_dflash_target(d_, target);
  if (fc_k_ != d_.fc_k(target.hidden))
    refuse(dir_, "fc.weight's K " + std::to_string(fc_k_) + " is not " + std::to_string(d_.taps.size()) +
                     " taps x the target's hidden " + std::to_string(target.hidden));
}

const uint16_t* DFlashCheckpoint::bf16(const std::string& c) const {
  auto it = std::lower_bound(canon_.begin(), canon_.end(), std::make_pair(c, std::string()));
  if (it == canon_.end() || it->first != c) refuse(dir_, "no tensor " + c);
  const TensorInfo& t = set_->tensors().at(it->second);
  return reinterpret_cast<const uint16_t*>(set_->data(t));
}

DFlashHostLinear DFlashCheckpoint::make_linear(const std::vector<std::string>& parts, uint32_t K, DraftFormat f,
                                               unsigned threads) const {
  // Concatenate the parts' rows ([N_i][K] each, row-major) into one [N][K], then the device form.
  std::vector<uint16_t> rows;
  for (const std::string& p : parts) {
    auto it = std::lower_bound(canon_.begin(), canon_.end(), std::make_pair(p, std::string()));
    const TensorInfo& t = set_->tensors().at(it->second);
    const uint16_t* src = bf16(p);
    rows.insert(rows.end(), src, src + (t.end - t.begin) / 2);
  }
  if (f == DraftFormat::Exact) refuse(dir_, "--draft-format exact is the LiLiCorr drafter's (spec 26b): load bf16");
  DFlashHostLinear h;
  h.K = K;
  h.N = uint32_t(rows.size() / K);
  if (h.N % 16 || K % 16) refuse(dir_, "a linear's N or K is not a multiple of 16 (the GEMV tiles): " + parts[0]);
  if (f == DraftFormat::Bf16) {
    h.kind = model::WeightKind::Bf16;
    h.bf16.resize(rows.size());
    common::repack_bf16_tiled(rows.data(), K, h.N, h.bf16.data());
  } else if (f == DraftFormat::Int4G64) {
    if (K % 64) refuse(dir_, "an int4 g64 linear's K is not a multiple of 64: " + parts[0]);
    h.kind = model::WeightKind::Int4;
    std::vector<uint32_t> qweight(size_t(K / 8) * h.N);
    std::vector<uint16_t> scales(size_t(K / 64) * h.N);
    rtn_int4_g64(rows.data(), K, h.N, qweight.data(), scales.data());
    h.int4.resize(size_t(h.N / 16) * (K / 64) * 136);
    common::repack_int4_layout1(qweight.data(), scales.data(), K, h.N, h.int4.data());
  } else {
    h.kind = model::WeightKind::Int8;
    h.i8.resize(rows.size());
    h.scales.resize(h.N);
    quantise_int8_tiled(rows.data(), K, h.N, h.i8.data(), h.scales.data(), threads);
  }
  return h;
}

DFlashHostLinear DFlashCheckpoint::fc(DraftFormat f, unsigned threads) const {
  return make_linear({"fc.weight"}, fc_k_, f, threads);
}

DFlashHostLinear DFlashCheckpoint::layer_linear(uint32_t l, DFlashLinear w, DraftFormat f, unsigned threads) const {
  if (l >= d_.layers) refuse(dir_, "no layer " + std::to_string(l));
  const std::string L = "layers." + std::to_string(l) + ".";
  switch (w) {
    case DFlashLinear::Qkv:
      return make_linear({L + "self_attn.q_proj.weight", L + "self_attn.k_proj.weight",
                          L + "self_attn.v_proj.weight"},
                         d_.hidden, f, threads);
    case DFlashLinear::O:
      return make_linear({L + "self_attn.o_proj.weight"}, d_.q_n(), f, threads);
    case DFlashLinear::GateUp:
      return make_linear({L + "mlp.gate_proj.weight", L + "mlp.up_proj.weight"}, d_.hidden, f, threads);
    case DFlashLinear::Down:
      return make_linear({L + "mlp.down_proj.weight"}, d_.inter, f, threads);
    case DFlashLinear::AttnKproj:
    case DFlashLinear::MlpKproj:
      if (!d_.v2()) refuse(dir_, "the conv kernel projections are DFlash 2's");
      return make_linear({L + (w == DFlashLinear::AttnKproj ? "attention_conv" : "mlp_conv") +
                          ".kernel_projection.weight"},
                         d_.hidden, f, threads);
  }
  refuse(dir_, "unknown linear");
}

DFlashHostLinear DFlashCheckpoint::selector_projection(DraftFormat f, unsigned threads) const {
  if (!d_.v2()) refuse(dir_, "the candidate selector is DFlash 2's");
  return make_linear({"candidate_selector.hidden_projection.weight"}, d_.hidden, f, threads);
}

namespace {
void widen_into(const uint16_t* src, size_t n, std::vector<float>& out) {
  for (size_t i = 0; i < n; ++i) out.push_back(common::bf16_to_f32(src[i]));
}
}  // namespace

std::vector<float> DFlashCheckpoint::hidden_norm() const {
  std::vector<float> v;
  widen_into(bf16("hidden_norm.weight"), d_.hidden, v);
  return v;
}
std::vector<float> DFlashCheckpoint::final_norm() const {
  std::vector<float> v;
  widen_into(bf16("norm.weight"), d_.hidden, v);
  return v;
}
std::vector<float> DFlashCheckpoint::layer_norms(uint32_t l) const {
  const std::string L = "layers." + std::to_string(l) + ".";
  std::vector<float> v;
  widen_into(bf16(L + "input_layernorm.weight"), d_.hidden, v);
  widen_into(bf16(L + "post_attention_layernorm.weight"), d_.hidden, v);
  return v;
}
std::vector<float> DFlashCheckpoint::qk_norms() const {
  std::vector<float> v;
  for (uint32_t l = 0; l < d_.layers; ++l) {
    const std::string L = "layers." + std::to_string(l) + ".self_attn.";
    widen_into(bf16(L + "q_norm.weight"), d_.head_dim, v);
    widen_into(bf16(L + "k_norm.weight"), d_.head_dim, v);
  }
  return v;
}
std::vector<float> DFlashCheckpoint::conv_bases() const {
  std::vector<float> v;
  if (!d_.v2()) return v;
  for (uint32_t l = 0; l < d_.layers; ++l)
    for (const char* cv : {"attention_conv", "mlp_conv"})
      widen_into(bf16("layers." + std::to_string(l) + "." + cv + ".base_kernel"), size_t(2) * d_.conv_taps * d_.hidden,
                 v);   // [side][tap][hidden], as stored
  return v;
}
const uint16_t* DFlashCheckpoint::codebook(bool successor) const {
  if (!d_.v2()) refuse(dir_, "the codebooks are DFlash 2's");
  return bf16(successor ? "candidate_selector.successor_codebook" : "candidate_selector.predecessor_codebook");
}
std::vector<float> DFlashCheckpoint::mask_embedding() const {
  std::vector<float> v;
  if (mask_path_.empty()) return v;
  MappedFile f(mask_path_);
  uint64_t hlen = 0;
  std::memcpy(&hlen, f.data(), 8);
  const auto entries = SafetensorsSet::parse_header(f.data(), f.size());
  const TensorInfo& t = entries[0].second;
  const uint8_t* p = f.data() + 8 + hlen + t.begin;
  v.resize(d_.hidden);
  if (t.dtype == "F32") {
    std::memcpy(v.data(), p, size_t(d_.hidden) * 4);
  } else {
    for (uint32_t i = 0; i < d_.hidden; ++i) {
      uint16_t h;
      std::memcpy(&h, p + 2 * i, 2);
      v[i] = common::bf16_to_f32(h);
    }
  }
  return v;
}

}  // namespace loader
