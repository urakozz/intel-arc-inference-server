#include "loader/bolt_repack.h"

#include <stdexcept>

#include "common/bf16.h"
#include "common/exp_torch.h"
#include "common/repack.h"
#include "loader/bolt_layout.h"

namespace loader {
namespace {

void add_int4(std::vector<std::string>& v, const std::string& base) {
  v.push_back(base + ".qweight");
  v.push_back(base + ".qzeros");
  v.push_back(base + ".scales");
}

}  // namespace

std::string bolt_trunk_prefix(const SafetensorsSet& set) {
  const bool model = set.tensors().count("language_model.model.embeddings.weight") != 0;
  const bool backbone = set.tensors().count("language_model.backbone.embeddings.weight") != 0;
  if (model && backbone)
    throw std::runtime_error("load_bolt: the checkpoint ships the embedding under both 'language_model.model.' and "
                             "'language_model.backbone.' - one trunk spelling per checkpoint");
  if (!model && !backbone)
    throw std::runtime_error("load_bolt: no 'language_model.model.embeddings.weight' (nor 'language_model.backbone."
                             "embeddings.weight') in the checkpoint - not a Nemotron 3.5 Bolt export");
  return model ? "language_model.model." : "language_model.backbone.";
}

bool bolt_is_vision(const std::string& name) {
  for (const std::string& p : bolt_vision_prefixes())
    if (name.rfind(p, 0) == 0) return true;
  return false;
}

std::vector<std::string> bolt_expected_names(const model::BoltDesc& d, const std::string& trunk) {
  std::vector<std::string> v = {trunk + "embeddings.weight", trunk + "norm_f.weight", kBoltLmHead};
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string lp = model::BoltDesc::layer_prefix(trunk, l), m = lp + "mixer.";
    v.push_back(lp + "norm.weight");
    if (d.is_mamba(l)) {
      for (const char* s : {"A_log", "D", "dt_bias", "conv1d.weight", "conv1d.bias", "norm.weight"}) v.push_back(m + s);
      add_int4(v, m + "in_proj");
      add_int4(v, m + "out_proj");
    } else if (d.is_attention(l)) {
      for (const char* s : {"q_proj", "k_proj", "v_proj", "o_proj"}) add_int4(v, m + s);
    } else {
      v.push_back(m + "gate.weight");
      v.push_back(m + "gate.e_score_correction_bias");
      for (const char* s : {"fc1_latent_proj", "fc2_latent_proj", "shared_experts.up_proj", "shared_experts.down_proj"})
        add_int4(v, m + s);
      for (uint32_t e = 0; e < d.experts.at(l); ++e) {
        add_int4(v, m + "experts." + std::to_string(e) + ".up_proj");
        add_int4(v, m + "experts." + std::to_string(e) + ".down_proj");
      }
    }
  }
  return v;
}

BoltHostLinear repack_bolt_linear(const model::BoltLinear& ln, const std::vector<LinearSrc>& part_srcs) {
  if (part_srcs.size() != ln.parts.size())
    throw std::invalid_argument("repack_bolt_linear: " + std::to_string(part_srcs.size()) + " sources for " +
                                std::to_string(ln.parts.size()) + " parts");
  std::vector<common::Part> parts;
  parts.reserve(part_srcs.size());
  uint32_t n = 0;
  for (const LinearSrc& s : part_srcs) {
    parts.push_back({s.qweight, s.scales, s.N});
    n += s.N;
  }
  if (n != ln.shape.N)
    throw std::invalid_argument("repack_bolt_linear: the parts' columns sum to " + std::to_string(n) + ", the descriptor says N=" +
                                std::to_string(ln.shape.N));
  const std::vector<common::ColSource> cols = common::cols_concat(parts);
  BoltHostLinear out;
  const uint32_t K = ln.shape.K, N = ln.shape.N;
  if (ln.shape.layout == 0) {
    out.words.resize(size_t(K / 8) * N);
    out.scales.resize(size_t(K / 64) * N);
    common::repack_int4_layout0_cols(K, N, cols, out.words.data(), out.scales.data());
  } else {
    out.words.resize(bolt::int4_l1_bytes(K, N) / 4);
    common::repack_int4_layout1_cols(K, N, cols, out.words.data());
  }
  return out;
}

void repack_bolt_expert(const LinearSrc& up, const LinearSrc& down, uint32_t* out) {
  if (up.K != down.N || up.N != down.K)
    throw std::invalid_argument("repack_bolt_expert: up [" + std::to_string(up.N) + "][" + std::to_string(up.K) +
                                "] and down [" + std::to_string(down.N) + "][" + std::to_string(down.K) + "] do not agree");
  common::repack_int4_layout1_cols(up.K, up.N, common::cols_concat({{up.qweight, up.scales, up.N}}), out);
  common::repack_int4_layout1_cols(down.K, down.N, common::cols_concat({{down.qweight, down.scales, down.N}}),
                                   out + bolt::expert_up_bytes(up.K, up.N) / 4);
}

std::vector<uint16_t> repack_bolt_router(const uint16_t* w, uint32_t hidden, uint32_t E) {
  if (E % 16 != 0) throw std::invalid_argument("repack_bolt_router: E " + std::to_string(E) + " is not a multiple of 16");
  std::vector<uint16_t> out(size_t(hidden) * E);
  common::repack_bf16_tiled(w, hidden, E, out.data());
  return out;
}

std::vector<float> widen_bolt_bf16(const uint16_t* w, size_t n) {
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = common::bf16_to_f32(w[i]);
  return out;
}

std::vector<float> bolt_neg_exp(const std::vector<float>& a_log) {
  std::vector<float> a(a_log.size());
  for (size_t i = 0; i < a.size(); ++i) a[i] = -common::exp_torch(a_log[i]);
  return a;
}

}  // namespace loader
