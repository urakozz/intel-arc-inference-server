#include "loader/lfm2moe_repack.h"

#include <stdexcept>

#include "common/bf16.h"
#include "common/repack.h"
#include "loader/lfm2moe_layout.h"

namespace loader {
namespace {

using model::Lfm2MoeDesc;
using model::LfmExpertNames;

const char* const kExpertNamesW[] = {"w1", "w3", "w2"};
const char* const kExpertNamesProj[] = {"gate_proj", "up_proj", "down_proj"};

const char* const* expert_names(LfmExpertNames n) {
  return n == LfmExpertNames::Proj ? kExpertNamesProj : kExpertNamesW;
}

void add_int4(std::vector<std::string>& v, const std::string& base) {
  v.push_back(base + ".qweight");
  v.push_back(base + ".qzeros");
  v.push_back(base + ".scales");
}

}  // namespace

std::vector<std::string> lfm_expected_names(const Lfm2MoeDesc& d, LfmExpertNames names) {
  std::vector<std::string> v = {d.prefix + "embed_tokens.weight", d.prefix + "embedding_norm.weight"};
  const char* const* ex = expert_names(names);
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string lp = d.layer_prefix(l);
    v.push_back(lp + "operator_norm.weight");
    v.push_back(lp + "ffn_norm.weight");
    if (d.is_attention(l)) {
      v.push_back(lp + "self_attn.q_layernorm.weight");
      v.push_back(lp + "self_attn.k_layernorm.weight");
      for (const char* s : {"q_proj", "k_proj", "v_proj", "out_proj"}) add_int4(v, lp + "self_attn." + s);
    } else {
      v.push_back(lp + "conv.conv.weight");
      add_int4(v, lp + "conv.in_proj");
      add_int4(v, lp + "conv.out_proj");
    }
    const std::string fp = lp + "feed_forward.";
    if (d.is_dense(l)) {
      for (const char* s : {"w1", "w2", "w3"}) add_int4(v, fp + s);
      continue;
    }
    v.push_back(fp + "gate.weight");
    v.push_back(fp + "expert_bias");
    for (uint32_t e = 0; e < d.experts; ++e)
      for (uint32_t s = 0; s < 3; ++s) add_int4(v, fp + "experts." + std::to_string(e) + "." + ex[s]);
  }
  return v;
}

LfmExpertNames detect_expert_names(const SafetensorsSet& set) {
  bool have_w = false, have_proj = false;
  for (const auto& [name, info] : set.tensors()) {
    (void)info;
    if (name.find(".experts.") == std::string::npos) continue;
    if (name.size() > 11 && name.compare(name.size() - 11, 11, ".w1.qweight") == 0) have_w = true;
    if (name.size() > 18 && name.compare(name.size() - 18, 18, ".gate_proj.qweight") == 0) have_proj = true;
  }
  if (have_w && have_proj)
    throw std::runtime_error("load_lfm2moe: the checkpoint mixes expert spellings (both 'w1.qweight' and "
                             "'gate_proj.qweight' tensors) - one format, one spelling (spec 24 §5)");
  if (!have_w && !have_proj)
    throw std::runtime_error("load_lfm2moe: no routed expert tensor ('.experts.*.w1.qweight' or "
                             "'.experts.*.gate_proj.qweight') found in the checkpoint");
  return have_proj ? LfmExpertNames::Proj : LfmExpertNames::W;
}

ExpertRepacked repack_expert(const LinearSrc& w1, const LinearSrc& w3, const LinearSrc& w2) {
  if (w1.K != w3.K || w1.N != w3.N || w1.K != w2.N)
    throw std::invalid_argument("repack_expert: w1 [" + std::to_string(w1.N) + "][" + std::to_string(w1.K) +
                                "], w3 [" + std::to_string(w3.N) + "][" + std::to_string(w3.K) + "], w2 [" +
                                std::to_string(w2.N) + "][" + std::to_string(w2.K) + "] do not agree");
  ExpertRepacked r;
  r.gate_up.resize(lfm::int4_l1_bytes(w1.K, 2 * w1.N) / 4);
  common::repack_int4_layout1_cols(
      w1.K, 2 * w1.N, common::cols_interleave16({w1.qweight, w1.scales, w1.N}, {w3.qweight, w3.scales, w3.N}),
      r.gate_up.data());
  r.down.resize(lfm::int4_l1_bytes(w2.K, w2.N) / 4);
  common::repack_int4_layout1_cols(w2.K, w2.N, common::cols_concat({{w2.qweight, w2.scales, w2.N}}),
                                   r.down.data());
  return r;
}

namespace {
// The column order alone - "a column map valid in both" layouts (Review Focus 2): independent of
// how the resulting columns are then packed.
std::vector<common::ColSource> in_proj_cols(const uint32_t* qweight, const uint16_t* scales, uint32_t N,
                                            bool conv_tiles) {
  const uint32_t H = N / 3;
  std::vector<common::ColSource> cols(N);
  if (!conv_tiles) {
    for (uint32_t n = 0; n < N; ++n) cols[n] = {qweight, scales, n, N};
  } else {
    uint32_t idx = 0;
    for (uint32_t t = 0; t < H / 16; ++t)
      for (uint32_t part = 0; part < 3; ++part)
        for (uint32_t l = 0; l < 16; ++l) cols[idx++] = {qweight, scales, part * H + t * 16 + l, N};
  }
  return cols;
}

HostLinear pack_cols(uint32_t K, uint32_t N, const std::vector<common::ColSource>& cols, uint32_t layout) {
  HostLinear out;
  if (layout == 0) {
    out.words.resize(size_t(K / 8) * N);
    out.scales.resize(size_t(K / 64) * N);
    common::repack_int4_layout0_cols(K, N, cols, out.words.data(), out.scales.data());
  } else {
    out.words.resize(lfm::int4_l1_bytes(K, N) / 4);
    common::repack_int4_layout1_cols(K, N, cols, out.words.data());
  }
  return out;
}
}  // namespace

HostLinear repack_in_proj(const uint32_t* qweight, const uint16_t* scales, uint32_t K, uint32_t N, bool conv_tiles,
                          uint32_t layout) {
  return pack_cols(K, N, in_proj_cols(qweight, scales, N, conv_tiles), layout);
}

HostLinear repack_fused_linear(const model::LfmLinear& ln, const std::vector<LinearSrc>& part_srcs) {
  std::vector<common::Part> parts;
  parts.reserve(part_srcs.size());
  for (const LinearSrc& s : part_srcs) parts.push_back({s.qweight, s.scales, s.N});
  const std::vector<common::ColSource> cols = (ln.fuse == model::Fuse::Interleave16 && parts.size() == 2)
                                                  ? common::cols_interleave16(parts[0], parts[1])
                                                  : common::cols_concat(parts);
  return pack_cols(ln.shape.K, ln.shape.N, cols, ln.shape.layout);
}

std::vector<float> widen_bf16(const uint16_t* w, size_t n) {
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = common::bf16_to_f32(w[i]);
  return out;
}

}  // namespace loader
