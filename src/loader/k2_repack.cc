#include "loader/k2_repack.h"

#include <cstring>
#include <stdexcept>

#include "common/bf16.h"
#include "common/repack.h"
#include "loader/k2_layout.h"

namespace loader {
namespace {

using model::K2Desc;

bool ends_with(const std::string& s, const char* suf) {
  const size_t n = std::strlen(suf);
  return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// The four suffixes of a GPTQ linear; the first two are required.
void add_int4(std::set<std::string>& s, const std::string& prefix, bool required_only) {
  s.insert(prefix + ".qweight");
  s.insert(prefix + ".scales");
  if (!required_only) {
    s.insert(prefix + ".qzeros");
    s.insert(prefix + ".g_idx");
  }
}

std::vector<common::ColSource> fuse_cols(model::Fuse fuse, const std::vector<LinearSrc>& srcs,
                                         const std::string& id) {
  std::vector<common::Part> parts;
  for (const LinearSrc& s : srcs) parts.push_back({s.qweight, s.scales, s.N});
  if (fuse == model::Fuse::Interleave16) {
    if (parts.size() != 2 || parts[0].N != parts[1].N || parts[0].N % 16 != 0)
      throw std::runtime_error(id + ": interleave16 needs two parts of equal, 16-divisible N");
    return common::cols_interleave16(parts[0], parts[1]);
  }
  return common::cols_concat(parts);
}

size_t int4_src_bytes(const LinearSrc& s) {
  return size_t(s.K / 8) * s.N * 4 + size_t(s.K / s.group) * s.N * 2;
}

}  // namespace

K2Checkpoint::K2Checkpoint(const K2Desc& d, const SafetensorsSet& set) : d_(d), set_(set) {}

std::set<std::string> K2Checkpoint::expected_names(bool required_only) const {
  std::set<std::string> s;
  for (uint32_t l = 0; l < d_.layers; ++l) {
    const std::string lp = K2Desc::layer_prefix(l);
    for (model::K2LinearId id : d_.layer_linears(l))
      for (const std::string& part : d_.linear(id).parts) add_int4(s, lp + part, required_only);
    for (const model::K2SmallTensor& t : d_.small_tensors(l)) s.insert(lp + t.name);
    if (d_.is_dense(l)) continue;
    s.insert(lp + "mlp.gate.weight");
    for (const model::K2ExpertGroup& g : d_.expert_groups()) {
      const uint32_t routed = g.id == model::K2ExpertId::Value ? g.blocks : g.blocks - 1;
      for (uint32_t e = 0; e < routed; ++e)
        for (const std::string& t : g.parts) add_int4(s, lp + K2Desc::expert_part(t, e), required_only);
      for (const std::string& p : g.shared) add_int4(s, lp + p, required_only);
    }
  }
  s.insert("model.embed_tokens.weight");
  s.insert("model.norm.weight");
  s.insert("lm_head.weight");
  return s;
}

void K2Checkpoint::check_names() const {
  const std::set<std::string> all = expected_names(false), req = expected_names(true);
  std::string bad;
  size_t n_bad = 0;
  for (const auto& [name, info] : set_.tensors()) {
    (void)info;
    if (all.count(name)) continue;
    if (n_bad++ < 5) bad += (bad.empty() ? "" : ", ") + name;
  }
  if (n_bad != 0)
    throw std::runtime_error(
        "load_k2: the checkpoint has " + std::to_string(n_bad) + " tensor(s) " + d_.name +
        " (model/k2_horizon.h) does not describe: " + bad + (n_bad > 5 ? ", ..." : "") +
        " - refusing to guess what they are (a quantised lm_head, a q/k norm, an mtp head and "
        "an extra expert are all outside this engine)");
  for (const std::string& name : req)
    if (!set_.tensors().count(name))
      throw std::runtime_error("load_k2: the checkpoint has no tensor '" + name + "', which " +
                               d_.name + " needs");
}

LinearSrc K2Checkpoint::int4(const std::string& prefix, uint32_t K, uint32_t N) {
  LinearSrc s = LinearSrc::classify(set_, prefix);
  if (s.kind != WKind::Int4)
    throw std::runtime_error("load_k2: '" + prefix + "' is bf16; " + d_.name +
                             " reads it as int4 g64 (the checkpoint's quantisation)");
  if (s.K != K || (N != 0 && s.N != N))
    throw std::runtime_error("load_k2: '" + prefix + "' is K=" + std::to_string(s.K) + " N=" +
                             std::to_string(s.N) + ", the descriptor says K=" + std::to_string(K) +
                             (N ? " N=" + std::to_string(N) : std::string()));
  consumed_.insert(prefix + ".qweight");
  consumed_.insert(prefix + ".scales");
  return s;
}

TensorInfo K2Checkpoint::take(const std::string& name, const char* dtype, uint64_t elems) {
  auto it = set_.tensors().find(name);
  if (it == set_.tensors().end()) throw std::runtime_error("load_k2: no tensor '" + name + "'");
  const TensorInfo t = it->second;   // a copy: gcc 13's -Wdangling-reference (loader.cc's note)
  if (t.dtype != dtype)
    throw std::runtime_error("load_k2: '" + name + "' is " + t.dtype + ", expected " + dtype);
  uint64_t n = 1;
  for (uint64_t x : t.shape) n *= x;
  if (n != elems)
    throw std::runtime_error("load_k2: '" + name + "' has " + std::to_string(n) +
                             " elements, expected " + std::to_string(elems));
  consumed_.insert(name);
  return t;
}

// 16-bit words widened to fp32 verbatim: "BF16" or "F16" - the source dtype is the bake.
void K2Checkpoint::widen(const TensorInfo& t, const std::string& name, float* dst) const {
  const uint8_t* p = set_.data(t);
  check_align(p, 2, name);
  const uint16_t* w = reinterpret_cast<const uint16_t*>(p);
  const size_t n = set_.bytes(t) / 2;
  const bool f16 = t.dtype == "F16";
  for (size_t i = 0; i < n; ++i) dst[i] = f16 ? common::f16_to_f32(w[i]) : common::bf16_to_f32(w[i]);
}

void K2Checkpoint::repack_layer(uint32_t layer, K2HostLayer& out) {
  const std::string lp = K2Desc::layer_prefix(layer);
  const std::string what = "layer " + std::to_string(layer);
  out.src_int4_bytes = out.src_bf16_bytes = 0;

  // --- the non-expert linears, layout 0, in execution order --------------------------
  const std::vector<model::K2LinearId> ids = d_.layer_linears(layer);
  out.linears.resize(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    const model::K2Linear fl = d_.linear(ids[i]);
    const model::GemvShape& sh = fl.shape;
    std::vector<LinearSrc> srcs;
    uint32_t n_sum = 0;
    for (const std::string& part : fl.parts) {
      srcs.push_back(int4(lp + part, sh.K, 0));
      n_sum += srcs.back().N;
      out.src_int4_bytes += int4_src_bytes(srcs.back());
    }
    if (n_sum != sh.N)
      throw std::runtime_error("load_k2: " + lp + fl.parts[0] + "...: parts sum to N=" +
                               std::to_string(n_sum) + ", the descriptor says " + std::to_string(sh.N));
    K2HostLinear& h = out.linears[i];
    h.words.resize(size_t(sh.K / 8) * sh.N);
    h.scales.resize(size_t(sh.K / 64) * sh.N);
    common::repack_int4_layout0_cols(sh.K, sh.N, fuse_cols(fl.fuse, srcs, lp + fl.parts[0]),
                                     h.words.data(), h.scales.data());
  }

  // --- the small tensors -------------------------------------------------------------
  out.norms.assign(d_.norms_bytes() / 4, 0.0f);
  out.route.assign(d_.is_dense(layer) ? 0 : d_.route_bytes() / 4, 0.0f);
  for (const model::K2SmallTensor& t : d_.small_tensors(layer)) {
    std::vector<float>& blk = t.block == model::K2SmallBlock::Norms ? out.norms : out.route;
    if (size_t(t.offset) + size_t(t.elems) * 4 > blk.size() * 4)
      throw std::logic_error("load_k2: " + lp + t.name + " overruns its block");
    const TensorInfo info = take(lp + t.name, t.dtype, t.elems);
    widen(info, lp + t.name, blk.data() + t.offset / 4);
    out.src_bf16_bytes += size_t(t.elems) * 2;
  }

  if (d_.is_dense(layer)) {
    out.value.clear();
    out.gate_up.clear();
    out.down.clear();
    out.router.clear();
    return;
  }

  // --- the MoE router: bf16 [experts][hidden], zero rows to router_n, tiled -----------
  {
    const LinearSrc r = LinearSrc::classify(set_, lp + "mlp.gate");
    if (r.kind != WKind::Bf16)
      throw std::runtime_error("load_k2: " + lp + "mlp.gate is int4; the MoE router is read as "
                               "bf16 (the checkpoint's 45 dynamic exclusions keep it so)");
    if (r.N != d_.experts || r.K != d_.hidden)
      throw std::runtime_error("load_k2: " + lp + "mlp.gate.weight is [" + std::to_string(r.N) +
                               "][" + std::to_string(r.K) + "], the descriptor says [" +
                               std::to_string(d_.experts) + "][" + std::to_string(d_.hidden) + "]");
    consumed_.insert(lp + "mlp.gate.weight");
    std::vector<uint16_t> rows(size_t(d_.router_n()) * d_.hidden, 0);
    std::memcpy(rows.data(), r.weight, size_t(d_.experts) * d_.hidden * 2);
    out.router.resize(rows.size());
    common::repack_bf16_tiled(rows.data(), d_.hidden, d_.router_n(), out.router.data());
    out.src_bf16_bytes += size_t(d_.experts) * d_.hidden * 2;
  }

  // --- the expert groups, layout 1, block e at e x stride -----------------------------
  for (const model::K2ExpertGroup& g : d_.expert_groups()) {
    std::vector<uint32_t>& dst = g.id == model::K2ExpertId::Value     ? out.value
                                 : g.id == model::K2ExpertId::MoeGateUp ? out.gate_up
                                                                        : out.down;
    const size_t blk_words = k2_layout1_bytes(g.K, g.N) / 4;
    dst.resize(blk_words * g.blocks);
    const bool fused = g.parts.size() == 2;   // gate || up, interleave16
    const uint32_t part_n = fused ? g.N / 2 : g.N;
    for (uint32_t b = 0; b < g.blocks; ++b) {
      const bool shared = !g.shared.empty() && b == g.blocks - 1;
      std::vector<LinearSrc> srcs;
      for (size_t p = 0; p < g.parts.size(); ++p) {
        const std::string part = shared ? g.shared[p] : K2Desc::expert_part(g.parts[p], b);
        srcs.push_back(int4(lp + part, g.K, part_n));
        out.src_int4_bytes += int4_src_bytes(srcs.back());
      }
      common::repack_int4_layout1_cols(
          g.K, g.N,
          fuse_cols(fused ? model::Fuse::Interleave16 : model::Fuse::Single, srcs,
                    what + " expert " + std::to_string(b)),
          dst.data() + size_t(b) * blk_words);
    }
  }
}

const uint16_t* K2Checkpoint::embed() {
  const TensorInfo t =
      take("model.embed_tokens.weight", "BF16", uint64_t(d_.vocab) * d_.hidden);
  check_align(set_.data(t), 2, "model.embed_tokens.weight");
  return reinterpret_cast<const uint16_t*>(set_.data(t));
}

std::vector<float> K2Checkpoint::final_norm() {
  std::vector<float> w(d_.hidden);
  const TensorInfo t = take("model.norm.weight", "BF16", d_.hidden);
  widen(t, "model.norm.weight", w.data());
  return w;
}

const uint16_t* K2Checkpoint::lm_head() {
  const LinearSrc s = LinearSrc::classify(set_, "lm_head");
  if (s.kind != WKind::Bf16)
    throw std::runtime_error("load_k2: lm_head is int4; " + d_.name +
                             " runs a bf16 head or quantises it to int8 at load (spec 9)");
  if (s.N != d_.vocab || s.K != d_.hidden)
    throw std::runtime_error("load_k2: lm_head.weight is [" + std::to_string(s.N) + "][" +
                             std::to_string(s.K) + "], expected [" + std::to_string(d_.vocab) +
                             "][" + std::to_string(d_.hidden) + "]");
  consumed_.insert("lm_head.weight");
  return s.weight;
}

size_t K2Checkpoint::unconsumed(std::string* names) const {
  size_t n = 0;
  for (const auto& [name, info] : set_.tensors()) {
    (void)info;
    if (consumed_.count(name) || ends_with(name, ".qzeros") || ends_with(name, ".g_idx")) continue;
    if (names && n < 5) *names += (names->empty() ? "" : ", ") + name;
    ++n;
  }
  return n;
}

QuantConfig check_k2_checkpoint_config(const model::K2Desc& d, const common::json::Value& config) {
  model::check_k2_config(d, config);
  return QuantConfig::parse(config);
}

}  // namespace loader
