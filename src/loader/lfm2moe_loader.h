#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "common/bf16.h"
#include "common/repack.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/lfm2moe_layout.h"
#include "loader/lfm2moe_repack.h"
#include "loader/lm_head_int8.h"
#include "loader/loader.h"   // DeviceWeight - reused, not re-declared
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/lfm2moe.h"

// Spec 24b Task 4: LFM2.5-24B-A2B on one or two B70s - the checkpoint's tensors in the layouts
// loader/lfm2moe_layout.h sizes (loader/lfm2moe_repack.h is the host half). A loader BESIDE
// loader::load, load_k2, load_kolibri1, load_qwen4exp and load_muse_glimmer (F0: their files
// untouched); the public pieces are reused - resolve_snapshot, QuantConfig::parse, SafetensorsSet,
// assert_quant_invariants, LinearSrc::classify, quantise_int8_tiled.
//
// **Both expert spellings, one format.** detect_expert_names reads the checkpoint's own spelling
// once; every expert of every layer is read in it (a mix is refused by lfm2moe_repack.h before any
// device work). **The tied head.** No `lm_head.*` tensor is required; an optional one must be
// bitwise `model.embed_tokens.weight` (checked, then skipped) or the load refuses naming both. The
// embedding is uploaded once, in gemv_bf16's tiled layout, on device 0 (the gather); the int8 head
// (when asked) is built from the SAME host bytes and uploaded on the last device, as spec 9's rule.
//
// **Placement at load**: each layer is repacked from the shards and uploaded straight onto the
// device the placement gives it (spec 16b's load-then-place would need the whole ~13.4 GB model on
// device 0 first). Device 0 also holds the embedding, the last device the final norm and the head,
// and every device holding an attention layer its own RoPE table.
namespace loader {

struct LfmLayer {
  std::unique_ptr<DeviceWeight> in_proj, out_proj;      // conv layers (in_proj in the B/C/x layout,
                                                          // or the arm-c conv tiles when asked)
  std::unique_ptr<DeviceWeight> qkv, o_proj;            // attention layers (q||k||v fused, N 3072)
  std::unique_ptr<DeviceWeight> dense_gate_up, dense_down;   // dense layers (is_dense)
  std::unique_ptr<l0::Mem> experts_gate_up, experts_down;    // MoE layers: 64 blocks, one allocation each
  std::unique_ptr<DeviceWeight> router;                 // bf16 tiles {hidden, experts} (MoE layers)
  std::unique_ptr<l0::Mem> small;                       // lfm::small_block(desc, l)'s fp32 bytes
};

struct LfmDevicePart {
  uint32_t device = 0, first = 0, end = 0;              // layers [first, end) live here
  std::vector<LfmLayer> layers;                         // index = layer - first
  std::unique_ptr<l0::Mem> embed_tiles;                 // device 0 only (the gather; also the bf16 head there)
  std::unique_ptr<l0::Mem> final_norm;                  // last device only, fp32 [hidden]
  std::unique_ptr<DeviceWeight> head_int8;              // last device only, when int8_head
  std::unique_ptr<l0::Mem> rope;                        // any device holding an attention layer
  size_t bytes = 0;                                     // weights, == sum of lfm::layer_weight_bytes + its share
  size_t rope_bytes = 0;
  const LfmLayer& layer(uint32_t l) const { return layers.at(l - first); }
};

struct LfmLoaded {
  model::Lfm2MoeDesc desc;
  model::LfmPlacement place;
  model::LfmExpertNames names;
  std::vector<LfmDevicePart> parts;                     // one per device
  uint32_t max_len = 0, trained_max_len = 0;
  bool int8_head = false, conv_tiles = false;
  uint32_t checkpoint_layers = 0;                        // the checkpoint's own count (>= desc.layers)
  size_t unconsumed = 0;
  size_t subnormal_scales = 0;
  double seconds = 0, lm_head_quant_seconds = 0;
  std::vector<std::string> skipped;                      // by design: the tied lm_head, --layers' later tensors
  size_t total_bytes() const;
};

struct LfmLoadOptions {
  bool int8_head = true;
  bool conv_tiles = false;
  uint32_t max_len = 0;
  uint32_t layers = 0;   // 0 = every layer the checkpoint has; > 0 truncates (development mode)
};

// One layer's host-repacked bytes (loader::Lfm2MoeCheckpoint::repack_layer's output), exactly what
// load_lfm2moe uploads - exposed so tests/loader/lfm2moe_synth_host_test.cc can check them against
// loader/lfm2moe_repack.h's functions over a checkpoint's own tensors, with NO Level Zero. Each
// HostLinear's `.scales` is empty at layout 1 (inline), populated at layout 0 - whichever
// `model::Lfm2MoeDesc::linears` says for that id (PROVISIONAL, switchable; spec 24 "24b as built"'s
// §13 correction).
struct HostLayer {
  HostLinear in_proj, out_proj;                         // conv
  HostLinear qkv, o_proj;                               // attention
  HostLinear dense_gate_up, dense_down;                 // dense
  std::vector<uint32_t> experts_gate_up, experts_down;  // MoE: all E blocks, concatenated, ALWAYS layout 1
  std::vector<uint16_t> router;                         // bf16 tiles {hidden, experts}
  std::vector<float> small;                             // fp32, lfm::small_block(d, l).bytes / 4
};

inline bool lfm_ends_with(const std::string& s, const char* suf) {
  const size_t n = std::strlen(suf);
  return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// The host-side checkpoint driver: reads and repacks one layer at a time, tracking every tensor it
// consumes. Host only (no Level Zero) - load_lfm2moe (lfm2moe_loader.cc) drives it per device;
// lfm2moe_synth_host_test.cc drives it directly, with no device at all.
class Lfm2MoeCheckpoint {
 public:
  Lfm2MoeCheckpoint(const model::Lfm2MoeDesc& d, const SafetensorsSet& set, model::LfmExpertNames names)
      : d_(d), set_(set), names_(names) {}

  // Throws naming up to five unexpected tensors, else the first missing one. Before any repack.
  void check_names() const {
    std::vector<std::string> exp = lfm_expected_names(d_, names_);
    const std::set<std::string> all(exp.begin(), exp.end());
    std::string bad;
    size_t n_bad = 0;
    for (const auto& [name, info] : set_.tensors()) {
      (void)info;
      if (all.count(name)) continue;
      if (name == "lm_head.weight") continue;   // the optional tied duplicate (checked after)
      if (lfm_ends_with(name, ".g_idx") && all.count(name.substr(0, name.size() - 6) + ".qweight")) continue;
      if (n_bad++ < 5) bad += (bad.empty() ? "" : ", ") + name;
    }
    if (n_bad != 0)
      throw std::runtime_error("load_lfm2moe: the checkpoint has " + std::to_string(n_bad) + " tensor(s) " +
                               d_.name + " (model/lfm2moe.h, expert spelling '" +
                               model::lfm_expert_names_name(names_) +
                               "') does not describe: " + bad + (n_bad > 5 ? ", ..." : "") +
                               " - refusing to guess what they are");
    for (const std::string& name : exp) {
      if (lfm_ends_with(name, ".qzeros")) continue;
      if (!set_.tensors().count(name))
        throw std::runtime_error("load_lfm2moe: the checkpoint has no tensor '" + name + "', which " + d_.name +
                                 " needs");
    }
  }

  // bf16 tensor `name` ([elems]) widened into out.small at byte offset `off`.
  void widen_small(HostLayer& out, const std::string& name, uint64_t elems, size_t off,
                   const std::vector<uint64_t>& want_shape = {}) {
    const std::vector<float> w = widen_bf16(take_bf16(name, elems, want_shape), elems);
    std::memcpy(out.small.data() + off / 4, w.data(), elems * 4);
  }

  // The generic case - every linear but InProj (its own permutation, Review Focus 2) and Router
  // (bf16): classify each part against `ln.shape.K` / `ln.part_n[i]` (M3: the descriptor is the one
  // source of shapes), then pack at `ln.shape.layout` via loader/lfm2moe_repack.h's one function.
  void fused_linear(const std::string& lp, const model::LfmLinear& ln, HostLinear& out) {
    std::vector<LinearSrc> srcs;
    srcs.reserve(ln.parts.size());
    for (size_t i = 0; i < ln.parts.size(); ++i) srcs.push_back(int4(lp + ln.parts[i], ln.shape.K, ln.part_n[i]));
    out = repack_fused_linear(ln, srcs);
  }

  void repack_layer(uint32_t layer, bool conv_tiles, HostLayer& out) {
    if (layer >= d_.layers)
      throw std::out_of_range("load_lfm2moe: layer " + std::to_string(layer) + " of " + std::to_string(d_.layers));
    const std::string lp = d_.layer_prefix(layer);
    const lfm::SmallBlock sb = lfm::small_block(d_, layer);
    out.small.assign(sb.bytes / 4, 0.0f);
    widen_small(out, lp + "operator_norm.weight", d_.hidden, sb.operator_norm);
    widen_small(out, lp + "ffn_norm.weight", d_.hidden, sb.ffn_norm);
    if (d_.is_attention(layer)) {
      widen_small(out, lp + "self_attn.q_layernorm.weight", d_.head_dim, sb.q_norm);
      widen_small(out, lp + "self_attn.k_layernorm.weight", d_.head_dim, sb.k_norm);
    } else {
      widen_small(out, lp + "conv.conv.weight", size_t(d_.hidden) * d_.conv_taps, sb.conv_taps,
                 {d_.hidden, 1, d_.conv_taps});   // M2: the shape, not just the element count
    }

    for (const model::LfmLinear& ln : d_.linears(layer)) {
      switch (ln.id) {
        case model::LfmLinearId::InProj: {
          const LinearSrc in = int4(lp + ln.parts[0], ln.shape.K, ln.shape.N);
          out.in_proj = repack_in_proj(in.qweight, in.scales, in.K, in.N, conv_tiles, ln.shape.layout);
          break;
        }
        case model::LfmLinearId::OutProj:
          fused_linear(lp, ln, out.out_proj);
          break;
        case model::LfmLinearId::Qkv:
          fused_linear(lp, ln, out.qkv);
          break;
        case model::LfmLinearId::OProj:
          fused_linear(lp, ln, out.o_proj);
          break;
        case model::LfmLinearId::DenseGateUp:
          fused_linear(lp, ln, out.dense_gate_up);
          break;
        case model::LfmLinearId::DenseDown:
          fused_linear(lp, ln, out.dense_down);
          break;
        case model::LfmLinearId::Router: {
          const TensorInfo gate = take(lp + ln.parts[0] + ".weight", size_t(ln.shape.N) * ln.shape.K, "BF16",
                                       {ln.shape.N, ln.shape.K});   // M2: [experts, hidden], not transposed
          out.router.resize(size_t(ln.shape.K) * ln.shape.N);
          common::repack_bf16_tiled(reinterpret_cast<const uint16_t*>(set_.data(gate)), ln.shape.K, ln.shape.N,
                                    out.router.data());
          break;
        }
        default:
          break;
      }
    }
    if (d_.is_dense(layer)) return;

    // The routed experts: always layout 1 (loader/lfm2moe_repack.h's rule), both spellings.
    const std::string fp = lp + "feed_forward.";
    const TensorInfo bias = take(fp + "expert_bias", d_.experts, "F32");
    std::memcpy(out.small.data() + sb.expert_bias / 4, set_.data(bias), size_t(d_.experts) * 4);
    static const char* const kW[] = {"w1", "w3", "w2"};
    static const char* const kProj[] = {"gate_proj", "up_proj", "down_proj"};
    const char* const* ex = names_ == model::LfmExpertNames::Proj ? kProj : kW;
    out.experts_gate_up.resize(size_t(d_.experts) * lfm::expert_gate_up_bytes(d_) / 4);
    out.experts_down.resize(size_t(d_.experts) * lfm::expert_down_bytes(d_) / 4);
    for (uint32_t e = 0; e < d_.experts; ++e) {
      const std::string ep = fp + "experts." + std::to_string(e) + ".";
      const LinearSrc g = int4(ep + ex[0], d_.hidden, d_.moe_inter);
      const LinearSrc u = int4(ep + ex[1], d_.hidden, d_.moe_inter);
      const LinearSrc dn = int4(ep + ex[2], d_.moe_inter, d_.hidden);
      const ExpertRepacked r = repack_expert(g, u, dn);
      std::memcpy(out.experts_gate_up.data() + lfm::expert_gate_up_offset(d_, e) / 4, r.gate_up.data(),
                 r.gate_up.size() * 4);
      std::memcpy(out.experts_down.data() + lfm::expert_down_offset(d_, e) / 4, r.down.data(), r.down.size() * 4);
    }
  }

  const uint16_t* embed() {
    const TensorInfo t = take(d_.prefix + "embed_tokens.weight", size_t(d_.vocab) * d_.hidden, "BF16",
                              {d_.vocab, d_.hidden});   // M2: [vocab, hidden], not transposed
    embed_bytes_ = set_.bytes(t);
    embed_ptr_ = set_.data(t);
    return reinterpret_cast<const uint16_t*>(embed_ptr_);
  }

  // Must run after embed(). Checks an optional tied lm_head.weight bitwise and marks it skipped;
  // refuses a different one naming both. Returns a note line, or "" when none was shipped.
  std::string check_tied_lm_head() {
    const std::string name = "lm_head.weight";   // never under the "model." prefix (sibling of it)
    auto it = set_.tensors().find(name);
    if (it == set_.tensors().end()) return "";
    if (it->second.dtype != "BF16" || set_.bytes(it->second) != embed_bytes_ ||
        std::memcmp(set_.data(it->second), embed_ptr_, embed_bytes_) != 0)
      throw std::runtime_error("load_lfm2moe: '" + name +
                               "' is shipped and differs from model.embed_tokens.weight - LFM2.5 ties the "
                               "head (tie_word_embeddings); refusing to guess which is right");
    consumed_.insert(name);
    return name + " (tied, bitwise embed_tokens, skipped)";
  }

  std::vector<float> final_norm() {
    const TensorInfo t = take(d_.prefix + "embedding_norm.weight", d_.hidden, "BF16");
    return widen_bf16(reinterpret_cast<const uint16_t*>(set_.data(t)), d_.hidden);
  }

  size_t skip_layers_from(uint32_t from) {
    size_t n = 0;
    const std::string prefix = d_.prefix + "layers.";
    for (const auto& [name, info] : set_.tensors()) {
      (void)info;
      if (name.rfind(prefix, 0) != 0) continue;
      const uint32_t l = uint32_t(std::stoul(name.substr(prefix.size())));
      if (l >= from && consumed_.insert(name).second) ++n;
    }
    return n;
  }

  // Tensors nothing read (qzeros / g_idx excepted), up to five names in `names`.
  size_t unconsumed(std::string* names = nullptr) const {
    size_t n = 0;
    for (const auto& [name, info] : set_.tensors()) {
      (void)info;
      if (consumed_.count(name) || lfm_ends_with(name, ".qzeros") || lfm_ends_with(name, ".g_idx")) continue;
      if (names && n < 5) *names += (names->empty() ? "" : ", ") + name;
      ++n;
    }
    return n;
  }

 private:
  LinearSrc int4(const std::string& prefix, uint32_t K, uint32_t N) {
    LinearSrc s = LinearSrc::classify(set_, prefix);
    if (s.kind != WKind::Int4 || s.packing != Int4Packing::Gptq)
      throw std::runtime_error("load_lfm2moe: '" + prefix + "' is not an int4 GPTQ linear; " + d_.name +
                               " reads it as int4 g64 (auto_round:auto_gptq)");
    if (s.K != K || s.N != N)
      throw std::runtime_error("load_lfm2moe: '" + prefix + "' is K=" + std::to_string(s.K) +
                               " N=" + std::to_string(s.N) + ", the descriptor says K=" + std::to_string(K) +
                               " N=" + std::to_string(N));
    for (const std::string& suffix : s.suffixes()) consumed_.insert(prefix + suffix);
    return s;
  }
  // `want_shape` empty: only the element count is checked (most small tensors - a norm's [2048] or
  // [1][2048] are both fine). Non-empty (M2): the exact shape is checked too - conv.conv.weight
  // [hidden, 1, conv_taps], the router [experts, hidden], the embedding [vocab, hidden], where a
  // transposed or reshaped tensor of the same element count must still be refused.
  TensorInfo take(const std::string& name, uint64_t elems, const char* dtype,
                  const std::vector<uint64_t>& want_shape = {}) {
    auto it = set_.tensors().find(name);
    if (it == set_.tensors().end()) throw std::runtime_error("load_lfm2moe: no tensor '" + name + "'");
    const TensorInfo t = it->second;
    if (t.dtype != dtype)
      throw std::runtime_error("load_lfm2moe: '" + name + "' is " + t.dtype + ", expected " + dtype);
    uint64_t n = 1;
    for (uint64_t x : t.shape) n *= x;
    if (n != elems)
      throw std::runtime_error("load_lfm2moe: '" + name + "' has " + std::to_string(n) + " elements, expected " +
                               std::to_string(elems));
    if (!want_shape.empty() && t.shape != want_shape) {
      std::string got, want;
      for (uint64_t x : t.shape) got += (got.empty() ? "" : ",") + std::to_string(x);
      for (uint64_t x : want_shape) want += (want.empty() ? "" : ",") + std::to_string(x);
      throw std::runtime_error("load_lfm2moe: '" + name + "' has shape [" + got + "], expected [" + want + "]");
    }
    consumed_.insert(name);
    return t;
  }
  const uint16_t* take_bf16(const std::string& name, uint64_t elems, const std::vector<uint64_t>& want_shape = {}) {
    return reinterpret_cast<const uint16_t*>(set_.data(take(name, elems, "BF16", want_shape)));
  }

  model::Lfm2MoeDesc d_;
  const SafetensorsSet& set_;
  model::LfmExpertNames names_;
  std::set<std::string> consumed_;
  const uint8_t* embed_ptr_ = nullptr;
  size_t embed_bytes_ = 0;
};

bool is_lfm2moe_checkpoint(const std::string& snapshot_dir);

// The descriptor load_lfm2moe would build, from config.json alone (no device, no tensor read): the
// CLI's planner needs it before the load (--pp, --max-len auto).
model::Lfm2MoeDesc lfm2moe_checkpoint_desc(const std::string& snapshot_or_repo, uint32_t layers_limit = 0);

// Loads the checkpoint (resolve_snapshot rules) onto `devices` (devices.size() == place.devices).
// Before a tensor is read: config.json held to model::lfm2moe() at the checkpoint's layer count,
// o.max_len within the trained context, the quantisation invariants, the expert spelling from the
// names (detect_expert_names), every name accounted for both ways (lfm_expected_names, plus the
// optional tied lm_head). `o.layers` > 0 loads only layers [0, o.layers) (development mode) and
// makes desc.layers = o.layers - the later layers' tensors are consumed by design and listed in
// `skipped`. The placement is over desc.layers (after the truncation). After: zero unconsumed
// tensors, and each part's weight bytes equal to the sum of loader::lfm::layer_weight_bytes over its
// layers plus its share of the embedding / head / final norm.
LfmLoaded load_lfm2moe(const std::vector<l0::Context*>& devices, const std::string& snapshot_or_repo,
                       const LfmLoadOptions& o, const model::LfmPlacement& place);

}  // namespace loader
