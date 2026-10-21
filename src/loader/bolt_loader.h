#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/bf16.h"
#include "common/repack.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/bolt_layout.h"
#include "loader/bolt_repack.h"
#include "loader/lm_head_int8.h"
#include "loader/loader.h"   // DeviceWeight - reused, not re-declared
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/bolt.h"

// Spec 25b Tasks 4-5: Nemotron 3.5 Bolt on one or two B70s - the checkpoint's tensors in the layouts
// loader/bolt_layout.h sizes (loader/bolt_repack.h is the host half). A loader BESIDE loader::load, load_k2,
// load_kolibri1, load_qwen4exp, load_muse_glimmer and load_lfm2moe (their files untouched); the public pieces are reused -
// resolve_snapshot, QuantConfig::parse, SafetensorsSet, assert_quant_invariants, LinearSrc::classify,
// quantise_int8_tiled.
//
// **Either int4 spelling, one device layout.** SergiioB's g128 GPTQ export (through LinearSrc::classify's exact g128 -> g64
// expansion; `g_idx` shipped, a restating `+:` rule, signed F16 scales) and our g64 export (25q) go through the same
// repack; a linear's group is decided by its scale rows, never by a label, and the device buffers of the two spellings of
// one set of weights are byte-identical. **The vision tower** (849 tensors in SergiioB's: `vision_model.*`,
// `vision_projector.*`, `mlp1.*`) is skipped by prefix and counted; every other tensor is consumed (unconsumed = 0).
// **The trunk** is `language_model.model.`; NVIDIA's own `language_model.backbone.` is accepted only when the checkpoint
// ships it (bolt_trunk_prefix). **The untied head**: `language_model.lm_head.weight` is its own bf16 tensor; the int8 head
// (spec 9, `--lm-head int8` default, Decision 18) is built from it on the last device, or its bf16 tiles are uploaded
// (`--lm-head bf16`); the embedding is uploaded once, in gemv_bf16's tiles, on device 0.
//
// **Placement at load**: each block is repacked from the shards and uploaded straight onto the device the placement gives
// it. Device 0 also holds the embedding, the last device norm_f and the head.
namespace loader {

struct BoltLayer {
  std::unique_ptr<DeviceWeight> in_proj, out_proj;                       // Mamba
  std::unique_ptr<DeviceWeight> qkv, o_proj;                             // attention (q||k||v fused, N 4608)
  std::unique_ptr<DeviceWeight> fc1_shared_up, fc2, shared_down;         // MoE (fc1||shared_up fused, N 6400)
  std::unique_ptr<l0::Mem> experts;                                      // MoE: E blocks (up || down, layout 1), ONE allocation
  std::unique_ptr<DeviceWeight> router;                                  // MoE: bf16 tiles {hidden, E}
  std::unique_ptr<l0::Mem> small;                                        // bolt::small_block(desc, l)'s fp32 bytes
};

struct BoltDevicePart {
  uint32_t device = 0, first = 0, end = 0;                               // blocks [first, end) live here
  std::vector<BoltLayer> layers;                                         // index = layer - first
  std::unique_ptr<l0::Mem> embed_tiles;                                  // device 0 only (the gather)
  std::unique_ptr<l0::Mem> final_norm;                                   // last device only, fp32 [hidden]
  std::unique_ptr<DeviceWeight> head;                                    // last device only: int8 (+ fp32 scales) or bf16 tiles
  size_t bytes = 0;                                                      // weights, == bolt::device_weight_bytes
  const BoltLayer& layer(uint32_t l) const { return layers.at(l - first); }
};

struct BoltLoaded {
  model::BoltDesc desc;                                                  // the descriptor at the LOADED block count
  model::BoltPlacement place;
  std::vector<BoltDevicePart> parts;                                     // one per device
  uint32_t max_len = 0;
  bool int8_head = false;
  uint32_t checkpoint_layers = 0;                                        // the checkpoint's own block count (>= desc.layers)
  size_t unconsumed = 0;
  size_t vision_skipped = 0;                                             // tensors skipped by the vision prefixes
  size_t subnormal_scales = 0;
  uint32_t group_mask = 0;                                               // bit 0: a g64 linear was read, bit 1: a g128 one
  std::string trunk;
  double seconds = 0, lm_head_quant_seconds = 0;
  std::vector<std::string> skipped;                                      // by design: vision, --layers' later tensors
  size_t total_bytes() const;
};

struct BoltLoadOptions {
  bool int8_head = true;
  uint32_t max_len = 0;
  uint32_t layers = 0;   // 0 = every block the checkpoint has; > 0 truncates (development mode)
};

// One block's host-repacked bytes (BoltCheckpoint::repack_layer's output), exactly what load_bolt uploads - exposed so
// tests/loader/bolt_synth_host_test.cc can check them over a checkpoint's own tensors, with NO Level Zero.
struct BoltHostLayer {
  BoltHostLinear in_proj, out_proj;                         // Mamba
  BoltHostLinear qkv, o_proj;                               // attention
  BoltHostLinear fc1_shared_up, fc2, shared_down;           // MoE
  std::vector<uint32_t> experts;                            // MoE: all E blocks, concatenated (up || down each), layout 1
  std::vector<uint16_t> router;                             // bf16 tiles {hidden, E}
  std::vector<float> small;                                 // fp32, bolt::small_block(d, l).bytes / 4
};

inline bool bolt_ends_with(const std::string& s, const char* suf) {
  const size_t n = std::strlen(suf);
  return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// The host-side checkpoint driver: reads and repacks one block at a time, tracking every tensor it consumes. Host only (no
// Level Zero) - load_bolt (bolt_loader.cc) drives it per device; bolt_synth_host_test.cc drives it directly.
class BoltCheckpoint {
 public:
  // `d` is the descriptor at the CHECKPOINT's block count (model::bolt_desc of its config.json); `trunk` from
  // bolt_trunk_prefix.
  BoltCheckpoint(const model::BoltDesc& d, const SafetensorsSet& set, std::string trunk)
      : d_(d), set_(set), trunk_(std::move(trunk)) {}

  const std::string& trunk() const { return trunk_; }

  // Throws naming up to five unexpected tensors, else the first missing one. Before any repack. The vision tower's tensors
  // (bolt_is_vision) and the .qzeros / .g_idx beside a linear are not "unexpected".
  void check_names() const {
    std::vector<std::string> exp = bolt_expected_names(d_, trunk_);
    const std::set<std::string> all(exp.begin(), exp.end());
    std::string bad;
    size_t n_bad = 0;
    for (const auto& [name, info] : set_.tensors()) {
      (void)info;
      if (all.count(name) || bolt_is_vision(name)) continue;
      if (bolt_ends_with(name, ".g_idx") && all.count(name.substr(0, name.size() - 6) + ".qweight")) continue;
      if (n_bad++ < 5) bad += (bad.empty() ? "" : ", ") + name;
    }
    if (n_bad != 0)
      throw std::runtime_error("load_bolt: the checkpoint has " + std::to_string(n_bad) + " tensor(s) " + d_.name +
                               " (model/bolt.h) does not describe: " + bad + (n_bad > 5 ? ", ..." : "") +
                               " - refusing to guess what they are");
    for (const std::string& name : exp) {
      if (bolt_ends_with(name, ".qzeros")) continue;
      if (!set_.tensors().count(name))
        throw std::runtime_error("load_bolt: the checkpoint has no tensor '" + name + "', which " + d_.name + " needs");
    }
  }

  // The vision tower / projector tensors, consumed by design; returns how many.
  size_t skip_vision() {
    size_t n = 0;
    for (const auto& [name, info] : set_.tensors()) {
      (void)info;
      if (bolt_is_vision(name) && consumed_.insert(name).second) ++n;
    }
    return n;
  }

  // Blocks [from, ...)'s tensors, consumed by design (--layers N on a longer checkpoint).
  size_t skip_layers_from(uint32_t from) {
    size_t n = 0;
    const std::string prefix = trunk_ + "layers.";
    for (const auto& [name, info] : set_.tensors()) {
      (void)info;
      if (name.rfind(prefix, 0) != 0) continue;
      const uint32_t l = uint32_t(std::stoul(name.substr(prefix.size())));
      if (l >= from && consumed_.insert(name).second) ++n;
    }
    return n;
  }

  void repack_layer(uint32_t layer, BoltHostLayer& out) {
    if (layer >= d_.layers)
      throw std::out_of_range("load_bolt: block " + std::to_string(layer) + " of " + std::to_string(d_.layers));
    const std::string lp = model::BoltDesc::layer_prefix(trunk_, layer), m = lp + "mixer.";
    const bolt::SmallBlock sb = bolt::small_block(d_, layer);
    out.small.assign(sb.bytes / 4, 0.0f);
    put(out, sb.norm, widen(take(lp + "norm.weight", {"BF16"}, {d_.hidden})));
    if (d_.is_mamba(layer)) {
      put(out, sb.conv_w, widen(take(m + "conv1d.weight", {"BF16"}, {d_.m_conv_dim, 1, d_.m_conv})));
      put(out, sb.conv_b, widen(take(m + "conv1d.bias", {"BF16"}, {d_.m_conv_dim})));
      put(out, sb.A, bolt_neg_exp(finite_f32(m + "A_log", {d_.m_heads}, 80.0f)));
      put(out, sb.D, finite_f32(m + "D", {d_.m_heads}));
      put(out, sb.dt_bias, finite_f32(m + "dt_bias", {d_.m_heads}));
      put(out, sb.gnorm, widen(take(m + "norm.weight", {"BF16"}, {d_.m_inner})));
    }
    for (const model::BoltLinear& ln : d_.linears(layer)) {
      switch (ln.id) {
        case model::BoltLinearId::InProj: fused_linear(m, ln, out.in_proj); break;
        case model::BoltLinearId::OutProj: fused_linear(m, ln, out.out_proj); break;
        case model::BoltLinearId::Qkv: fused_linear(m, ln, out.qkv); break;
        case model::BoltLinearId::OProj: fused_linear(m, ln, out.o_proj); break;
        case model::BoltLinearId::Fc1SharedUp: fused_linear(m, ln, out.fc1_shared_up); break;
        case model::BoltLinearId::Fc2: fused_linear(m, ln, out.fc2); break;
        case model::BoltLinearId::SharedDown: fused_linear(m, ln, out.shared_down); break;
        case model::BoltLinearId::Router: {
          const TensorInfo gate = take(m + "gate.weight", {"BF16"}, {ln.shape.N, ln.shape.K});   // [E][hidden], not transposed
          out.router = repack_bolt_router(reinterpret_cast<const uint16_t*>(set_.data(gate)), ln.shape.K, ln.shape.N);
          break;
        }
        default: break;   // the routed experts: below
      }
    }
    if (!d_.is_moe(layer)) return;

    const uint32_t E = d_.experts.at(layer), I = d_.inter.at(layer);
    const TensorInfo bias = take(m + "gate.e_score_correction_bias", {"F32"}, {E});
    std::memcpy(out.small.data() + sb.e_bias / 4, set_.data(bias), size_t(E) * 4);
    for (uint32_t i = 0; i < E; ++i) {
      float v;
      std::memcpy(&v, out.small.data() + sb.e_bias / 4 + i, 4);
      if (!std::isfinite(v))
        throw std::runtime_error("load_bolt: '" + m + "gate.e_score_correction_bias'[" + std::to_string(i) + "] is not finite");
    }
    const size_t block = bolt::expert_block_bytes(d_.latent, I);
    out.experts.assign(size_t(E) * block / 4, 0u);
    for (uint32_t e = 0; e < E; ++e) {
      const std::string ep = m + "experts." + std::to_string(e) + ".";
      const LinearSrc up = int4(ep + "up_proj", d_.latent, I);
      const LinearSrc dn = int4(ep + "down_proj", I, d_.latent);
      repack_bolt_expert(up, dn, out.experts.data() + bolt::expert_up_offset(d_.latent, I, e) / 4);
    }
  }

  const uint16_t* embed() {
    const TensorInfo t = take(trunk_ + "embeddings.weight", {"BF16"}, {d_.vocab, d_.hidden});   // [vocab, hidden], not transposed
    embed_bytes_ = set_.bytes(t);
    return reinterpret_cast<const uint16_t*>(set_.data(t));
  }
  const uint16_t* lm_head() {
    const TensorInfo t = take(kBoltLmHead, {"BF16"}, {d_.vocab, d_.hidden});
    return reinterpret_cast<const uint16_t*>(set_.data(t));
  }
  std::vector<float> final_norm() { return widen(take(trunk_ + "norm_f.weight", {"BF16"}, {d_.hidden})); }

  // Tensors nothing read (qzeros / g_idx excepted), up to five names in `names`.
  size_t unconsumed(std::string* names = nullptr) const {
    size_t n = 0;
    for (const auto& [name, info] : set_.tensors()) {
      (void)info;
      if (consumed_.count(name) || bolt_ends_with(name, ".qzeros") || bolt_ends_with(name, ".g_idx")) continue;
      if (names && n < 5) *names += (names->empty() ? "" : ", ") + name;
      ++n;
    }
    return n;
  }
  // bit 0: a linear read at g64, bit 1: at g128 (expanded); what the report prints.
  uint32_t group_mask() const { return group_mask_; }

 private:
  static void put_floats(BoltHostLayer& out, size_t off, const std::vector<float>& v) {
    std::memcpy(out.small.data() + off / 4, v.data(), v.size() * 4);
  }
  static void put(BoltHostLayer& out, size_t off, const std::vector<float>& v) { put_floats(out, off, v); }

  std::vector<float> widen(const TensorInfo& t) {
    return widen_bolt_bf16(reinterpret_cast<const uint16_t*>(set_.data(t)), set_.bytes(t) / 2);
  }
  // A kept fp32 tensor shipped F32 (SergiioB's: bf16 values in F32) or BF16 (widened, exact); finite, and |v| <= `bound`.
  std::vector<float> finite_f32(const std::string& name, const std::vector<uint64_t>& shape, float bound = 3.0e38f) {
    const TensorInfo t = take(name, {"F32", "BF16"}, shape);
    std::vector<float> v;
    if (t.dtype == "BF16") {
      v = widen(t);
    } else {
      v.resize(set_.bytes(t) / 4);
      std::memcpy(v.data(), set_.data(t), v.size() * 4);
    }
    for (size_t i = 0; i < v.size(); ++i)
      if (!std::isfinite(v[i]) || std::fabs(v[i]) > bound)
        throw std::runtime_error("load_bolt: '" + name + "'[" + std::to_string(i) + "] = " + std::to_string(v[i]) +
                                 " is not a finite value within " + std::to_string(bound));
    return v;
  }

  // Every fused linear: classify each part against its shape (the descriptor is the one source of shapes), then pack at
  // `ln.shape.layout`.
  void fused_linear(const std::string& m, const model::BoltLinear& ln, BoltHostLinear& out) {
    std::vector<LinearSrc> srcs;
    srcs.reserve(ln.parts.size());
    for (size_t i = 0; i < ln.parts.size(); ++i) srcs.push_back(int4(m + ln.parts[i], ln.shape.K, ln.part_n[i]));
    out = repack_bolt_linear(ln, srcs);
  }

  LinearSrc int4(const std::string& prefix, uint32_t K, uint32_t N) {
    LinearSrc s = LinearSrc::classify(set_, prefix);
    if (s.kind != WKind::Int4 || s.packing != Int4Packing::Gptq)
      throw std::runtime_error("load_bolt: '" + prefix + "' is not an int4 GPTQ linear; " + d_.name +
                               " reads it as int4 g64 or g128 (GPTQ packing)");
    if (s.K != K || s.N != N)
      throw std::runtime_error("load_bolt: '" + prefix + "' is K=" + std::to_string(s.K) + " N=" + std::to_string(s.N) +
                               ", the descriptor says K=" + std::to_string(K) + " N=" + std::to_string(N));
    group_mask_ |= s.group == 128 ? 2u : 1u;
    for (const std::string& suffix : s.suffixes()) consumed_.insert(prefix + suffix);
    return s;
  }

  // The exact shape and one of `dtypes` are checked (a transposed or reshaped tensor of the same element count must be
  // refused).
  TensorInfo take(const std::string& name, std::initializer_list<const char*> dtypes, const std::vector<uint64_t>& shape) {
    auto it = set_.tensors().find(name);
    if (it == set_.tensors().end()) throw std::runtime_error("load_bolt: no tensor '" + name + "'");
    const TensorInfo t = it->second;
    bool ok = false;
    std::string want;
    for (const char* dt : dtypes) {
      ok = ok || t.dtype == dt;
      want += (want.empty() ? "" : " or ") + std::string(dt);
    }
    if (!ok) throw std::runtime_error("load_bolt: '" + name + "' is " + t.dtype + ", expected " + want);
    if (t.shape != shape) {
      std::string got, exp;
      for (uint64_t x : t.shape) got += (got.empty() ? "" : ",") + std::to_string(x);
      for (uint64_t x : shape) exp += (exp.empty() ? "" : ",") + std::to_string(x);
      throw std::runtime_error("load_bolt: '" + name + "' has shape [" + got + "], expected [" + exp + "]");
    }
    consumed_.insert(name);
    return t;
  }

  model::BoltDesc d_;
  const SafetensorsSet& set_;
  std::string trunk_;
  std::set<std::string> consumed_;
  size_t embed_bytes_ = 0;
  uint32_t group_mask_ = 0;
};

// The descriptor load_bolt would build, from config.json alone (no device, no tensor read): the CLI's planner needs it before
// the load (--pp, --max-len auto).
model::BoltDesc bolt_checkpoint_desc(const std::string& snapshot_or_repo, uint32_t layers_limit = 0);
bool is_bolt_checkpoint(const std::string& snapshot_dir);

// Loads the checkpoint (resolve_snapshot rules) onto `devices` (devices.size() == place.devices). Before a tensor is read:
// config.json held to model::bolt() (check_bolt_config), the quantisation config (QuantConfig::parse: g64 or g128 sym,
// a restating '+:' rule), the quantisation invariants over every tensor, every name accounted for both ways
// (bolt_expected_names + the vision prefixes), the memory plan against each device (runtime::bolt::require_fits).
// `o.layers` > 0 loads only blocks [0, o.layers) (development mode; the later blocks' tensors are consumed by design and
// listed in `skipped`; the placement is over the truncated count). After: zero unconsumed tensors, and each part's weight
// bytes equal to loader::bolt::device_weight_bytes.
BoltLoaded load_bolt(const std::vector<l0::Context*>& devices, const std::string& snapshot_or_repo, const BoltLoadOptions& o,
                     const model::BoltPlacement& place);

}  // namespace loader
