#include "loader/lfm2moe_loader.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "common/bf16.h"
#include "common/json.h"
#include "common/repack.h"
#include "l0/cmdlist.h"
#include "loader/lfm2moe_repack.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"
#include "loader/trained_context.h"
#include "runtime/lfm2moe/lfm2moe_sizes.h"

namespace loader {
namespace {

using model::Lfm2MoeDesc;
using model::LfmExpertNames;
using model::LfmPlacement;

common::json::Value read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream s;
  s << f.rdbuf();
  return common::json::parse(s.str());
}

std::string model_type_of(const common::json::Value& cfg) {
  if (!cfg.is_object()) return "";
  const common::json::Value* v = cfg.find("model_type");
  return v && v->is_string() ? v->str() : "";
}

Lfm2MoeDesc desc_of(const common::json::Value& cfg, uint32_t layers_limit) {
  Lfm2MoeDesc d = model::lfm2moe_desc(cfg, model::lfm2moe().vocab_used);
  if (layers_limit != 0) {
    if (layers_limit > d.layers)
      throw std::runtime_error("--layers " + std::to_string(layers_limit) + ": the checkpoint has " +
                               std::to_string(d.layers) + " layers");
    d.layers = layers_limit;
    d.kinds.resize(layers_limit);
  }
  return d;
}

void check_len(const Lfm2MoeDesc& d, uint32_t max_len, uint32_t trained) {
  if (max_len == 0) throw std::runtime_error(d.name + ": max_len 0 is not a model length");
  const uint32_t cap = trained ? trained : d.trained_max_len;
  if (max_len > cap)
    throw std::runtime_error(d.name + ": max_len " + std::to_string(max_len) + " exceeds the trained context " +
                             std::to_string(cap) + " (max_position_embeddings)");
}

l0::Mem upload(l0::Context& ctx, l0::CmdList& imm, const void* src, size_t bytes) {
  l0::Mem m(ctx, l0::MemKind::Device, bytes);
  imm.copy(m.ptr(), src, bytes);
  return m;
}
std::unique_ptr<l0::Mem> upload_u(l0::Context& ctx, l0::CmdList& imm, const void* src, size_t bytes) {
  return std::make_unique<l0::Mem>(upload(ctx, imm, src, bytes));
}
// `shape.layout` picks whether `scales` is a real, separate allocation (0: DeviceWeight::scales is
// the real thing, loader.h's rule) or null (1: inline in `words` already).
std::unique_ptr<DeviceWeight> upload_linear(l0::Context& ctx, l0::CmdList& imm, const HostLinear& ln,
                                            model::GemvShape shape) {
  std::unique_ptr<l0::Mem> scales =
      shape.layout == 0 ? upload_u(ctx, imm, ln.scales.data(), ln.scales.size() * 2) : nullptr;
  return std::make_unique<DeviceWeight>(
      DeviceWeight{upload(ctx, imm, ln.words.data(), ln.words.size() * 4), std::move(scales), shape,
                  model::WeightKind::Int4});
}
size_t linear_bytes(const HostLinear& ln) { return ln.words.size() * 4 + ln.scales.size() * 2; }
}  // namespace

size_t LfmLoaded::total_bytes() const {
  size_t t = 0;
  for (const LfmDevicePart& p : parts) t += p.bytes + p.rope_bytes;
  return t;
}

bool is_lfm2moe_checkpoint(const std::string& snapshot_dir) {
  return model::is_lfm2moe(read_json(snapshot_dir + "config.json"));
}

model::Lfm2MoeDesc lfm2moe_checkpoint_desc(const std::string& snapshot_or_repo, uint32_t layers_limit) {
  const std::string snap = resolve_snapshot(snapshot_or_repo);
  return desc_of(read_json(snap + "config.json"), layers_limit);
}

LfmLoaded load_lfm2moe(const std::vector<l0::Context*>& devices, const std::string& snapshot_or_repo,
                       const LfmLoadOptions& o, const LfmPlacement& place) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::string snap = resolve_snapshot(snapshot_or_repo);
  const common::json::Value cfg = read_json(snap + "config.json");
  const std::string mt = model_type_of(cfg);
  if (!model::is_lfm2moe(cfg))
    throw std::runtime_error("load_lfm2moe: " + snap + " is model_type '" + mt + "', not lfm2_moe");
  const QuantConfig qc = QuantConfig::parse(cfg);
  const uint32_t trained = trained_context(cfg);

  SafetensorsSet set(snap);
  const Lfm2MoeDesc full = model::lfm2moe_desc(cfg, model::lfm2moe().vocab_used);
  check_len(full, o.max_len, trained);
  const QuantScan scan = assert_quant_invariants(set);
  if (!qc.desc_act_declared && scan.g_idx_tensors != 0)
    throw std::runtime_error("load_lfm2moe: config.json declares no desc_act but the checkpoint ships " +
                             std::to_string(scan.g_idx_tensors) + " g_idx tensors");
  check_quant_scan(qc, scan);
  if (scan.ct_linears != 0)
    throw std::runtime_error("load_lfm2moe: compressed-tensors linears in an LFM2.5-24B-A2B checkpoint - spec "
                             "24 serves its own AutoRound auto_round:auto_gptq export only (§5)");
  const LfmExpertNames names = detect_expert_names(set);
  Lfm2MoeCheckpoint ck(full, set, names);
  ck.check_names();
  // M1: the optional tied lm_head is checked right after check_names, before any upload - embed()
  // just finds the tensor (no device work), so this costs nothing extra later (embed() is
  // idempotent: it is called again on device 0 and, for a bf16 head, on the last device).
  ck.embed();
  const std::string tied_note = ck.check_tied_lm_head();

  const Lfm2MoeDesc d = desc_of(cfg, o.layers);
  model::validate(place, d);
  if (devices.size() != place.devices)
    throw std::invalid_argument("load_lfm2moe: " + std::to_string(devices.size()) + " device(s) for a " +
                                std::to_string(place.devices) + "-device placement");
  // I2: the plan before the first allocation (muse_loader.cc's precedent) - every device's byte
  // total must fit before a single tensor is repacked or uploaded.
  for (uint32_t dev = 0; dev < place.devices; ++dev)
    runtime::lfm2moe::require_fits(runtime::lfm2moe::plan(d, place, dev, o.max_len, o.int8_head),
                                   devices[dev]->memory_bytes());

  LfmLoaded m;
  m.desc = d;
  m.place = place;
  m.names = names;
  m.max_len = o.max_len;
  m.trained_max_len = trained ? trained : d.trained_max_len;
  m.int8_head = o.int8_head;
  m.conv_tiles = o.conv_tiles;
  m.checkpoint_layers = full.layers;
  m.subnormal_scales = scan.subnormal_scales;
  if (d.layers < full.layers) {
    const size_t n = ck.skip_layers_from(d.layers);
    m.skipped.push_back("layers " + std::to_string(d.layers) + ".." + std::to_string(full.layers - 1) +
                        " not loaded (" + std::to_string(n) + " tensors consumed by design, --layers " +
                        std::to_string(d.layers) + ")");
  }

  const std::vector<float> rope = lfm_rope_table(d, o.max_len);
  loader::HostLayer host;
  m.parts.resize(place.devices);
  for (uint32_t dev = 0; dev < place.devices; ++dev) {
    l0::Context& ctx = *devices[dev];
    l0::CmdList imm = l0::CmdList::immediate(ctx);
    LfmDevicePart& P = m.parts[dev];
    P.device = dev;
    P.first = place.first(dev);
    P.end = place.end(dev);
    if (dev == 0) {
      std::vector<uint16_t> tiles(size_t(d.hidden) * d.vocab);
      common::repack_bf16_tiled(ck.embed(), d.hidden, d.vocab, tiles.data());
      P.embed_tiles = upload_u(ctx, imm, tiles.data(), tiles.size() * 2);
      P.bytes += tiles.size() * 2;
    }
    if (lfm::device_has_attention(d, place, dev)) {
      P.rope = upload_u(ctx, imm, rope.data(), rope.size() * 4);
      P.rope_bytes = rope.size() * 4;
      if (P.rope_bytes != lfm::rope_table_bytes(d, o.max_len))
        throw std::logic_error("load_lfm2moe: the RoPE table is not loader::lfm::rope_table_bytes");
    }
    for (uint32_t l = P.first; l < P.end; ++l) {
      ck.repack_layer(l, o.conv_tiles, host);
      LfmLayer L;
      // Every linear's upload SHAPE (incl. layout) comes from the descriptor - the same source
      // repack_layer read (M3: one source of shapes, not a second hardcoded copy here).
      for (const model::LfmLinear& ln : d.linears(l)) {
        switch (ln.id) {
          case model::LfmLinearId::InProj:
            L.in_proj = upload_linear(ctx, imm, host.in_proj, ln.shape);
            P.bytes += linear_bytes(host.in_proj);
            break;
          case model::LfmLinearId::OutProj:
            L.out_proj = upload_linear(ctx, imm, host.out_proj, ln.shape);
            P.bytes += linear_bytes(host.out_proj);
            break;
          case model::LfmLinearId::Qkv:
            L.qkv = upload_linear(ctx, imm, host.qkv, ln.shape);
            P.bytes += linear_bytes(host.qkv);
            break;
          case model::LfmLinearId::OProj:
            L.o_proj = upload_linear(ctx, imm, host.o_proj, ln.shape);
            P.bytes += linear_bytes(host.o_proj);
            break;
          case model::LfmLinearId::DenseGateUp:
            L.dense_gate_up = upload_linear(ctx, imm, host.dense_gate_up, ln.shape);
            P.bytes += linear_bytes(host.dense_gate_up);
            break;
          case model::LfmLinearId::DenseDown:
            L.dense_down = upload_linear(ctx, imm, host.dense_down, ln.shape);
            P.bytes += linear_bytes(host.dense_down);
            break;
          case model::LfmLinearId::Router:
            L.router = std::make_unique<DeviceWeight>(DeviceWeight{
                upload(ctx, imm, host.router.data(), host.router.size() * 2), nullptr, ln.shape,
                model::WeightKind::Bf16});
            P.bytes += host.router.size() * 2;
            break;
          default:
            break;
        }
      }
      if (!d.is_dense(l)) {
        L.experts_gate_up = upload_u(ctx, imm, host.experts_gate_up.data(), host.experts_gate_up.size() * 4);
        L.experts_down = upload_u(ctx, imm, host.experts_down.data(), host.experts_down.size() * 4);
        P.bytes += host.experts_gate_up.size() * 4 + host.experts_down.size() * 4;
      }
      L.small = upload_u(ctx, imm, host.small.data(), host.small.size() * 4);
      P.bytes += host.small.size() * 4;
      P.layers.push_back(std::move(L));
    }
    if (dev + 1 == place.devices) {
      const std::vector<float> fn = ck.final_norm();
      P.final_norm = upload_u(ctx, imm, fn.data(), fn.size() * 4);
      P.bytes += fn.size() * 4;
      if (o.int8_head) {
        const auto q0 = std::chrono::steady_clock::now();
        std::vector<int8_t> q(size_t(d.vocab) * d.hidden);
        std::vector<float> s(d.vocab);
        quantise_int8_tiled(ck.embed(), d.hidden, d.vocab, q.data(), s.data());
        m.lm_head_quant_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - q0).count();
        const model::GemvShape sh{d.hidden, d.vocab, 1, 0};
        P.head_int8 = std::make_unique<DeviceWeight>(
            DeviceWeight{upload(ctx, imm, q.data(), q.size()), upload_u(ctx, imm, s.data(), s.size() * 4), sh,
                        model::WeightKind::Int8});
        P.bytes += q.size() + s.size() * 4;
      } else if (dev != 0) {
        // the bf16 head IS the embedding tiles (spec 24 §4.4): a device without device 0's copy needs
        // its own (the two-card case, decision 10's placement).
        std::vector<uint16_t> tiles(size_t(d.hidden) * d.vocab);
        common::repack_bf16_tiled(ck.embed(), d.hidden, d.vocab, tiles.data());
        P.embed_tiles = upload_u(ctx, imm, tiles.data(), tiles.size() * 2);
        P.bytes += tiles.size() * 2;
      }
    }
    const size_t want = lfm::device_weight_bytes(d, place, dev, o.int8_head);
    if (P.bytes != want)
      throw std::logic_error("load_lfm2moe: device " + std::to_string(dev) + " allocated " +
                             std::to_string(P.bytes) + " B of weights, loader::lfm::device_weight_bytes says " +
                             std::to_string(want));
  }

  if (!tied_note.empty()) m.skipped.push_back(tied_note);

  std::string names_str;
  m.unconsumed = ck.unconsumed(&names_str);
  if (m.unconsumed != 0)
    throw std::runtime_error("load_lfm2moe: " + std::to_string(m.unconsumed) +
                             " checkpoint tensors were not loaded: " + names_str);
  m.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  const double gb = 1e9;
  std::printf("load_lfm2moe: %s\n  model       %s (%s): %u of %u layers, expert names '%s', int4 g%u sym, "
              "%zu subnormal f16 scales; %zu tensors, every name accounted for, 0 unconsumed\n",
              snap.c_str(), d.name.c_str(), d.architecture.c_str(), d.layers, full.layers,
              model::lfm_expert_names_name(names), qc.group_size, scan.subnormal_scales, set.tensors().size());
  for (const std::string& s : m.skipped) std::printf("  note        %s\n", s.c_str());
  for (const LfmDevicePart& P : m.parts)
    std::printf("  device %u    layers [%u, %u): weights %13zu B %7.3f GB%s%s, RoPE %zu B (max_len %u)\n",
               P.device, P.first, P.end, P.bytes, P.bytes / gb, P.embed_tiles ? " + embed" : "",
               P.head_int8 ? " + int8 head" : "", P.rope_bytes, o.max_len);
  std::printf("  load        %.1f s\n", m.seconds);
  return m;
}

}  // namespace loader
