#include "loader/bolt_loader.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "common/json.h"
#include "l0/cmdlist.h"
#include "loader/snapshot.h"
#include "runtime/bolt/bolt_sizes.h"

namespace loader {
namespace {

common::json::Value read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream s;
  s << f.rdbuf();
  return common::json::parse(s.str());
}

model::BoltDesc desc_of(const common::json::Value& cfg, uint32_t layers_limit) {
  model::BoltDesc d = model::bolt_desc(cfg);
  if (layers_limit != 0) {
    if (layers_limit > d.layers)
      throw std::runtime_error("--layers " + std::to_string(layers_limit) + ": the checkpoint has " + std::to_string(d.layers) +
                               " blocks");
    d = model::bolt_truncated(d, layers_limit);
  }
  return d;
}

l0::Mem upload(l0::Context& ctx, l0::CmdList& imm, const void* src, size_t bytes) {
  l0::Mem m(ctx, l0::MemKind::Device, bytes);
  imm.copy(m.ptr(), src, bytes);
  return m;
}
std::unique_ptr<l0::Mem> upload_u(l0::Context& ctx, l0::CmdList& imm, const void* src, size_t bytes) {
  return std::make_unique<l0::Mem>(upload(ctx, imm, src, bytes));
}
// `shape.layout` picks whether `scales` is a real, separate allocation (0: DeviceWeight::scales, loader.h's rule) or null (1:
// inline in `words` already).
std::unique_ptr<DeviceWeight> upload_linear(l0::Context& ctx, l0::CmdList& imm, const BoltHostLinear& ln, model::GemvShape shape) {
  std::unique_ptr<l0::Mem> scales = shape.layout == 0 ? upload_u(ctx, imm, ln.scales.data(), ln.scales.size() * 2) : nullptr;
  return std::make_unique<DeviceWeight>(
      DeviceWeight{upload(ctx, imm, ln.words.data(), ln.words.size() * 4), std::move(scales), shape, model::WeightKind::Int4});
}
size_t linear_bytes(const BoltHostLinear& ln) { return ln.words.size() * 4 + ln.scales.size() * 2; }

}  // namespace

size_t BoltLoaded::total_bytes() const {
  size_t t = 0;
  for (const BoltDevicePart& p : parts) t += p.bytes;
  return t;
}

bool is_bolt_checkpoint(const std::string& snapshot_dir) { return model::is_bolt(read_json(snapshot_dir + "config.json")); }

model::BoltDesc bolt_checkpoint_desc(const std::string& snapshot_or_repo, uint32_t layers_limit) {
  const std::string snap = resolve_snapshot(snapshot_or_repo);
  return desc_of(read_json(snap + "config.json"), layers_limit);
}

BoltLoaded load_bolt(const std::vector<l0::Context*>& devices, const std::string& snapshot_or_repo, const BoltLoadOptions& o,
                     const model::BoltPlacement& place) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::string snap = resolve_snapshot(snapshot_or_repo);
  const common::json::Value cfg = read_json(snap + "config.json");
  if (!model::is_bolt(cfg))
    throw std::runtime_error("load_bolt: " + snap + " is not a nemotron_h_omni / nemotron_h_puzzle checkpoint");
  const QuantConfig qc = QuantConfig::parse(cfg);   // g64, or g128 through the exact expansion; a restating '+:' rule accepted
  const model::BoltDesc full = model::bolt_desc(cfg);   // check_bolt_config first: held to llm_config key by key
  if (o.max_len == 0) throw std::runtime_error(full.name + ": max_len 0 is not a model length (resolve --max-len auto first)");
  if (o.max_len > full.max_pos)
    throw std::runtime_error(full.name + ": max_len " + std::to_string(o.max_len) + " exceeds the trained context " +
                             std::to_string(full.max_pos) + " (max_position_embeddings)");

  SafetensorsSet set(snap);
  const QuantScan scan = assert_quant_invariants(set);
  if (!qc.desc_act_declared && scan.g_idx_tensors != 0)
    throw std::runtime_error("load_bolt: config.json declares no desc_act but the checkpoint ships " +
                             std::to_string(scan.g_idx_tensors) + " g_idx tensors");
  check_quant_scan(qc, scan);
  if (scan.ct_linears != 0)
    throw std::runtime_error("load_bolt: compressed-tensors linears in a Bolt checkpoint - spec 25 serves AutoRound GPTQ "
                             "(g128 expanded, or our g64) only (§5)");
  const std::string trunk = bolt_trunk_prefix(set);
  BoltCheckpoint ck(full, set, trunk);
  ck.check_names();
  const size_t vision = ck.skip_vision();

  const model::BoltDesc d = desc_of(cfg, o.layers);
  model::validate(place, d);
  if (devices.size() != place.devices)
    throw std::invalid_argument("load_bolt: " + std::to_string(devices.size()) + " device(s) for a " +
                                std::to_string(place.devices) + "-device placement");
  // The plan before the first allocation: every device's byte total must fit before a tensor is repacked or uploaded.
  for (uint32_t dev = 0; dev < place.devices; ++dev)
    runtime::bolt::require_fits(
        runtime::bolt::device_plan(d, place, dev, o.max_len, o.int8_head, {}, size_t(runtime::kDefaultReserveGb * 1e9)), dev,
        o.max_len, devices[dev]->memory_bytes());

  BoltLoaded m;
  m.desc = d;
  m.place = place;
  m.max_len = o.max_len;
  m.int8_head = o.int8_head;
  m.checkpoint_layers = full.layers;
  m.subnormal_scales = scan.subnormal_scales;
  m.trunk = trunk;
  m.vision_skipped = vision;
  m.skipped.push_back(std::to_string(vision) + " vision / projector tensors skipped by prefix (vision_model. / "
                      "vision_projector. / mlp1.)");
  if (d.layers < full.layers) {
    const size_t n = ck.skip_layers_from(d.layers);
    m.skipped.push_back("blocks " + std::to_string(d.layers) + ".." + std::to_string(full.layers - 1) + " not loaded (" +
                        std::to_string(n) + " tensors consumed by design, --layers " + std::to_string(d.layers) + ")");
  }

  BoltHostLayer host;
  m.parts.resize(place.devices);
  for (uint32_t dev = 0; dev < place.devices; ++dev) {
    l0::Context& ctx = *devices[dev];
    l0::CmdList imm = l0::CmdList::immediate(ctx);
    BoltDevicePart& P = m.parts[dev];
    P.device = dev;
    P.first = place.first(dev);
    P.end = place.end(dev);
    if (dev == 0) {
      std::vector<uint16_t> tiles(size_t(d.hidden) * d.vocab);
      common::repack_bf16_tiled(ck.embed(), d.hidden, d.vocab, tiles.data());
      P.embed_tiles = upload_u(ctx, imm, tiles.data(), tiles.size() * 2);
      P.bytes += tiles.size() * 2;
    }
    for (uint32_t l = P.first; l < P.end; ++l) {
      ck.repack_layer(l, host);
      BoltLayer L;
      // Every linear's upload SHAPE (incl. layout) comes from the descriptor - the same source repack_layer read.
      for (const model::BoltLinear& ln : d.linears(l)) {
        auto up = [&](std::unique_ptr<DeviceWeight>& dst, const BoltHostLinear& h) {
          dst = upload_linear(ctx, imm, h, ln.shape);
          P.bytes += linear_bytes(h);
        };
        switch (ln.id) {
          case model::BoltLinearId::InProj: up(L.in_proj, host.in_proj); break;
          case model::BoltLinearId::OutProj: up(L.out_proj, host.out_proj); break;
          case model::BoltLinearId::Qkv: up(L.qkv, host.qkv); break;
          case model::BoltLinearId::OProj: up(L.o_proj, host.o_proj); break;
          case model::BoltLinearId::Fc1SharedUp: up(L.fc1_shared_up, host.fc1_shared_up); break;
          case model::BoltLinearId::Fc2: up(L.fc2, host.fc2); break;
          case model::BoltLinearId::SharedDown: up(L.shared_down, host.shared_down); break;
          case model::BoltLinearId::Router:
            L.router = std::make_unique<DeviceWeight>(DeviceWeight{upload(ctx, imm, host.router.data(), host.router.size() * 2),
                                                                     nullptr, ln.shape, model::WeightKind::Bf16});
            P.bytes += host.router.size() * 2;
            break;
          default: break;   // the routed experts: one allocation below
        }
      }
      if (d.is_moe(l)) {
        L.experts = upload_u(ctx, imm, host.experts.data(), host.experts.size() * 4);
        P.bytes += host.experts.size() * 4;
      }
      L.small = upload_u(ctx, imm, host.small.data(), host.small.size() * 4);
      P.bytes += host.small.size() * 4;
      P.layers.push_back(std::move(L));
    }
    if (dev + 1 == place.devices) {
      const std::vector<float> fn = ck.final_norm();
      P.final_norm = upload_u(ctx, imm, fn.data(), fn.size() * 4);
      P.bytes += fn.size() * 4;
      const model::GemvShape sh{d.hidden, d.vocab, 1, 0};
      if (o.int8_head) {
        const auto q0 = std::chrono::steady_clock::now();
        std::vector<int8_t> q(size_t(d.vocab) * d.hidden);
        std::vector<float> s(d.vocab);
        quantise_int8_tiled(ck.lm_head(), d.hidden, d.vocab, q.data(), s.data());
        m.lm_head_quant_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - q0).count();
        P.head = std::make_unique<DeviceWeight>(DeviceWeight{upload(ctx, imm, q.data(), q.size()),
                                                             upload_u(ctx, imm, s.data(), s.size() * 4), sh,
                                                             model::WeightKind::Int8});
        P.bytes += q.size() + s.size() * 4;
      } else {
        // the untied bf16 head: its OWN tiles (not the embedding's)
        std::vector<uint16_t> tiles(size_t(d.hidden) * d.vocab);
        common::repack_bf16_tiled(ck.lm_head(), d.hidden, d.vocab, tiles.data());
        P.head = std::make_unique<DeviceWeight>(
            DeviceWeight{upload(ctx, imm, tiles.data(), tiles.size() * 2), nullptr, sh, model::WeightKind::Bf16});
        P.bytes += tiles.size() * 2;
      }
    }
    const size_t want = bolt::device_weight_bytes(d, place, dev, o.int8_head);
    if (P.bytes != want)
      throw std::logic_error("load_bolt: device " + std::to_string(dev) + " allocated " + std::to_string(P.bytes) +
                             " B of weights, loader::bolt::device_weight_bytes says " + std::to_string(want));
  }

  std::string names_str;
  m.unconsumed = ck.unconsumed(&names_str);
  if (m.unconsumed != 0)
    throw std::runtime_error("load_bolt: " + std::to_string(m.unconsumed) + " checkpoint tensors were not loaded: " + names_str);
  m.group_mask = ck.group_mask();
  m.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  const double gb = 1e9;
  std::printf("load_bolt: %s\n  model       %s (%s over %s): %u of %u blocks, trunk '%s', int4 %s sym%s, %zu subnormal f16 "
              "scales%s; %zu tensors, every name accounted for, 0 unconsumed\n",
              snap.c_str(), d.name.c_str(), d.architecture.c_str(), d.llm_model_type.c_str(), d.layers, full.layers,
              trunk.c_str(), m.group_mask == 3 ? "g64 + g128 (expanded)" : m.group_mask == 2 ? "g128 (expanded to g64)" : "g64",
              qc.dynamic_restating ? (", " + std::to_string(qc.dynamic_restating) + " restating '+:' rule(s) accepted").c_str() : "",
              scan.subnormal_scales, scan.gptq_bf16_scales ? ", BF16 scales converted exactly" : "", set.tensors().size());
  for (const std::string& s : m.skipped) std::printf("  note        %s\n", s.c_str());
  for (const BoltDevicePart& P : m.parts)
    std::printf("  device %u    blocks [%u, %u): weights %13zu B %7.3f GB%s%s (max_len %u)\n", P.device, P.first, P.end, P.bytes,
                P.bytes / gb, P.embed_tiles ? " + embed" : "", P.head ? (o.int8_head ? " + int8 head" : " + bf16 head") : "",
                o.max_len);
  std::printf("  load        %.1f s\n", m.seconds);
  return m;
}

}  // namespace loader
