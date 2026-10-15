#include "loader/muse_loader.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "common/json.h"
#include "common/repack.h"
#include "l0/cmdlist.h"
#include "loader/snapshot.h"

namespace loader {
namespace {

common::json::Value read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream s;
  s << f.rdbuf();
  return common::json::parse(s.str());
}

l0::Mem upload(l0::Context& ctx, l0::CmdList& imm, const void* src, size_t bytes) {
  l0::Mem m(ctx, l0::MemKind::Device, bytes);
  imm.copy(m.ptr(), src, bytes);
  return m;
}
std::unique_ptr<l0::Mem> upload_u(l0::Context& ctx, l0::CmdList& imm, const void* src, size_t bytes) {
  return std::make_unique<l0::Mem>(upload(ctx, imm, src, bytes));
}
// One fused int4 linear: layout 0's words and scales as DeviceWeight::mem and ::scales; layout 1's tiles alone.
std::unique_ptr<DeviceWeight> upload_linear(l0::Context& ctx, l0::CmdList& imm, const MuseHostLinear& h,
                                            const model::GemvShape& sh) {
  std::unique_ptr<l0::Mem> sc;
  if (!h.scales.empty()) sc = upload_u(ctx, imm, h.scales.data(), h.scales.size() * 2);
  return std::make_unique<DeviceWeight>(
      DeviceWeight{upload(ctx, imm, h.words.data(), h.words.size() * 4), std::move(sc), sh, model::WeightKind::Int4});
}

std::string model_type_of(const common::json::Value& cfg) {
  if (!cfg.is_object()) return "";
  const common::json::Value* v = cfg.find("model_type");
  return v && v->is_string() ? v->str() : "";
}

}  // namespace

size_t MuseWeights::total_bytes() const {
  size_t t = 0;
  for (const MuseDevicePart& p : parts) t += p.bytes;
  return t;
}

bool is_muse_checkpoint(const std::string& snapshot_dir) {
  return model::is_muse_model_type(model_type_of(read_json(snapshot_dir + "config.json")));
}

model::MuseGlimmerDesc muse_checkpoint_desc(const std::string& snapshot_or_repo, uint32_t layers_override) {
  return model::check_muse_config(read_json(resolve_snapshot(snapshot_or_repo) + "config.json"), layers_override);
}

MuseWeights load_muse_glimmer(const std::vector<l0::Context*>& devices, const std::string& snapshot_or_repo,
                              uint32_t max_len, const model::MusePlacement& placement, bool int8,
                              uint32_t layers_override, const runtime::muse::MuseReserve& reserve) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::string snap = resolve_snapshot(snapshot_or_repo);
  // The host half's checks, before anything else: config.json, the quantisation, the names, the data asserts.
  const MuseCheckpoint ck(snap, layers_override);
  const model::MuseGlimmerDesc& d = ck.desc();
  model::validate(placement, d);
  if (devices.size() != placement.devices)
    throw std::invalid_argument("load_muse_glimmer: " + std::to_string(devices.size()) + " device(s) for a " +
                                std::to_string(placement.devices) + "-device placement");
  if (max_len == 0 || max_len > d.max_pos)
    throw std::invalid_argument("load_muse_glimmer: max_len " + std::to_string(max_len) + " outside [1, " +
                                std::to_string(d.max_pos) + "] (spec 23 decision 6)");
  // The plan before the first allocation.
  {
    std::vector<size_t> caps;
    for (l0::Context* c : devices) caps.push_back(c->memory_bytes());
    runtime::muse::require_fits(runtime::muse::plan_at(d, placement, max_len, int8, reserve, caps));
  }

  MuseWeights m;
  m.desc = d;
  m.placement = placement;
  m.max_len = max_len;
  m.int8_head = int8;
  m.form = ck.form();
  m.checkpoint_layers = ck.checkpoint_layers();
  m.tensors = ck.tensors();
  m.vision_skipped = ck.vision_skipped();
  m.skipped_layer_tensors = ck.skipped_layer_tensors();
  m.subnormal_scales = ck.subnormal_scales();

  MuseHostLayer host;
  m.parts.resize(placement.devices);
  for (uint32_t dev = 0; dev < placement.devices; ++dev) {
    l0::Context& ctx = *devices[dev];
    l0::CmdList imm = l0::CmdList::immediate(ctx);
    MuseDevicePart& P = m.parts[dev];
    P.device = dev;
    P.first = placement.first(dev);
    P.end = placement.end(dev);
    if (dev == 0) {
      P.embed = upload_u(ctx, imm, ck.embed(), muse_embed_bytes(d));
      P.bytes += muse_embed_bytes(d);
    }
    for (uint32_t l = P.first; l < P.end; ++l) {
      ck.layer(l, host);
      MuseLayer L;
      L.qgkv = upload_linear(ctx, imm, host.qgkv, d.linear(model::MuseLinearId::QGKV).shape);
      L.oproj = upload_linear(ctx, imm, host.oproj, d.linear(model::MuseLinearId::OProj).shape);
      L.gate_up = upload_linear(ctx, imm, host.gate_up, d.linear(model::MuseLinearId::GateUp).shape);
      L.down = upload_linear(ctx, imm, host.down, d.linear(model::MuseLinearId::Down).shape);
      const std::vector<float> nb = host.norm_block();
      L.norms = upload_u(ctx, imm, nb.data(), nb.size() * 4);
      if (host.bytes() != muse_layer_bytes(d).total())
        throw std::logic_error("load_muse_glimmer: layer " + std::to_string(l) + " repacked to " +
                               std::to_string(host.bytes()) + " B, loader::muse_layer_bytes says " +
                               std::to_string(muse_layer_bytes(d).total()));
      P.bytes += host.bytes();
      P.layers.push_back(std::move(L));
    }
    if (dev + 1 == placement.devices) {
      const std::vector<float> fn = ck.final_norm();
      P.final_norm = upload_u(ctx, imm, fn.data(), fn.size() * 4);
      P.bytes += fn.size() * 4;
      const uint16_t* w = ck.lm_head();
      const model::GemvShape sh{d.hidden, d.vocab, 1, 0};
      if (int8) {
        const auto q0 = std::chrono::steady_clock::now();
        std::vector<int8_t> q(size_t(d.vocab) * d.hidden);
        std::vector<float> s(d.vocab);
        quantise_int8_tiled(w, d.hidden, d.vocab, q.data(), s.data());
        m.lm_head_quant_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - q0).count();
        P.lm_head = std::make_unique<DeviceWeight>(DeviceWeight{
            upload(ctx, imm, q.data(), q.size()), upload_u(ctx, imm, s.data(), s.size() * 4), sh, model::WeightKind::Int8});
        P.bytes += q.size() + s.size() * 4;
      } else {
        std::vector<uint16_t> t(size_t(d.vocab) * d.hidden);
        common::repack_bf16_tiled(w, d.hidden, d.vocab, t.data());
        P.lm_head = std::make_unique<DeviceWeight>(
            DeviceWeight{upload(ctx, imm, t.data(), t.size() * 2), nullptr, sh, model::WeightKind::Bf16});
        P.bytes += t.size() * 2;
      }
    }
    const size_t want = muse_device_weight_bytes(d, placement, dev, int8);
    if (P.bytes != want)
      throw std::logic_error("load_muse_glimmer: device " + std::to_string(dev) + " allocated " + std::to_string(P.bytes) +
                             " B of weights, loader::muse_device_weight_bytes says " + std::to_string(want));
  }

  const std::vector<std::string> left = ck.unconsumed();
  m.unconsumed = left.size();
  if (!left.empty()) {
    std::string names;
    for (size_t i = 0; i < left.size() && i < 5; ++i) names += (i ? ", " : "") + left[i];
    throw std::runtime_error("load_muse_glimmer: " + std::to_string(left.size()) +
                             " checkpoint tensors were not loaded: " + names);
  }
  m.read_per_token = muse_read_per_token(d, int8);
  m.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  char line[512];
  std::snprintf(line, sizeof line,
                "Muse-Glimmer-30B: %u layers (int4 g%u, %s spelling), %zu vision tensors skipped, head %s, devices %u, "
                "%zu B",
                d.layers, ck.quant().group_size, muse_quant_form_name(m.form), m.vision_skipped, int8 ? "int8" : "bf16",
                placement.devices, m.total_bytes());
  m.line = line;
  std::printf("load_muse_glimmer: %s\n  %s\n", snap.c_str(), line);
  if (m.skipped_layer_tensors)
    std::printf("  note        development mode (--layers %u of %u): %zu tensors of the later layers consumed by "
                "design; the head reads layer %u's residual\n",
                d.layers, m.checkpoint_layers, m.skipped_layer_tensors, d.layers - 1);
  for (const MuseDevicePart& P : m.parts)
    std::printf("  device %u    layers [%u, %u): weights %13zu B %7.3f GB%s%s\n", P.device, P.first, P.end, P.bytes,
                P.bytes / 1e9, P.embed ? " + embed" : "", P.lm_head ? (int8 ? " + int8 head" : " + bf16 head") : "");
  std::printf("  checks      %zu tensors, every text name accounted for (0 unconsumed), %zu subnormal f16 scales\n"
              "  read/token  %13zu B %7.3f GB (derived)\n  load        %.1f s (int8 head %.2f s)\n",
              m.tensors, m.subnormal_scales, m.read_per_token, m.read_per_token / 1e9, m.seconds,
              m.lm_head_quant_seconds);
  return m;
}

}  // namespace loader
