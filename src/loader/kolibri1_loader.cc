#include "loader/kolibri1_loader.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "common/json.h"
#include "common/repack.h"
#include "l0/cmdlist.h"
#include "loader/kolibri1_repack.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"
#include "loader/trained_context.h"

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

void check_len(const model::Kolibri1Desc& d, uint32_t max_len, uint32_t trained) {
  if (max_len == 0) throw std::runtime_error(d.name + ": max_len 0 is not a model length");
  const uint32_t cap = trained ? trained : d.trained_max_len;
  if (max_len > cap)
    throw std::runtime_error(d.name + ": max_len " + std::to_string(max_len) + " exceeds the trained context " +
                             std::to_string(cap) +
                             " - a longer context is spec 20 decision 3 (open: 262144, or more on the NoPE "
                             "full layers' published 1M validation)");
}

// The descriptor from config.json at the checkpoint's layer count, the arm given; then the
// development-mode truncation.
model::Kolibri1Desc desc_of(const common::json::Value& cfg, model::KolAttnForm a, uint32_t layers_limit) {
  model::Kolibri1Desc d = model::kolibri1_desc(cfg, a);
  if (layers_limit != 0) {
    if (layers_limit > d.layers)
      throw std::runtime_error("--layers " + std::to_string(layers_limit) + ": the checkpoint has " +
                               std::to_string(d.layers) + " layers");
    d.layers = layers_limit;
  }
  return d;
}

std::string model_type_of(const common::json::Value& cfg) {
  if (!cfg.is_object()) return "";
  const common::json::Value* v = cfg.find("model_type");
  return v && v->is_string() ? v->str() : "";
}

}  // namespace

size_t KolLoadedModel::total_bytes() const {
  size_t t = 0;
  for (const KolDevicePart& p : parts) t += p.bytes + p.rope_bytes;
  return t;
}

bool is_kolibri1_checkpoint(const std::string& snapshot_dir) {
  return model::is_kolibri1_model_type(model_type_of(read_json(snapshot_dir + "config.json")));
}

model::Kolibri1Desc kolibri1_checkpoint_desc(const std::string& snapshot_or_repo, uint32_t layers_limit) {
  const std::string snap = resolve_snapshot(snapshot_or_repo);
  const common::json::Value cfg = read_json(snap + "config.json");
  // The arm from the index's names (no shard is opened); none shipped: int4, the lighter arm.
  model::KolAttnForm a = model::KolAttnForm::Int4;
  if (std::ifstream ix(snap + "model.safetensors.index.json"); ix) {
    std::stringstream s;
    s << ix.rdbuf();
    const common::json::Value idx = common::json::parse(s.str());
    if (const common::json::Value* wm = idx.find("weight_map"); wm && wm->is_object()) {
      if (wm->find("model.layers.0.self_attn.q_proj.weight")) a = model::KolAttnForm::Bf16;
    }
  }
  return desc_of(cfg, a, layers_limit);
}

KolLoadedModel load_kolibri1(const std::vector<l0::Context*>& devices, const std::string& snapshot_or_repo,
                             uint32_t max_len, const model::KolPlacement& placement, LmHeadForm lm_form,
                             uint32_t layers_limit) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::string snap = resolve_snapshot(snapshot_or_repo);
  const common::json::Value cfg = read_json(snap + "config.json");
  const std::string mt = model_type_of(cfg);
  if (!model::is_kolibri1_model_type(mt))
    throw std::runtime_error("load_kolibri1: " + snap + " is model_type '" + mt + "', not kolibri1");
  const QuantConfig qc = check_kolibri1_checkpoint_config(cfg);
  const uint32_t trained = trained_context(cfg);

  SafetensorsSet set(snap);
  // The checkpoint's own descriptor (every layer), then its arm from the names.
  model::Kolibri1Desc full = model::kolibri1_desc(cfg, model::KolAttnForm::Int4);
  full.attn = kol_attn_form(set, full);
  model::kolibri1_desc(cfg, full.attn);   // re-held at the arm (nothing in the config depends on it)
  check_len(full, max_len, trained);
  const QuantScan scan = assert_quant_invariants(set);
  if (!qc.desc_act_declared && scan.g_idx_tensors != 0)
    throw std::runtime_error("load_kolibri1: config.json declares no desc_act but the checkpoint ships " +
                             std::to_string(scan.g_idx_tensors) + " g_idx tensors");
  check_quant_scan(qc, scan);
  if (scan.ct_linears != 0)
    throw std::runtime_error("load_kolibri1: compressed-tensors linears in a Kolibri-1 checkpoint - spec 20 "
                             "serves its own AutoRound auto_round:auto_gptq export only (§3.1)");
  KolCheckpoint ck(full, set);
  ck.check_names();   // by name, both ways, before a byte is repacked

  const model::Kolibri1Desc d = desc_of(cfg, full.attn, layers_limit);
  model::validate(placement, d);
  if (devices.size() != placement.devices)
    throw std::invalid_argument("load_kolibri1: " + std::to_string(devices.size()) + " device(s) for a " +
                                std::to_string(placement.devices) + "-device placement");
  const bool int8 = lm_form == LmHeadForm::Int8;

  KolLoadedModel m;
  m.desc = d;
  m.placement = placement;
  m.max_len = max_len;
  m.trained_max_len = trained ? trained : d.trained_max_len;
  m.int8_head = int8;
  m.checkpoint_layers = full.layers;
  m.subnormal_scales = scan.subnormal_scales;
  size_t skipped = 0;
  if (d.layers < full.layers) skipped = ck.skip_layers_from(d.layers);

  const std::vector<float> rope = kol_rope_table(d, max_len);
  KolHostLayer host;
  m.parts.resize(placement.devices);
  for (uint32_t dev = 0; dev < placement.devices; ++dev) {
    l0::Context& ctx = *devices[dev];
    l0::CmdList imm = l0::CmdList::immediate(ctx);
    KolDevicePart& P = m.parts[dev];
    P.device = dev;
    P.first = placement.first(dev);
    P.end = placement.end(dev);
    if (dev == 0) {
      P.embed = upload_u(ctx, imm, ck.embed(), kol_embed_bytes(d));
      P.bytes += kol_embed_bytes(d);
    }
    if (kol_device_has_sliding(d, placement, dev)) {
      P.rope = upload_u(ctx, imm, rope.data(), rope.size() * 4);
      P.rope_bytes = rope.size() * 4;
      if (P.rope_bytes != d.rope_table_bytes(max_len))
        throw std::logic_error("load_kolibri1: the RoPE table is not Kolibri1Desc::rope_table_bytes");
    }
    for (uint32_t l = P.first; l < P.end; ++l) {
      ck.repack_layer(l, host);
      KolLayer L;
      const model::GemvShape qs = d.linear(model::KolLinearId::Qkv).shape, os = d.linear(model::KolLinearId::OProj).shape;
      if (d.attn == model::KolAttnForm::Int4) {
        L.qkv = std::make_unique<DeviceWeight>(DeviceWeight{upload(ctx, imm, host.qkv_words.data(), host.qkv_words.size() * 4),
                             upload_u(ctx, imm, host.qkv_scales.data(), host.qkv_scales.size() * 2), qs,
                             model::WeightKind::Int4});
        L.oproj = std::make_unique<DeviceWeight>(DeviceWeight{upload(ctx, imm, host.oproj_words.data(), host.oproj_words.size() * 4),
                               upload_u(ctx, imm, host.oproj_scales.data(), host.oproj_scales.size() * 2), os,
                               model::WeightKind::Int4});
        P.bytes += (host.qkv_words.size() + host.oproj_words.size()) * 4 +
                   (host.qkv_scales.size() + host.oproj_scales.size()) * 2;
      } else {
        L.qkv = std::make_unique<DeviceWeight>(DeviceWeight{upload(ctx, imm, host.qkv_bf16.data(), host.qkv_bf16.size() * 2), nullptr, qs,
                             model::WeightKind::Bf16});
        L.oproj = std::make_unique<DeviceWeight>(DeviceWeight{upload(ctx, imm, host.oproj_bf16.data(), host.oproj_bf16.size() * 2), nullptr, os,
                               model::WeightKind::Bf16});
        P.bytes += (host.qkv_bf16.size() + host.oproj_bf16.size()) * 2;
      }
      L.norms = upload_u(ctx, imm, host.norms.data(), host.norms.size() * 4);
      L.router = std::make_unique<DeviceWeight>(
          DeviceWeight{upload(ctx, imm, host.router.data(), host.router.size() * 2), nullptr,
                       model::GemvShape{d.hidden, d.router_n(), 1, 0}, model::WeightKind::Bf16});
      L.bias = upload_u(ctx, imm, host.bias.data(), host.bias.size() * 4);
      L.gate_up = upload_u(ctx, imm, host.gate_up.data(), host.gate_up.size() * 4);
      L.down = upload_u(ctx, imm, host.down.data(), host.down.size() * 4);
      L.shared_gate_up = upload_u(ctx, imm, host.shared_gate_up.data(), host.shared_gate_up.size() * 2);
      L.shared_down = upload_u(ctx, imm, host.shared_down.data(), host.shared_down.size() * 2);
      P.bytes += host.norms.size() * 4 + host.router.size() * 2 + host.bias.size() * 4 +
                 (host.gate_up.size() + host.down.size()) * 4 +
                 (host.shared_gate_up.size() + host.shared_down.size()) * 2;
      P.layers.push_back(std::move(L));
    }
    if (dev + 1 == placement.devices) {
      const std::vector<float> fn = ck.final_norm();
      P.final_norm = upload_u(ctx, imm, fn.data(), fn.size() * 4);
      P.bytes += fn.size() * 4;
      // lm_head: bf16 tiled for gemv_bf16, or int8 rows + fp32 scales for gemv_i8w (spec 9).
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
    const size_t want = kol_device_weight_bytes(d, placement, dev, int8);
    if (P.bytes != want)
      throw std::logic_error("load_kolibri1: device " + std::to_string(dev) + " allocated " + std::to_string(P.bytes) +
                             " B of weights, loader::kol_device_weight_bytes says " + std::to_string(want));
  }

  std::string names;
  m.unconsumed = ck.unconsumed(&names);
  if (m.unconsumed != 0)
    throw std::runtime_error("load_kolibri1: " + std::to_string(m.unconsumed) +
                             " checkpoint tensors were not loaded: " + names);
  m.read_per_token = size_t(d.layers) * kol_layer_read_per_token(d) + kol_final_norm_bytes(d) +
                     kol_lm_head_bytes(d, int8);
  m.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  const double gb = 1e9;
  std::printf("load_kolibri1: %s\n  model       %s (%s): %u of %u layers (%u full / NoPE), %s attention, "
              "int4 g%u sym, %zu subnormal f16 scales; %zu tensors, every name accounted for, 0 unconsumed\n",
              snap.c_str(), d.name.c_str(), d.architecture.c_str(), d.layers, full.layers, d.full_before(d.layers),
              model::kol_attn_form_name(d.attn), qc.group_size, scan.subnormal_scales, set.tensors().size());
  if (skipped)
    std::printf("  note        development mode (--layers %u): layers %u..%u not loaded (%zu tensors consumed by "
                "design); the head reads layer %u's residual\n",
                d.layers, d.layers, full.layers - 1, skipped, d.layers - 1);
  for (const KolDevicePart& P : m.parts)
    std::printf("  device %u    layers [%u, %u): weights %13zu B %7.3f GB%s%s, RoPE %zu B (max_len %u)\n", P.device,
                P.first, P.end, P.bytes, P.bytes / gb, P.embed ? " + embed" : "",
                P.lm_head ? (int8 ? " + int8 head" : " + bf16 head") : "", P.rope_bytes, max_len);
  std::printf("  read/token  %13zu B %7.3f GB   (top-%u + shared experts, attention, routers, head; derived)\n"
              "  load        %.1f s\n",
              m.read_per_token, m.read_per_token / gb, d.top_k, m.seconds);
  return m;
}

}  // namespace loader
