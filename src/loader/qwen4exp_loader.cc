#include "loader/qwen4exp_loader.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "common/json.h"
#include "common/repack.h"
#include "l0/cmdlist.h"
#include "loader/qwen4exp_repack.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"
#include "runtime/memory_plan.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

namespace loader {
namespace {

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

l0::Mem upload(l0::Context& ctx, l0::CmdList& imm, const void* src, size_t bytes) {
  l0::Mem m(ctx, l0::MemKind::Device, bytes);
  imm.copy(m.ptr(), src, bytes);
  return m;
}
std::unique_ptr<l0::Mem> upload_u(l0::Context& ctx, l0::CmdList& imm, const void* src, size_t bytes) {
  return std::make_unique<l0::Mem>(upload(ctx, imm, src, bytes));
}
std::unique_ptr<DeviceWeight> upload_w(l0::Context& ctx, l0::CmdList& imm, const Q4HostWeight& w, size_t& bytes) {
  bytes += w.bytes();
  if (w.kind == model::WeightKind::Int4)
    return std::make_unique<DeviceWeight>(DeviceWeight{upload(ctx, imm, w.words.data(), w.words.size() * 4),
                                                       upload_u(ctx, imm, w.scales.data(), w.scales.size() * 2), w.shape,
                                                       model::WeightKind::Int4});
  return std::make_unique<DeviceWeight>(
      DeviceWeight{upload(ctx, imm, w.tiles.data(), w.tiles.size() * 2), nullptr, w.shape, model::WeightKind::Bf16});
}

// One host layer onto its device (the fields of the other layer kind stay null).
Q4Layer upload_layer(l0::Context& ctx, l0::CmdList& imm, const Q4HostLayer& h, size_t& bytes) {
  Q4Layer L;
  const auto blob = [&](const std::vector<uint8_t>& v) {
    bytes += v.size();
    return v.empty() ? nullptr : upload_u(ctx, imm, v.data(), v.size());
  };
  L.hc_attn = blob(h.hc_attn);
  L.hc_mlp = blob(h.hc_mlp);
  if (h.gdn_qkvz.bytes()) {
    L.gdn_qkvz = upload_w(ctx, imm, h.gdn_qkvz, bytes);
    L.gdn_ab = upload_w(ctx, imm, h.gdn_ab, bytes);
    L.gdn_out = upload_w(ctx, imm, h.gdn_out, bytes);
    L.gdn_small = blob(h.gdn_small);
  }
  if (h.qsa_qkvg.bytes()) {
    L.qsa_qkvg = upload_w(ctx, imm, h.qsa_qkvg, bytes);
    L.qsa_idx = upload_w(ctx, imm, h.qsa_idx, bytes);
    L.qsa_o = upload_w(ctx, imm, h.qsa_o, bytes);
    L.qsa_small = blob(h.qsa_small);
  }
  L.router = upload_w(ctx, imm, h.router, bytes);
  L.gate_up = upload_u(ctx, imm, h.gate_up.data(), h.gate_up.size() * 4);
  L.down = upload_u(ctx, imm, h.down.data(), h.down.size() * 4);
  bytes += (h.gate_up.size() + h.down.size()) * 4;
  L.shared = blob(h.shared);
  L.shared_down_offset = h.shared_gate_up;
  L.ple = blob(h.ple);
  return L;
}

// The checkpoint's PLE I64 tensors against the formula for `d` (Review Focus 3): refused naming the tensor.
void check_ple_constants(const model::Qwen4ExpDesc& d, const Q4PleConstants& c) {
  const std::string p = model::Qwen4ExpDesc::layer_prefix(d.ple_layer) + "ple.ple_embedding.";
  if (c.multipliers != q4_ple_multipliers(d.vocab, d.ngram, 0, d.ple_seed))
    throw std::runtime_error("load_qwen4exp: '" + p + "layer_multipliers' differs from the formula (M:1040-1049, seed " +
                             std::to_string(d.ple_seed) + ")");
  const std::vector<uint64_t> sizes = q4_ple_primes(d.ple_base, d.ple_heads, 0);
  if (c.sizes != sizes)
    throw std::runtime_error("load_qwen4exp: '" + p + "ngram_heads_vocab_sizes' differs from the primes after " +
                             std::to_string(d.ple_base) + " (ngram_vocab_size_base)");
  if (c.offsets != q4_ple_offsets(sizes))
    throw std::runtime_error("load_qwen4exp: '" + p + "ngram_heads_offsets' is not the head sizes' running sum");
}

struct Opened {
  std::string snap;
  common::json::Value cfg;
};
Opened open(const std::string& snapshot_or_repo) {
  Opened o;
  o.snap = resolve_snapshot(snapshot_or_repo);
  o.cfg = read_json(o.snap + "config.json");
  const std::string mt = model_type_of(o.cfg);
  if (!model::is_qwen4exp_model_type(mt))
    throw std::runtime_error("load_qwen4exp: " + o.snap + " is model_type '" + mt + "', not qwen4_exp");
  return o;
}

model::Qwen4ExpDesc limited(const model::Qwen4ExpDesc& full, uint32_t layers_limit) {
  if (layers_limit == 0) return full;
  if (layers_limit > full.layers)
    throw std::runtime_error("--layers " + std::to_string(layers_limit) + ": the checkpoint has " +
                             std::to_string(full.layers) + " layers");
  return runtime::qwen4exp::truncated(full, layers_limit);
}

}  // namespace

size_t Q4LoadedModel::total_bytes() const {
  size_t t = 0;
  for (const Q4DevicePart& p : parts) t += p.bytes + p.rope_bytes;
  return t;
}

bool is_qwen4exp_checkpoint(const std::string& snapshot_dir) {
  return model::is_qwen4exp_model_type(model_type_of(read_json(snapshot_dir + "config.json")));
}

model::Qwen4ExpDesc qwen4exp_checkpoint_desc(const std::string& snapshot_or_repo, uint32_t layers_limit) {
  const Opened o = open(snapshot_or_repo);
  const SafetensorsSet set(o.snap);   // the headers only: the forms are read from the names
  const model::Qwen4ExpDesc pre = model::qwen4exp_desc(o.cfg, model::Q4Forms{});
  return limited(model::qwen4exp_desc(o.cfg, q4_forms(set, pre)), layers_limit);
}

Q4LoadedModel load_qwen4exp(const std::vector<l0::Context*>& devices, const std::string& snapshot_or_repo,
                            uint32_t max_len, const model::Q4Placement& placement, LmHeadForm lm_form,
                            uint32_t layers_limit, bool mtp, const std::string& ple_dir_arg) {
  const auto t0 = std::chrono::steady_clock::now();
  const Opened o = open(snapshot_or_repo);
  const QuantConfig qc = check_qwen4exp_checkpoint_config(o.cfg);
  SafetensorsSet set(o.snap);
  // The checkpoint's own descriptor (every layer), then its forms from the names.
  const model::Qwen4ExpDesc pre = model::qwen4exp_desc(o.cfg, model::Q4Forms{});
  const model::Qwen4ExpDesc full = model::qwen4exp_desc(o.cfg, q4_forms(set, pre));
  if (max_len == 0) throw std::runtime_error(full.name + ": max_len 0 is not a model length");
  if (max_len > full.trained_max_len)
    throw std::runtime_error(full.name + ": max_len " + std::to_string(max_len) + " exceeds the trained context " +
                             std::to_string(full.trained_max_len) + " (context against cache is spec 21 decision 9)");
  if (max_len % runtime::kMaxLenQuantum != 0)
    throw std::runtime_error(full.name + ": max_len " + std::to_string(max_len) + " is not a multiple of " +
                             std::to_string(runtime::kMaxLenQuantum) + " (the indexer's blocks of 4 and the prefill grid)");
  const QuantScan scan = assert_quant_invariants(set);
  if (!qc.desc_act_declared && scan.g_idx_tensors != 0)
    throw std::runtime_error("load_qwen4exp: config.json declares no desc_act but the checkpoint ships " +
                             std::to_string(scan.g_idx_tensors) + " g_idx tensors");
  check_quant_scan(qc, scan);
  if (scan.ct_linears != 0)
    throw std::runtime_error("load_qwen4exp: compressed-tensors linears in a qwen4_exp checkpoint - spec 21 serves "
                             "AutoRound auto_round:auto_gptq exports only (§5, one format)");
  if (qc.group_size != full.forms.expert_group)
    throw std::runtime_error("load_qwen4exp: quantization_config.group_size is " + std::to_string(qc.group_size) +
                             " but the routed experts' scales are g" + std::to_string(full.forms.expert_group) +
                             " - the label and the bytes disagree");
  Q4Checkpoint ck(full, set);
  ck.check_names(mtp);   // by name, both ways, before a byte is repacked

  const model::Qwen4ExpDesc d = limited(full, layers_limit);
  model::validate(placement, d);
  if (devices.size() != placement.devices)
    throw std::invalid_argument("load_qwen4exp: " + std::to_string(devices.size()) + " device(s) for a " +
                                std::to_string(placement.devices) + "-device placement");
  const bool int8 = lm_form == LmHeadForm::Int8;
  // The plan before the first allocation: the full model is refused here, naming spec 22.
  {
    std::array<size_t, runtime::kPpDevices> caps{};
    for (uint32_t i = 0; i < placement.devices; ++i) caps[i] = devices[i]->memory_bytes();
    runtime::qwen4exp::require_fits(runtime::qwen4exp::plan(d, placement, max_len, int8, mtp), caps,
                                    size_t(runtime::kDefaultReserveGb * 1e9));
  }
  const std::string ple_dir = ple_dir_arg.empty() ? q4_ple_dir(o.snap) : ple_dir_arg;
  check_ple_constants(d, ck.ple_constants());

  Q4LoadedModel m;
  m.desc = d;
  m.placement = placement;
  m.max_len = max_len;
  m.int8_head = int8;
  m.mtp = mtp;
  m.checkpoint_layers = full.layers;
  m.subnormal_scales = scan.subnormal_scales;
  if (d.layers < full.layers) m.skipped += ck.skip_layers_from(d.layers);
  const size_t visual = ck.skip_prefix("model.visual."), shards = ck.skip_ple_shards();
  const size_t mtp_skipped = mtp ? 0 : ck.skip_prefix("mtp.");
  // The PLE table, once, through the context of the device holding the PLE layer.
  m.ple_device = placement.device_of(d.ple_layer);
  m.ple = load_q4_ple(*devices[m.ple_device], d, ple_dir);

  const std::vector<float> rope = q4_rope_table(d, max_len);
  Q4HostLayer host;
  m.parts.resize(placement.devices);
  for (uint32_t dev = 0; dev < placement.devices; ++dev) {
    l0::Context& ctx = *devices[dev];
    l0::CmdList imm = l0::CmdList::immediate(ctx);
    Q4DevicePart& P = m.parts[dev];
    P.device = dev;
    P.first = placement.first(dev);
    P.end = placement.end(dev);
    if (dev == 0) {
      P.embed = upload_u(ctx, imm, ck.embed(), q4_embed_bytes(d));
      P.bytes += q4_embed_bytes(d);
    }
    if (q4_device_has_qsa(d, placement, dev, mtp)) {
      P.rope = upload_u(ctx, imm, rope.data(), rope.size() * 4);
      P.rope_bytes = rope.size() * 4;
      if (P.rope_bytes != d.rope_table_bytes(max_len))
        throw std::logic_error("load_qwen4exp: the RoPE table is not Qwen4ExpDesc::rope_table_bytes");
    }
    for (uint32_t l = P.first; l < P.end; ++l) {
      ck.repack_layer(l, host);
      P.layers.push_back(upload_layer(ctx, imm, host, P.bytes));
    }
    if (dev + 1 == placement.devices) {
      const std::vector<uint8_t> fm = ck.final_mixer();
      P.final_mixer = upload_u(ctx, imm, fm.data(), fm.size());
      P.bytes += fm.size();
      // lm_head: spec 9's int8 rows + fp32 scales for gemv_i8w, or bf16 tiles for gemv_bf16.
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
      if (mtp) {
        ck.repack_mtp(host);
        P.mtp = std::make_unique<Q4Layer>(upload_layer(ctx, imm, host, P.bytes));
        const std::vector<uint8_t> fc = ck.mtp_fc(), mm = ck.mtp_mixer();
        P.mtp_fc = upload_u(ctx, imm, fc.data(), fc.size());
        P.mtp_mixer = upload_u(ctx, imm, mm.data(), mm.size());
        P.bytes += fc.size() + mm.size();
      }
    }
    const size_t want = q4_device_weight_bytes(d, placement, dev, int8, mtp);
    if (P.bytes != want)
      throw std::logic_error("load_qwen4exp: device " + std::to_string(dev) + " allocated " + std::to_string(P.bytes) +
                             " B of weights, loader::q4_device_weight_bytes says " + std::to_string(want));
  }

  std::string names;
  m.unconsumed = ck.unconsumed(&names);
  if (m.unconsumed != 0)
    throw std::runtime_error("load_qwen4exp: " + std::to_string(m.unconsumed) + " checkpoint tensors were not loaded: " +
                             names);
  for (uint32_t l = 0; l < d.layers; ++l) m.read_per_token += q4_layer_read_per_token(d, l);
  m.read_per_token += q4_final_mixer_bytes(d) + q4_lm_head_bytes(d, int8);
  m.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  const double gb = 1e9;
  std::printf("load_qwen4exp: %s\n  model       %s (%s): %u of %u layers (%u QSA), dense %s, shared %s, routed experts "
              "int4 g%u%s, %zu subnormal f16 scales; %zu tensors, every name accounted for, 0 unconsumed\n",
              o.snap.c_str(), d.name.c_str(), d.architecture.c_str(), d.layers, full.layers, d.qsa_before(d.layers),
              model::q4_form_name(d.forms.dense), model::q4_form_name(d.forms.shared), d.forms.expert_group,
              d.forms.expert_group == 128 ? " (expanded exactly to g64)" : "", scan.subnormal_scales,
              set.tensors().size());
  std::printf("  skipped     by design: %zu visual tensors, %zu bf16 PLE shards (the int8 file replaces them)%s\n", visual,
              shards, mtp ? "" : (", " + std::to_string(mtp_skipped) + " mtp.* (the head not loaded)").c_str());
  if (m.skipped)
    std::printf("  note        development mode (--layers %u): layers %u..%u not loaded (%zu tensors consumed by "
                "design); the final mixer reads layer %u's streams\n",
                d.layers, d.layers, full.layers - 1, m.skipped, d.layers - 1);
  for (const Q4DevicePart& P : m.parts)
    std::printf("  device %u    layers [%u, %u): weights %13zu B %7.3f GB%s%s%s, RoPE %zu B (max_len %u)\n", P.device, P.first,
                P.end, P.bytes, P.bytes / gb, P.embed ? " + embed" : "",
                P.lm_head ? (int8 ? " + int8 head" : " + bf16 head") : "", P.mtp ? " + MTP head (RTN int4 g64 experts)" : "",
                P.rope_bytes, max_len);
  std::printf("  host PLE    %u + %u host-USM ranges via device %u: %13zu B %7.3f GB (%s scales), pinned in %.1f s; %zu "
              "page tags read back, %zu rows compared; MemAvailable %.1f GB before, %.1f GB after\n",
              d.ple_heads, d.ple_heads, m.ple_device, m.ple.bytes, m.ple.bytes / gb, q4_ple_scale_name(m.ple.scale),
              m.ple.seconds, m.ple.page_words.size(), m.ple.rows_checked, m.ple.mem_before / gb, m.ple.mem_after / gb);
  std::printf("  read/token  %13zu B %7.3f GB   (top-%u + shared experts, HC, mixers, routers, head; derived)\n"
              "  load        %.1f s\n",
              m.read_per_token, m.read_per_token / gb, d.top_k, m.seconds);
  return m;
}

}  // namespace loader
