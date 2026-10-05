#include "loader/k2_loader.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "common/json.h"
#include "common/repack.h"
#include "l0/cmdlist.h"
#include "loader/k2_repack.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"
#include "loader/trained_context.h"

namespace loader {
namespace {

common::json::Value read_config(const std::string& snap) {
  std::ifstream cf(snap + "config.json");
  if (!cf) throw std::runtime_error("cannot read " + snap + "config.json");
  std::stringstream cs;
  cs << cf.rdbuf();
  return common::json::parse(cs.str());
}

l0::Mem upload(l0::Context& ctx, l0::CmdList& imm, const void* src, size_t bytes) {
  l0::Mem m(ctx, l0::MemKind::Device, bytes);
  imm.copy(m.ptr(), src, bytes);
  return m;
}
std::unique_ptr<l0::Mem> upload_u(l0::Context& ctx, l0::CmdList& imm, const void* src,
                                  size_t bytes) {
  return std::make_unique<l0::Mem>(upload(ctx, imm, src, bytes));
}

void check_len(const model::K2Desc& d, uint32_t max_len, uint32_t trained) {
  if (max_len == 0) throw std::runtime_error(d.name + ": max_len 0 is not a model length");
  if (trained != 0 && max_len > trained)
    throw std::runtime_error(d.name + ": max_len " + std::to_string(max_len) +
                             " exceeds the trained context " + std::to_string(trained) +
                             " (config.json max_position_embeddings)");
}

// One bucket of the report against the planner's formula (k2_layout.h): the loader and
// runtime/k2/k2_plan.h must agree to the byte, or `--max-len auto` plans another model.
void same(const char* what, size_t got, size_t want) {
  if (got != want)
    throw std::logic_error(std::string("load_k2: ") + what + " allocated " + std::to_string(got) +
                           " B, loader::k2_weight_bytes says " + std::to_string(want));
}

}  // namespace

bool is_k2_checkpoint(const std::string& snapshot_dir) {
  return model::is_k2_model_type(model::model_type_of(read_config(snapshot_dir)));
}

K2LoadedModel load_k2(l0::Context& ctx, const std::string& snapshot_or_repo, uint32_t max_len,
                      LmHeadForm lm_form) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::string snap = resolve_snapshot(snapshot_or_repo);
  const common::json::Value cfg = read_config(snap);
  const std::string mt = model::model_type_of(cfg);
  if (!model::is_k2_model_type(mt))
    throw std::runtime_error("load_k2: " + snap + " is model_type '" + mt + "', not k2_horizon");
  const model::K2Desc& d = model::k2();
  const QuantConfig qc = check_k2_checkpoint_config(d, cfg);
  const uint32_t trained = trained_context(cfg);
  check_len(d, max_len, trained);

  SafetensorsSet set(snap);
  const QuantScan scan = assert_quant_invariants(set);
  if (!qc.desc_act_declared && scan.g_idx_tensors != 0)
    throw std::runtime_error("load_k2: config.json declares no desc_act but the checkpoint ships " +
                             std::to_string(scan.g_idx_tensors) + " g_idx tensors");
  check_quant_scan(qc, scan);
  const std::string quant_note = ct_conversion_note(scan);
  if (!quant_note.empty()) std::fprintf(stderr, "load_k2: %s\n", quant_note.c_str());
  K2Checkpoint ck(d, set);
  ck.check_names();   // by name, both ways, before a byte is repacked
  const bool int8 = lm_form == LmHeadForm::Int8;

  K2LoadedModel m;
  m.desc = &d;
  m.max_len = max_len;
  m.trained_max_len = trained;
  K2LoadReport& r = m.report;
  r.subnormal_scales = scan.subnormal_scales;
  r.quant_note = quant_note;
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  m.embed = upload_u(ctx, imm, ck.embed(), size_t(d.vocab) * d.hidden * 2);
  r.bytes.embed = size_t(d.vocab) * d.hidden * 2;
  r.src_bf16_bytes += r.bytes.embed;
  {
    const std::vector<float> fn = ck.final_norm();
    m.final_norm = upload_u(ctx, imm, fn.data(), fn.size() * 4);
    r.bytes.small += fn.size() * 4;
    r.src_bf16_bytes += fn.size() * 2;
  }
  {
    const std::vector<float> rope = k2_rope_table(d, max_len);
    m.rope = upload_u(ctx, imm, rope.data(), rope.size() * 4);
    r.rope_bytes = rope.size() * 4;
    if (r.rope_bytes != d.rope_table_bytes(max_len))
      throw std::logic_error("load_k2: the RoPE table is not K2Desc::rope_table_bytes");
  }

  K2HostLayer host;
  m.layers.reserve(d.layers);
  for (uint32_t l = 0; l < d.layers; ++l) {
    ck.repack_layer(l, host);
    K2Layer L;
    const std::vector<model::K2LinearId> ids = d.layer_linears(l);
    for (size_t i = 0; i < ids.size(); ++i) {
      const model::GemvShape sh = d.linear(ids[i]).shape;
      const K2HostLinear& h = host.linears[i];
      L.linears.push_back(DeviceWeight{upload(ctx, imm, h.words.data(), h.words.size() * 4),
                                       upload_u(ctx, imm, h.scales.data(), h.scales.size() * 2),
                                       sh, model::WeightKind::Int4});
      r.bytes.linears += h.words.size() * 4 + h.scales.size() * 2;
    }
    L.norms = upload_u(ctx, imm, host.norms.data(), host.norms.size() * 4);
    r.bytes.small += host.norms.size() * 4;
    if (!d.is_dense(l)) {
      L.route = upload_u(ctx, imm, host.route.data(), host.route.size() * 4);
      r.bytes.small += host.route.size() * 4;
      L.router = std::make_unique<DeviceWeight>(
          DeviceWeight{upload(ctx, imm, host.router.data(), host.router.size() * 2), nullptr,
                       model::GemvShape{d.hidden, d.router_n(), 1, 0}, model::WeightKind::Bf16});
      r.bytes.routers += host.router.size() * 2;
      L.value = upload_u(ctx, imm, host.value.data(), host.value.size() * 4);
      L.gate_up = upload_u(ctx, imm, host.gate_up.data(), host.gate_up.size() * 4);
      L.down = upload_u(ctx, imm, host.down.data(), host.down.size() * 4);
      r.bytes.experts += (host.value.size() + host.gate_up.size() + host.down.size()) * 4;
    }
    r.src_int4_bytes += host.src_int4_bytes;
    r.src_bf16_bytes += host.src_bf16_bytes;
    m.layers.push_back(std::move(L));
  }

  // lm_head: the checkpoint's bf16 rows tiled for gemv_bf16, or int8 rows + fp32 scales
  // for gemv_i8w (spec 9: quantised per row on the host, loader/lm_head_int8.h).
  {
    const uint16_t* w = ck.lm_head();
    const model::GemvShape sh{d.hidden, d.vocab, 1, 0};
    r.src_bf16_bytes += size_t(d.vocab) * d.hidden * 2;
    if (int8) {
      const auto q0 = std::chrono::steady_clock::now();
      std::vector<int8_t> q(size_t(d.vocab) * d.hidden);
      std::vector<float> s(d.vocab);
      quantise_int8_tiled(w, d.hidden, d.vocab, q.data(), s.data());
      r.lm_head_quant_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - q0).count();
      m.lm_head = std::make_unique<DeviceWeight>(
          DeviceWeight{upload(ctx, imm, q.data(), q.size()), upload_u(ctx, imm, s.data(), s.size() * 4),
                       sh, model::WeightKind::Int8});
      r.bytes.lm_head = q.size() + s.size() * 4;
    } else {
      std::vector<uint16_t> t(size_t(d.vocab) * d.hidden);
      common::repack_bf16_tiled(w, d.hidden, d.vocab, t.data());
      m.lm_head = std::make_unique<DeviceWeight>(
          DeviceWeight{upload(ctx, imm, t.data(), t.size() * 2), nullptr, sh, model::WeightKind::Bf16});
      r.bytes.lm_head = t.size() * 2;
    }
  }

  std::string names;
  r.unconsumed = ck.unconsumed(&names);
  if (r.unconsumed != 0)
    throw std::runtime_error("load_k2: " + std::to_string(r.unconsumed) +
                             " checkpoint tensors were not loaded: " + names);
  const K2WeightBytes want = k2_weight_bytes(d, int8);
  same("the int4 linears", r.bytes.linears, want.linears);
  same("the MoE routers", r.bytes.routers, want.routers);
  same("the expert blocks", r.bytes.experts, want.experts);
  same("the small blocks", r.bytes.small, want.small);
  same("embed_tokens", r.bytes.embed, want.embed);
  same("lm_head", r.bytes.lm_head, want.lm_head);
  r.read_per_token = r.bytes.linears + r.bytes.small + r.bytes.lm_head +
                     size_t(d.sparse_layers()) * k2_expert_bytes(d).per_token(d);
  r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  const double gb = 1e9;
  const K2ExpertBytes eb = k2_expert_bytes(d);
  std::printf(
      "load_k2: %s\n"
      "  model       %s (%s): %u layers = %u dense + %u MoVA/MoE, hidden %u (%u norm groups)\n"
      "  quant       int4 g%u sym desc_act=false, %zu dynamic exclusion rules, %zu subnormal f16 scales\n"
      "  tensors     %zu in the index, every name accounted for, 0 unconsumed (qzeros / g_idx dropped)\n"
      "  linears     %13zu B %7.3f GB   int4 layout 0, non-expert\n"
      "  routers     %13zu B %7.3f GB   bf16 [%u][%u], rows %u.. zero\n"
      "  experts     %13zu B %7.3f GB   per layer: value %u x %zu + gate||up %u x %zu + down %u x %zu\n"
      "  small       %13zu B %7.3f GB   norms (plain w) + route biases + final norm, fp32\n"
      "  embed       %13zu B %7.3f GB   (gathered, not per-token)\n"
      "  lm_head     %13zu B %7.3f GB   %s\n"
      "  rope        %13zu B %7.3f GB   (max_len %u)\n"
      "  total       %13zu B %7.3f GB\n"
      "  read/token  %13zu B %7.3f GB   (top-%u + shared MoE, top-%u value experts; derived)\n"
      "  load        %.1f s\n",
      snap.c_str(), d.name.c_str(), d.architecture.c_str(), d.layers, d.dense_layers,
      d.sparse_layers(), d.hidden, d.norm_groups, qc.group_size, qc.dynamic_rule_count,
      scan.subnormal_scales, set.tensors().size(), r.bytes.linears, r.bytes.linears / gb,
      r.bytes.routers, r.bytes.routers / gb, d.router_n(), d.hidden, d.experts, r.bytes.experts,
      r.bytes.experts / gb, eb.value_blocks, eb.value_block, eb.moe_blocks, eb.gate_up_block,
      eb.moe_blocks, eb.down_block, r.bytes.small, r.bytes.small / gb, r.bytes.embed,
      r.bytes.embed / gb, r.bytes.lm_head, r.bytes.lm_head / gb,
      int8 ? "int8 per row + fp32 scales, quantised at load" : "bf16, as shipped", r.rope_bytes,
      r.rope_bytes / gb, max_len, r.total(), r.total() / gb, r.read_per_token,
      r.read_per_token / gb, d.top_k, d.value_top_k, r.seconds);
  return m;
}

void set_max_len_k2(l0::Context& ctx, K2LoadedModel& m, uint32_t max_len) {
  check_len(*m.desc, max_len, m.trained_max_len);
  const std::vector<float> rope = k2_rope_table(*m.desc, max_len);
  const size_t bytes = rope.size() * 4;
  if (bytes != m.desc->rope_table_bytes(max_len))
    throw std::logic_error("set_max_len_k2: the RoPE table is not K2Desc::rope_table_bytes");
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  auto fresh = upload_u(ctx, imm, rope.data(), bytes);   // before the old one is freed
  m.rope = std::move(fresh);
  m.report.rope_bytes = bytes;
  m.max_len = max_len;
}

}  // namespace loader
