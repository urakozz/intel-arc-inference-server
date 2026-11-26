#include "runtime/inkling2/inkling2_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "kernels/inkling2_kernels.h"
#include "runtime/dflash/dflash_control.h"

namespace runtime::inkling2 {
namespace {
bool bf16_glue() {
  const char* e = std::getenv("B70_INK_GLUE");
  if (e == nullptr || *e == '\0' || std::strcmp(e, "f32") == 0) return false;
  if (std::strcmp(e, "bf16") == 0) return true;
  throw std::runtime_error(std::string("B70_INK_GLUE=") + e + ": expected bf16 or f32 (unset: f32)");
}

using D = model::Inkling2Desc;
namespace k = kernels::inkling2;
constexpr size_t f32 = sizeof(float), bf16 = sizeof(uint16_t);
const std::string glue(k::kGlue), prep(std::string(k::kPrep) + "_FUSED"), root(model::kInkling2Prefix);
const std::string controller_glue(k::kControllerGlue);
const std::string draft_glue(k::kDraftRecurrent), draft_prep(k::kDraftPrep);

void* at(const l0::Mem& mem, size_t offset = 0) { return static_cast<std::byte*>(mem.ptr()) + offset; }

struct Pending {
  void* k;
  void* v;
  void* attn;
  void* mlp;
};

Pending pending(const ScratchBuffers& s, size_t visit, size_t m) {
  const size_t kv = m * D::kv_n * f32, hidden = m * D::hidden * f32;
  const size_t base = visit * 2 * (kv + hidden);
  const auto& mem = *s.pending_conv;
  return {at(mem, base), at(mem, base + kv), at(mem, base + 2 * kv), at(mem, base + 2 * kv + hidden)};
}

}  // namespace

Engine::Engine(l0::Context& ctx, const loader::Inkling2Checkpoint& checkpoint, uint32_t max_len, bool draft,
               uint32_t prefill_rows, bool kv8)
    : ctx_(ctx), desc_(checkpoint.desc()), weights_(checkpoint.layout()), max_len_(max_len), prefill_rows_(prefill_rows),
      kv8_(kv8), buffers_(ctx, sizes(desc_, weights_, max_len, draft, prefill_rows, kv8), weights_),
      ctrl_(static_cast<Control*>(buffers_.persistent.control.ptr())), queue_(ctx), bf16_glue_(bf16_glue()) {
  buffers_.upload(ctx_, checkpoint);
  reset();
}

void Engine::reset() {
  queue_.synchronize();
  buffers_.zero(ctx_);
  committed_ = pending_ = 0;
  prefill_logits_ = false;
}

Engine::Modules::iterator Engine::add_module(const std::string& name) {
  return modules_.emplace(name, std::make_unique<l0::Module>(ctx_, kernels::path(name))).first;
}

const void* Engine::weight(const std::string& name) const {
  const auto& w = weights_.weights.at(name);
  return at(buffers_.weights.arena(w.arena), w.offset);
}

void Engine::linear(l0::CmdList& list, const std::string& name, uint32_t rows, const void* input, void* output) {
  const auto& w = weights_.weights.at(name);
  const auto& mem = buffers_.weights.arena(w.arena);
  if (!w.scales) {
    const auto variant = kernels::gemv_bf16_variant(rows, w.K, w.N, {64, 1});
    launch(list, variant, "gemv_bf16", 64, w.N / 64, 1, at(mem, w.offset), input, output);
  } else if (rows == 1) {
    launch(list, k::int8(rows, w.K, w.N), "gemv_i8w", 64, w.N / 64, 1,
           at(mem, w.offset), at(mem, w.scales), input, output);
  } else {
    launch(list, k::int8(rows, w.K, w.N), "gemm_sm_i8w", 64, w.N / 64, 1,
           at(mem, w.offset), at(mem, w.scales), input, output);
  }
}

void Engine::record_target(l0::CmdList& list, uint32_t m) {
  const auto& s = buffers_.scratch;
  const auto& p = buffers_.persistent;
  const size_t hidden = m * D::hidden * f32, arow = m * D::hidden * bf16;
  const size_t qn = m * D::q_n * f32, rn = m * D::q_heads * D::relative_dim * f32;
  const auto residual = [&](uint32_t slot) { return at(*s.residual, slot * hidden); };
  const auto bank = [&](uint32_t slot) { return at(*s.block_bank, slot * hidden); };
  const auto activ = [&](uint32_t slot) { return at(*s.inputs, slot * arow); };
  void* current = residual(0);
  void* next = residual(1);
  launch(list, glue, "ink_embed_norm", 128, m, 1, ctrl_, weight(root + "embed.weight"),
         weight(root + "embed_norm.weight"), current, desc_.eps);

  uint32_t finished = 0, visit = 0;
  const auto mix_norm = [&](const void* last, const void* query, uint32_t queries, const void* q,
                            const void* k = nullptr, const void* v = nullptr) {
    launch(list, glue, "ink_depth_mix_norm", 128, m, queries, bank(0), query, activ(0),
           finished + 1, m, desc_.eps, q, k, v, last);
  };
  const auto control = [&](const void* last, const void* query, const void* messages,
                           const std::string& layer, std::array<const char*, 3> names,
                           std::array<const void*, 3> norms, uint32_t streams, bool shared_mix,
                           const void* raw = nullptr, const void* history = nullptr, const void* taps = nullptr,
                           const void* residual = nullptr) {
    auto* mixed = s.controller->ptr();
    auto* cat = at(*s.controller, 3 * hidden);
    auto* fg = at(*s.controller, 6 * hidden);
    auto* low = at(*s.controller, 12 * hidden);
    std::array<const void*, 3> down{}, forget{}, gain{};
    for (uint32_t i = 0; i < streams; ++i) {
      const auto prefix = layer + "controllers." + names[i] + ".";
      down[i] = weight(prefix + "down_weight");
      forget[i] = weight(prefix + "forget_weight");
      gain[i] = weight(prefix + "gain_weight");
    }
    launch(list, controller_glue, "ink_mix_cat", 128, m, streams, bank(0), last, query, messages,
           mixed, cat, finished + 1, m, uint32_t(shared_mix), desc_.eps, raw, history, taps, residual);
    launch_3d(list, k::controller(m, true), "gemv_bf16", 64, 1, streams, 1,
              down[0], down[1], down[2], nullptr, nullptr, nullptr, cat, low);
    launch_3d(list, k::controller(m, false), "gemv_bf16", 64, D::hidden / 64, streams, 2,
              forget[0], forget[1], forget[2], gain[0], gain[1], gain[2], low, fg);
    launch(list, controller_glue, "ink_controller_norm", 128, m, streams, mixed, messages, fg,
           activ(0), norms[0], norms[1], norms[2], m, desc_.eps);
  };
  for (const auto& v : desc_.visits) {
    const auto l = v.layer;
    const auto layer = model::inkling2_layer_prefix(l);
    const auto a = layer + "attn.";
    // Every core pass starts at the prelude residual. Passes two and three
    // also mix the previous pass's midpoint and output into their controllers.
    if (l == D::core_start) {
      if (v.pass == 0) {
        list.copy(residual(2), current, hidden);
      } else {
        list.copy(residual(4), current, hidden);
        list.copy(current, residual(2), hidden);
        finished = D::core_start / D::block_layers;
        launch(list, controller_glue, "ink_cross_mix", 128, m, 4 * (D::core_end - D::core_start),
               residual(3), weight(root + "middle_cycle_cross_pass_queries"),
               weight(root + "middle_cycle_cross_pass_kv_queries"), s.messages->ptr(), m);
      }
    }
    if (l == D::core_start + D::block_layers) list.copy(residual(3), current, hidden);
    const auto msg = [&](uint32_t stream) -> const void* {
      return v.controller ? at(*s.messages, (4 * (l - D::core_start) + stream) * hidden) : nullptr;
    };

    const auto qnorm = layer + (v.window ? "attn_norm.weight" : "q_input_norm.weight");
    const auto knorm = v.window ? qnorm : layer + "k_input_norm.weight";
    const auto vnorm = v.window ? qnorm : layer + "v_input_norm.weight";
    if (l == 0) {
      launch(list, glue, "ink_norm", 128, m, 1, current, nullptr, weight(qnorm), activ(0), desc_.eps);
    } else {
      const auto* query = weight(root + "attn_res.attn_queries." + std::to_string(l - 1));
      if (v.controller) {
        control(current, query, msg(0), layer, {"attn", "k", "v"},
                {weight(qnorm), weight(knorm), weight(vnorm)}, 3, bool(v.window));
      } else {
        mix_norm(current, query, v.window ? 1 : 3, weight(qnorm),
                 v.window ? nullptr : weight(knorm), v.window ? nullptr : weight(vnorm));
      }
    }
    const void* ak = v.window && !v.controller ? activ(0) : activ(1);
    const void* av = v.window && !v.controller ? activ(0) : activ(2);

    void* q = at(*s.projections), *g = at(*s.projections, qn);
    void* relative = at(*s.projections, 2 * qn), *sinks = at(*s.projections, 2 * qn + rn);
    const auto fresh = pending(s, visit, m);
    const auto& wq = weights_.weights.at(a + "wq_du.weight");
    const auto& wk = weights_.weights.at(a + "wk_dv.weight");
    const auto& wv = weights_.weights.at(a + "wv_dv.weight");
    const auto& qkv_arena = buffers_.weights.arena(wq.arena);
    const auto& wg = weights_.weights.at(a + "gate_attn.weight");
    const auto& wr = weights_.weights.at(a + "wr_du.weight");
    const auto& ws = weights_.weights.at(a + "attn_sink_proj.weight");
    launch(list, k::qkv(m) + "_FULL", m == 1 ? "gemv_i8w" : "gemm_sm_i8w", 64,
           (2 * D::q_n + 2 * D::kv_n + D::q_heads * D::relative_dim + 64) / 64, 1,
           at(qkv_arena, wq.offset), at(qkv_arena, wq.scales), activ(0), q,
           at(qkv_arena, wk.offset), at(qkv_arena, wk.scales), ak, fresh.k,
           at(qkv_arena, wv.offset), at(qkv_arena, wv.scales), av, fresh.v,
           at(qkv_arena, wg.offset), at(qkv_arena, wg.scales), g,
           at(qkv_arena, wr.offset), at(qkv_arena, wr.scales), relative,
           at(qkv_arena, ws.offset), at(qkv_arena, ws.scales), sinks);
    void* kt = nullptr;
    void* vt = nullptr;
    if (!v.window) {
      kt = at(*s.full_taps);
      vt = at(*s.full_taps, size_t(m) * 64 * f32);
      linear(list, a + "k_sconv.weight_proj", m, ak, kt);
      linear(list, a + "v_sconv.weight_proj", m, av, vt);
    }

    const bool kv8 = kv8_ && !v.window;
    const size_t kv_off = size_t(v.kv_slot) * (v.window ? kRing : max_len_) * D::kv_n * (kv8 ? 1 : bf16);
    void* ck = at(v.window ? p.ring_k : p.full_k, kv_off), *cv = at(v.window ? p.ring_v : p.full_v, kv_off);
    const size_t kh = size_t(visit) * 3 * D::kv_n * f32, ah = size_t(visit) * 3 * D::hidden * f32;
    const char* taps = v.window ? "weight" : "static_w";
    const uint32_t ring = v.window ? kRing : 0;
    const auto attn = k::attn(m, bool(v.window)) + (kv8 ? "_KV8" : "");
    auto* part = s.attention_parts->ptr();
    if (kv8) {
      const size_t scale_off = size_t(v.kv_slot) * max_len_ * D::kv_heads * bf16;
      void* ks = at(*p.full_k_scale, scale_off), *vs = at(*p.full_v_scale, scale_off);
      launch(list, prep + "_KV8", "ink_qkv_prep", 128, D::q_heads + 2 * D::kv_heads, m,
             q, relative, weight(a + "q_l2_clip.scale"), weight(a + "rel_l2_clip.scale"),
             fresh.k, fresh.v, at(p.conv_k, kh), at(p.conv_v, kh), weight(a + "k_sconv." + taps),
             weight(a + "v_sconv." + taps), kt, vt, ck, cv, ks, vs, ctrl_, ring, desc_.eps, desc_.qk_clip,
             desc_.relative_clip, desc_.log_floor, desc_.log_alpha);
      launch(list, attn, "kol_attn_decode", 128, D::kv_heads, k::kPartitions, ctrl_, q, ck, cv, ks, vs, part,
             relative, weight(a + "rel_logits_proj.proj"));
    } else {
      launch(list, prep, "ink_qkv_prep", 128, D::q_heads + 2 * D::kv_heads, m,
             q, relative, weight(a + "q_l2_clip.scale"), weight(a + "rel_l2_clip.scale"),
             fresh.k, fresh.v, at(p.conv_k, kh), at(p.conv_v, kh), weight(a + "k_sconv." + taps),
             weight(a + "v_sconv." + taps), kt, vt, ck, cv, ctrl_, ring, desc_.eps, desc_.qk_clip,
             desc_.relative_clip, desc_.log_floor, desc_.log_alpha);
      launch(list, attn, "kol_attn_decode", 128, D::kv_heads, k::kPartitions, ctrl_, q, ck, cv, part,
             relative, weight(a + "rel_logits_proj.proj"));
    }
    launch(list, attn, "kol_attn_reduce", 128, D::q_heads, m, ctrl_, part, s.attn_output->ptr(), sinks, g);
    linear(list, a + "wo_ud.weight", m, s.attn_output->ptr(), fresh.attn);
    if (v.block_boundary) list.copy(bank(finished++), current, hidden);
    const auto* mlp_query = weight(root + "attn_res.mlp_queries." + std::to_string(l));
    const auto* mlp_norm = weight(layer + "mlp_norm.weight");
    const auto* history = at(p.conv_attn, ah);
    const auto* post_taps = weight(layer + "attn_sconv.weight");
    const auto* post_residual = v.block_boundary ? nullptr : current;
    if (v.controller) {
      control(next, mlp_query, msg(3), layer, {"mlp", nullptr, nullptr},
              {mlp_norm, nullptr, nullptr}, 1, true, fresh.attn, history, post_taps, post_residual);
    } else {
      launch(list, controller_glue, "ink_post_mix_norm", 128, m, 1, bank(0), next, mlp_query,
             activ(0), mlp_norm, finished + 1, m, desc_.eps, fresh.attn, history, post_taps, post_residual);
    }

    auto* route = at(*s.routes, size_t(visit) * m * 32 * sizeof(uint32_t));
    const auto* route_bias = weight(layer + "mlp.gate.bias");
    if (m == 1) {
      launch(list, k::route() + "_M1_FUSED", "kol_route", 128, 1, 1,
             weight(layer + "mlp.gate.weight"), activ(0), route_bias, route);
    } else {
      linear(list, layer + "mlp.gate.weight", m, activ(0), q);
      launch(list, k::route(), "kol_route", 128, 1, m, q, route_bias, route);
    }
    const auto moe = k::moe(m, desc_.expert_format[l]);
    auto* expert = s.expert_hidden->ptr();
    void* draft_tap = nullptr;
    if (s.draft_taps)
      for (uint32_t t = 0; t + 1 < desc_.draft.taps.size(); ++t)
        if (visit + 1 == desc_.draft.taps[t]) draft_tap = at(*s.draft_taps, size_t(t) * D::hidden * bf16);
    if (m == 1) {
      launch(list, moe, "kol_moe_gate_up", 256, D::top_k * D::inter / 32, m, route, activ(0),
             weight(layer + "mlp.experts.w13_weight"), nullptr, expert);
      launch(list, moe + "_POST", "kol_moe_down", 128, D::hidden / 16, m, route, expert,
             weight(layer + "mlp.experts.w2_weight"), nullptr, fresh.mlp,
             at(p.conv_mlp, ah), weight(layer + "mlp_sconv.weight"), next, draft_tap);
    } else {
      auto* plan = s.expert_union->ptr();
      auto* expert_out = s.expert_output->ptr();
      launch(list, moe, "kol_moe_union_plan", D::experts, 1, 1, route, plan);
      launch(list, moe, "kol_moe_union_gate_up", 256, D::inter / 32, m * D::top_k,
             plan, activ(0), weight(layer + "mlp.experts.w13_weight"), expert);
      launch(list, moe, "kol_moe_union_down", 64, D::hidden / 64, m * D::top_k,
             plan, expert, weight(layer + "mlp.experts.w2_weight"), expert_out);
      launch(list, moe, "kol_moe_union_finish", 128, D::hidden / 128, 1, route, expert_out,
             at(p.conv_mlp, ah), weight(layer + "mlp_sconv.weight"), fresh.mlp, next, draft_tap);
    }
    std::swap(current, next);
    ++visit;
  }

  mix_norm(current, weight(root + "attn_res.output_query"), 1, weight(root + "norm.weight"));
  if (s.draft_taps)
    launch(list, glue, "ink_tap", 128, D::hidden / 128, m, nullptr, activ(0), s.draft_taps->ptr(), 3u);
  linear(list, root + "unembed.weight", m, activ(0), s.logits->ptr());
  const auto argmax = k::argmax(m);
  launch(list, argmax, "argmax_stage1", 256, (D::head_rows + 1023) / 1024, m, s.logits->ptr(), s.argmax_parts->ptr());
  launch(list, argmax, "argmax_stage2", 256, 1, 1, ctrl_, s.argmax_parts->ptr());
}

void Engine::record_commit(l0::CmdList& list, uint32_t m, uint32_t accepted) {
  // Target replay writes tentative KV rows past the cursor. The next replay
  // overwrites any rejected suffix, while convolution must keep only the
  // accepted raw rows in each visit's three-token history.
  const auto& p = buffers_.persistent;
  launch(list, glue, "ink_commit_conv", 128, 2 * (D::kv_n + D::hidden) / 128, D::visit_count,
         buffers_.scratch.pending_conv->ptr(), p.conv_k.ptr(), p.conv_v.ptr(), p.conv_attn.ptr(),
         p.conv_mlp.ptr(), m, accepted);
  if (buffers_.weights.draft) record_draft_context(list, accepted);
}

std::span<const uint32_t> Engine::verify(std::span<const uint32_t> ids) {
  if (pending_) throw std::logic_error("Commit pending Inkling rows before verifying again");
  if (ids.empty() || ids.size() > kRows || ids.size() > max_len_ - committed_)
    throw std::invalid_argument("Inkling input rows exceed the context or verify width");
  for (const auto id : ids)
    if (id >= D::vocab) throw std::invalid_argument("An Inkling input token is outside its embedding table");
  const auto m = uint32_t(ids.size());
  auto& list = target_[m - 1];
  if (!list) {
    list = std::make_unique<l0::CmdList>(l0::CmdList::regular(ctx_));
    const auto first = kernels_.size();
    record_target(*list, m);
    // A single row has exactly one legal commit. Keep its state update on
    // the target list and publish the cursor only when the caller commits.
    if (m == 1) record_commit(*list, m, 1);
    target_launches_[m - 1] = kernels_.size() - first;
    list->close();
  }
  std::copy(ids.begin(), ids.end(), ctrl_->cur_token);
  ctrl_->pos = committed_;
  ctrl_->n_active = m;
  if (m == 1) set_commit_control(1);
  queue_.execute(*list);
  queue_.synchronize();
  pending_ = m;
  prefill_logits_ = false;
  return {ctrl_->out_token, pending_};
}

void Engine::read_logits(uint32_t row, std::span<float> out) {
  if (row >= pending_ || out.size() != D::vocab_used)
    throw std::invalid_argument("Inkling logit read requires a pending row and the live vocabulary");
  copy_logits(row, out);
}

void Engine::copy_logits(uint32_t row, std::span<float> out) {
  auto list = l0::CmdList::immediate(ctx_);
  list.copy(out.data(), at(*buffers_.scratch.logits, size_t(row) * D::head_rows * f32), out.size_bytes());
  for (float& value : out) value = D::logit_cap * std::tanh(value / D::logit_cap);
}

void Engine::set_commit_control(uint32_t rows) {
  if (buffers_.weights.draft) {
    auto* dctl = static_cast<dflash::DFlashCtl*>(buffers_.draft.draft_control->ptr());
    dctl->proj_pos = committed_;
    dctl->proj_n = rows;
  }
}

void Engine::commit(uint32_t rows) {
  if (rows == 0 || rows > pending_) throw std::invalid_argument("Inkling commit must consume a prefix of the pending rows");
  if (pending_ > 1) {
    auto& list = commits_[pending_ - 1][rows - 1];
    if (!list) {
      list = std::make_unique<l0::CmdList>(l0::CmdList::regular(ctx_));
      const auto first = kernels_.size();
      record_commit(*list, pending_, rows);
      commit_launches_[pending_ - 1][rows - 1] = kernels_.size() - first;
      list->close();
    }
    set_commit_control(rows);
    queue_.execute(*list);
    queue_.synchronize();
  }
  committed_ += rows;
  pending_ = 0;
  ctrl_->pos = committed_;
}

void Engine::record_draft_context(l0::CmdList& list, uint32_t rows) {
  const auto& s = buffers_.scratch;
  const auto& d = buffers_.draft;
  constexpr size_t tap_row = 4 * D::hidden * bf16;
  constexpr size_t kv_row = D::kv_n * bf16;
  auto* taps = s.draft_taps->ptr();
  list.copy(at(*s.draft_taps, kRows * tap_row), at(*s.draft_taps, (rows - 1) * tap_row), tap_row);
  auto* mix = at(*d.context_mix);
  auto* kv_k = at(*d.context_mix, size_t(kRows) * D::hidden * bf16);
  auto* kv_v = at(*d.context_mix, size_t(kRows) * (D::hidden * bf16 + D::kv_n * f32));
  const auto* fuse = static_cast<const std::byte*>(weight("fuse_weight"));
  for (uint32_t l = 0, ring_index = 0; l < desc_.draft.layers; ++l) {
    const auto layer = "layers." + std::to_string(l) + ".";
    const auto a = layer + "self_attn.";
    const bool local = desc_.draft.window[l] != 0;
    const size_t off = local ? ring_index++ * kDraftRing * kv_row : 0;
    launch(list, draft_glue, "ink_draft_mix", 256, rows, 1, taps, fuse + l * 4 * f32,
           weight(layer + "dflare_norm.weight"), mix, desc_.draft.eps);
    linear(list, a + "k_proj_ctx.weight", rows, mix, kv_k);
    linear(list, a + "v_proj_ctx.weight", rows, mix, kv_v);
    launch(list, draft_prep, "df_ink_context", 128, D::kv_heads, rows, d.draft_control->ptr(),
           kv_k, kv_v, weight(a + "k_norm_ctx.weight"), d.rope->ptr(),
           at(local ? *d.draft_ring_k : *d.draft_full_k, off),
           at(local ? *d.draft_ring_v : *d.draft_full_v, off), local ? kDraftRing : 0u);
  }
}

void Engine::record_draft(l0::CmdList& list) {
  const auto& d = buffers_.draft;
  constexpr uint32_t m = kRows;
  constexpr size_t HR = m * D::hidden * f32, QR = m * D::q_n * f32, KR = m * D::kv_n * f32;
  constexpr size_t FF = m * D::draft_inter * f32, AR = m * D::draft_inter * bf16;
  auto* resid = at(*d.block_hidden);
  auto* norm = at(*d.block_hidden, HR);
  auto* act = at(*d.block_hidden, 2 * HR);
  auto* attention = at(*d.block_hidden, 2 * HR + AR);
  auto* q = at(*d.block_projections), *kk = at(*d.block_projections, QR);
  auto* vv = at(*d.block_projections, QR + KR);
  auto* u = at(*d.block_projections, FF);
  auto* dctl = d.draft_control->ptr();
  launch(list, draft_glue, "ink_draft_embed", 256, m, 1, ctrl_, weight(root + "embed.weight"),
         weight(root + "embed_norm.weight"), weight("mask_embed"), resid, desc_.eps);
  for (uint32_t l = 0, ring_index = 0; l < desc_.draft.layers; ++l) {
    const auto layer = "layers." + std::to_string(l) + ".";
    const auto a = layer + "self_attn.";
    const bool local = desc_.draft.window[l] != 0;
    const size_t off = local ? ring_index++ * kDraftRing * D::kv_n * bf16 : 0;
    const auto module = std::string(local ? k::kDraftWindow : k::kDraftGlobal);
    launch(list, draft_prep, "df_norm", 256, 1, m, resid, weight(layer + "input_layernorm.weight"), norm, act);
    linear(list, a + "q_proj.weight", m, act, q);
    linear(list, a + "k_proj.weight", m, act, kk);
    linear(list, a + "v_proj.weight", m, act, vv);
    launch(list, draft_prep, "df_ink_qkv", 128, D::q_heads + 2 * D::kv_heads, m, ctrl_, q, kk, vv,
           weight(a + "q_norm.weight"), weight(a + "k_norm.weight"), d.rope->ptr(), q, kk, vv);
    const float scale = 1.f / std::sqrt(float(D::head_dim));
    launch(list, module, "df_attn_part", 128, D::kv_heads, local ? kDraftRing / 128 : kPartitions,
           ctrl_, dctl, q, at(local ? *d.draft_ring_k : *d.draft_full_k, off),
           at(local ? *d.draft_ring_v : *d.draft_full_v, off), scale, d.draft_parts->ptr());
    launch(list, module, "df_attn_reduce", 128, D::q_heads, m,
           ctrl_, dctl, d.draft_parts->ptr(), q, kk, vv, scale, attention);
    linear(list, a + "o_proj.weight", m, attention, q);
    launch(list, draft_prep, "df_add", 256, D::hidden / 256, m, q, resid);
    launch(list, draft_prep, "df_norm", 256, 1, m, resid, weight(layer + "post_attention_layernorm.weight"), norm, act);
    linear(list, layer + "mlp.gate_proj.weight", m, act, q);
    linear(list, layer + "mlp.up_proj.weight", m, act, u);
    launch(list, draft_glue, "ink_draft_silu", 128, m * D::draft_inter / 128, 1, q, u, act);
    linear(list, layer + "mlp.down_proj.weight", m, act, q);
    launch(list, draft_prep, "df_add", 256, D::hidden / 256, m, q, resid);
  }

  launch(list, draft_prep, "df_norm", 256, 1, m, resid, weight("norm.weight"), norm, act);
  constexpr uint32_t rank = D::markov_rank;
  constexpr size_t head_row = size_t(D::head_rows) * f32;
  const auto* hidden = static_cast<const std::byte*>(act);
  linear(list, root + "unembed.weight", m - 1, hidden + D::hidden * bf16, d.draft_logits->ptr());
  auto* fold = at(*d.draft_logits, (m - 1) * head_row);
  auto* state = at(*d.markov);
  auto* latent = at(*d.markov, rank * bf16);
  auto* rnn = at(*d.markov, 2 * rank * bf16);
  auto* raw = at(*d.markov, (4 * rank + 2 * D::hidden) * bf16);
  list.fill(state, 0, rank * bf16);
  constexpr size_t tap_rows = m * 4 * D::hidden * bf16;
  auto* context = at(*d.context_mix);
  const auto* fuse = static_cast<const std::byte*>(weight("fuse_weight"));
  launch(list, draft_glue, "ink_draft_mix", 256, 1, 1, at(*buffers_.scratch.draft_taps, tap_rows),
         fuse + D::draft_layers * 4 * f32, weight("markov_head.dflare_norm.weight"), context, desc_.draft.eps);
  for (uint32_t j = 0; j < m - 1; ++j) {
    launch(list, draft_glue, "ink_markov_input", 128, (2 * rank + 2 * D::hidden) / 128, 1,
           state, weight("markov_head.markov_w1.weight"), hidden + size_t(j + 1) * D::hidden * bf16,
           context, ctrl_->cur_token + j, rnn);
    linear(list, "markov_head.rnn_proj.weight", 1, rnn, raw);
    launch(list, draft_glue, "ink_markov_step", 128, rank / 128, 1,
           raw, weight("markov_head.rnn_proj.bias"), state, latent);
    linear(list, "markov_head.fold.weight", 1, latent, fold);
    launch(list, draft_glue, "ink_draft_argmax", 256, 1, 1, at(*d.draft_logits, j * head_row),
           fold, weight("markov_head.fold.bias"), ctrl_->cur_token + j + 1);
  }
}

std::span<const uint32_t> Engine::propose(uint32_t anchor, uint32_t count) {
  if (!buffers_.weights.draft || pending_ || !committed_ || anchor >= D::vocab ||
      count >= kRows || count + 1 > max_len_ - committed_)
    throw std::invalid_argument("Inkling draft is outside the committed context or the draft width");
  if (!draft_list_) {
    draft_list_ = std::make_unique<l0::CmdList>(l0::CmdList::regular(ctx_));
    record_draft(*draft_list_);
    draft_list_->close();
  }
  ctrl_->cur_token[0] = anchor;
  ctrl_->pos = committed_;
  static_cast<dflash::DFlashCtl*>(buffers_.draft.draft_control->ptr())->k = kRows - 1;
  queue_.execute(*draft_list_);
  queue_.synchronize();
  return {ctrl_->cur_token + 1, count};
}

uint32_t Engine::step_drafted(uint32_t anchor, uint32_t max_drafts, std::vector<uint32_t>& output,
                              double* proposal_ms) {
  if (max_drafts > 0 && max_len_ - committed_ > 1 && buffers_.weights.draft && committed_) {
    const uint32_t k = std::min({max_drafts, kRows - 1, max_len_ - committed_ - 1});
    std::array<uint32_t, kRows> ids{};
    ids[0] = anchor;
    const auto start = proposal_ms ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto proposed = propose(anchor, k);
    if (proposal_ms)
      *proposal_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::copy(proposed.begin(), proposed.end(), ids.begin() + 1);
    const auto predicted = verify({ids.data(), k + 1});
    uint32_t accepted = 1;
    while (accepted <= k && ids[accepted] == predicted[accepted - 1]) ++accepted;
    const auto next = predicted[accepted - 1];
    output.insert(output.end(), ids.begin(), ids.begin() + accepted);
    commit(accepted);
    return next;
  }
  output.push_back(anchor);
  return append(anchor);
}

uint32_t Engine::append(uint32_t token) {
  const uint32_t next = verify({&token, 1})[0];
  commit(1);
  return next;
}

}  // namespace runtime::inkling2
