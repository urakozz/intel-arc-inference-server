#include "runtime/inkling2/inkling2_engine.h"

#include <algorithm>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <functional>
#include <print>
#include <stdexcept>

#include "kernels/dflash_kernels.h"
#include "kernels/inkling2_kernels.h"
#include "loader/dflash_repack.h"
#include "loader/lm_head_int8.h"
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
const std::string controller_glue(k::kControllerGlue), split_glue(k::kSplitGlue);
const std::string draft_glue(k::kDraftRecurrent), draft_prep(k::kDraftPrep);
namespace kd = kernels::dflash;

void* at(const l0::Mem& mem, size_t offset = 0) { return static_cast<std::byte*>(mem.ptr()) + offset; }

struct Pending {
  void* k;
  void* v;
  void* attn;
  void* mlp;
};

// kv_n: the visit's k / v channels on this card (Tp::kv_n).
Pending pending(const ScratchBuffers& s, size_t visit, size_t m, size_t kv_n) {
  const size_t kv = m * kv_n * f32, hidden = m * D::hidden * f32;
  const size_t base = visit * 2 * (kv + hidden);
  const auto& mem = *s.pending_conv;
  return {at(mem, base), at(mem, base + kv), at(mem, base + 2 * kv), at(mem, base + 2 * kv + hidden)};
}

}  // namespace

Engine::Engine(l0::Context& ctx, const loader::Inkling2Checkpoint& checkpoint, uint32_t max_len, bool draft,
               uint32_t prefill_rows, bool kv8, bool int8_pass)
    : Engine(ctx, checkpoint, max_len, draft, prefill_rows, kv8, int8_pass, Stage{}, checkpoint.layout()) {
  buffers_.upload(ctx_, checkpoint);
  loaded();
}

Engine::Engine(l0::Context& ctx, const loader::Inkling2Checkpoint& checkpoint, uint32_t max_len, bool draft,
               uint32_t prefill_rows, bool kv8, bool int8_pass, const Stage& stage,
               const loader::Inkling2WeightLayout& layout, const Tp& tp)
    : ctx_(ctx), desc_(checkpoint.desc()), weights_(layout), max_len_(max_len), prefill_rows_(prefill_rows),
      kv8_(kv8), int8_pass_(int8_pass), stage_(stage), tp_(tp), share_(stage_share(desc_, weights_, stage)),
      buffers_(ctx, sizes(desc_, weights_, max_len, draft, prefill_rows, kv8, int8_pass, stage, tp), weights_, share_),
      ctrl_(static_cast<Control*>(buffers_.persistent.control.ptr())), queue_(ctx),
      bf16_glue_(int8_pass || bf16_glue()) {}

void Engine::loaded() {
  if (stage_.head()) build_candidate_head();
  if (int8_pass_ && prefill_rows_) rotate_int8_linears();
  reset();
}

std::array<std::pair<void*, size_t>, 6> Engine::handoff_rows(uint32_t m, bool prompt) const {
  const auto& residual = prompt ? *buffers_.prefill.residual : *buffers_.scratch.residual;
  const auto& bank = prompt ? *buffers_.prefill.block_bank : *buffers_.scratch.block_bank;
  const auto* taps = prompt ? buffers_.prefill.draft_taps.get() : buffers_.scratch.draft_taps.get();
  const size_t rows = size_t(m) * D::hidden * f32;
  return {{{at(residual), rows}, {at(bank), rows}, {at(bank, rows), rows}, {at(residual, 3 * rows), rows},
           {at(residual, 4 * rows), rows}, {taps ? taps->ptr() : nullptr, taps ? 4 * size_t(m) * D::hidden * bf16 : 0}}};
}

// The int8 pass's linears rotated and re-quantised once, from the int8 weights decode reads (ink_int8.cl).
void Engine::rotate_int8_linears() {
  auto list = l0::CmdList::immediate(ctx_);
  auto* base = static_cast<std::byte*>(buffers_.weights.prompt_int8->ptr());
  list.fill(base, 0, buffers_.weights.prompt_int8->size());   // the columns past N
  size_t offset = 0;
  for (uint32_t l = 0; l < D::layers; ++l)
    for (const auto& lin : int8_prompt_linears(tp_)) {
      if (!share_.layers[l]) continue;
      const auto name = model::inkling2_layer_prefix(l) + lin.name;
      const auto& w = weights_.weights.at(name);
      contract_assert(w.K == lin.K && w.N == lin.N && w.scales != 0);
      const auto& mem = buffers_.weights.arena(w.arena);
      auto it = modules_.find("ink_int8_K" + std::to_string(lin.K));
      if (it == modules_.end()) it = add_module("ink_int8_K" + std::to_string(lin.K));
      auto k = it->second->kernel("ink_rot_weight");
      k.group_size(16);
      k.arg_ptr(0, at(mem, w.offset));
      k.arg_ptr(1, at(mem, w.scales));
      k.arg_ptr(2, base + offset);
      const uint32_t columns = int8_columns(lin);
      k.arg_ptr(3, base + offset + size_t(lin.K) * columns);
      k.arg(4, columns);
      list.launch(k, lin.N);
      int8_linears_.emplace(name, RotatedLinear{offset, lin.K, lin.N, columns});
      offset += int8_linear_bytes(lin);
    }
  contract_assert(offset == buffers_.weights.prompt_int8->size());
}

void Engine::reset() {
  queue_.synchronize();
  buffers_.zero(ctx_, written_, max_len_, share_.full);
  committed_ = pending_ = written_ = 0;
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
  if (w.arena == loader::Inkling2WeightLayout::kDraft && weights_.draft_format == loader::DraftFormat::Int4G64) {
    // gemm_sm where kSmallMTuning has the shape (gemv's arguments, grid and output slices), else gemv (one row)
    const uint32_t slices = kd::int4_slices(w.K, w.N);
    const auto* packed = at(mem, w.offset);
    const auto kind = kernels::gemv_kind(rows);
    const bool sm = kind == kernels::LinearKind::SmallM && kernels::small_m_tuning(w.K, w.N, slices, 1);
    launch(list, kd::linear_variant(2, rows, w.K, w.N, kind), sm ? "gemm_sm" : "gemv", 64, w.N / 64, slices, packed,
           packed, input, output);
  } else if (!w.scales) {
    const auto t = k::bf16_tiling(w.N);
    launch(list, k::bf16_linear(rows, w.K, w.N), "gemv_bf16", t.cols * t.ksplit, w.N / t.cols, 1, at(mem, w.offset),
           input, output);
  } else if (rows == 1 || w.N == k::kTapColumns) {
    // The full visits' 64-column tap projections stay on gemv_i8w at every width (one gemm_sm work-group took 52 us
    // at 2 rows), their K split in 16 (kTapKSplit) at every row count.
    const bool tap = w.N == k::kTapColumns;
    const uint32_t wg = tap ? 64 * k::kTapKSplit : 64;
    launch(list, kernels::gemv_i8w_variant(rows, w.K, w.N) + (tap && rows == 1 ? "_KS" : ""), "gemv_i8w", wg,
           w.N / 64, 1, at(mem, w.offset), at(mem, w.scales), input, output);
  } else {
    const auto& t = *kernels::small_m_tuning(w.K, w.N, 1, 2);
    launch(list, k::int8(rows, w.K, w.N), "gemm_sm_i8w", kernels::gemm_sm_wg_size(t), w.N / kernels::gemm_sm_cols(t),
           1, at(mem, w.offset), at(mem, w.scales), input, output);
  }
}

void Engine::record_target(l0::CmdList& list, uint32_t m, bool candidates, bool deep) {
  const auto& s = buffers_.scratch;
  const auto& p = buffers_.persistent;
  const size_t hidden = m * D::hidden * f32, arow = m * D::hidden * bf16;
  const size_t qn = m * tp_.q_n() * f32, rn = m * tp_.rel_n() * f32;
  const auto residual = [&](uint32_t slot) { return at(*s.residual, slot * hidden); };
  const auto bank = [&](uint32_t slot) { return at(*s.block_bank, slot * hidden); };
  const auto activ = [&](uint32_t slot) { return at(*s.inputs, slot * arow); };
  // `--parallel tp`: this card's first head of each kind, its per-head weights' offsets into the whole tensors.
  const size_t q0 = size_t(tp_.rank) * tp_.q_heads(), kv0 = size_t(tp_.rank) * tp_.kv_heads();
  void* current = residual(0);
  void* next = residual(1);
  const auto stage = [&](const char* name) { profile_stage_ = name; };
  stage("embed");
  if (stage_.first == 0)
    launch(list, glue, "ink_embed_norm", 128, m, 1, ctrl_, weight(root + "embed.weight"),
           weight(root + "embed_norm.weight"), current, desc_.eps);

  uint32_t finished = 0, visit = 0;
  // The mixes and norms a row over kGlueSplit one-sub-group work-groups, bitwise the one-work-group kernels: the
  // scale sums, then the mixed row (fp32, in the controller's `mixed` rows) and its norm sums, then the norm.
  auto* parts = s.glue_parts->ptr();
  auto* norm_parts = at(*s.glue_parts, size_t(3) * kRows * 2 * kGlueBank * kGlueLanes * f32);
  auto* mixed = s.controller->ptr();
  const auto split = [&](const char* entry, uint32_t streams, auto... args) {
    launch_3d(list, split_glue, entry, 16, kGlueSplit, m, streams, args...);
  };
  const auto mix_norm = [&](const void* last, const void* query, uint32_t queries, const void* q,
                            const void* k = nullptr, const void* v = nullptr) {
    stage("mix");
    split("ink_split_sums", queries, bank(0), last, query, parts, finished + 1, m, 0u);
    launch_3d(list, split_glue, "ink_split_mix", 128, D::hidden / 128, m, queries, bank(0), last, parts, mixed,
              finished + 1, m);
    split("ink_split_squares", queries, mixed, norm_parts, m);
    split("ink_split_norm", queries, mixed, norm_parts, activ(0), m, desc_.eps, q, k, v);
  };
  const auto control = [&](const void* last, const void* query, const void* messages,
                           const std::string& layer, std::array<const char*, 3> names,
                           std::array<const void*, 3> norms, uint32_t streams, bool shared_mix,
                           const void* raw = nullptr) {
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
    stage(streams == 1 ? "controller mlp" : "controller qkv");
    // The attention residual is already in `last` (o_proj's epilogue or ink_post_conv).
    contract_assert(raw == nullptr);
    split("ink_split_sums", streams, bank(0), last, query, parts, finished + 1, m, uint32_t(shared_mix));
    launch_3d(list, split_glue, "ink_split_mix_cat", 128, D::hidden / 128, m, streams, bank(0), last, parts, mixed,
              finished + 1, m);
    split("ink_split_cat_squares", streams, mixed, messages, norm_parts, m);
    split("ink_split_cat", streams, mixed, messages, norm_parts, cat, m, desc_.eps);
    launch_3d(list, k::controller(m, true), "gemv_bf16", k::kControllerDown.cols * k::kControllerDown.ksplit,
              D::controller_rank / k::kControllerDown.cols, streams, 1,
              down[0], down[1], down[2], nullptr, nullptr, nullptr, cat, low);
    launch_3d(list, k::controller(m, false), "gemv_bf16", 64, D::hidden / 64, streams, 2,
              forget[0], forget[1], forget[2], gain[0], gain[1], gain[2], low, fg);
    launch_3d(list, split_glue, "ink_split_blend", 128, D::hidden / 128, m, streams, mixed, messages, fg, m, streams);
    split("ink_split_squares", streams, mixed, norm_parts, m);
    split("ink_split_norm", streams, mixed, norm_parts, activ(0), m, desc_.eps, norms[0], norms[1], norms[2]);
  };
  for (const auto& v : desc_.visits) {
    const auto l = v.layer;
    if (visit == stage_.end) break;
    if (visit < stage_.first) {   // the card before's visit: the walk's bookkeeping only
      if (l == D::core_start && v.pass != 0) finished = D::core_start / D::block_layers;
      finished += v.block_boundary;
      std::swap(current, next);
      ++visit;
      continue;
    }
    if (visit == stage_.first && visit) {
      // The split (handoff_rows' rows are in place): the prelude residual is finished block 1, and this pass's
      // controller messages are mixed again from the previous pass's layer-12 input and output - the same launch,
      // so the same values as on the card before.
      contract_assert(current == residual(0) && v.controller && l != D::core_start);
      list.copy(residual(2), bank(1), hidden);
      launch(list, controller_glue, "ink_cross_mix", 128, m, 4 * (D::core_end - D::core_start),
             residual(3), weight(root + "middle_cycle_cross_pass_queries"),
             weight(root + "middle_cycle_cross_pass_kv_queries"), s.messages->ptr(), m);
    }
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
        stage("controller");
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
      stage("mix");
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
    const uint32_t here = visit - stage_.first;   // the visit's index among this card's
    const auto fresh = pending(s, here, m, tp_.kv_n());
    const auto& wq = weights_.weights.at(a + "wq_du.weight");
    const auto& wk = weights_.weights.at(a + "wk_dv.weight");
    const auto& wv = weights_.weights.at(a + "wv_dv.weight");
    const auto& qkv_arena = buffers_.weights.arena(wq.arena);
    const auto& wg = weights_.weights.at(a + "gate_attn.weight");
    const auto& wr = weights_.weights.at(a + "wr_du.weight");
    const auto& ws = weights_.weights.at(a + "attn_sink_proj.weight");
    stage("qkv");
    launch(list, tp_.on() ? k::tp_qkv(m) : k::qkv(m) + "_FULL", m == 1 ? "gemv_i8w" : "gemm_sm_i8w", 64,
           (2 * tp_.q_n() + 2 * tp_.kv_n() + tp_.rel_n() + 64) / 64, 1,
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
      stage("taps");
      linear(list, a + "k_sconv.weight_proj", m, ak, kt);
      linear(list, a + "v_sconv.weight_proj", m, av, vt);
      // tp: every head's taps are projected (64 columns a row); this card's kv heads read theirs, 4 a head
      kt = static_cast<std::byte*>(kt) + kv0 * D::conv_taps * f32;
      vt = static_cast<std::byte*>(vt) + kv0 * D::conv_taps * f32;
    }

    const bool kv8 = kv8_ && !v.window;
    const size_t kv_off = size_t(slot(v)) * (v.window ? kRing : max_len_) * tp_.kv_n() * (kv8 ? 1 : bf16);
    void* ck = at(v.window ? p.ring_k : p.full_k, kv_off), *cv = at(v.window ? p.ring_v : p.full_v, kv_off);
    const size_t kh = size_t(here) * 3 * tp_.kv_n() * f32, ah = size_t(here) * 3 * D::hidden * f32;
    const char* taps = v.window ? "weight" : "static_w";
    // the convolution taps of this card's channels: [channel][4] on window visits, [4][kv_n] (from the first) on full
    const size_t tap_off = (v.window ? kv0 * D::head_dim * D::conv_taps : kv0 * D::head_dim) * f32;
    const auto* k_taps = static_cast<const std::byte*>(weight(a + "k_sconv." + taps)) + tap_off;
    const auto* v_taps = static_cast<const std::byte*>(weight(a + "v_sconv." + taps)) + tap_off;
    const auto* q_scale = static_cast<const std::byte*>(weight(a + "q_l2_clip.scale")) + q0 * D::head_dim * f32;
    const auto* r_scale = static_cast<const std::byte*>(weight(a + "rel_l2_clip.scale")) + q0 * D::relative_dim * f32;
    const uint32_t ring = v.window ? kRing : 0;
    const auto attn = (tp_.on() ? k::tp_attn(m, bool(v.window)) : k::attn(m, bool(v.window))) + (kv8 ? "_KV8" : "");
    const uint32_t row_split = tp_.on() && m > 1 ? k::kTpAttnRowSplit : 1;   // the rows over grid z (ATTN_RSPLIT)
    const auto attn_decode = attn + (row_split > 1 ? "_RS2" : "");
    const std::string prep_variant = tp_.on() ? std::string(k::kTpPrep) : prep;
    auto* part = s.attention_parts->ptr();
    // A deep verify's full visit: the split prompt flash over the same q, keys and partials area (kVerifySpans x
    // kRows rows of {m, l, O} is kol_attn's kPartitions x kRows), then its merge into attn_output.
    const bool flash = deep && !v.window && !tp_.on();   // tp: its half heads stay on kol_attn
    const auto verify = k::prefill_attn(false) + (kv8 ? "_KV8_PAIR_VERIFY" : "_PAIR_VERIFY");
    static_assert(kVerifySpans == k::kPartitions);
    stage("attention");
    if (kv8) {
      const size_t scale_off = size_t(slot(v)) * max_len_ * tp_.kv_heads() * bf16;
      void* ks = at(*p.full_k_scale, scale_off), *vs = at(*p.full_v_scale, scale_off);
      launch(list, prep_variant + "_KV8", "ink_qkv_prep", 128, tp_.q_heads() + 2 * tp_.kv_heads(), m,
             q, relative, q_scale, r_scale, fresh.k, fresh.v, at(p.conv_k, kh), at(p.conv_v, kh), k_taps, v_taps,
             kt, vt, ck, cv, ks, vs, ctrl_, ring, desc_.eps, desc_.qk_clip, desc_.relative_clip, desc_.log_floor,
             desc_.log_alpha);
      if (flash)
        launch_3d(list, verify, "kol_pf_flash_attn", k::kPrefillAttnGroup, 1, D::kv_heads, kVerifySpans, q, ck, cv, ks,
                  vs, s.attn_output->ptr(), ctrl_, relative, weight(a + "rel_logits_proj.proj"), sinks, g, part);
      else
        launch_3d(list, attn_decode, "kol_attn_decode", 128, tp_.kv_heads(), k::kPartitions, row_split, ctrl_, q, ck,
                  cv, ks, vs, part, relative, weight(a + "rel_logits_proj.proj"));
    } else {
      launch(list, prep_variant, "ink_qkv_prep", 128, tp_.q_heads() + 2 * tp_.kv_heads(), m,
             q, relative, q_scale, r_scale, fresh.k, fresh.v, at(p.conv_k, kh), at(p.conv_v, kh), k_taps, v_taps,
             kt, vt, ck, cv, ctrl_, ring, desc_.eps, desc_.qk_clip, desc_.relative_clip, desc_.log_floor,
             desc_.log_alpha);
      if (flash)
        launch_3d(list, verify, "kol_pf_flash_attn", k::kPrefillAttnGroup, 1, D::kv_heads, kVerifySpans, q, ck, cv,
                  s.attn_output->ptr(), ctrl_, relative, weight(a + "rel_logits_proj.proj"), sinks, g, part);
      else
        launch_3d(list, attn_decode, "kol_attn_decode", 128, tp_.kv_heads(), k::kPartitions, row_split, ctrl_, q, ck,
                  cv, part, relative, weight(a + "rel_logits_proj.proj"));
    }
    if (flash)
      launch(list, verify, "kol_pf_flash_merge", D::head_dim, D::q_heads, m, part, s.attn_output->ptr(), ctrl_,
             kVerifySpans, sinks, g);
    else   // tp: the sinks' projection is whole (every head's logit, 64 a row); this card's heads read theirs
      launch(list, attn, "kol_attn_reduce", 128, tp_.q_heads(), m, ctrl_, part, s.attn_output->ptr(),
             static_cast<std::byte*>(sinks) + q0 * f32, g);
    stage("o_proj");
    const auto* history = at(p.conv_attn, ah);
    const auto* post_taps = weight(layer + "attn_sconv.weight");
    const auto* post_residual = v.block_boundary ? nullptr : current;
    // The output convolution and the partial sum before the MLP mix, which then starts from `next`: o_proj's
    // epilogue at one row; ink_post_conv at more (a row's convolution reads the earlier rows' outputs).
    const void* raw = fresh.attn;
    if (tp_.on()) {
      // This card's heads' share of o_proj, the all-reduce into the row, then its convolution and partial sum.
      const auto& wo = weights_.weights.at(a + "wo_ud.weight");
      const auto& wo_arena = buffers_.weights.arena(wo.arena);
      auto* share = buffers_.tp.tp_part->ptr();
      launch(list, k::tp_o_proj(m), "gemm_sm_i8w", 256, D::hidden / 32, 1, at(wo_arena, wo.offset),
             at(wo_arena, wo.scales), s.attn_output->ptr(), share);
      record_all_reduce(list, m, share, fresh.attn);
      launch(list, glue, "ink_post_conv", 128, D::hidden / 128, m, fresh.attn, history, post_taps, post_residual,
             next, nullptr);
      raw = nullptr;
    } else if (m == 1) {
      const auto& wo = weights_.weights.at(a + "wo_ud.weight");
      const auto& wo_arena = buffers_.weights.arena(wo.arena);
      launch(list, k::attn_post(), "gemm_sm_i8w", 256, D::hidden / 32, 1, at(wo_arena, wo.offset),
             at(wo_arena, wo.scales), s.attn_output->ptr(), fresh.attn, history, post_taps, post_residual, next);
      raw = nullptr;
    } else {
      linear(list, a + "wo_ud.weight", m, s.attn_output->ptr(), fresh.attn);
      launch(list, glue, "ink_post_conv", 128, D::hidden / 128, m, fresh.attn, history, post_taps, post_residual,
             next, nullptr);
      raw = nullptr;
    }
    if (v.block_boundary) list.copy(bank(finished++), current, hidden);
    const auto* mlp_query = weight(root + "attn_res.mlp_queries." + std::to_string(l));
    const auto* mlp_norm = weight(layer + "mlp_norm.weight");
    if (v.controller) {
      control(next, mlp_query, msg(3), layer, {"mlp", nullptr, nullptr},
              {mlp_norm, nullptr, nullptr}, 1, true, raw);
    } else {
      stage("mix");
      contract_assert(raw == nullptr);
      split("ink_split_sums", 1, bank(0), next, mlp_query, parts, finished + 1, m, 0u);
      launch_3d(list, split_glue, "ink_split_mix", 128, D::hidden / 128, m, 1, bank(0), next, parts, mixed,
                finished + 1, m);
      split("ink_split_squares", 1, mixed, norm_parts, m);
      split("ink_split_norm", 1, mixed, norm_parts, activ(0), m, desc_.eps, mlp_norm, nullptr, nullptr);
    }

    auto* route = at(*s.routes, size_t(here) * m * 32 * sizeof(uint32_t));
    const auto* route_bias = weight(layer + "mlp.gate.bias");
    stage("router");
    // The router's logits a work-group a tile (bitwise the fused kernel's sums), then the selection; the logits
    // pass through the head's buffer, which the head writes only after the last visit.
    launch(list, k::route() + "_M1_FUSED", "kol_route_logits", 16 * k::kRouteSplit, 8, m,
           weight(layer + "mlp.gate.weight"), activ(0), s.logits->ptr());
    launch(list, k::route_select(), "kol_route", 128, 1, m, s.logits->ptr(), route_bias, route);
    const auto moe = tp_.on() ? k::tp_moe(m, desc_.expert_format[l]) : k::moe(m, desc_.expert_format[l]);
    auto* expert = s.expert_hidden->ptr();
    void* draft_tap = nullptr;
    if (s.draft_taps)
      for (uint32_t t = 0; t + 1 < desc_.draft.taps.size(); ++t)
        if (visit + 1 == desc_.draft.taps[t]) draft_tap = at(*s.draft_taps, size_t(t) * D::hidden * bf16);
    stage(desc_.expert_format[l] == desc_.expert_format[0] ? "experts fp4" : "experts fp8");
    if (tp_.on()) {
      // This card's half of every routed expert's width, the weighted sum of its parts, the all-reduce into the
      // row, then the convolution and the partial sum (one row: kol_moe_down's sum; more: the union's finish stops
      // at the sum, INK_MOE_TP_SUM).
      auto* share = buffers_.tp.tp_part->ptr();
      if (m == 1) {
        launch(list, moe, "kol_moe_gate_up", 256, D::top_k * tp_.inter() / 32, m, route, activ(0),
               weight(layer + "mlp.experts.w13_weight"), nullptr, expert);
        launch(list, moe, "kol_moe_down", 128, D::hidden / 16, m, route, expert,
               weight(layer + "mlp.experts.w2_weight"), nullptr, share);
      } else {
        auto* plan = s.expert_union->ptr();
        auto* expert_out = s.expert_output->ptr();
        launch(list, moe, "kol_moe_union_plan", D::experts, 1, 1, route, plan);
        launch(list, moe, "kol_moe_union_gate_up", 256, tp_.inter() / 32, m * D::top_k, plan, activ(0),
               weight(layer + "mlp.experts.w13_weight"), expert);
        launch(list, moe, "kol_moe_union_down", 64, D::hidden / 64, m * D::top_k, plan, expert,
               weight(layer + "mlp.experts.w2_weight"), expert_out);
        launch(list, moe, "kol_moe_union_finish", 128, D::hidden / 128, 1, route, expert_out,
               at(p.conv_mlp, ah), weight(layer + "mlp_sconv.weight"), share, next, nullptr);
      }
      record_all_reduce(list, m, share, fresh.mlp);
      launch(list, glue, "ink_post_conv", 128, D::hidden / 128, m, fresh.mlp, at(p.conv_mlp, ah),
             weight(layer + "mlp_sconv.weight"), next, next, draft_tap);
    } else if (m == 1) {
      launch(list, moe, "kol_moe_gate_up", 256, D::top_k * D::inter / 32, m, route, activ(0),
             weight(layer + "mlp.experts.w13_weight"), nullptr, expert);
      launch(list, moe + "_POST", "kol_moe_down", 128, D::hidden / 16, m, route, expert,
             weight(layer + "mlp.experts.w2_weight"), nullptr, fresh.mlp,
             at(p.conv_mlp, ah), weight(layer + "mlp_sconv.weight"), next, draft_tap);
    } else {
      // Each expert of the rows' union read once for every row that chose it (consecutive tokens share about half
      // their experts), each row's sums the one-row kernels', bitwise.
      auto* plan = s.expert_union->ptr();
      auto* expert_out = s.expert_output->ptr();
      launch(list, moe, "kol_moe_union_plan", D::experts, 1, 1, route, plan);
      launch(list, moe, "kol_moe_union_gate_up", 256, D::inter / 32, m * D::top_k, plan, activ(0),
             weight(layer + "mlp.experts.w13_weight"), expert);
      launch(list, moe, "kol_moe_union_down", 64, D::hidden / 64, m * D::top_k, plan, expert,
             weight(layer + "mlp.experts.w2_weight"), expert_out);
      launch(list, moe, "kol_moe_union_finish", 128, D::hidden / 128, m, route, expert_out,
             at(p.conv_mlp, ah), weight(layer + "mlp_sconv.weight"), fresh.mlp, next, draft_tap);
    }
    std::swap(current, next);
    ++visit;
  }
  if (!stage_.head()) {   // the rows cross (handoff_rows), PpEngine's copies
    contract_assert(current == residual(0));
    return;
  }

  mix_norm(current, weight(root + "attn_res.output_query"), 1, weight(root + "norm.weight"));
  stage("head");
  if (s.draft_taps)
    launch(list, glue, "ink_tap", 128, D::hidden / 128, m, nullptr, activ(0), s.draft_taps->ptr(), 3u);
  const auto argmax = k::argmax(m);
  if (candidates) {
    record_candidate_head(list, m);
  } else {
    linear(list, root + "unembed.weight", m, activ(0), s.logits->ptr());
    launch(list, argmax, "argmax_stage1", 256, (D::head_rows + 1023) / 1024, m, s.logits->ptr(),
           s.argmax_parts->ptr());
  }
  launch(list, argmax, "argmax_stage2", 256, 1, 1, ctrl_, s.argmax_parts->ptr());
  profile_stage_.clear();
}

// PERF-123 on a greedy step: the 2-bit head over the rows into `logits`, the top kCandidates of each row, those columns
// of the int8 head gathered and rescored with the full head's own kernel (gemv_i8w at one row, gemm_sm_i8w at more:
// a column's score does not depend on the width, so each is bitwise the full head's), then the best of them into
// argmax_parts for argmax_stage2. The pick equals the full head's whenever its winner is among the candidates.
void Engine::record_candidate_head(l0::CmdList& list, uint32_t m) {
  const auto& s = buffers_.scratch;
  const auto& head = weights_.weights.at(root + "unembed.weight");
  const auto& head_arena = buffers_.weights.arena(head.arena);
  auto* part = at(*s.head_candidates);
  auto* cand = at(*s.head_candidates, kCandidateParts);
  auto* cw = at(*s.head_candidates, kCandidateParts + kCandidateIds);
  auto* cs = at(*s.head_candidates, kCandidateParts + kCandidateIds + kCandidateWeights);
  auto* score = at(*s.head_candidates, kCandidateParts + kCandidateIds + kCandidateWeights + kCandidateScales);
  const auto* activ = s.inputs->ptr();
  const auto& t2 = *kernels::small_m_tuning(D::hidden, D::head_rows, 1, 3);
  launch(list, kernels::gemm_sm_variant(m, t2), "gemm_sm", kernels::gemm_sm_wg_size(t2),
         D::head_rows / kernels::gemm_sm_cols(t2), kernels::gemm_sm_grid_y(t2), buffers_.weights.candidate_head->ptr(),
         buffers_.weights.candidate_head->ptr(), activ, s.logits->ptr());
  launch(list, std::string(k::kHeadCandidates), "df_candidate_select1", 256, uint32_t(kCandidateChunks), m,
         s.logits->ptr(), part);
  launch(list, std::string(k::kHeadCandidates), "df_candidate_stage2", 256, m, 1, part, cand);
  launch(list, k::head_candidates(), "head_cand_gather", 256, m * kCandidates * (D::hidden / 16) / 256, 1,
         at(head_arena, head.offset), at(head_arena, head.scales), cand, cw, cs);
  if (m == 1) {
    launch(list, k::int8(1, D::hidden, kCandidateColumns), "gemv_i8w", 64, kCandidateColumns / 64, 1, cw, cs, activ,
           score);
  } else {
    const auto& tx = *kernels::small_m_tuning(D::hidden, kCandidateColumns, 1, 2);
    launch(list, k::int8(m, D::hidden, kCandidateColumns), "gemm_sm_i8w", kernels::gemm_sm_wg_size(tx),
           kCandidateColumns / kernels::gemm_sm_cols(tx), 1, cw, cs, activ, score);
  }
  launch(list, k::head_candidates(), "head_cand_argmax", 256, m, 1, score, cand, s.argmax_parts->ptr());
}

void Engine::record_all_reduce(l0::CmdList& list, uint32_t m, const void* part, void* sum) {
  static_assert(kTpFoldG == k::kTpFoldG);
  const auto& t = buffers_.tp;
  const auto cross = k::tp_cross(m, tp_.rank);
  const std::string stage = profile_stage_;
  profile_stage_ = "all-reduce";
  launch(list, cross, "tp_send", 256, kTpFoldG, m, part, peer_landing_, t.tp_send_ctr->ptr(), nullptr);
  launch(list, cross, "tp_sum", 256, kTpFoldG, m, part, t.tp_landing->ptr(), sum, t.tp_sum_ctr->ptr(),
         t.tp_state->ptr(), spin_limit_);
  profile_stage_ = stage;
}

// The 2-bit copy of the int8 head (loader::dflash_candidate_head2's quantisation of its fp32 values, per 64 k).
void Engine::build_candidate_head() {
  const auto& head = weights_.weights.at(root + "unembed.weight");
  const auto& head_arena = buffers_.weights.arena(head.arena);
  auto list = l0::CmdList::immediate(ctx_);
  std::vector<int8_t> rows(size_t(D::head_rows) * D::hidden);
  std::vector<float> scales(D::head_rows);
  list.copy(rows.data(), at(head_arena, head.offset), rows.size());
  list.copy(scales.data(), at(head_arena, head.scales), scales.size() * sizeof(float));
  const auto h = loader::dflash_candidate_head2(rows.data(), scales.data(), D::hidden, D::head_rows);
  contract_assert(h.int4.size() * sizeof(uint32_t) == buffers_.weights.candidate_head->size());
  list.copy(buffers_.weights.candidate_head->ptr(), h.int4.data(), buffers_.weights.candidate_head->size());
}

void Engine::record_commit(l0::CmdList& list, uint32_t m, uint32_t accepted, bool on_device, const uint32_t* count) {
  // Target replay writes tentative KV rows past the cursor. The next replay
  // overwrites any rejected suffix, while convolution must keep only the
  // accepted raw rows in each visit's three-token history. On the device the accepted count is ink_accept's
  // projection rows, and the drafter's context runs on all m rows (df_ink_context stores the accepted ones).
  const auto& p = buffers_.persistent;
  auto* dctl = on_device ? static_cast<dflash::DFlashCtl*>(buffers_.draft.draft_control->ptr()) : nullptr;
  if (on_device) {
    launch(list, draft_glue, "ink_accept", 1, 1, 1, ctrl_, dctl);
    count = &dctl->proj_n;
  }
  launch(list, tp_.on() ? std::string(k::kTpGlue) : glue, "ink_commit_conv", 128, 2 * (tp_.kv_n() + D::hidden) / 128,
         share_.visits(),
         buffers_.scratch.pending_conv->ptr(), p.conv_k.ptr(), p.conv_v.ptr(), p.conv_attn.ptr(),
         p.conv_mlp.ptr(), m, accepted, count);
  if (buffers_.weights.draft) record_draft_context(list, on_device ? m : accepted, on_device);
}

std::span<const uint32_t> Engine::verify(std::span<const uint32_t> ids, bool candidates) {
  if (pending_) throw std::logic_error("Commit pending Inkling rows before verifying again");
  if (ids.empty() || ids.size() > kRows || ids.size() > max_len_ - committed_)
    throw std::invalid_argument("Inkling input rows exceed the context or verify width");
  for (const auto id : ids)
    if (id >= D::vocab) throw std::invalid_argument("An Inkling input token is outside its embedding table");
  const auto m = uint32_t(ids.size());
  const bool deep = deep_verify(m, committed_);
  auto& list = target_[candidates][deep][m - 1];
  if (!list) {
    list = std::make_unique<l0::CmdList>(l0::CmdList::regular(ctx_));
    const auto first = kernels_.size();
    record_target(*list, m, candidates, deep);
    // A single row has exactly one legal commit. Keep its state update on
    // the target list and publish the cursor only when the caller commits.
    if (m == 1) record_commit(*list, m, 1);
    if (!candidates) target_launches_[m - 1] = kernels_.size() - first;
    list->close();
  }
  std::copy(ids.begin(), ids.end(), ctrl_->cur_token);
  ctrl_->pos = committed_;
  ctrl_->n_active = m;
  written_ = std::max(written_, committed_ + m);
  if (m == 1) set_commit_control(1);
  queue_.execute(*list);
  queue_.synchronize();
  pending_ = m;
  prefill_logits_ = false;
  return {ctrl_->out_token, pending_};
}

void Engine::profile_target(std::span<const uint32_t> ids) {
  const auto m = uint32_t(ids.size());
  std::copy(ids.begin(), ids.end(), ctrl_->cur_token);
  ctrl_->pos = committed_;
  ctrl_->n_active = m;
  profile_list("decode profile: " + std::to_string(m) + "-row target at " + std::to_string(committed_),
               [&](l0::CmdList& list) { record_target(list, m, true); });
  // The expert weights the step read: each visit's union of its rows' choices, each expert once (the union kernels).
  std::vector<uint32_t> words(size_t(D::visit_count) * m * 32);
  {
    auto list = l0::CmdList::immediate(ctx_);
    list.copy(words.data(), buffers_.scratch.routes->ptr(), words.size() * sizeof(uint32_t));
  }
  std::array<uint64_t, 2> visits{}, experts{}, bytes{};
  for (uint32_t v = 0; v < D::visit_count; ++v) {
    std::bitset<D::experts> chosen;
    for (uint32_t r = 0; r < m; ++r)
      for (uint32_t j = 0; j < D::top_k; ++j) chosen.set(words[(size_t(v) * m + r) * 32 + j]);
    const auto format = desc_.expert_format[desc_.visits[v].layer];
    const uint32_t i = format == common::ExpertFpFormat::Nvfp4 ? 0 : 1;
    const uint64_t tiles = 2 * D::inter / 16 * (D::hidden / 64) + D::hidden / 16 * (D::inter / 64);
    ++visits[i];
    experts[i] += chosen.count();
    // NVFP4: each matrix's fp32 outer scale after its tiles
    bytes[i] += chosen.count() * (tiles * common::expert_fp_tile_words(format) * sizeof(uint32_t) + (1 - i) * 2 * f32);
  }
  for (uint32_t i = 0; i < 2; ++i)
    if (visits[i])
      std::print(stderr, "Inkling {}-row experts {}: {} visits, {:.2f} experts a visit, {:.1f} MB of weights read\n", m,
                 i ? "fp8" : "fp4", visits[i], double(experts[i]) / visits[i], bytes[i] / 1e6);
}

void Engine::profile_draft(uint32_t anchor) {
  ctrl_->cur_token[0] = anchor;
  ctrl_->pos = committed_;
  static_cast<dflash::DFlashCtl*>(buffers_.draft.draft_control->ptr())->k = kRows - 1;
  profile_list("draft profile at " + std::to_string(committed_), [&](l0::CmdList& list) { record_draft(list); });
}

void Engine::profile_list(const std::string& what, const std::function<void(l0::CmdList&)>& record) {
  constexpr uint32_t events = 1u << 12;
  profile_pool_ = std::make_unique<l0::EventPool>(ctx_, events);
  profile_events_.reserve(events);
  const auto first = kernels_.size();
  const auto start = std::chrono::steady_clock::now();
  {
    auto list = l0::CmdList::immediate_async(ctx_);
    record(list);
    list.host_synchronize();
  }
  report_profile(what, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
  profile_stage_.clear();
  while (kernels_.size() > first) kernels_.pop_back();
}

std::array<uint32_t, 8 * D::visit_count> Engine::routes() {
  contract_assert(stage_.whole());
  std::array<uint32_t, 8 * D::visit_count> ids{};
  std::vector<uint32_t> words(size_t(D::visit_count) * 32);
  auto list = l0::CmdList::immediate(ctx_);
  list.copy(words.data(), buffers_.scratch.routes->ptr(), words.size() * sizeof(uint32_t));
  for (uint32_t v = 0; v < D::visit_count; ++v) std::copy_n(words.begin() + v * 32, 8, ids.begin() + v * 8);
  return ids;
}

void Engine::read_logits(uint32_t row, std::span<float> out) {
  if (row >= pending_ || out.size() != D::vocab_used)
    throw std::invalid_argument("Inkling logit read requires a pending row and the live vocabulary");
  copy_logits(row, out);
}

void Engine::read_taps(bool prompt, uint32_t first, uint32_t rows, void* host) {
  constexpr size_t tap_row = 4 * D::hidden * bf16;
  const auto& taps = prompt ? *buffers_.prefill.draft_taps : *buffers_.scratch.draft_taps;
  if (size_t(first + rows) * tap_row > taps.size())
    throw std::invalid_argument("Inkling tap read past the step's rows");
  auto list = l0::CmdList::immediate(ctx_);
  list.copy(host, at(taps, first * tap_row), rows * tap_row);
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

void Engine::record_draft_context(l0::CmdList& list, uint32_t rows, bool on_device) {
  if (dflash()) return record_dflash_context(list, rows);   // ink_accept / set_commit_control set its count
  const auto& s = buffers_.scratch;
  const auto& d = buffers_.draft;
  constexpr size_t tap_row = 4 * D::hidden * bf16;
  const size_t kv_row = tp_.draft_kv_n() * bf16;
  auto* taps = s.draft_taps->ptr();
  if (on_device)
    launch(list, draft_glue, "ink_tap_keep", 256, 4 * D::hidden / 256, 1, d.draft_control->ptr(), taps,
           d.draft_tap->ptr());
  else
    list.copy(d.draft_tap->ptr(), at(*s.draft_taps, (rows - 1) * tap_row), tap_row);
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
    launch(list, tp_.on() ? std::string(k::kTpDraftPrep) : draft_prep, "df_ink_context", 128,
           tp_.draft_kv_heads(), rows, d.draft_control->ptr(),
           kv_k, kv_v, weight(a + "k_norm_ctx.weight"), d.rope->ptr(),
           at(local ? *d.draft_ring_k : *d.draft_full_k, off),
           at(local ? *d.draft_ring_v : *d.draft_full_v, off), local ? kDraftRing : 0u);
  }
}

std::pair<void*, void*> Engine::head_half(const std::string& name) const {
  const size_t first = size_t(tp_.rank) * k::kTpHeadHalf;
  const auto& w = weights_.weights.at(name);
  const auto& mem = buffers_.weights.arena(w.arena);
  return {at(mem, w.offset + first / 16 * loader::kInt8HeadKBlock * 256), at(mem, w.scales + first * f32)};
}

void Engine::record_draft(l0::CmdList& list) {
  if (dflash()) return record_dflash_draft(list);
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
  // tp: the card's heads and MLP columns, o_proj's and down's partial sums all-reduced into u.
  const bool split = tp_.on();
  const std::string prep = split ? std::string(k::kTpDraftPrep) : draft_prep;
  launch(list, draft_glue, "ink_draft_embed", 256, m, 1, ctrl_, weight(root + "embed.weight"),
         weight(root + "embed_norm.weight"), weight("mask_embed"), resid, desc_.eps);
  for (uint32_t l = 0, ring_index = 0; l < desc_.draft.layers; ++l) {
    const auto layer = "layers." + std::to_string(l) + ".";
    const auto a = layer + "self_attn.";
    const bool local = desc_.draft.window[l] != 0;
    const size_t off = local ? ring_index++ * kDraftRing * tp_.draft_kv_n() * bf16 : 0;
    const auto module = std::string(local ? (split ? k::kTpDraftWindow : k::kDraftWindow)
                                          : (split ? k::kTpDraftGlobal : k::kDraftGlobal));
    profile_stage_ = "draft attention";
    launch(list, prep, "df_norm", 256, 1, m, resid, weight(layer + "input_layernorm.weight"), norm, act);
    linear(list, a + "q_proj.weight", m, act, q);
    linear(list, a + "k_proj.weight", m, act, kk);
    linear(list, a + "v_proj.weight", m, act, vv);
    launch(list, prep, "df_ink_qkv", 128, tp_.draft_q_heads() + 2 * tp_.draft_kv_heads(), m, ctrl_, q, kk, vv,
           weight(a + "q_norm.weight"), weight(a + "k_norm.weight"), d.rope->ptr(), q, kk, vv);
    const float scale = 1.f / std::sqrt(float(D::head_dim));
    // The full layer's part on DPAS (a work-group of one sub-group a q head), its reduce the scalar binary's.
    const auto part = local ? module : std::string(split ? k::kTpDraftGlobalDpas : k::kDraftGlobalDpas);
    launch(list, part, "df_attn_part", local ? 128 : 16 * D::q_heads / D::kv_heads, tp_.draft_kv_heads(),
           local ? kDraftRing / 128 : kDraftPartitions,
           ctrl_, dctl, q, at(local ? *d.draft_ring_k : *d.draft_full_k, off),
           at(local ? *d.draft_ring_v : *d.draft_full_v, off), scale, d.draft_parts->ptr());
    launch(list, module, "df_attn_reduce", 128, tp_.draft_q_heads(), m,
           ctrl_, dctl, d.draft_parts->ptr(), q, kk, vv, scale, attention);
    linear(list, a + "o_proj.weight", m, attention, q);
    if (split) record_all_reduce(list, m, q, u);
    launch(list, prep, "df_add", 256, D::hidden / 256, m, split ? u : q, resid);
    profile_stage_ = "draft mlp";
    launch(list, prep, "df_norm", 256, 1, m, resid, weight(layer + "post_attention_layernorm.weight"), norm, act);
    linear(list, layer + "mlp.gate_proj.weight", m, act, q);
    linear(list, layer + "mlp.up_proj.weight", m, act, u);
    launch(list, draft_glue, "ink_draft_silu", 128, m * tp_.draft_inter() / 128, 1, q, u, act);
    linear(list, layer + "mlp.down_proj.weight", m, act, q);
    if (split) record_all_reduce(list, m, q, u);
    launch(list, prep, "df_add", 256, D::hidden / 256, m, split ? u : q, resid);
  }

  profile_stage_ = "draft head";
  launch(list, prep, "df_norm", 256, 1, m, resid, weight("norm.weight"), norm, act);
  constexpr uint32_t rank = D::markov_rank;
  constexpr size_t head_row = size_t(D::head_rows) * f32;
  const auto* hidden = static_cast<const std::byte*>(act);
  // tp: the card's kTpHeadHalf rows of the head and the fold, each draft's pick over them, then
  // an all-reduce of the two cards' best (ink_draft_argmax2_tp's row in tp_part, its sum after the fold's rows).
  const bool head_split = tp_.on();
  const size_t first = size_t(tp_.rank) * k::kTpHeadHalf, row = head_split ? size_t(k::kTpHeadHalf) * f32 : head_row;
  if (head_split) {
    const auto [w, s] = head_half(root + "unembed.weight");
    launch(list, std::string(k::kTpDraftHead), "gemm_sm_i8w", 64, k::kTpHeadHalf / 64, 1, w, s,
           hidden + D::hidden * bf16, d.draft_logits->ptr());
  } else {
    linear(list, root + "unembed.weight", m - 1, hidden + D::hidden * bf16, d.draft_logits->ptr());
  }
  auto* fold = at(*d.draft_logits, (m - 1) * row);
  auto* state = at(*d.markov);
  auto* latent = at(*d.markov, rank * bf16);
  auto* rnn = at(*d.markov, 2 * rank * bf16);
  auto* raw = at(*d.markov, (4 * rank + 2 * D::hidden) * bf16);
  list.fill(state, 0, rank * bf16);
  auto* context = at(*d.context_mix);
  const auto* fuse = static_cast<const std::byte*>(weight("fuse_weight"));
  launch(list, draft_glue, "ink_draft_mix", 256, 1, 1, d.draft_tap->ptr(),
         fuse + D::draft_layers * 4 * f32, weight("markov_head.dflare_norm.weight"), context, desc_.draft.eps);
  profile_stage_ = "draft markov";
  for (uint32_t j = 0; j < m - 1; ++j) {
    launch(list, draft_glue, "ink_markov_input", 128, (2 * rank + 2 * D::hidden) / 128, 1,
           state, weight("markov_head.markov_w1.weight"), hidden + size_t(j + 1) * D::hidden * bf16,
           context, ctrl_->cur_token + j, rnn);
    linear(list, "markov_head.rnn_proj.weight", 1, rnn, raw);
    launch(list, draft_glue, "ink_markov_step", 128, rank / 128, 1,
           raw, weight("markov_head.rnn_proj.bias"), state, latent);
    if (head_split) {
      const auto [w, s] = head_half("markov_head.fold.weight");
      const auto recurrent = k::tp_draft_recurrent(tp_.rank);
      auto* parts = buffers_.scratch.argmax_parts->ptr();
      auto* best = buffers_.tp.tp_part->ptr();
      auto* both = at(*d.draft_logits, m * row);
      launch(list, std::string(k::kTpDraftFold), "gemv_i8w", 64, k::kTpHeadHalf / 64, 1, w, s, latent, fold);
      launch(list, recurrent, "ink_draft_argmax1_tp", 256, (k::kTpHeadHalf + 1023) / 1024, 1,
             at(*d.draft_logits, j * row), fold,
             static_cast<const std::byte*>(weight("markov_head.fold.bias")) + first * f32, parts);
      launch(list, recurrent, "ink_draft_argmax2_tp", 256, 1, 1, parts, best);
      record_all_reduce(list, 1, best, both);
      launch(list, recurrent, "ink_draft_pick", 1, 1, 1, both, ctrl_->cur_token + j + 1);
      continue;
    }
    linear(list, "markov_head.fold.weight", 1, latent, fold);
    launch(list, draft_glue, "ink_draft_argmax1", 256, k::kDraftArgmaxChunks, 1, at(*d.draft_logits, j * head_row),
           fold, weight("markov_head.fold.bias"), buffers_.scratch.argmax_parts->ptr());
    launch(list, draft_glue, "ink_draft_argmax2", 256, 1, 1, buffers_.scratch.argmax_parts->ptr(),
           ctrl_->cur_token + j + 1);
  }
}

// The DFlash 2 drafter's scratch (inkling2_sizes.cc's plan 41k block).
namespace {
struct Df2Scratch {
  void *resid, *h, *x, *xa, *co, *gout, *q, *kb, *vb, *hid, *cand, *unary, *tpart;
};
Df2Scratch df2_scratch(const DraftBuffers& d, const model::DFlashDesc& f, bool int4) {
  constexpr size_t m = kRows;
  const size_t H = f.hidden, wide = std::max({H, size_t(f.q_n()), size_t(f.inter)});
  const size_t HR = m * H * f32, XR = m * H * bf16, AR = m * wide * bf16;
  const size_t QR = m * f.q_n() * f32, KR = m * f.kv_n() * f32;
  const size_t GR = d.block_projections->size() - QR - 2 * KR, CR = (m - 1) * kd::kTopK * f32;
  const size_t RR = m * f.sel_rank * f32 * (int4 ? kd::int4_slices(f.hidden, f.sel_rank) : 1);
  return {at(*d.block_hidden), at(*d.block_hidden, HR), at(*d.block_hidden, 2 * HR), at(*d.block_hidden, 2 * HR + XR),
          at(*d.block_hidden, 2 * HR + XR + AR), at(*d.block_projections), at(*d.block_projections, GR),
          at(*d.block_projections, GR + QR), at(*d.block_projections, GR + QR + KR), at(*d.markov),
          at(*d.markov, RR), at(*d.markov, RR + CR), at(*d.markov, RR + 2 * CR)};
}
}  // namespace

void Engine::record_dflash_draft(l0::CmdList& list) {
  const auto& d = buffers_.draft;
  const auto& f = desc_.draft;
  constexpr uint32_t m = kRows;
  const uint32_t H = f.hidden, groups = (H + 255) / 256;
  const bool int4 = weights_.draft_format == loader::DraftFormat::Int4G64;
  const auto df2_prep = kd::prep_variant(kd::kInkling, int4);
  const auto s = df2_scratch(d, f, int4);
  auto* dctl = d.draft_control->ptr();
  const size_t ring = size_t(kDraftRing) * f.kv_n() * bf16;
  const float scale = 1.f / std::sqrt(float(f.head_dim));
  profile_stage_ = "draft embed";
  launch(list, draft_glue, "ink_draft_embed", 256, m, 1, ctrl_, weight(root + "embed.weight"),
         weight(root + "embed_norm.weight"), weight("df2.mask_embed"), s.resid, desc_.eps);
  for (uint32_t l = 0; l < f.layers; ++l) {
    const auto layer = "layers." + std::to_string(l) + ".";
    const auto a = layer + "self_attn.";
    const auto* attn_base = weight(layer + "attention_conv.base_kernel");
    const auto* mlp_base = weight(layer + "mlp_conv.base_kernel");
    profile_stage_ = "draft attention";
    launch(list, df2_prep, "df_norm", 256, 1, m, s.resid, weight(layer + "input_layernorm.weight"), s.h, s.x);
    linear(list, layer + "attention_conv.kernel_projection.weight", m, s.x, s.co);
    launch(list, df2_prep, "df_conv_prep", 256, groups, m, s.h, s.co, attn_base, s.xa);
    linear(list, a + "qkv_proj.weight", m, s.xa, s.gout);
    launch(list, df2_prep, "df_qkv_prep", f.head_dim, f.q_heads + 2 * f.kv_heads, m, ctrl_, s.gout,
           weight(a + "qk_norm"), d.rope->ptr(), s.q, s.kb, s.vb);
    launch(list, kd::attn_dpas_variant(kd::kInkling, 2), "df_attn_part", 16 * (f.q_heads / f.kv_heads) * 2, f.kv_heads,
           kDraftRing / kd::kAttnPb, ctrl_, dctl, s.q, at(*d.draft_ring_k, l * ring), at(*d.draft_ring_v, l * ring),
           scale, d.draft_parts->ptr());
    launch(list, kd::attn_variant(kd::kInkling, 1, 4), "df_attn_reduce", f.head_dim, f.q_heads, m, ctrl_, dctl,
           d.draft_parts->ptr(), s.q, s.kb, s.vb, scale, s.xa);
    linear(list, a + "o_proj.weight", m, s.xa, s.gout);
    launch(list, df2_prep, "df_conv_add", 256, groups, m, s.gout, s.co, attn_base, s.resid);
    profile_stage_ = "draft mlp";
    launch(list, df2_prep, "df_norm", 256, 1, m, s.resid, weight(layer + "post_attention_layernorm.weight"), s.h, s.x);
    linear(list, layer + "mlp_conv.kernel_projection.weight", m, s.x, s.co);
    launch(list, df2_prep, "df_conv_prep", 256, groups, m, s.h, s.co, mlp_base, s.xa);
    linear(list, layer + "mlp.gate_up_proj.weight", m, s.xa, s.gout);
    launch(list, df2_prep, "df_silu_mul", 256, (f.inter + 255) / 256, m, s.gout, s.xa);
    linear(list, layer + "mlp.down_proj.weight", m, s.xa, s.gout);
    launch(list, df2_prep, "df_conv_add", 256, groups, m, s.gout, s.co, mlp_base, s.resid);
  }
  profile_stage_ = "draft head";
  launch(list, df2_prep, "df_norm", 256, 1, m, s.resid, weight("norm.weight"), s.h, s.x);
  // rows 1..7 through the target's head; the top 16 of its 200058 live rows, raw (the cap is the selector's)
  auto* rows = static_cast<std::byte*>(s.x) + H * bf16;
  if (tp_.on()) {
    // tp: one card's form on the card's half of the head's rows (the 2-bit copy's tiles of those rows, its top
    // kCandidates a row, their exact int8 dots reranked to the half's top 16), the two cards' lists through one
    // all-reduce row into the stage-1 parts (spent by then), merged on both into the whole head's top 16
    const auto& t2 = *kernels::small_m_tuning(D::hidden, k::kTpHeadHalf, 1, 3);
    auto* half2 = at(*buffers_.weights.candidate_head, kCandidateHeadBytes / D::head_rows * tp_.rank * k::kTpHeadHalf);
    auto* cpart = at(*buffers_.scratch.head_candidates);
    auto* ids = at(*buffers_.scratch.head_candidates, kCandidateParts);
    launch(list, kernels::gemm_sm_variant(m - 1, t2), "gemm_sm", kernels::gemm_sm_wg_size(t2),
           k::kTpHeadHalf / kernels::gemm_sm_cols(t2), kernels::gemm_sm_grid_y(t2), half2, half2, rows,
           d.draft_logits->ptr());
    const std::string cands = kd::inkling_tp_candidate_variant(tp_.rank);
    launch(list, cands, "df_candidate_select1", 256, (k::kTpHeadHalf + kCandidateChunk - 1) / kCandidateChunk, m - 1,
           d.draft_logits->ptr(), cpart);
    launch(list, cands, "df_candidate_select2", 256, m - 1, 1, cpart, ids);
    const auto [w, sc] = head_half(root + "unembed.weight");
    launch(list, kd::inkling_tp_rerank_variant(tp_.rank), "df_rerank", kCandidates * 4, m - 1, 1, w, sc, rows, ids,
           1.f, 0.f, s.cand, s.unary);
    const std::string recurrent = k::tp_draft_recurrent(tp_.rank);
    auto* part = buffers_.tp.tp_part->ptr();
    launch(list, recurrent, "ink_df_topk_tp", 256, D::hidden / 256, 1, s.cand, s.unary, part);
    record_all_reduce(list, 1, part, s.tpart);
    launch(list, recurrent, "ink_df_topk_merge", 16, 1, 1, s.tpart, f.out_mult, s.cand, s.unary);
  } else {
    // the head's 2-bit copy over the rows, its top kCandidates a row, their exact int8 dots reranked to the top 16
    // (df_rerank); the candidate scratch is the greedy head's, which the round's verify reuses after the draft
    const auto& t2 = *kernels::small_m_tuning(D::hidden, D::head_rows, 1, 3);
    const auto& head = weights_.weights.at(root + "unembed.weight");
    const auto& head_arena = buffers_.weights.arena(head.arena);
    auto* part = at(*buffers_.scratch.head_candidates);
    auto* ids = at(*buffers_.scratch.head_candidates, kCandidateParts);
    launch(list, kernels::gemm_sm_variant(m - 1, t2), "gemm_sm", kernels::gemm_sm_wg_size(t2),
           D::head_rows / kernels::gemm_sm_cols(t2), kernels::gemm_sm_grid_y(t2), buffers_.weights.candidate_head->ptr(),
           buffers_.weights.candidate_head->ptr(), rows, d.draft_logits->ptr());
    launch(list, std::string(k::kHeadCandidates), "df_candidate_select1", 256, uint32_t(kCandidateChunks), m - 1,
           d.draft_logits->ptr(), part);
    // the chunks' lists merged by the radix select (df_rerank does not read their order)
    launch(list, std::string(k::kHeadCandidates), "df_candidate_select2", 256, m - 1, 1, part, ids);
    launch(list, kd::inkling_rerank_variant(), "df_rerank", kCandidates * 4, m - 1, 1, at(head_arena, head.offset),
           at(head_arena, head.scales), rows, ids, f.out_mult, 0.f, s.cand, s.unary);
  }
  profile_stage_ = "draft selector";
  linear(list, "candidate_selector.hidden_projection.weight", m, s.x, s.hid);
  launch(list, kd::inkling_select_variant(int4), "df_select", 256, 1, 1, ctrl_, dctl, s.cand, s.unary, s.hid,
         weight("candidate_selector.predecessor_codebook"), weight("candidate_selector.successor_codebook"),
         static_cast<std::byte*>(dctl) + sizeof(dflash::DFlashCtl));
  profile_stage_.clear();
}

void Engine::record_dflash_context(l0::CmdList& list, uint32_t rows) {
  const auto& d = buffers_.draft;
  const auto& f = desc_.draft;
  const bool int4 = weights_.draft_format == loader::DraftFormat::Int4G64;
  const auto df2_prep = kd::prep_variant(kd::kInkling, int4);
  const size_t H = f.hidden, fc_slices = int4 ? kd::int4_slices(f.fc_k(D::hidden), f.hidden) : 1;
  auto* fc = at(*d.context_mix);
  auto* fcx = at(*d.context_mix, size_t(kRows) * H * f32 * fc_slices);
  auto* kv = at(*d.context_mix, size_t(kRows) * H * (f32 * fc_slices + bf16));
  const size_t ring = size_t(kDraftRing) * f.kv_n() * bf16;
  linear(list, "fc.weight", rows, buffers_.scratch.draft_taps->ptr(), fc);
  // The folded norm writes fp32 scratch as well as fc's bf16 rows. Reuse the KV region before its projection.
  launch(list, df2_prep, int4 ? "df_norm_fc" : "df_norm", 256, 1, rows, fc, weight("hidden_norm.weight"),
         int4 ? kv : fc, fcx);
  for (uint32_t l = 0; l < f.layers; ++l) {
    const auto a = "layers." + std::to_string(l) + ".self_attn.";
    linear(list, a + "kv_proj.weight", rows, fcx, kv);
    launch(list, df2_prep, "df_ctx_store", f.head_dim, 2 * f.kv_heads, rows, buffers_.draft.draft_control->ptr(), kv,
           weight(a + "qk_norm"), d.rope->ptr(), at(*d.draft_ring_k, l * ring), at(*d.draft_ring_v, l * ring));
  }
}

std::span<const uint32_t> Engine::propose(uint32_t anchor, uint32_t count, float temperature, uint64_t seed) {
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
  auto* dctl = static_cast<dflash::DFlashCtl*>(buffers_.draft.draft_control->ptr());
  dctl->k = kRows - 1;
  dctl->temperature = temperature;
  dctl->seed_lo = uint32_t(seed);
  dctl->seed_hi = uint32_t(seed >> 32);
  queue_.execute(*draft_list_);
  queue_.synchronize();
  return {ctrl_->cur_token + 1, count};
}

uint32_t Engine::step_drafted(uint32_t anchor, uint32_t max_drafts, std::vector<uint32_t>& output) {
  if (max_drafts > 0 && max_len_ - committed_ > 1 && buffers_.weights.draft && committed_) {
    if (pending_) throw std::logic_error("Commit pending Inkling rows before a drafted step");
    if (anchor >= D::vocab) throw std::invalid_argument("An Inkling input token is outside its embedding table");
    // One submission: the draft, the verify of the anchor and k drafts, and the commit of the accepted prefix,
    // which ink_accept counts on the device.
    const uint32_t k = std::min({max_drafts, kRows - 1, max_len_ - committed_ - 1});
    const bool deep = deep_verify(k + 1, committed_);
    auto& list = (deep ? deep_rounds_ : rounds_)[k - 1];
    if (!list) {
      list = std::make_unique<l0::CmdList>(l0::CmdList::regular(ctx_));
      record_step_inputs(*list);
      record_draft(*list);
      record_target(*list, k + 1, true, deep);
      record_commit(*list, k + 1, 0, true);
      record_step_outputs(*list, true);
      list->close();
    }
    auto* io = step_io();
    io->cur_token[0] = anchor;
    io->pos = io->proj_pos = committed_;
    io->n_active = k + 1;
    io->k = kRows - 1;
    io->proj_n = 0;
    written_ = std::max(written_, committed_ + k + 1);
    queue_.execute(*list);
    queue_.synchronize();
    const uint32_t accepted = io->accepted;
    uint32_t host_accepted = 1;
    while (host_accepted <= k && io->drafts[host_accepted] == io->out_token[host_accepted - 1]) ++host_accepted;
    contract_assert(accepted == host_accepted);
    // drafts[0] is not the anchor by now: argmax_stage2 wrote the last row's prediction there (the next step's input).
    output.push_back(anchor);
    output.insert(output.end(), io->drafts + 1, io->drafts + accepted);
    committed_ += accepted;
    prefill_logits_ = false;
    return io->out_token[accepted - 1];
  }
  output.push_back(anchor);
  return append(anchor);
}

// A greedy plain step: the target list with the candidate head and the one-row commit, one submission.
uint32_t Engine::append(uint32_t token) {
  if (pending_) throw std::logic_error("Commit pending Inkling rows before appending");
  if (committed_ >= max_len_) throw std::invalid_argument("Inkling input rows exceed the context");
  if (token >= D::vocab) throw std::invalid_argument("An Inkling input token is outside its embedding table");
  if (!plain_) {
    plain_ = std::make_unique<l0::CmdList>(l0::CmdList::regular(ctx_));
    record_step_inputs(*plain_);
    record_target(*plain_, 1, true);
    record_commit(*plain_, 1, 1);
    record_step_outputs(*plain_, false);
    plain_->close();
  }
  auto* io = step_io();
  io->pos = io->proj_pos = committed_;
  io->n_active = io->proj_n = 1;
  io->cur_token[0] = token;
  io->k = kRows - 1;
  written_ = std::max(written_, committed_ + 1);
  queue_.execute(*plain_);
  queue_.synchronize();
  committed_ += 1;
  prefill_logits_ = false;
  return io->out_token[0];
}

StepIo* Engine::step_io() const { return static_cast<StepIo*>(buffers_.scratch.step_io->ptr()); }

// A step's inputs into Control and the draft control (their first words, StepIo's order), from host memory.
void Engine::record_step_inputs(l0::CmdList& list) {
  static_assert(offsetof(Control, pos) == offsetof(StepIo, pos) && offsetof(Control, cur_token) == offsetof(StepIo, cur_token));
  static_assert(offsetof(dflash::DFlashCtl, proj_n) - offsetof(dflash::DFlashCtl, k) ==
                offsetof(StepIo, proj_n) - offsetof(StepIo, k));
  auto* io = step_io();
  list.copy(ctrl_, io, offsetof(StepIo, k));
  if (buffers_.weights.draft) list.copy(buffers_.draft.draft_control->ptr(), &io->k, 3 * sizeof(uint32_t));
}

// Its outputs back: the predictions; a round's drafts (Control's cur_token) and accepted rows too.
void Engine::record_step_outputs(l0::CmdList& list, bool round) {
  auto* io = step_io();
  list.copy(io->out_token, ctrl_->out_token, sizeof(io->out_token));
  if (!round) return;
  list.copy(io->drafts, ctrl_->cur_token, sizeof(io->drafts));
  list.copy(&io->accepted, &static_cast<dflash::DFlashCtl*>(buffers_.draft.draft_control->ptr())->proj_n,
            sizeof(uint32_t));
}

}  // namespace runtime::inkling2
