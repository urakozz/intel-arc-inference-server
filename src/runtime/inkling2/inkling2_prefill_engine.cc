#include "runtime/inkling2/inkling2_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <print>

#include "kernels/inkling2_kernels.h"
#include "kernels/prefill/pf_kernels.h"

namespace runtime::inkling2 {
namespace {
using D = model::Inkling2Desc;
namespace k = kernels::inkling2;
constexpr size_t f32 = sizeof(float), bf16 = sizeof(uint16_t);
const std::string root(model::kInkling2Prefix);

void* at(const l0::Mem& mem, size_t offset = 0) { return static_cast<std::byte*>(mem.ptr()) + offset; }
}  // namespace

uint32_t Engine::prefill(std::span<const uint32_t> ids, uint32_t cap) {
  auto* staging = static_cast<uint32_t*>(buffers_.prefill.ids->ptr());
  for (const auto id : ids) contract_assert(id < D::vocab);
  std::copy(ids.begin(), ids.end(), staging);

  const auto first = kernels_.size();
  const uint32_t end = committed_ + uint32_t(ids.size());
  ctrl_->pos = end - 1;
  ctrl_->n_active = 1;
  const char* profile = std::getenv("B70_PREFILL_PROFILE");
  if (profile && std::strcmp(profile, "2") == 0) {
    constexpr uint32_t events = 1u << 18;
    profile_pool_ = std::make_unique<l0::EventPool>(ctx_, events);
    profile_events_.reserve(events);
  }
  const auto start = std::chrono::steady_clock::now();
  {
    auto list = l0::CmdList::immediate_async(ctx_);
    for (const auto& chunk : prefill_chunks(desc_, committed_, uint32_t(ids.size()), cap))
      record_prefill_chunk(list, chunk, staging + (chunk.position - committed_), chunk.position + chunk.rows == end);
    list.host_synchronize();
  }
  if (profile_pool_) {
    const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::map<std::string, std::pair<uint32_t, double>> each;
    double busy = 0;
    for (size_t i = 0; i < profile_events_.size(); ++i) {
      const double ms = profile_events_[i].duration_us() / 1000;
      auto& [launches, total] = each[profile_names_[i]];
      ++launches;
      total += ms;
      busy += ms;
    }
    const double span = profile_events_.back().elapsed_us(profile_events_.front()) / 1000;
    std::print(stderr, "Inkling prompt profile: {} ids, {} launches, {:.2f} ms wall, {:.2f} ms device span, "
               "{:.2f} ms inside kernels\n", ids.size(), profile_events_.size(), wall, span, busy);
    std::vector<std::pair<std::string, std::pair<uint32_t, double>>> rows(each.begin(), each.end());
    std::ranges::sort(rows, [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
    for (const auto& [name, value] : rows)
      std::print(stderr, "  {:9.3f} ms {:6} launches  {}\n", value.second, value.first, name);
    profile_events_.clear();
    profile_names_.clear();
    profile_pool_.reset();
  }
  while (kernels_.size() > first) kernels_.pop_back();
  committed_ = ctrl_->pos = end;
  prefill_logits_ = true;
  return ctrl_->out_token[0];
}

void Engine::read_prefill_logits(std::span<float> out) { copy_logits(0, out); }

void Engine::prefill_linear(l0::CmdList& list, const std::string& name, uint32_t rows, const void* input, void* output) {
  const auto& w = weights_.weights.at(name);
  const auto& mem = buffers_.weights.arena(w.arena);
  const auto* weights = at(mem, w.offset);
  const auto& s = buffers_.prefill;
  const auto stage = profile_stage_;
  if (profile_pool_) {
    const auto layers = name.find("layers.");
    const auto cut = layers == std::string::npos ? 0 : name.find('.', layers + 7) + 1;
    profile_stage_ = "linear " + name.substr(cut);
  }
  auto* slab = s.linear_slab->ptr();
  const std::string unpack = "pf_ink_slab";
  // The shared GEMMs address complete row tiles. Keep the padding separate
  // from the other live matrices packed immediately after each input/output.
  const bool large = rows > 1024 && w.N % 256 == 0;
  const uint32_t tile = large ? 256 : 16;
  const uint32_t whole = rows / tile * tile, tail = rows - whole;
  auto* padded_input = static_cast<std::byte*>(s.linear_tail_input->ptr());
  auto* padded_output = s.linear_tail_output->ptr();
  if (tail) {
    const size_t input_row = w.K * bf16;
    list.copy(padded_input, static_cast<const std::byte*>(input) + whole * input_row, tail * input_row);
    list.fill(padded_input + tail * input_row, 0, (tile - tail) * input_row);
  }
  for (uint32_t n0 = 0; n0 < w.N; n0 += kPrefillSlab) {
    const uint32_t width = std::min(kPrefillSlab, w.N - n0);
    auto* dest = static_cast<float*>(output) + n0;
    launch(list, unpack, "pf_ink_slab", 64, width / 64, w.K / 8,
           weights, slab, w.K, n0, width, uint32_t(w.scales != 0));
    const auto project = [&](const void* a, void* c, uint32_t height, uint32_t pitch) {
      if (large && w.scales) {   // the channel scales in the epilogue
        launch(list, "pf_gemm_T0_CS", "pf_gemm", 512, height / tile, width / 256, a, slab, c, height, w.K, width,
               w.K, width, pitch, uint64_t(0), uint64_t(0), uint64_t(0),
               static_cast<const float*>(at(mem, w.scales)) + n0);
      } else if (large) {
        launch(list, kernels::pf_gemm_variant(false), "pf_gemm", 512, height / tile, width / 256,
               a, slab, c, height, w.K, width, w.K, width, pitch, uint64_t(0), uint64_t(0), uint64_t(0));
      } else {
        const uint32_t sn = width % 128 == 0 ? 4 : 2;
        const auto gemm = kernels::pf_gemm_small_variant(2, 2, sn, false) + (sn == 2 ? "_INK" : "");
        launch(list, gemm, "pf_gemm_small", 16 * sn, width / (32 * sn), height / tile,
               a, slab, c, height, w.K, width, w.K, width, pitch);
      }
    };
    if (whole) project(input, dest, whole, w.N);
    if (tail) {
      project(padded_input, padded_output, tile, width);
      launch(list, unpack, "pf_ink_tail", 64, width / 64, tail,
             padded_output, dest + size_t(whole) * w.N, width, w.N);
    }
  }
  if (w.scales && !large) launch(list, unpack, "pf_ink_scale", 64, w.N / 64, rows, output, at(mem, w.scales), w.N);
  profile_stage_ = stage;
}

void Engine::record_prefill_chunk(l0::CmdList& list, const PrefillChunk& chunk, const uint32_t* ids, bool head) {
  const auto& s = buffers_.prefill;
  const auto& p = buffers_.persistent;
  const uint32_t m = chunk.rows;
  const size_t hidden = size_t(m) * D::hidden * f32, arow = size_t(m) * D::hidden * bf16;
  const size_t qn = size_t(m) * D::q_n * f32, rn = size_t(m) * D::q_heads * D::relative_dim * f32;
  const size_t kv = size_t(m) * D::kv_n * f32;
  // The glue's messages, controller mixes and expert rows are bf16 under B70_INK_GLUE=bf16 (bf16_glue_).
  const std::string glue = bf16_glue_ ? "ink_glue_H2304_PREFILL_BF16" : "ink_glue_H2304_PREFILL";
  const size_t message = size_t(m) * D::hidden * (bf16_glue_ ? bf16 : f32);
  const auto residual = [&](uint32_t slot) { return at(*s.residual, slot * hidden); };
  const auto bank = [&](uint32_t slot) { return at(*s.block_bank, slot * hidden); };
  const auto activ = [&](uint32_t slot) { return at(*s.inputs, slot * arow); };
  auto* current = residual(0);
  auto* next = residual(1);
  auto* midpoint = residual(3);
  auto* prev_midpoint = residual(5);

  launch(list, glue, "ink_embed_norm", 128, m, 1, ids, weight(root + "embed.weight"),
         weight(root + "embed_norm.weight"), current, desc_.eps);
  uint32_t finished = 0;
  const auto mix_norm = [&](const void* last, const void* query, uint32_t queries, const void* q,
                            const void* k = nullptr, const void* v = nullptr) {
    launch(list, glue, "ink_depth_mix_norm", 128, queries, m, bank(0), query, activ(0),
           finished + 1, m, desc_.eps, q, k, v, last);
  };
  const auto controller = [&](void* last, const void* query, const void* messages,
                             const std::string& layer, std::span<const char* const> names,
                             std::array<const void*, 3> norms, bool shared_mix,
                             const void* raw = nullptr, const void* history = nullptr,
                             const void* taps = nullptr, const void* prior = nullptr) {
    auto* mixed = at(*s.controller);
    auto* cat = at(*s.controller, 3 * hidden);
    auto* low = at(*s.controller, 12 * hidden);
    auto* low_f32 = s.route_logits->ptr();
    const auto streams = uint32_t(names.size());
    launch(list, glue, "ink_mix_cat", 128, streams, m, bank(0), last, query, messages,
           mixed, cat, finished + 1, m, uint32_t(shared_mix), desc_.eps, raw, history, taps, prior);
    for (uint32_t i = 0; i < streams; ++i) {
      const auto prefix = layer + "controllers." + names[i] + ".";
      auto* rank = static_cast<std::byte*>(low) + size_t(i) * m * D::controller_rank * bf16;
      prefill_linear(list, prefix + "down_weight", m, static_cast<std::byte*>(cat) + 2 * i * arow, low_f32);
      launch(list, glue, "ink_pack", 64, m * D::controller_rank / 64, 1, low_f32, rank);
    }
    std::array<const void*, 6> gates{};
    for (uint32_t i = 0; i < streams; ++i) {
      const auto prefix = layer + "controllers." + names[i] + ".";
      gates[i] = weight(prefix + "forget_weight");
      gates[3 + i] = weight(prefix + "gain_weight");
    }
    launch(list, glue, "ink_prefill_controller", 128, (m + 7) / 8, streams, low, mixed, messages, activ(0),
           gates[0], gates[1], gates[2], gates[3], gates[4], gates[5], norms[0], norms[1], norms[2], m, desc_.eps);
  };

  for (const auto& v : chunk.visits) {
    const auto l = v.layer;
    const auto layer = model::inkling2_layer_prefix(l);
    const auto a = layer + "attn.";
    if (l == D::core_start) {
      if (v.pass == 0) {
        list.copy(residual(2), current, hidden);
      } else {
        std::swap(midpoint, prev_midpoint);
        list.copy(residual(4), current, hidden);
        list.copy(current, residual(2), hidden);
        finished = D::core_start / D::block_layers;
      }
    }
    if (l == D::core_start + D::block_layers) list.copy(midpoint, current, hidden);
    if (v.controller)
      launch(list, glue, "ink_cross_mix", 128, 4, m, prev_midpoint, weight(root + "middle_cycle_cross_pass_queries"),
             weight(root + "middle_cycle_cross_pass_kv_queries"), s.messages->ptr(), m, residual(4), l - D::core_start);

    const auto qnorm = layer + (v.window ? "attn_norm.weight" : "q_input_norm.weight");
    const auto knorm = v.window ? qnorm : layer + "k_input_norm.weight";
    const auto vnorm = v.window ? qnorm : layer + "v_input_norm.weight";
    if (l == 0) {
      launch(list, glue, "ink_norm", 128, m, 1, current, nullptr, weight(qnorm), activ(0), desc_.eps);
    } else {
      const auto* query = weight(root + "attn_res.attn_queries." + std::to_string(l - 1));
      if (v.controller) {
        controller(current, query, s.messages->ptr(), layer, std::array{"attn", "k", "v"},
                   {weight(qnorm), weight(knorm), weight(vnorm)}, bool(v.window));
      } else {
        mix_norm(current, query, v.window ? 1 : 3, weight(qnorm),
                 v.window ? nullptr : weight(knorm), v.window ? nullptr : weight(vnorm));
      }
    }
    const auto* ak = activ(v.window && !v.controller ? 0 : 1);
    const auto* av = activ(v.window && !v.controller ? 0 : 2);
    auto* q = at(*s.projections);
    auto* gate = at(*s.projections, qn);
    auto* relative = at(*s.projections, 2 * qn);
    auto* sinks = at(*s.projections, 2 * qn + rn);
    auto* raw_k = at(*s.conv), *raw_v = at(*s.conv, kv);
    auto* raw_attn = at(*s.conv, 2 * kv), *raw_mlp = at(*s.conv, 2 * kv + hidden);
    prefill_linear(list, a + "wq_du.weight", m, activ(0), q);
    prefill_linear(list, a + "wk_dv.weight", m, ak, raw_k);
    prefill_linear(list, a + "wv_dv.weight", m, av, raw_v);
    prefill_linear(list, a + "gate_attn.weight", m, activ(0), gate);
    prefill_linear(list, a + "wr_du.weight", m, activ(0), relative);
    prefill_linear(list, a + "attn_sink_proj.weight", m, activ(0), sinks);
    void *kt = nullptr, *vt = nullptr;
    if (!v.window) {
      kt = at(*s.full_taps);
      vt = at(*s.full_taps, size_t(m) * 64 * f32);
      prefill_linear(list, a + "k_sconv.weight_proj", m, ak, kt);
      prefill_linear(list, a + "v_sconv.weight_proj", m, av, vt);
    }

    const uint32_t ring = v.window ? kRing : 0;
    const bool kv8 = kv8_ && !ring;
    const size_t kv_off = size_t(v.kv_slot) * (ring ? ring : max_len_) * D::kv_n * (kv8 ? 1 : bf16);
    auto* ck = at(ring ? p.ring_k : p.full_k, kv_off);
    auto* cv = at(ring ? p.ring_v : p.full_v, kv_off);
    const size_t scale_off = size_t(v.kv_slot) * max_len_ * D::kv_heads * bf16;
    void* ks = kv8 ? at(*p.full_k_scale, scale_off) : nullptr;
    void* vs = kv8 ? at(*p.full_v_scale, scale_off) : nullptr;
    const size_t kh = size_t(v.conv_slot) * 3 * D::kv_n * f32, ah = size_t(v.conv_slot) * 3 * D::hidden * f32;
    auto* hist_k = at(p.conv_k, kh), *hist_v = at(p.conv_v, kh);
    auto* hist_attn = at(p.conv_attn, ah), *hist_mlp = at(p.conv_mlp, ah);
    const char* taps = ring ? "weight" : "static_w";
    // The ring holds kPrefillAttentionRows rows and their window: a longer chunk writes and reads its keys in
    // parts, each part's convolution history the previous part's last rows.
    for (uint32_t r0 = 0; r0 < m; r0 += kPrefillAttentionRows) {
      const uint32_t n = std::min(kPrefillAttentionRows, m - r0);
      const auto row = [&](void* base, size_t width, size_t bytes = f32) {
        return base ? static_cast<std::byte*>(base) + size_t(r0) * width * bytes : nullptr;
      };
      auto* part_k = row(raw_k, D::kv_n), *part_v = row(raw_v, D::kv_n);
      void* part_hk = r0 ? static_cast<std::byte*>(part_k) - 3 * D::kv_n * f32 : hist_k;
      void* part_hv = r0 ? static_cast<std::byte*>(part_v) - 3 * D::kv_n * f32 : hist_v;
      auto* part_q = row(q, D::q_n), *part_rel = row(relative, D::q_heads * D::relative_dim);
      if (kv8) {
        launch(list, std::string(k::kPrefillPrep) + "_KV8", "ink_qkv_prep", 128, D::q_heads + 2 * D::kv_heads, n,
               part_q, part_rel, weight(a + "q_l2_clip.scale"), weight(a + "rel_l2_clip.scale"),
               part_k, part_v, part_hk, part_hv, weight(a + "k_sconv." + taps), weight(a + "v_sconv." + taps),
               row(kt, 64), row(vt, 64), ck, cv, ks, vs, chunk.position + r0, ring, desc_.eps, desc_.qk_clip,
               desc_.relative_clip, desc_.log_floor, desc_.log_alpha);
        if (profile_pool_) profile_stage_ = "full";
        launch_3d(list, k::prefill_attn(false) + "_KV8_PAIR", "kol_pf_flash_attn", k::kPrefillAttnGroup,
                  (n + 15) / 16, D::kv_heads, 1, part_q, ck, cv, ks, vs, row(s.attention->ptr(), D::q_n, bf16),
                  chunk.position + r0, n, part_rel, weight(a + "rel_logits_proj.proj"), row(sinks, 64), row(gate, D::q_n));
      } else {
        launch(list, std::string(k::kPrefillPrep), "ink_qkv_prep", 128, D::q_heads + 2 * D::kv_heads, n,
               part_q, part_rel, weight(a + "q_l2_clip.scale"), weight(a + "rel_l2_clip.scale"),
               part_k, part_v, part_hk, part_hv, weight(a + "k_sconv." + taps), weight(a + "v_sconv." + taps),
               row(kt, 64), row(vt, 64), ck, cv, chunk.position + r0, ring, desc_.eps, desc_.qk_clip, desc_.relative_clip,
               desc_.log_floor, desc_.log_alpha);
        if (profile_pool_) profile_stage_ = ring ? "window" : "full";
        launch_3d(list, k::prefill_attn(bool(ring)) + "_PAIR", "kol_pf_flash_attn", k::kPrefillAttnGroup,
                  (n + 15) / 16, D::kv_heads, 1, part_q, ck, cv, row(s.attention->ptr(), D::q_n, bf16), chunk.position + r0, n,
                  part_rel, weight(a + "rel_logits_proj.proj"), row(sinks, 64), row(gate, D::q_n));
      }
      profile_stage_.clear();
    }
    prefill_linear(list, a + "wo_ud.weight", m, s.attention->ptr(), raw_attn);

    if (v.block_boundary) list.copy(bank(finished++), current, hidden);
    const auto* mlp_query = weight(root + "attn_res.mlp_queries." + std::to_string(l));
    const auto* mlp_norm = weight(layer + "mlp_norm.weight");
    const auto* post_taps = weight(layer + "attn_sconv.weight");
    const auto* prior = v.block_boundary ? nullptr : current;
    if (v.controller) {
      controller(next, mlp_query, at(*s.messages, 3 * message), layer, std::array{"mlp"},
                 {mlp_norm, nullptr, nullptr}, true, raw_attn, hist_attn, post_taps, prior);
    } else {
      launch(list, glue, "ink_post_mix_norm", 128, m, 1, bank(0), next, mlp_query, activ(0), mlp_norm,
             finished + 1, m, desc_.eps, raw_attn, hist_attn, post_taps, prior);
    }

    prefill_linear(list, layer + "mlp.gate.weight", m, activ(0), s.route_logits->ptr());
    launch(list, k::route(), "kol_route", 128, 1, m, s.route_logits->ptr(), weight(layer + "mlp.gate.bias"), s.routes->ptr());
    const std::string sorter(k::kPrefillSort), moe = std::string(k::kPrefillMoe) + (bf16_glue_ ? "_BF16" : "");
    const uint32_t tiles = (m * D::top_k + D::experts * (kPrefillExpertTile - 1)) / kPrefillExpertTile;
    constexpr uint32_t item_rows = k::kPrefillExpertRows;
    const uint32_t items = (m * D::top_k + D::experts * (item_rows - 1)) / item_rows;
    const uint32_t groups = (m + kPrefillExpertTile - 1) / kPrefillExpertTile;
    auto* cnt = s.expert_counts->ptr();
    auto* hdr = s.expert_header->ptr();
    auto* row_tok = s.expert_rows->ptr();
    auto* pair_row = s.expert_pairs->ptr();
    auto* work = s.expert_items->ptr();
    launch(list, sorter, "pf_moe_hist", 128, groups, 1, s.routes->ptr(), cnt, m);
    launch(list, sorter, "pf_moe_scan", 128, 1, 1, cnt, hdr, nullptr, row_tok, work, m, 0u, items);
    launch(list, sorter, "pf_moe_scatter", 128, groups, 1, s.routes->ptr(), cnt, row_tok, pair_row, m);
    launch(list, moe, "kol_pf_gather", 64, tiles * kPrefillExpertTile, 1, activ(0), hdr, row_tok, s.expert_inputs->ptr());
    const auto fmt = desc_.expert_format[l];
    if (profile_pool_) profile_stage_ = std::string(fmt == desc_.expert_format[0] ? "fp4" : "fp8") + " w13";
    launch(list, k::prefill_expert(fmt, true, bf16_glue_), "pf_moe_q4gemm", k::kPrefillExpertGroup, 2 * D::inter / 128, items,
           work, s.expert_inputs->ptr(), weight(layer + "mlp.experts.w13_weight"), nullptr, s.expert_output->ptr());
    launch(list, moe, "ink_prefill_swiglu", 96, D::inter / (4 * 96), tiles * kPrefillExpertTile,
           s.expert_output->ptr(), s.expert_hidden->ptr());
    if (profile_pool_) profile_stage_.replace(profile_stage_.size() - 3, 3, "w2");
    launch(list, k::prefill_expert(fmt, false, bf16_glue_), "pf_moe_q4gemm", k::kPrefillExpertGroup, D::hidden / 128, items,
           work, s.expert_hidden->ptr(), weight(layer + "mlp.experts.w2_weight"), nullptr, s.expert_output->ptr());
    profile_stage_.clear();
    void* draft_tap = nullptr;
    if (s.draft_taps)
      for (uint32_t t = 0; t + 1 < desc_.draft.taps.size(); ++t)
        if (v.conv_slot + 1 == desc_.draft.taps[t]) draft_tap = at(*s.draft_taps, size_t(t) * D::hidden * bf16);
    launch(list, moe, "ink_prefill_moe_combine", 256, D::hidden / 256, groups, s.routes->ptr(), pair_row,
           s.expert_output->ptr(), weight(layer + "mlp_sconv.weight"), hist_mlp, raw_mlp, next, draft_tap, m);
    launch(list, glue, "ink_prefill_commit_conv", 128, 2 * (D::kv_n + D::hidden) / 128, 1,
           raw_k, hist_k, hist_v, hist_attn, hist_mlp, m);
    if (head && trace_prelude_ && v.conv_slot < D::core_start) {
      constexpr size_t row_bytes = D::hidden * f32;
      list.copy(at(*s.prelude_trace, v.conv_slot * row_bytes), static_cast<std::byte*>(next) + (m - 1) * row_bytes,
                row_bytes);
    }
    std::swap(current, next);
  }

  if (head || s.draft_taps)
    mix_norm(current, weight(root + "attn_res.output_query"), 1, weight(root + "norm.weight"));
  if (s.draft_taps) {
    auto* taps = s.draft_taps->ptr();
    launch(list, glue, "ink_tap", 128, D::hidden / 128, m, nullptr, activ(0), taps, 3u);
    constexpr size_t tap_row = 4 * D::hidden * bf16, kv_row = D::kv_n * bf16;
    list.copy(at(*buffers_.scratch.draft_taps, kRows * tap_row), at(*s.draft_taps, (m - 1) * tap_row), tap_row);
    const auto& d = buffers_.draft;
    auto* mix = at(*s.draft_projection);
    auto* key = at(*s.draft_projection, arow);
    auto* value = at(*s.draft_projection, arow + kv);
    const auto* fuse = static_cast<const std::byte*>(weight("fuse_weight"));
    for (uint32_t l = 0, ring_index = 0; l < desc_.draft.layers; ++l) {
      const auto layer = "layers." + std::to_string(l) + ".";
      const auto a = layer + "self_attn.";
      const bool local = desc_.draft.window[l] != 0;
      const size_t off = local ? ring_index++ * kDraftRing * kv_row : 0;
      launch(list, std::string(k::kDraftRecurrent), "ink_draft_mix", 256, m, 1, taps, fuse + l * 4 * f32,
             weight(layer + "dflare_norm.weight"), mix, desc_.draft.eps);
      prefill_linear(list, a + "k_proj_ctx.weight", m, mix, key);
      prefill_linear(list, a + "v_proj_ctx.weight", m, mix, value);
      for (uint32_t r0 = 0; r0 < m; r0 += kDraftRing)   // a ring's rows a launch: no two rows of one write a slot
        launch(list, "df_prep_H2304_INK_CONTEXT_PREFILL", "df_ink_context", 128, D::kv_heads, std::min(kDraftRing, m - r0),
               chunk.position + r0, static_cast<std::byte*>(key) + size_t(r0) * D::kv_n * f32,
               static_cast<std::byte*>(value) + size_t(r0) * D::kv_n * f32, weight(a + "k_norm_ctx.weight"), d.rope->ptr(),
               at(local ? *d.draft_ring_k : *d.draft_full_k, off),
               at(local ? *d.draft_ring_v : *d.draft_full_v, off), local ? kDraftRing : 0u);
    }
  }
  if (!head) return;
  const auto& decode = buffers_.scratch;
  linear(list, root + "unembed.weight", 1, at(*s.inputs, (m - 1) * D::hidden * bf16), decode.logits->ptr());
  const auto argmax = k::argmax(1);
  launch(list, argmax, "argmax_stage1", 256, (D::head_rows + 1023) / 1024, 1, decode.logits->ptr(),
         decode.argmax_parts->ptr());
  launch(list, argmax, "argmax_stage2", 256, 1, 1, ctrl_, decode.argmax_parts->ptr());
}

}  // namespace runtime::inkling2
