#include "runtime/inkling2/inkling2_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <print>

#include "kernels/dflash_kernels.h"
#include "kernels/inkling2_kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "runtime/prefill_chunks.h"

namespace runtime::inkling2 {
namespace {
using D = model::Inkling2Desc;
namespace k = kernels::inkling2;
constexpr size_t f32 = sizeof(float), bf16 = sizeof(uint16_t);
const std::string root(model::kInkling2Prefix);
const std::string df2_wide = kernels::dflash::proj_wide_variant(kernels::dflash::kInkling);

void* at(const l0::Mem& mem, size_t offset = 0) { return static_cast<std::byte*>(mem.ptr()) + offset; }
}  // namespace

void Engine::report_profile(const std::string& what, double wall) {
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
  std::print(stderr, "Inkling {}, {} launches, {:.2f} ms wall, {:.2f} ms device span, {:.2f} ms inside kernels\n",
             what, profile_events_.size(), wall, span, busy);
  std::vector<std::pair<std::string, std::pair<uint32_t, double>>> rows(each.begin(), each.end());
  std::ranges::sort(rows, [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
  for (const auto& [name, value] : rows)
    std::print(stderr, "  {:9.3f} ms {:6} launches  {}\n", value.second, value.first, name);
  profile_events_.clear();
  profile_names_.clear();
  profile_pool_.reset();
}

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
  const uint32_t base = committed_;
  written_ = std::max(written_, end);
  prefill_end_ = end;
  // With the prefix cache's hook every chunk ends at a snapshot point or the prompt's end (runtime::prefill_chunk_rows,
  // the wide stride: a 4096-row chunk over 2048-position blocks), and the chunks up to a mid-prompt block end run
  // before the hook stores them.
  std::vector<PrefillChunk> chunks;
  if (block_hook_) {
    for (uint32_t pos = base; pos < end;) {
      const uint32_t rows = prefill_chunk_rows(pos, end - pos, cap, true, kBlock, /*wide_stride=*/true);
      chunks.push_back({pos, rows, desc_.visits});
      pos += rows;
    }
  } else {
    chunks = prefill_chunks(desc_, base, uint32_t(ids.size()), cap);
  }
  {
    auto list = l0::CmdList::immediate_async(ctx_);
    for (const auto& chunk : chunks) {
      const uint32_t chunk_end = chunk.position + chunk.rows;
      record_prefill_chunk(list, chunk, staging + (chunk.position - base), chunk_end == end);
      if (block_hook_ && chunk_end < end && chunk_end % kBlock == 0) {
        list.host_synchronize();
        committed_ = chunk_end;   // what the hook stores; a hook that throws leaves the chunks written
        overlap_kv_ = true;
        try {
          block_hook_(chunk_end, true);
        } catch (...) {
          overlap_kv_ = false;
          if (kv_store_) kv_store_->host_synchronize();
          throw;
        }
        overlap_kv_ = false;
      }
    }
    list.host_synchronize();
    if (kv_store_) kv_store_->host_synchronize();   // PERF-89: the mid-prompt KV stores, beside the chunks after them
  }
  if (profile_pool_)
    report_profile("prompt profile: " + std::to_string(ids.size()) + " ids",
                   std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
  while (kernels_.size() > first) kernels_.pop_back();
  committed_ = ctrl_->pos = end;
  prefill_logits_ = true;
  if (block_hook_) block_hook_(end, end % kBlock == 0);   // the prompt's end: its first id chosen
  return ctrl_->out_token[0];
}

void Engine::read_prefill_logits(std::span<float> out) { copy_logits(0, out); }

void Engine::prefill_linear(l0::CmdList& list, const std::string& name, uint32_t rows, const void* input, void* output) {
  const auto& w = weights_.weights.at(name);
  const auto& mem = buffers_.weights.arena(w.arena);
  const auto* weights = at(mem, w.offset);
  const auto& s = *pf_;
  const auto stage = profile_stage_;
  if (profile_pool_) {
    const auto layers = name.find("layers.");
    const auto cut = layers == std::string::npos ? 0 : name.find('.', layers + 7) + 1;
    profile_stage_ = "linear " + name.substr(cut);
  }
  if (const auto it = int8_linears_.find(name); it != int8_linears_.end()) {
    // The int8 pass: the input rotated and quantised once (rows reused by the linears that share it), then
    // pf_gemm_i8 against the rotated int8 copy; rows past `rows` and columns past N read as zeros and are not stored.
    const auto [offset, K, N, columns] = it->second;
    if (quantized_ != input) {
      launch(list, "ink_int8_K" + std::to_string(K), "ink_quant_rot", 16 * K / 256, rows, 1, input,
             s.int8_rows->ptr(), s.int8_scales->ptr());
      quantized_ = input;
    }
    auto* rotated = static_cast<std::byte*>(buffers_.weights.prompt_int8->ptr()) + offset;
    launch(list, kernels::pf_gemm_i8_variant(false), "pf_gemm_i8", 512, (rows + 255) / 256,
           columns / kernels::kPfGemmI8WgN, s.int8_rows->ptr(), s.int8_scales->ptr(), rotated, rotated + size_t(K) * columns,
           output, rows, K, N, K / 2, columns, N);
    profile_stage_ = stage;
    return;
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
  const bool int4 = w.arena == loader::Inkling2WeightLayout::kDraft &&
                    weights_.draft_format == loader::DraftFormat::Int4G64;
  // The packed path fits two full-precision weight parts and two output planes in the same slab.
  static_assert(2 * (4 * D::hidden * bf16 + kPrefillRows * f32) * kDraftInt4Slab <=
                4 * D::hidden * bf16 * kPrefillSlab);
  const size_t bytes_per_col = 2 * (size_t(w.K) * bf16 + size_t(rows) * f32);
  const uint32_t slab_width = int4 ? std::min<size_t>(kPrefillSlab, s.linear_slab->size() / bytes_per_col /
                                                                     kDraftInt4Slab * kDraftInt4Slab)
                                  : kPrefillSlab;
  for (uint32_t n0 = 0; n0 < w.N; n0 += slab_width) {
    const uint32_t width = std::min(slab_width, w.N - n0);
    auto* dest = static_cast<float*>(output) + n0;
    const auto project = [&](const void* a, const void* b, void* c, uint32_t height, uint32_t pitch) {
      if (large && w.scales) {   // the channel scales in the epilogue
        launch(list, "pf_gemm_T0_CS", "pf_gemm", 512, height / tile, width / 256, a, b, c, height, w.K, width,
               w.K, width, pitch, uint64_t(0), uint64_t(0), uint64_t(0),
               static_cast<const float*>(at(mem, w.scales)) + n0);
      } else if (large) {
        launch(list, kernels::pf_gemm_variant(false), "pf_gemm", 512, height / tile, width / 256,
               a, b, c, height, w.K, width, w.K, width, pitch, uint64_t(0), uint64_t(0), uint64_t(0));
      } else {
        const uint32_t sn = width % 128 == 0 ? 4 : 2;
        const auto gemm = kernels::pf_gemm_small_variant(2, 2, sn, false) + (sn == 2 ? "_INK" : "");
        launch(list, gemm, "pf_gemm_small", 16 * sn, width / (32 * sn), height / tile,
               a, b, c, height, w.K, width, w.K, width, pitch);
      }
    };
    const auto run = [&](const void* b, float* c, uint32_t pitch, uint32_t full_rows) {
      if (full_rows) project(input, b, c, full_rows, pitch);
      if (tail) {
        project(padded_input, b, padded_output, tile, width);
        launch(list, unpack, "pf_ink_tail", 64, width / 64, tail,
               padded_output, c + size_t(whole) * pitch, width, pitch);
      }
    };
    if (int4) {
      // Preserve the full int4 dequantized value with the shared wide projection's hi and lo slabs.
      auto* hi = static_cast<uint16_t*>(slab);
      auto* lo = hi + size_t(w.K) * width;
      auto* partials = reinterpret_cast<float*>(lo + size_t(w.K) * width);
      const uint64_t plane = uint64_t(rows) * width;
      launch(list, df2_wide, "df_w_slab_q4", 256, width / 256, w.K / 64, weights, hi, n0, w.K, width);
      if (large && whole) {
        launch_3d(list, kernels::pf_gemm_variant(false), "pf_gemm", 512, whole / tile, width / 256, 2,
                  input, hi, partials, whole, w.K, width, w.K, width, width, uint64_t(0), uint64_t(w.K) * width, plane);
      }
      run(hi, partials, width, large ? 0 : whole);
      run(lo, partials + plane, width, large ? 0 : whole);
      launch(list, df2_wide, "df_w_add", 256, width / 256, rows, partials, dest, width, w.N, plane);
    } else {
      launch(list, unpack, "pf_ink_slab", 64, width / 64, w.K / 8, weights, slab, w.K, n0, width,
             uint32_t(w.scales != 0));
      run(slab, dest, w.N, whole);
    }
  }
  if (w.scales && !large) launch(list, unpack, "pf_ink_scale", 64, w.N / 64, rows, output, at(mem, w.scales), w.N);
  profile_stage_ = stage;
}

void Engine::record_dflash_prompt_context(l0::CmdList& list, const PrefillChunk& chunk, const PrefillBuffers& s) {
  const auto& d = buffers_.draft;
  const auto& f = desc_.draft;
  const uint32_t first = prefill_end_ > kDraftRing ? std::max(chunk.position, prefill_end_ - kDraftRing) : chunk.position;
  if (first >= chunk.position + chunk.rows) return;   // the chunk ends before the rings' reach
  const uint32_t r0 = first - chunk.position, n = chunk.rows - r0;
  const size_t H = f.hidden, m = chunk.rows;
  auto* fc = at(*s.draft_projection);
  auto* fcx = at(*s.draft_projection, m * H * f32);
  auto* kv = at(*s.draft_projection, m * H * (f32 + bf16));
  const size_t ring = size_t(kDraftRing) * f.kv_n() * bf16;
  const auto* taps = static_cast<const std::byte*>(s.draft_taps->ptr()) + size_t(r0) * 4 * D::hidden * bf16;
  prefill_linear(list, "fc.weight", n, taps, fc);
  launch(list, df2_wide, "df_w_norm", 256, 1, n, fc, nullptr, weight("hidden_norm.weight"), fcx, 0u);
  for (uint32_t l = 0; l < f.layers; ++l) {
    const auto a = "layers." + std::to_string(l) + ".self_attn.";
    prefill_linear(list, a + "kv_proj.weight", n, fcx, kv);
    launch(list, df2_wide, "df_w_ctx_store", f.head_dim, 2 * f.kv_heads, n, kv, nullptr,
           static_cast<const std::byte*>(weight(a + "qk_norm")) + f.head_dim * f32, d.rope->ptr(),
           at(*d.draft_ring_k, l * ring), at(*d.draft_ring_v, l * ring), first, 0u);
  }
}

void Engine::record_prefill_chunk(l0::CmdList& list, const PrefillChunk& chunk, const uint32_t* ids, bool head) {
  auto walk = prefill_walk(chunk, ids, head, buffers_.prefill);
  while (!walk.done) {
    const auto crossing = record_prefill_part(list, walk);
    contract_assert(!crossing);   // only tp's walk crosses
  }
}

Engine::PrefillWalk Engine::prefill_walk(const PrefillChunk& chunk, const uint32_t* ids, bool head, PrefillBuffers& s) {
  const size_t hidden = size_t(chunk.rows) * D::hidden * f32;
  PrefillWalk w{chunk, ids, head, &s};
  w.current = at(*s.residual);
  w.next = at(*s.residual, hidden);
  w.midpoint = at(*s.residual, 3 * hidden);
  w.prev_midpoint = at(*s.residual, 5 * hidden);
  return w;
}

std::optional<Engine::Crossing> Engine::record_prefill_part(l0::CmdList& list, PrefillWalk& w) {
  pf_ = w.s;
  quantized_ = w.quantized;
  struct Keep {   // the walk's int8 input, kept for its next part
    PrefillWalk& w;
    const void*& quantized;
    ~Keep() { w.quantized = quantized; }
  } keep{w, quantized_};
  const auto& s = *w.s;
  const auto& p = buffers_.persistent;
  const auto& chunk = w.chunk;
  const uint32_t m = chunk.rows;
  const size_t hidden = size_t(m) * D::hidden * f32, arow = size_t(m) * D::hidden * bf16;
  const size_t qn = size_t(m) * tp_.q_n() * f32, rn = size_t(m) * tp_.rel_n() * f32;
  const size_t kv = size_t(m) * tp_.kv_n() * f32;
  // tp: this card's first head of each kind (its per-head weights' offsets into the whole tensors)
  const size_t q0 = size_t(tp_.rank) * tp_.q_heads(), kv0 = size_t(tp_.rank) * tp_.kv_heads();
  // The glue's messages, controller mixes and expert rows are bf16 under B70_INK_GLUE=bf16 (bf16_glue_).
  const std::string glue = tp_.on() ? k::tp_prefill_glue(bf16_glue_)
                                    : bf16_glue_ ? "ink_glue_H2304_PREFILL_BF16" : "ink_glue_H2304_PREFILL";
  const size_t message = size_t(m) * D::hidden * (bf16_glue_ ? bf16 : f32);
  const auto residual = [&](uint32_t slot) { return at(*s.residual, slot * hidden); };
  const auto bank = [&](uint32_t slot) { return at(*s.block_bank, slot * hidden); };
  const auto activ = [&](uint32_t slot) { return at(*s.inputs, slot * arow); };
  void*& current = w.current;
  void*& next = w.next;
  void*& midpoint = w.midpoint;
  void*& prev_midpoint = w.prev_midpoint;
  uint32_t& finished = w.finished;

  if (w.visit == 0 && w.phase == 0 && stage_.first == 0)
    launch(list, glue, "ink_embed_norm", 128, m, 1, w.ids, weight(root + "embed.weight"),
           weight(root + "embed_norm.weight"), current, desc_.eps);
  const auto mix_norm = [&](const void* last, const void* query, uint32_t queries, const void* q,
                            const void* k = nullptr, const void* v = nullptr) {
    if (profile_pool_) profile_stage_ = queries == 1 ? "1 query" : "3 queries";
    launch(list, glue, "ink_depth_mix_norm", 128, queries, m, bank(0), query, activ(0),
           finished + 1, m, desc_.eps, q, k, v, last);
    profile_stage_.clear();
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
    if (profile_pool_) profile_stage_ = streams == 1 ? "mlp" : shared_mix ? "attn shared" : "attn";
    if (shared_mix && streams == 3)   // a window visit's attention controller: one x for its three streams
      launch(list, glue, "ink_prefill_mix_cat", 128, m, 1, bank(0), last, query, messages, mixed, cat, finished + 1,
             m, desc_.eps);
    else
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
    profile_stage_.clear();
  };

  for (; w.visit < chunk.visits.size(); ++w.visit, w.phase = 0) {
    const auto& v = chunk.visits[w.visit];
    const auto l = v.layer;
    if (v.conv_slot == stage_.end) break;
    if (v.conv_slot < stage_.first) {   // the card before's visit: the walk's bookkeeping only
      if (l == D::core_start && v.pass != 0) {
        std::swap(midpoint, prev_midpoint);
        finished = D::core_start / D::block_layers;
      }
      finished += v.block_boundary;
      std::swap(current, next);
      continue;
    }
    const auto layer = model::inkling2_layer_prefix(l);
    const auto a = layer + "attn.";
    auto* raw_k = at(*s.conv), *raw_v = at(*s.conv, kv);
    auto* raw_attn = at(*s.conv, 2 * kv), *raw_mlp = at(*s.conv, 2 * kv + hidden);
    const size_t here = v.conv_slot - stage_.first;   // the visit's index among this card's
    const size_t kh = here * 3 * tp_.kv_n() * f32, ah = here * 3 * D::hidden * f32;
    auto* hist_k = at(p.conv_k, kh), *hist_v = at(p.conv_v, kh);
    auto* hist_attn = at(p.conv_attn, ah), *hist_mlp = at(p.conv_mlp, ah);
    void* draft_tap = nullptr;
    if (s.draft_taps)
      for (uint32_t t = 0; t + 1 < desc_.draft.taps.size(); ++t)
        if (v.conv_slot + 1 == desc_.draft.taps[t]) draft_tap = at(*s.draft_taps, size_t(t) * D::hidden * bf16);

    if (w.phase == 0) {
      if (v.conv_slot == stage_.first && v.conv_slot) {   // the split: handoff_rows' rows are in place
        contract_assert(current == residual(0) && prev_midpoint == residual(3) && v.controller);
        list.copy(residual(2), bank(1), hidden);   // the prelude residual is finished block 1
      }
      quantized_ = nullptr;   // this visit rewrites the linears' inputs
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
        launch(list, glue, "ink_prefill_cross_mix", 128, m, 1, prev_midpoint,
               weight(root + "middle_cycle_cross_pass_queries"), weight(root + "middle_cycle_cross_pass_kv_queries"),
               s.messages->ptr(), m, residual(4), l - D::core_start);

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
      prefill_linear(list, a + "wq_du.weight", m, activ(0), q);
      prefill_linear(list, a + "gate_attn.weight", m, activ(0), gate);
      prefill_linear(list, a + "wr_du.weight", m, activ(0), relative);   // the int8 pass: q's rows, before k / v's
      prefill_linear(list, a + "wk_dv.weight", m, ak, raw_k);
      prefill_linear(list, a + "wv_dv.weight", m, av, raw_v);
      prefill_linear(list, a + "attn_sink_proj.weight", m, activ(0), sinks);
      void *kt = nullptr, *vt = nullptr;
      if (!v.window) {
        kt = at(*s.full_taps);
        vt = at(*s.full_taps, size_t(m) * 64 * f32);
        prefill_linear(list, a + "k_sconv.weight_proj", m, ak, kt);
        prefill_linear(list, a + "v_sconv.weight_proj", m, av, vt);
        // tp: every head's taps are projected (64 columns a row); this card's kv heads read theirs, 4 a head
        kt = static_cast<std::byte*>(kt) + kv0 * D::conv_taps * f32;
        vt = static_cast<std::byte*>(vt) + kv0 * D::conv_taps * f32;
      }

      const uint32_t ring = v.window ? kRing : 0;
      const bool kv8 = kv8_ && !ring;
      const size_t kv_off = size_t(slot(v)) * (ring ? ring : max_len_) * tp_.kv_n() * (kv8 ? 1 : bf16);
      auto* ck = at(ring ? p.ring_k : p.full_k, kv_off);
      auto* cv = at(ring ? p.ring_v : p.full_v, kv_off);
      const size_t scale_off = size_t(slot(v)) * max_len_ * tp_.kv_heads() * bf16;
      void* ks = kv8 ? at(*p.full_k_scale, scale_off) : nullptr;
      void* vs = kv8 ? at(*p.full_v_scale, scale_off) : nullptr;
      const char* taps = ring ? "weight" : "static_w";
      // the convolution taps of this card's channels: [channel][4] on window visits, [4][kv_n] (from the first) on full
      const size_t tap_off = (ring ? kv0 * D::head_dim * D::conv_taps : kv0 * D::head_dim) * f32;
      const auto* k_taps = static_cast<const std::byte*>(weight(a + "k_sconv." + taps)) + tap_off;
      const auto* v_taps = static_cast<const std::byte*>(weight(a + "v_sconv." + taps)) + tap_off;
      const auto* q_scale = static_cast<const std::byte*>(weight(a + "q_l2_clip.scale")) + q0 * D::head_dim * f32;
      const auto* r_scale = static_cast<const std::byte*>(weight(a + "rel_l2_clip.scale")) + q0 * D::relative_dim * f32;
      const std::string prep_variant(tp_.on() ? k::kTpPrefillPrep : k::kPrefillPrep);
      const auto attn_variant = [&](bool local) { return tp_.on() ? k::tp_prefill_attn(local) : k::prefill_attn(local); };
      const uint32_t heads = tp_.q_heads() + 2 * tp_.kv_heads();
      // The ring holds kPrefillAttentionRows rows and their window: a longer chunk writes and reads its keys in
      // parts, each part's convolution history the previous part's last rows.
      for (uint32_t r0 = 0; r0 < m; r0 += kPrefillAttentionRows) {
        const uint32_t n = std::min(kPrefillAttentionRows, m - r0);
        const auto row = [&](void* base, size_t width, size_t bytes = f32) {
          return base ? static_cast<std::byte*>(base) + size_t(r0) * width * bytes : nullptr;
        };
        auto* part_k = row(raw_k, tp_.kv_n()), *part_v = row(raw_v, tp_.kv_n());
        void* part_hk = r0 ? static_cast<std::byte*>(part_k) - 3 * tp_.kv_n() * f32 : hist_k;
        void* part_hv = r0 ? static_cast<std::byte*>(part_v) - 3 * tp_.kv_n() * f32 : hist_v;
        auto* part_q = row(q, tp_.q_n()), *part_rel = row(relative, tp_.rel_n());
        auto* part_sinks = static_cast<std::byte*>(row(sinks, 64)) + q0 * f32;
        auto* attn_out = row(s.attention->ptr(), tp_.q_n(), bf16);
        const uint32_t at_pos = chunk.position + r0;
        // PERF-12: a short part deep in the cache walks its keys in spans (a full visit, one card or pp; the window's
        // keys are few, tp's half heads keep one span).
        const uint32_t spans = ring || tp_.on() ? 1 : attention_spans(n, at_pos), span = at_pos / spans / 64 * 64;
        if (kv8) {
          launch(list, prep_variant + "_KV8", "ink_qkv_prep", 128, heads, n, part_q, part_rel, q_scale, r_scale,
                 part_k, part_v, part_hk, part_hv, k_taps, v_taps, row(kt, 64), row(vt, 64), ck, cv, ks, vs,
                 chunk.position + r0, ring, desc_.eps, desc_.qk_clip, desc_.relative_clip, desc_.log_floor,
                 desc_.log_alpha);
          if (profile_pool_) profile_stage_ = "full";
          const auto flash = attn_variant(false) + (int8_pass_ ? "_KV8_I8S_PAIR" : "_KV8_PAIR");
          if (spans > 1) {
            launch_3d(list, flash + "_SPLIT", "kol_pf_flash_attn", k::kPrefillAttnGroup, (n + 15) / 16, D::kv_heads,
                      spans, part_q, ck, cv, ks, vs, attn_out, at_pos, n, part_rel, weight(a + "rel_logits_proj.proj"),
                      part_sinks, row(gate, D::q_n), span, s.split_parts->ptr());
            launch(list, flash + "_SPLIT", "kol_pf_flash_merge", D::head_dim, D::q_heads, n, s.split_parts->ptr(),
                   attn_out, n, spans, part_sinks, row(gate, D::q_n));
          } else {
            launch_3d(list, flash, "kol_pf_flash_attn", k::kPrefillAttnGroup, (n + 15) / 16, tp_.kv_heads(), 1, part_q,
                      ck, cv, ks, vs, attn_out, at_pos, n, part_rel, weight(a + "rel_logits_proj.proj"), part_sinks,
                      row(gate, tp_.q_n()));
          }
        } else {
          launch(list, prep_variant, "ink_qkv_prep", 128, heads, n, part_q, part_rel, q_scale, r_scale,
                 part_k, part_v, part_hk, part_hv, k_taps, v_taps, row(kt, 64), row(vt, 64), ck, cv,
                 chunk.position + r0, ring, desc_.eps, desc_.qk_clip, desc_.relative_clip, desc_.log_floor,
                 desc_.log_alpha);
          if (profile_pool_) profile_stage_ = ring ? "window" : "full";
          const auto flash = attn_variant(bool(ring)) + "_PAIR";
          if (spans > 1) {
            launch_3d(list, flash + "_SPLIT", "kol_pf_flash_attn", k::kPrefillAttnGroup, (n + 15) / 16, D::kv_heads,
                      spans, part_q, ck, cv, attn_out, at_pos, n, part_rel, weight(a + "rel_logits_proj.proj"),
                      part_sinks, row(gate, D::q_n), span, s.split_parts->ptr());
            launch(list, flash + "_SPLIT", "kol_pf_flash_merge", D::head_dim, D::q_heads, n, s.split_parts->ptr(),
                   attn_out, n, spans, part_sinks, row(gate, D::q_n));
          } else {
            launch_3d(list, flash, "kol_pf_flash_attn", k::kPrefillAttnGroup, (n + 15) / 16, tp_.kv_heads(), 1, part_q,
                      ck, cv, attn_out, at_pos, n, part_rel, weight(a + "rel_logits_proj.proj"), part_sinks,
                      row(gate, tp_.q_n()));
          }
        }
        profile_stage_.clear();
      }
      if (tp_.on()) {   // this card's heads' share of o_proj; both cards' into raw_attn
        prefill_linear(list, a + "wo_ud.weight", m, s.attention->ptr(), s.tp_part->ptr());
        w.phase = 1;
        return Crossing{s.tp_part->ptr(), raw_attn, m};
      }
      prefill_linear(list, a + "wo_ud.weight", m, s.attention->ptr(), raw_attn);
      w.phase = 1;
    }

    if (w.phase == 1) {
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
      launch(list, k::route(), "kol_route", 128, 1, m, s.route_logits->ptr(), weight(layer + "mlp.gate.bias"),
             s.routes->ptr());
      const std::string sorter(k::kPrefillSort);
      const std::string moe = std::string(tp_.on() ? k::kTpPrefillMoe : k::kPrefillMoe) + (bf16_glue_ ? "_BF16" : "");
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
      launch(list, moe, "kol_pf_gather", 64, tiles * kPrefillExpertTile, 1, activ(0), hdr, row_tok,
             s.expert_inputs->ptr());
      const auto fmt = desc_.expert_format[l];
      const auto expert = [&](bool gate_up) {
        return tp_.on() ? k::tp_prefill_expert(fmt, gate_up, bf16_glue_) : k::prefill_expert(fmt, gate_up, bf16_glue_);
      };
      if (profile_pool_) profile_stage_ = std::string(fmt == desc_.expert_format[0] ? "fp4" : "fp8") + " w13";
      launch(list, expert(true), "pf_moe_q4gemm", k::kPrefillExpertGroup, 2 * tp_.inter() / 128, items,
             work, s.expert_inputs->ptr(), weight(layer + "mlp.experts.w13_weight"), nullptr, s.expert_output->ptr());
      const uint32_t swiglu = tp_.on() ? k::kTpSwigluGroup : 96;
      launch(list, moe, "ink_prefill_swiglu", swiglu, tp_.inter() / (4 * swiglu), tiles * kPrefillExpertTile,
             s.expert_output->ptr(), s.expert_hidden->ptr());
      if (profile_pool_) profile_stage_.replace(profile_stage_.size() - 3, 3, "w2");
      launch(list, expert(false), "pf_moe_q4gemm", k::kPrefillExpertGroup, D::hidden / 128, items,
             work, s.expert_hidden->ptr(), weight(layer + "mlp.experts.w2_weight"), nullptr, s.expert_output->ptr());
      profile_stage_.clear();
      if (tp_.on()) {   // this card's part of the experts' sum (the combine stops there); both cards' into raw_mlp
        launch(list, moe, "ink_prefill_moe_combine", 256, D::hidden / 256, groups, s.routes->ptr(), pair_row,
               s.expert_output->ptr(), weight(layer + "mlp_sconv.weight"), hist_mlp, s.tp_part->ptr(), next,
               nullptr, m);
        w.phase = 2;
        return Crossing{s.tp_part->ptr(), raw_mlp, m};
      }
      launch(list, moe, "ink_prefill_moe_combine", 256, D::hidden / 256, groups, s.routes->ptr(), pair_row,
             s.expert_output->ptr(), weight(layer + "mlp_sconv.weight"), hist_mlp, raw_mlp, next, draft_tap, m);
    }

    if (tp_.on())   // the experts' convolution and partial sum over the all-reduced rows
      launch(list, glue, "ink_post_conv", 128, D::hidden / 128, m, raw_mlp, hist_mlp, weight(layer + "mlp_sconv.weight"),
             next, next, draft_tap);
    launch(list, glue, "ink_prefill_commit_conv", 128, 2 * (tp_.kv_n() + D::hidden) / 128, 1,
           raw_k, hist_k, hist_v, hist_attn, hist_mlp, m);
    if (w.head && trace_prelude_ && v.conv_slot < D::core_start) {
      constexpr size_t row_bytes = D::hidden * f32;
      list.copy(at(*s.prelude_trace, v.conv_slot * row_bytes), static_cast<std::byte*>(next) + (m - 1) * row_bytes,
                row_bytes);
    }
    std::swap(current, next);
  }
  w.done = true;
  if (!stage_.head()) {   // the rows cross (handoff_rows), PpEngine's copies
    contract_assert(current == residual(0) && prev_midpoint == residual(3));
    return std::nullopt;
  }

  const bool drafter = buffers_.weights.draft != nullptr;
  if (w.head || drafter)
    mix_norm(current, weight(root + "attn_res.output_query"), 1, weight(root + "norm.weight"));
  if (drafter && dflash()) {
    launch(list, glue, "ink_tap", 128, D::hidden / 128, m, nullptr, activ(0), s.draft_taps->ptr(), 3u);
    record_dflash_prompt_context(list, chunk, s);
  } else if (drafter) {
    auto* taps = s.draft_taps->ptr();
    launch(list, glue, "ink_tap", 128, D::hidden / 128, m, nullptr, activ(0), taps, 3u);
    constexpr size_t tap_row = 4 * D::hidden * bf16;
    const size_t kv_row = tp_.draft_kv_n() * bf16, kv_f32 = tp_.draft_kv_n() * f32;   // the card's heads (tp)
    list.copy(buffers_.draft.draft_tap->ptr(), at(*s.draft_taps, (m - 1) * tap_row), tap_row);
    const auto& d = buffers_.draft;
    auto* mix = at(*s.draft_projection);
    auto* key = at(*s.draft_projection, arow);
    auto* value = at(*s.draft_projection, arow + size_t(m) * D::kv_n * f32);
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
        launch(list, tp_.on() ? std::string(k::kTpDraftContextPrefill) : "df_prep_H2304_INK_CONTEXT_PREFILL",
               "df_ink_context", 128, tp_.draft_kv_heads(), std::min(kDraftRing, m - r0), chunk.position + r0,
               static_cast<std::byte*>(key) + r0 * kv_f32, static_cast<std::byte*>(value) + r0 * kv_f32,
               weight(a + "k_norm_ctx.weight"), d.rope->ptr(),
               at(local ? *d.draft_ring_k : *d.draft_full_k, off),
               at(local ? *d.draft_ring_v : *d.draft_full_v, off), local ? kDraftRing : 0u);
    }
  }
  if (!w.head) return std::nullopt;
  const auto& decode = buffers_.scratch;
  linear(list, root + "unembed.weight", 1, at(*s.inputs, (m - 1) * D::hidden * bf16), decode.logits->ptr());
  const auto argmax = k::argmax(1);
  launch(list, argmax, "argmax_stage1", 256, (D::head_rows + 1023) / 1024, 1, decode.logits->ptr(),
         decode.argmax_parts->ptr());
  launch(list, argmax, "argmax_stage2", 256, 1, 1, ctrl_, decode.argmax_parts->ptr());
  return std::nullopt;
}

}  // namespace runtime::inkling2
