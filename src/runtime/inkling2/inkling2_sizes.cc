#include "runtime/inkling2/inkling2_sizes.h"

#include <algorithm>
#include <stdexcept>

#include "runtime/control.h"
#include "kernels/dflash_dims.h"
#include "runtime/dflash/dflash_control.h"
#include "runtime/dflash/dflash_sizes.h"
#include "runtime/inkling2/inkling2_prefill.h"
#include "runtime/inkling2/inkling2_tp_split.h"
#include "runtime/tp_protocol.h"

namespace runtime::inkling2 {
namespace {
using D = model::Inkling2Desc;
using L = loader::Inkling2WeightLayout;
}  // namespace

MemoryComponents Sizes::memory() const {
  MemoryComponents m;
  m.model = weights.model + weights.layers + weights.prompt_int8 + weights.candidate_head;
  m.kv = gs::bytes(persistent, gs::Group::Kv);
  m.decode_state = gs::bytes(persistent, gs::Group::State) + gs::bytes(scratch) + handoff.landing + gs::bytes(tp);
  m.prefill_scratch = gs::bytes(prefill) + gs::bytes(prefill_b) + handoff.prefill_landing + handoff.prefill_outbox +
                      handoff.snapshot_shadow;
  m.drafter = weights.draft + gs::bytes(draft);
  return m;
}

StageShare stage_share(const D& d, const L& layout, const Stage& stage) {
  StageShare share;
  for (uint32_t i = 0; i < stage.end; ++i) {
    const auto& v = d.visits[i];
    if (i < stage.first) {
      ++(v.window ? share.window_base : share.full_base);
    } else {
      ++(v.window ? share.window : share.full);
      share.layers[v.layer] = true;
    }
  }
  for (uint32_t l = 0; l < D::layers; ++l)
    if (share.layers[l]) share.layer_bytes += layout.bytes[L::kFirstLayer + l];
  return share;
}

std::array<Stage, 2> pp_stages(bool draft) { return {Stage{0, kPpSplit, draft}, Stage{kPpSplit, D::visit_count}}; }

Sizes sizes(const D& d, const L& layout, uint32_t max_len, bool draft, uint32_t prefill_rows, bool kv8, bool int8_pass,
            const Stage& stage, const Tp& tp) {
  if (!max_len || max_len > D::trained_max_len)
    throw std::invalid_argument("Inkling --max-len must be within its trained context");
  if (draft && !layout.bytes[L::kDraft])
    throw std::invalid_argument("Inkling drafter weights are missing. Use --spec off or restore mtp.safetensors.");
  contract_assert(!draft || stage.head());
  const auto share = stage_share(d, layout, stage);
  const bool taps = draft || stage.taps;   // this card writes the drafter's taps

  Sizes out;
  const bool dflash = d.draft.form == model::DFlashForm::Vllm;   // plan 41k: the DFlash 2 drafter, not DSpark
  if (prefill_rows) out.prefill = prefill_sizes(max_len, prefill_rows, draft, taps, tp, dflash);
  out.weights = {layout.bytes[L::kModel], share.layer_bytes, draft ? layout.bytes[L::kDraft] : 0};
  if (stage.head()) out.weights.candidate_head = kCandidateHeadBytes;
  if (int8_pass && prefill_rows) {
    const size_t rows = (size_t(prefill_rows) + 255) / 256 * 256;   // the int8 GEMM's 256-row tiles
    uint32_t layers = 0;
    for (const bool held : share.layers) layers += held;
    out.weights.prompt_int8 = layers * int8_layer_bytes(tp);
    out.prefill.int8_rows = rows * D::q_n;
    out.prefill.int8_scales = rows * sizeof(float);
  }
  auto& p = out.persistent;
  const size_t kv_row = tp.kv_n() * 2;
  constexpr size_t conv_rows = (D::conv_taps - 1) * sizeof(float);
  p.control = sizeof(Control);
  p.full_k = p.full_v = size_t(share.full) * max_len * (kv8 ? tp.kv_n() : kv_row);
  if (kv8) p.full_k_scale = p.full_v_scale = size_t(share.full) * max_len * tp.kv_heads() * 2;
  p.ring_k = p.ring_v = size_t(share.window) * kRing * kv_row;
  p.conv_k = p.conv_v = share.visits() * tp.kv_n() * conv_rows;
  p.conv_attn = p.conv_mlp = share.visits() * D::hidden * conv_rows;
  if (!stage.whole()) {
    auto& h = out.handoff;
    if (stage.first) {   // this card receives the previous card's rows
      h.landing = kRows * handoff_row_bytes(stage.taps || draft);
      if (prefill_rows) h.prefill_landing = size_t(prefill_rows) * handoff_row_bytes(stage.taps || draft);
    }
    if (!stage.head() && prefill_rows) h.prefill_outbox = size_t(prefill_rows) * handoff_row_bytes(taps);
  }

  // The developer prompt path uses verify-width rows. Pending convolution
  // inputs are separate from the committed three-row histories.
  auto& s = out.scratch;
  constexpr size_t m = kRows;
  s.residual = m * D::hidden * 6 * 4;
  s.block_bank = m * D::hidden * (D::layers / D::block_layers + 1) * 4;
  s.messages = m * (D::core_end - D::core_start) * 4 * D::hidden * 4;
  s.controller = m * (D::hidden * 12 * sizeof(float) + 3 * D::controller_rank * sizeof(uint16_t));
  s.inputs = m * (3 * D::hidden + D::q_n) * 2;
  s.projections = m * (2 * tp.q_n() + 2 * tp.kv_n() + tp.rel_n() + 64) * 4;
  s.full_taps = m * 2 * 64 * 4;
  s.attention_parts = m * tp.q_heads() * kPartitions * (D::head_dim + 2) * 4;
  s.attn_output = m * tp.q_n() * 2;
  s.routes = share.visits() * m * 32 * sizeof(uint32_t);
  s.expert_union = m * D::top_k * 2 * sizeof(uint32_t);
  s.expert_hidden = m * D::top_k * tp.inter() * 2;
  s.expert_output = m * D::top_k * D::hidden * 4;
  s.pending_conv = share.visits() * m * 2 * (tp.kv_n() + D::hidden) * 4;
  s.logits = m * D::head_rows * 4;
  s.argmax_parts = m * ((D::head_rows + 1023) / 1024) * 2 * sizeof(uint32_t);
  s.step_io = sizeof(StepIo);
  s.glue_parts = m * 3 * (2 * kGlueBank + 2) * kGlueLanes * sizeof(float);   // up to 3 streams a row
  if (taps) s.draft_taps = m * d.draft.taps.size() * D::hidden * 2;
  if (stage.head())
    s.head_candidates = kCandidateParts + kCandidateIds + kCandidateWeights + kCandidateScales + kCandidateScores;
  if (tp.on()) {
    const runtime::tp::Layout cross{D::hidden, 1, kTpFoldG};
    out.tp = {cross.total_bytes(), cross.counter_bytes(), cross.counter_bytes(), runtime::tp::kStateWords * sizeof(uint32_t),
              m * D::hidden * sizeof(float)};
  }
  if (draft && dflash) {
    // Plan 41k: five rings of the last 2048 positions and no per-position cache (every layer slides); the block's
    // scratch in df_prep's form (dflash_capture.cc's): block_hidden holds resid and h (fp32), x and xa (bf16) and the
    // conv coefficients; block_projections the linears' fp32 outputs, q and the block's own k / v; context_mix the commit's
    // fc rows (fp32), their hidden_norm (bf16) and a layer's k / v; markov the selector's hidden projection, candidates,
    // unary logits and top-k parts; draft_logits the head over the 7 draft rows.
    namespace kd = kernels::dflash;
    const auto& f = d.draft;
    const auto sz = dflash::sizes(f, D::hidden, D::head_rows, max_len, layout.draft_format);
    const bool int4 = layout.draft_format == loader::DraftFormat::Int4G64;
    const uint32_t fc_slices = int4 ? kd::int4_slices(f.fc_k(D::hidden), f.hidden) : 1;
    const size_t H = f.hidden;
    const size_t R = m - 1, chunks = (D::head_rows + kd::kTopkChunk - 1) / kd::kTopkChunk;
    auto& x = out.draft;
    x.draft_control = sizeof(dflash::DFlashCtl) + sizeof(dflash::DFlashDraftOut);
    x.draft_ring_k = x.draft_ring_v = size_t(f.layers) * kDraftRing * f.kv_n() * 2;
    x.rope = f.head_dim / 2 * 4;   // inv_freq: df_prep / df_proj_wide compute the angles (ROPE_INV)
    x.block_hidden = sz.resid + sz.h + sz.x + sz.xa + sz.co;
    x.block_projections = sz.gout + sz.q + sz.kb + sz.vb;
    x.draft_parts = size_t(f.q_heads) * (kDraftRing / kd::kAttnPb) * m * (f.head_dim + 2) * 4;
    x.context_mix = m * H * (fc_slices * 4 + 2) + sz.kv / f.layers;
    x.markov = sz.hid + 2 * R * kd::kTopK * 4 + R * chunks * kd::kTopK * 2 * 4;
    x.draft_logits = R * D::head_rows * 4;
  } else if (draft) {
    const size_t draft_kv_row = tp.draft_kv_n() * 2;   // the card's heads of the drafter's caches
    auto& x = out.draft;
    x.draft_control = sizeof(dflash::DFlashCtl) + sizeof(dflash::DFlashDraftOut);
    x.draft_ring_k = x.draft_ring_v = 2 * kDraftRing * draft_kv_row;
    x.draft_tap = d.draft.taps.size() * D::hidden * 2;
    x.draft_full_k = x.draft_full_v = size_t(max_len) * draft_kv_row;
    x.rope = (size_t(max_len) + d.draft.block) * D::head_dim * 4;
    x.block_hidden = m * (D::hidden * 3 + D::draft_inter) * 4;
    x.block_projections = m * (D::q_n + 2 * D::kv_n + 2 * D::draft_inter) * 4;
    x.draft_parts = m * D::q_heads * kDraftPartitions * (D::head_dim + 2) * 4;
    x.context_mix = m * (D::hidden + 2 * D::kv_n) * 4;
    x.markov = (m * 3 + 3) * D::markov_rank * 4;
    x.draft_logits = (m + 2) * D::head_rows * 4;
  }
  if (tp.on() && prefill_rows) {   // two images of the state buffers (Engine::shadow_image)
    const auto& x = out.draft;
    out.handoff.snapshot_shadow = 2 * (p.ring_k + p.ring_v + p.conv_k + p.conv_v + p.conv_attn + p.conv_mlp +
                                       x.draft_ring_k + x.draft_ring_v + x.draft_tap);
  }
  if (!stage.whole() && prefill_rows) {   // Engine::state_runs' bytes on this card, two block ends of them
    const auto& x = out.draft;
    const size_t state = 2 * size_t(share.window) * (D::window - 1) * kv_row + p.conv_k + p.conv_v + p.conv_attn +
                         p.conv_mlp + x.draft_ring_k + x.draft_ring_v + x.draft_tap;
    out.handoff.snapshot_shadow = 2 * state;
  }
  if (prefill_rows && tp.on()) out.prefill_b = out.prefill;
  return out;
}

std::array<Sizes, 2> tp_sizes(const D& d, const L& layout, uint32_t max_len, bool draft, uint32_t prefill_rows, bool kv8,
                              bool int8_pass) {
  std::array<Sizes, 2> out;
  for (uint32_t r = 0; r < 2; ++r)
    out[r] = sizes(d, tp_weights(d, layout, r).layout, max_len, draft, prefill_rows, kv8, int8_pass, {}, Tp{r, 2});
  return out;
}

std::array<Sizes, 2> pp_sizes(const D& d, const L& layout, uint32_t max_len, bool draft, uint32_t prefill_rows, bool kv8,
                              bool int8_pass) {
  const auto stages = pp_stages(draft);
  return {sizes(d, layout, max_len, false, prefill_rows, kv8, int8_pass, stages[0]),
          sizes(d, layout, max_len, draft, prefill_rows, kv8, int8_pass, stages[1])};
}

namespace {
template <class Fits>
uint32_t longest(Fits fits) {
  uint32_t lo = kMinAutoMaxLen / kMaxLenQuantum, hi = D::trained_max_len / kMaxLenQuantum;
  if (!fits(lo)) return 0;
  while (lo < hi) {
    const uint32_t mid = lo + (hi - lo + 1) / 2;
    if (fits(mid)) lo = mid;
    else hi = mid - 1;
  }
  return lo * kMaxLenQuantum;
}
}  // namespace

uint32_t pp_max_len_that_fits(const D& d, const L& layout, size_t capacity, size_t reserve, bool draft,
                              uint32_t prefill_rows, bool kv8, bool int8_pass) {
  if (capacity < reserve) return 0;
  return longest([&](uint32_t pages) {
    for (const auto& s : pp_sizes(d, layout, pages * kMaxLenQuantum, draft, prefill_rows, kv8, int8_pass))
      if (s.memory().total() > capacity - reserve) return false;
    return true;
  });
}

uint32_t pp_prefill_rows(const D& d, const L& layout, size_t capacity, size_t reserve, uint32_t max_len, bool draft,
                         bool kv8, bool int8_pass) {
  for (const uint32_t rows : {kPrefillRows, 2048u})
    if (pp_max_len_that_fits(d, layout, capacity, reserve, draft, rows, kv8, int8_pass) >= max_len) return rows;
  return 0;
}

uint32_t tp_max_len_that_fits(const D& d, const L& layout, size_t capacity, size_t reserve, bool draft,
                              uint32_t prefill_rows, bool kv8, bool int8_pass) {
  if (capacity < reserve) return 0;
  const std::array<L, 2> cards{tp_weights(d, layout, 0).layout, tp_weights(d, layout, 1).layout};
  return longest([&](uint32_t pages) {
    for (uint32_t r = 0; r < 2; ++r)
      if (sizes(d, cards[r], pages * kMaxLenQuantum, draft, prefill_rows, kv8, int8_pass, {}, Tp{r, 2}).memory().total() >
          capacity - reserve)
        return false;
    return true;
  });
}

uint32_t max_len_that_fits(const D& d, const L& layout, size_t capacity, size_t reserve, bool draft,
                           uint32_t prefill_rows, bool kv8, bool int8_pass) {
  if (capacity < reserve) return 0;
  return longest([&](uint32_t pages) {
    return sizes(d, layout, pages * kMaxLenQuantum, draft, prefill_rows, kv8, int8_pass).memory().total() <=
           capacity - reserve;
  });
}

}  // namespace runtime::inkling2
