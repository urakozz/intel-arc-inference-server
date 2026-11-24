#include "runtime/dflash/dflash_sizes.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "kernels/dflash_dims.h"
#include "runtime/buffer_sizes.h"
#include "runtime/dflash/dflash_control.h"

namespace runtime::dflash {

namespace kd = kernels::dflash;

std::vector<float> rope_table(double theta, uint32_t hd, uint32_t rows) {
  const uint32_t half = hd / 2;
  std::vector<double> inv(half);
  for (uint32_t j = 0; j < half; ++j) inv[j] = 1.0 / std::pow(theta, double(2 * j) / double(hd));
  std::vector<float> t(size_t(rows) * hd);
  for (uint32_t p = 0; p < rows; ++p)
    for (uint32_t j = 0; j < half; ++j) {
      const double a = double(p) * inv[j];
      t[size_t(p) * hd + j] = float(std::cos(a));
      t[size_t(p) * hd + half + j] = float(std::sin(a));
    }
  return t;
}

uint32_t ring_slots(const model::DFlashDesc& d, uint32_t layer, uint32_t max_len) {
  return d.window.at(layer) ? d.window[layer] : max_len;
}

DFlashSizes sizes(const model::DFlashDesc& d, uint32_t target_hidden, uint32_t head_rows, uint32_t max_len,
                  loader::DraftFormat format, bool candidate_head, uint32_t rows) {
  const size_t M = rows, R = rows - 1, H = d.hidden;
  DFlashSizes s;
  s.dctl = sizeof(DFlashCtl);
  s.out = sizeof(DFlashDraftOut);
  s.rope = (size_t(max_len) + kd::kM) * d.head_dim * 4;   // a block's rows reach pos + kM - 1
  for (uint32_t l = 0; l < d.layers; ++l) s.ring_k += size_t(ring_slots(d, l, max_len)) * d.kv_n() * 2;
  s.ring_v = s.ring_k;
  s.taps = size_t(kd::kM) * d.fc_k(target_hidden) * 2;   // a session's: one verify list's rows
  s.resid = s.h = M * H * 4;
  s.x = M * H * 2;
  s.xa = M * std::max({size_t(H), size_t(d.q_n()), size_t(d.inter)}) * 2;
  s.gout = M * std::max({size_t(d.qkv_n()), size_t(H), size_t(d.gate_up_n())}) * 4;
  s.co = d.v2() ? M * d.kproj_n() * 4 : 0;
  s.q = M * d.q_n() * 4;
  s.kb = s.vb = M * d.kv_n() * 4;
  // df_attn_part's blocks of 128 ring offsets over the largest ring (a full layer: max_len)
  uint32_t slots = 0;
  for (uint32_t l = 0; l < d.layers; ++l) slots = std::max(slots, ring_slots(d, l, max_len));
  s.part = size_t(d.q_heads) * ((slots + kd::kAttnPb - 1) / kd::kAttnPb) * M * (d.head_dim + 2) * 4;
  s.kv = size_t(d.layers) * M * 2 * d.kv_n() * 4;
  s.logits = R * head_rows * 4;
  s.tpart = R * ((head_rows + kd::kTopkChunk - 1) / kd::kTopkChunk) * kd::kTopK * 2 * 4;
  s.cand = s.unary = R * kd::kTopK * 4;
  s.hid = (d.v2() || d.dspark()) ? M * d.sel_rank * 4 : 0;   // Kolibri DSpark: ones (the Markov walk, df_select V2)
  if (format == loader::DraftFormat::Int4G64) {
    s.gout = M * std::max({size_t(d.qkv_n()) * kd::int4_slices(d.hidden, d.qkv_n()),
                           H * kd::int4_slices(d.q_n(), d.hidden),
                           H * kd::int4_slices(d.inter, d.hidden),
                           H * kd::int4_slices(d.fc_k(target_hidden), d.hidden),
                           size_t(d.gate_up_n()) * kd::int4_slices(d.hidden, d.gate_up_n())}) * 4;
    s.co *= kd::int4_slices(d.hidden, d.kproj_n());
    s.kv *= kd::int4_slices(d.hidden, 2 * d.kv_n());
    s.hid *= kd::int4_slices(d.hidden, d.sel_rank);
  }
  if (candidate_head) {
    s.tpart *= 5;   // up to top80 rather than top16 (spec 19 v3d: 80 candidates for the 2-bit head)
    s.candidate = R * 80 * 4;
  }
  return s;
}

DrafterPlan drafter_plan(const model::DFlashDesc& d, uint32_t target_hidden, uint32_t head_rows,
                         loader::DraftFormat f, bool candidate_head) {
  DrafterPlan p;
  p.weights = loader::dflash_weight_bytes(d, target_hidden, f).total();
  // the 2-bit candidate head (spec 19 v3d): 72 u32 a 16-column tile and 64-row group
  if (candidate_head) p.weights += size_t(head_rows / 16) * (target_hidden / 64) * 72 * 4;
  // Linear in max_len: the difference between two lengths is the per-position term.
  const size_t a = sizes(d, target_hidden, head_rows, 4096, f, candidate_head).total();
  const size_t b = sizes(d, target_hidden, head_rows, 8192, f, candidate_head).total();
  p.per_position = (b - a) / 4096;
  p.fixed = a - p.per_position * 4096;
  p.n_taps = uint32_t(d.taps.size());   // spec 19c: the engine's prefill tap staging
  p.window = d.all_sliding() ? d.max_window() : 0;
  // Plan 19v3b: a sequence's session - ring, taps, control, readback - the same way (a full layer's ring grows).
  const auto session = [&](uint32_t L) {
    return sizes(d, target_hidden, head_rows, L, f, candidate_head).SessionSizes::total();
  };
  p.session_per_position = (session(8192) - session(4096)) / 4096;
  p.session_fixed = session(4096) - p.session_per_position * 4096;
  // Spec 39: and beside several slots its row tables and row taps (DFlashBuffers::add_sessions; unplanned before);
  // plan 40a Task 4: and the scratch's growth to the batched pass's rows.
  const DFlashSizes one = sizes(d, target_hidden, head_rows, 4096, f, candidate_head);
  p.rows = rows_sizes(one, 2).total() +
           (sizes(d, target_hidden, head_rows, 4096, f, candidate_head, seqs_rows(f, candidate_head)).total() -
            one.total());
  return p;
}

RowsSizes rows_sizes(const DFlashSizes& s, uint32_t seqs) {
  RowsSizes r;
  if (seqs < 2) return r;
  r.seqs = r.proj_seqs = (size_t(kd::kSeqsRows) + 8 * kd::kSeqs) * 4;   // the widest list's table
  r.rows_taps = s.taps / kd::kM * DecodeScratchDims::kWideRowsM;   // plan 40a Task 3: the 32-row lists' rows
  return r;
}

size_t ring_snapshot_bytes(const model::DFlashDesc& d, uint32_t max_len) {
  size_t k = 0;
  for (uint32_t l = 0; l < d.layers; ++l) k += size_t(ring_slots(d, l, max_len)) * d.kv_n() * 2;
  return 2 * k;   // ring_k, then ring_v: sizes()' two terms
}

namespace {
struct Fnv {
  uint64_t h = 0xCBF29CE484222325ull;
  void bytes(const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001B3ull;
  }
  void u32(uint32_t v) { bytes(&v, sizeof v); }
  void f64(double v) { bytes(&v, sizeof v); }
};
}  // namespace

uint64_t drafter_key(const model::DFlashDesc& d, loader::DraftFormat f, const void* norms, size_t norm_bytes,
                     bool packed) {
  Fnv h;
  h.bytes(d.architecture.data(), d.architecture.size());
  for (uint32_t v : {d.version, d.layers, d.hidden, d.q_heads, d.kv_heads, d.head_dim, d.inter, d.vocab, d.block,
                     d.mask_id, d.conv_group, d.conv_taps, d.sel_rank, d.sel_top_k})
    h.u32(v);
  h.u32(uint32_t(d.taps.size()));
  for (uint32_t t : d.taps) h.u32(t);
  for (uint32_t w : d.window) h.u32(w);
  h.f64(d.theta);
  h.f64(d.eps);
  h.u32(f == loader::DraftFormat::Int4G64 ? 4u : f == loader::DraftFormat::Int8 ? 8u : 16u);
  if (packed) h.u32(0x504B4434u);   // "PKD4": the other forms' keys stay as they were
  if (norm_bytes != 0) h.bytes(norms, norm_bytes);
  return h.h != 0 ? h.h : 1;
}

size_t draft_launches(const model::DFlashDesc& d, bool candidate_head) {
  return 1 + size_t(d.layers) * (d.v2() ? 16 : 12) + (d.v2() ? 6 : 5) + size_t(candidate_head);
}
size_t project_launches(const model::DFlashDesc& d) { return 3 + d.layers; }
size_t tail_launches(const model::DFlashDesc& d) { return d.v2() ? 2 : 1; }
size_t tp_draft_launches(const model::DFlashDesc& d) {
  return draft_launches(d, true) - tail_launches(d) + 4 * size_t(d.layers);
}

void set_draft_params(DFlashCtl& c, const model::DFlashDesc& d, uint32_t k, float temperature, uint64_t seed) {
  if (k < 1 || k > kd::kR || k > d.max_k())
    throw std::invalid_argument("DFlash draft: K " + std::to_string(k) + " outside 1.." +
                                std::to_string(std::min(kd::kR, d.max_k())));
  if (!(temperature >= 0.0f)) throw std::invalid_argument("DFlash draft: the temperature must be >= 0");
  if (!d.v2() && temperature != 0.0f)
    throw std::invalid_argument("DFlash draft: a DFlash v1 drafter drafts greedily only (temperature 0): its sampled "
                                "q is a full-vocabulary softmax the 16-candidate readback cannot carry");
  c.k = k;
  c.temperature = temperature;
  c.seed_lo = uint32_t(seed);
  c.seed_hi = uint32_t(seed >> 32);
}

}  // namespace runtime::dflash
