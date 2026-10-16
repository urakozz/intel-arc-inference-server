#include "runtime/dflash/dflash_sizes.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "kernels/dflash_dims.h"
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

DFlashSizes sizes(const model::DFlashDesc& d, uint32_t target_hidden, uint32_t head_rows, uint32_t max_len) {
  const size_t M = kd::kM, R = kd::kR, H = d.hidden;
  DFlashSizes s;
  s.dctl = sizeof(DFlashCtl);
  s.out = sizeof(DFlashDraftOut);
  s.rope = (size_t(max_len) + M) * d.head_dim * 4;
  for (uint32_t l = 0; l < d.layers; ++l) s.ring_k += size_t(ring_slots(d, l, max_len)) * d.kv_n() * 2;
  s.ring_v = s.ring_k;
  s.taps = M * d.fc_k(target_hidden) * 2;
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
  s.hid = d.v2() ? M * d.sel_rank * 4 : 0;
  return s;
}

DrafterPlan drafter_plan(const model::DFlashDesc& d, uint32_t target_hidden, uint32_t head_rows,
                         loader::DraftFormat f) {
  DrafterPlan p;
  p.weights = loader::dflash_weight_bytes(d, target_hidden, f).total();
  // Linear in max_len: the difference between two lengths is the per-position term.
  const size_t a = sizes(d, target_hidden, head_rows, 4096).total();
  const size_t b = sizes(d, target_hidden, head_rows, 8192).total();
  p.per_position = (b - a) / 4096;
  p.fixed = a - p.per_position * 4096;
  return p;
}

size_t draft_launches(const model::DFlashDesc& d) {
  return 1 + size_t(d.layers) * (d.v2() ? 16 : 12) + (d.v2() ? 6 : 5);
}
size_t project_launches(const model::DFlashDesc& d) { return 3 + d.layers; }

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
