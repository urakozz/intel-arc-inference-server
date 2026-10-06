#include "runtime/kolibri/kolibri_sizes.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "loader/kolibri1_layout.h"
#include "runtime/control.h"

namespace runtime::kolibri {
namespace {
// The kernels' constants the sizes depend on (src/kernels/kolibri_kernels.h is their home on the host;
// this file stays free of the kernel-dir define - kolibri_capture.cc checks them against it).
constexpr uint32_t kNormG = 20, kAttnTgt = 32, kAttnPart = 130, kArgmaxChunk = 1024, kSlots = 7;
constexpr size_t kLine = 64;   // an empty group's allocation
size_t line(size_t b) { return b ? b : kLine; }

uint32_t count_full(const model::Kolibri1Desc& d, uint32_t first, uint32_t end) {
  uint32_t n = 0;
  for (uint32_t l = first; l < end; ++l) n += d.is_sliding(l) ? 0 : 1;
  return n;
}
}  // namespace

KolAttn kolibri_attn() {
  const char* v = std::getenv("B70_KOLIBRI_ATTN");
  if (v == nullptr || *v == '\0') return kDefaultKolAttn;
  if (std::strcmp(v, "flash") == 0) return KolAttn::Flash;
  if (std::strcmp(v, "eager") == 0) return KolAttn::Eager;
  throw std::runtime_error(std::string("B70_KOLIBRI_ATTN=") + v + ": expected flash or eager");
}

const char* kol_attn_name(KolAttn a) { return a == KolAttn::Eager ? "eager" : "flash"; }

PersistentSizes persistent_sizes(const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t dev,
                                 uint32_t max_len) {
  const uint32_t full = count_full(d, p.first(dev), p.end(dev)), sliding = p.count(dev) - full;
  PersistentSizes s;
  s.control = sizeof(Control);
  s.full_k = s.full_v = line(size_t(full) * full_layer_rows_bytes(d, max_len));
  s.ring_k = s.ring_v = line(size_t(sliding) * ring_layer_rows_bytes(d));
  return s;
}

size_t partials_floats(const model::Kolibri1Desc& d) {
  size_t m = 0;
  for (model::KolLinearId id : d.layer_linears()) {
    const model::GemvShape s = d.linear(id).shape;
    m = std::max(m, size_t(s.S) * s.N);
  }
  return m;
}

ScratchSizes scratch_sizes(const model::Kolibri1Desc& d) {
  ScratchSizes s;
  s.resid = s.x = s.a = s.mo = size_t(kM) * d.hidden * 2;
  s.partials = partials_floats(d) * kM * 4;
  s.sumsq_a = s.sumsq_r = size_t(kNormG) * kM * 4;
  s.attn_q = size_t(kM) * d.q_n() * 4;
  s.attn_part = size_t(d.q_heads) * kAttnTgt * kM * kAttnPart * 4;
  s.attn_out = size_t(kM) * d.q_n() * 2;
  s.logits_r = size_t(kM) * d.router_n() * 4;
  s.routes = size_t(d.layers) * kM * kRouteWords * 4;
  s.moe_h = size_t(kM) * kSlots * d.moe_inter * 2;
  s.logits = size_t(kM) * d.vocab * 4;
  s.argmax_part = size_t(kM) * ((d.vocab + kArgmaxChunk - 1) / kArgmaxChunk) * 2 * 4;
  return s;
}

size_t attn_scores_bytes(const model::Kolibri1Desc& d, uint32_t max_len, KolAttn a) {
  return a == KolAttn::Eager ? size_t(kM) * d.q_heads * max_len * 4 : 0;
}

PpLandingLayout landing_layout(const model::Kolibri1Desc& d) {
  return pp_landing_layout(size_t(kM) * d.hidden * 2, size_t(kNormG) * kM * 4);
}
size_t link_bytes(const model::Kolibri1Desc& d, uint32_t dev) {
  return dev == 0 ? kPpStateWords * 4 : landing_layout(d).total + kPpStateWords * 4;
}

size_t device_launches(const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t dev, KolAttn a,
                       PpHandoff h) {
  (void)d;
  const size_t per_layer = a == KolAttn::Eager ? 17 : 15;
  size_t n = size_t(p.count(dev)) * per_layer;
  if (dev == 0) n += 2;                     // embed_gather, prep_res_fold SP0
  if (dev + 1 == p.devices) n += 4;         // the head
  if (p.devices > 1 && h == PpHandoff::Peer) n += 1;   // pp_send (device 0) / pp_recv (device 1)
  return n;
}

size_t decode_launches(const model::Kolibri1Desc& d, const model::KolPlacement& p, KolAttn a, PpHandoff h) {
  size_t n = 0;
  for (uint32_t dev = 0; dev < p.devices; ++dev) n += device_launches(d, p, dev, a, h);
  return n;
}

// --- spec 20d: the prefill chunk ---------------------------------------------------------------------
namespace {
constexpr uint32_t kPfHdrWords = (4 + 384 + 1 + 15) / 16 * 16;   // kernels::kolibri::pf_hdr::words()
constexpr uint32_t kSlabLinears = 2;                              // q||k||v, o_proj
}  // namespace

uint32_t pf_tiles(const model::Kolibri1Desc& d, uint32_t C) {
  return (C * d.top_k + d.experts * (kPfTm - 1)) / kPfTm + (C + kPfTm - 1) / kPfTm;
}
size_t pf_block_gu_bytes(const model::Kolibri1Desc& d) { return size_t(d.hidden) * 2 * d.moe_inter * 2; }
size_t pf_block_dn_bytes(const model::Kolibri1Desc& d) { return size_t(d.moe_inter) * d.hidden * 2; }
uint32_t pf_batch_blocks_gu(const model::Kolibri1Desc& d) { return uint32_t(kPfBatchBytes / pf_block_gu_bytes(d)); }
uint32_t pf_batch_blocks_dn(const model::Kolibri1Desc& d) { return uint32_t(kPfBatchBytes / pf_block_dn_bytes(d)); }
uint32_t pf_batches_gu(const model::Kolibri1Desc& d) {
  const uint32_t per = pf_batch_blocks_gu(d);
  return (d.experts + 1 + per - 1) / per;
}
uint32_t pf_batches_dn(const model::Kolibri1Desc& d) {
  const uint32_t per = pf_batch_blocks_dn(d);
  return (d.experts + 1 + per - 1) / per;
}

PrefillSizes prefill_sizes(const model::Kolibri1Desc& d) {
  const size_t C = kPfC, T = pf_tiles(d, kPfC), R = T * kPfTm;
  uint32_t ld = 0, kmax = 0;
  for (model::KolLinearId id : d.layer_linears()) {
    const model::GemvShape s = d.linear(id).shape;
    ld = std::max(ld, pf_ld(s.N));
    kmax = std::max(kmax, s.K);
  }
  PrefillSizes s;
  s.ids = C * 4;
  s.resid = s.x = s.a = s.mo = C * d.hidden * 2;
  s.partials = C * ld * 4;
  s.slab = size_t(kmax) * kPfSlab * 2;
  s.sumsq_a = s.sumsq_r = size_t(kNormG) * C * 4;
  s.attn_q = C * d.q_n() * 4;
  s.attn_out = C * d.q_n() * 2;
  s.logits = C * d.router_n() * 4;
  s.routes = size_t(d.layers) * C * kRouteWords * 4;
  s.hdr = size_t(kPfHdrWords) * 4;
  s.tiles = T * 2 * 4;
  s.row_tok = R * 4;
  s.pair_row = C * d.top_k * 4;
  s.xg = R * d.hidden * 2;
  s.h = R * d.moe_inter * 2;
  s.w = kPfBatchBytes;
  return s;
}

PpLandingLayout pf_landing_layout(const model::Kolibri1Desc& d) {
  return pp_landing_layout(size_t(kPfC) * d.hidden * 2, size_t(kNormG) * kPfC * 4);
}
size_t pf_link_bytes(const model::Kolibri1Desc& d, uint32_t dev) {
  return dev == 0 ? kPpStateWords * 4 : pf_landing_layout(d).total + kPpStateWords * 4;
}

size_t prefill_device_launches(const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t dev) {
  size_t slabs = 0;
  for (model::KolLinearId id : d.layer_linears()) slabs += pf_slabs(d.linear(id).shape.N);
  static_assert(kSlabLinears == 2, "q||k||v and o_proj");
  // norm, 2 x slabs, prep, flash, fold _Z + post_add, norm, router, route, sort, gather, gate||up and
  // down batches x 2, combine, fold SP0 + post_add
  const size_t per_layer = 1 + 2 * slabs + 1 + 1 + 2 + 1 + 1 + 1 + 1 + 1 + 2 * (pf_batches_gu(d) + pf_batches_dn(d)) + 1 + 2;
  size_t n = size_t(p.count(dev)) * per_layer;
  if (dev == 0) n += 2;   // pf_embed_gather, pf_res_fold SP0
  return n;
}
size_t prefill_chunk_launches(const model::Kolibri1Desc& d, const model::KolPlacement& p) {
  size_t n = 0;
  for (uint32_t dev = 0; dev < p.devices; ++dev) n += prefill_device_launches(d, p, dev);
  return n;
}

std::vector<size_t> pp_layer_bytes(const model::Kolibri1Desc& d, uint32_t max_len) {
  const size_t w = loader::kol_layer_bytes(d).total();
  std::vector<size_t> v(d.layers);
  for (uint32_t l = 0; l < d.layers; ++l)
    v[l] = w + (d.is_sliding(l) ? ring_bytes_per_layer(d) : full_bytes_per_layer(d, max_len));
  return v;
}

namespace {
size_t fixed_state(const model::Kolibri1Desc& d, uint32_t max_len, KolAttn a) {
  return sizeof(Control) + scratch_sizes(d).total() + attn_scores_bytes(d, max_len, a);
}
// Spec 20d: a device's prefill scratch and its prefill link (two devices), when the run prefills.
size_t prefill_state(const model::Kolibri1Desc& d, uint32_t devices, uint32_t dev, bool prefill) {
  if (!prefill) return 0;
  return prefill_sizes(d).total() + (devices > 1 ? pf_link_bytes(d, dev) : 0);
}
}  // namespace

PpBalance pp_split(const model::Kolibri1Desc& d, uint32_t max_len, bool int8_head, PpHandoff h, KolAttn a,
                   bool prefill) {
  (void)h;   // both hand-offs allocate the same buffers (PipelineLink: the event is not device memory)
  if (d.layers < 2) throw std::invalid_argument(d.name + ": " + std::to_string(d.layers) + " layer(s), nothing to split");
  const size_t rope = d.rope_table_bytes(max_len), state = fixed_state(d, max_len, a);
  const size_t dev0 = loader::kol_embed_bytes(d) + rope + state + link_bytes(d, 0) + prefill_state(d, 2, 0, prefill);
  const size_t dev1 = loader::kol_final_norm_bytes(d) + loader::kol_lm_head_bytes(d, int8_head) + rope + state +
                      link_bytes(d, 1) + prefill_state(d, 2, 1, prefill);
  return pp_balance(pp_layer_bytes(d, max_len), dev0, dev1);
}

std::vector<DevicePlan> plan(const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t max_len,
                             bool int8_head, bool debug_tap, KolAttn a, bool prefill) {
  model::validate(p, d);
  std::vector<DevicePlan> out(p.devices);
  for (uint32_t dev = 0; dev < p.devices; ++dev) {
    DevicePlan& dp = out[dev];
    dp.device = dev;
    dp.weights = loader::kol_device_weight_bytes(d, p, dev, int8_head);
    dp.rope = loader::kol_device_has_sliding(d, p, dev) ? d.rope_table_bytes(max_len) : 0;
    dp.model = dp.weights + dp.rope;
    const PersistentSizes ps = persistent_sizes(d, p, dev, max_len);
    dp.full_kv = ps.full_k + ps.full_v;
    dp.rings = ps.ring_k + ps.ring_v;
    dp.kv = ps.kv();
    dp.link = p.devices > 1 ? link_bytes(d, dev) : 0;
    dp.decode_state = ps.control + scratch_sizes(d).total() + attn_scores_bytes(d, max_len, a) +
                      (debug_tap ? tap_bytes(d) : 0) + dp.link;
    dp.prefill_scratch = prefill_state(d, p.devices, dev, prefill);
  }
  return out;
}

uint32_t max_len_that_fits(const model::Kolibri1Desc& d, const model::KolPlacement& p, bool int8_head,
                           const std::array<size_t, kPpDevices>& device_bytes, size_t reserve, uint32_t cap,
                           bool prefill) {
  if (cap == 0 || cap > d.trained_max_len) cap = d.trained_max_len;
  if (cap < kMaxLenQuantum)
    throw std::invalid_argument("kolibri::max_len_that_fits: the cap " + std::to_string(cap) + " is below one " +
                                std::to_string(kMaxLenQuantum) + "-position quantum");
  const auto fits = [&](uint32_t len) {
    const std::vector<DevicePlan> pl = plan(d, p, len, int8_head, false, kolibri_attn(), prefill);
    for (uint32_t i = 0; i < p.devices; ++i)
      if (pl[i].total() + reserve > device_bytes[i]) return false;
    return true;
  };
  // Every term is non-decreasing in max_len on every device: the quanta that fit are a prefix.
  uint32_t lo = std::min(kMinAutoMaxLen, cap) / kMaxLenQuantum, hi = cap / kMaxLenQuantum;
  if (!fits(lo * kMaxLenQuantum)) return 0;
  while (lo < hi) {
    const uint32_t mid = lo + (hi - lo + 1) / 2;
    if (fits(mid * kMaxLenQuantum))
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo * kMaxLenQuantum;
}

PpChoice pp_split_and_len(const model::Kolibri1Desc& d, bool int8_head,
                          const std::array<size_t, kPpDevices>& device_bytes, size_t reserve, uint32_t cap,
                          bool prefill) {
  uint32_t best_len = 0;
  for (uint32_t s = 1; s < d.layers; ++s) {
    const uint32_t len =
        max_len_that_fits(d, model::KolPlacement::two(d, s), int8_head, device_bytes, reserve, cap, prefill);
    best_len = std::max(best_len, len);
  }
  if (best_len == 0) return {};
  // At that length, the split by bytes - 16b's rule - among the splits that reach it.
  const PpBalance b = pp_split(d, best_len, int8_head, kDefaultPpHandoff, kolibri_attn(), prefill);
  if (max_len_that_fits(d, model::KolPlacement::two(d, b.split), int8_head, device_bytes, reserve, cap, prefill) ==
      best_len)
    return {b.split, best_len};
  for (uint32_t s = 1; s < d.layers; ++s)   // the balanced split does not reach it: the first that does
    if (max_len_that_fits(d, model::KolPlacement::two(d, s), int8_head, device_bytes, reserve, cap, prefill) == best_len)
      return {s, best_len};
  return {};
}

bool fits_one_card(const model::Kolibri1Desc& d, bool int8_head, size_t device_bytes, size_t reserve, bool prefill) {
  const uint32_t len = std::min(kMinAutoMaxLen, d.trained_max_len);
  const std::vector<DevicePlan> pl = plan(d, model::KolPlacement::one(d), len, int8_head, false, kolibri_attn(), prefill);
  return pl[0].total() + reserve <= device_bytes;
}

std::string describe(const std::vector<DevicePlan>& p, const model::KolPlacement& pl, uint32_t max_len,
                     const std::array<size_t, kPpDevices>& device_bytes, size_t reserve) {
  char head[256];
  if (pl.devices > 1)
    std::snprintf(head, sizeof head, "pipeline plan at max_len %u, split %u (device 0 layers [0, %u), device 1 "
                  "layers [%u, %u)):", max_len, pl.split, pl.split, pl.split, pl.layers);
  else
    std::snprintf(head, sizeof head, "plan at max_len %u (one card, layers [0, %u)):", max_len, pl.layers);
  std::string out = head;
  for (const DevicePlan& dp : p) {
    const std::string label = "\n  device " + std::to_string(dp.device);
    out += format_memory(label.c_str(), dp, device_bytes[dp.device]);
    char tail[200];
    std::snprintf(tail, sizeof tail, "; + reserve %.3f GB = %.3f GB (weights %.3f GB, RoPE %.3f GB, full KV %.3f GB, "
                  "rings %.3f GB, hand-off %zu B)", reserve / 1e9, (dp.total() + reserve) / 1e9, dp.weights / 1e9,
                  dp.rope / 1e9, dp.full_kv / 1e9, dp.rings / 1e9, dp.link);
    out += tail;
  }
  return out;
}

// --- spec 20e: prefix-cache snapshots ---------------------------------------------------------------
std::vector<SnapRun> state_runs(const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t pos) {
  model::validate(p, d);
  const uint32_t W = state_positions(d), R = model::Kolibri1Desc::kRing;
  if (W > R)
    throw std::invalid_argument("runtime::kolibri::state_runs: the ring's " + std::to_string(R) +
                                " slots cannot hold the window's " + std::to_string(W) + " positions");
  const size_t row = size_t(d.kv_n()) * 2;
  std::vector<SnapRun> runs;
  for (SnapTensor t : {SnapTensor::RingK, SnapTensor::RingV}) {
    for (uint32_t l = 0; l < d.layers; ++l) {
      if (!d.is_sliding(l)) continue;
      const uint32_t dev = p.device_of(l);
      const size_t base = size_t(d.sliding_before(l) - d.sliding_before(p.first(dev))) * ring_layer_rows_bytes(d);
      // Positions q in [pos - W, pos): those below 0 first (one zero run over their slots
      // [R - (W - pos), R)), then the rest through the ring, split where the slots wrap.
      int64_t q = int64_t(pos) - int64_t(W);
      const int64_t end = int64_t(pos);
      if (q < 0) {
        const uint32_t n = uint32_t(-q);
        runs.push_back({dev, t, base + size_t(R - n) * row, size_t(n) * row, true});
        q = 0;
      }
      while (q < end) {
        const uint32_t slot = uint32_t(q) & (R - 1);
        const uint32_t n = uint32_t(std::min<int64_t>(end - q, int64_t(R - slot)));
        runs.push_back({dev, t, base + size_t(slot) * row, size_t(n) * row, false});
        q += n;
      }
    }
  }
  return runs;
}

std::vector<SnapRun> kv_runs(const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t max_len,
                             uint32_t begin, uint32_t end) {
  model::validate(p, d);
  if (begin > end || end > max_len)
    throw std::invalid_argument("runtime::kolibri::kv_runs: range [" + std::to_string(begin) + ", " +
                                std::to_string(end) + ") is not within [0, max_len " + std::to_string(max_len) + ")");
  std::vector<SnapRun> runs;
  if (begin == end) return runs;
  const size_t row = size_t(d.kv_n()) * 2;
  for (SnapTensor t : {SnapTensor::FullK, SnapTensor::FullV}) {
    for (uint32_t l = 0; l < d.layers; ++l) {
      if (d.is_sliding(l)) continue;
      const uint32_t dev = p.device_of(l);
      const size_t base = size_t(d.full_before(l) - d.full_before(p.first(dev))) * full_layer_rows_bytes(d, max_len);
      runs.push_back({dev, t, base + size_t(begin) * row, size_t(end - begin) * row, false});
    }
  }
  return runs;
}

}  // namespace runtime::kolibri
