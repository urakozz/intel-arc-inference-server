#include "runtime/tp_plan.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "kernels/prefill/pf_kernels.h"
#include "model/model_desc.h"
#include "runtime/buffer_sizes.h"
#include "runtime/prefill_chunks.h"
#include "runtime/tp_protocol.h"

namespace runtime {

std::vector<std::array<uint32_t, 2>> tp_prefill_chunks(uint32_t pos, size_t n, uint32_t chunk, uint32_t block,
                                                       uint32_t tail) {
  if (chunk == 0 || chunk % 64 != 0)
    throw std::invalid_argument("runtime::tp_prefill_chunks: chunk " + std::to_string(chunk) +
                                " is not a positive multiple of 64");
  if (block % 64 != 0)
    throw std::invalid_argument("runtime::tp_prefill_chunks: block " + std::to_string(block) +
                                " is not a multiple of 64");
  std::vector<std::array<uint32_t, 2>> out;
  size_t off = 0;
  while (off < n) {
    // A pair takes up to two chunks' rows; split evenly on the 64-grid so both halves of the wavefront have work,
    // A never shorter than B. Spec 17d: never past the next snapshot point (open-work 29: prefill_snapshot_point at
    // the pair's stride).
    const uint32_t p = pos + uint32_t(off);
    size_t r = std::min<size_t>(n - off, size_t(kTpPrefillChunks) * chunk);
    if (block != 0)
      r = std::min<uint64_t>(r, prefill_snapshot_point(p, uint64_t(pos) + n,
                                                       prefill_hook_stride(kTpPrefillChunks * chunk, block, true),
                                                       block, tail) - p);
    uint32_t a = uint32_t(r);
    if (r > 64) a = uint32_t(std::min<size_t>(chunk, (r / 2 + 63) / 64 * 64));
    out.push_back({pos + uint32_t(off), a});
    if (r > a) out.push_back({pos + uint32_t(off) + a, uint32_t(r - a)});
    off += r;
  }
  return out;
}

TpPfDeviceSizes tp_pf_device_sizes(const model::ModelDesc& card, uint32_t chunk_cap, bool mtp, uint32_t n_taps,
                                   uint32_t window) {
  TpPfDeviceSizes s;
  const size_t row = size_t(chunk_cap) * card.hidden;
  s.resid_b = row * 2;                                                  // chunk B's residual, bf16
  s.own = size_t(kTpPrefillChunks) * row * 4;                           // the chunks' own halves, fp32
  s.land = size_t(kTpPrefillChunks) * kTpPrefillLandSlots * row * 4;    // their landing slots, fp32
  if (mtp) {
    s.hid = (size_t(chunk_cap) + 1) * card.hidden * 2;
    s.hslab = size_t(2) * card.hidden * kernels::kPfSlabWidth * 2;      // fc's K: [2 x hidden][1024]
  }
  if (n_taps) s.df_taps = TapVerifyDims::prefill_taps_bytes(card, n_taps, window);
  return s;
}

size_t tp_prefill_extra_bytes(const model::ModelDesc& card, uint32_t chunk_cap) {
  const TpPfDeviceSizes s = tp_pf_device_sizes(card, chunk_cap, false);
  return s.resid_b + s.own + s.land;
}

TpCardSizes tp_card_sizes(const model::ModelDesc& card, const tp::Layout& L, bool folded, bool tap, bool drafter) {
  TpCardSizes s;
  s.land = L.total_bytes();
  s.send_ctr = s.fold_ctr = L.counter_bytes();
  s.state = tp::kStateWords * 4;
  if (folded) s.own = size_t(tp::kMRows) * card.hidden * 4;
  if (tap) s.tap = size_t(card.layers) * card.hidden * 2;
  if (drafter) s.df_slots = TapVerifyDims::gdn_spec_bytes(card);
  return s;
}

size_t tp_crossing_bytes(const model::ModelDesc& card, model::TpCrossing c) {
  return tp_card_sizes(card, tp::layout(card, c), c == model::TpCrossing::Folded, false, false).total();
}

size_t tp_prefill_mtp_bytes(const model::ModelDesc& card, uint32_t chunk_cap) {
  const TpPfDeviceSizes s = tp_pf_device_sizes(card, chunk_cap, true);
  return s.hid + s.hslab;
}

MemoryPlan tp_card_plan(const model::ModelDesc& card, model::TpCrossing c, uint32_t max_len, size_t model_bytes,
                        KvCache kv, const PrefillPath& path, bool mtp, const DrafterPlan& drafter) {
  MemoryPlan p = plan(card, max_len, /*mtp=*/false, model_bytes, path, {}, kv, drafter);
  p.decode_state += tp_crossing_bytes(card, c);
  if (path.prefill) p.prefill_scratch += tp_prefill_extra_bytes(card, path.chunk_cap);
  // Spec 39: a drafter's tap staging is the prompt's last window here (TpPfCard::df_taps), not plan()'s chunk.
  if (path.prefill && drafter.on() && drafter.window) {
    const size_t taps = tp_pf_device_sizes(card, path.chunk_cap, false, drafter.n_taps, drafter.window).df_taps;
    p.prefill_scratch = p.prefill_scratch - p.drafter_taps + taps;
    p.drafter_taps = taps;
  }
  if (mtp) {
    p.mtp = true;
    p.mtp_buffers = MtpDims::tp_sizes(max_len, card, model::qwen38(), kv).total();
    p.mtp_hidden = path.prefill ? tp_prefill_mtp_bytes(card, path.chunk_cap) : 0;
    p.decode_state += p.mtp_buffers + p.mtp_hidden;
  }
  return p;
}

uint32_t tp_max_len_that_fits(const model::ModelDesc& card, model::TpCrossing c,
                              const std::array<size_t, 2>& model_bytes, const std::array<size_t, 2>& device_bytes,
                              size_t reserve_bytes, uint32_t cap, KvCache kv, const PrefillPath& path, bool mtp,
                              const DrafterPlan& drafter) {
  // The crossings (and spec 17c's prefill buffers beside the one-card scratch plan() counts) do not grow with
  // max_len; spec 17d's head buffers do (its KV layer), so the search holds the whole card plan to each device.
  const auto fits = [&](uint32_t len) {
    for (uint32_t r = 0; r < model::kTpSize; ++r)
      if (tp_card_plan(card, c, len, model_bytes[r], kv, path, mtp, drafter).total() + reserve_bytes > device_bytes[r])
        return false;
    return true;
  };
  if (!mtp) {
    uint32_t best = cap;
    const size_t fixed = tp_crossing_bytes(card, c) + (path.prefill ? tp_prefill_extra_bytes(card, path.chunk_cap) : 0);
    for (uint32_t r = 0; r < model::kTpSize; ++r) {
      if (device_bytes[r] < fixed) return 0;
      best = std::min(best, max_len_that_fits(card, false, model_bytes[r], device_bytes[r] - fixed, reserve_bytes,
                                              cap, path, {}, kv, drafter));
    }
    return best;
  }
  // max_len_that_fits' search over the whole card plan: the largest multiple of kMaxLenQuantum at most `cap`, 0 when
  // not even min(kMinAutoMaxLen, cap) fits (every term is non-decreasing in max_len).
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

}  // namespace runtime
