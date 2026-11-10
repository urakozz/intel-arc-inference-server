#include "runtime/tp_plan.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "kernels/prefill/pf_kernels.h"
#include "model/model_desc.h"
#include "runtime/buffer_sizes.h"
#include "runtime/tp_protocol.h"

namespace runtime {

std::vector<std::array<uint32_t, 2>> tp_prefill_chunks(uint32_t pos, size_t n, uint32_t chunk, uint32_t stride) {
  if (chunk == 0 || chunk % 64 != 0)
    throw std::invalid_argument("runtime::tp_prefill_chunks: chunk " + std::to_string(chunk) +
                                " is not a positive multiple of 64");
  if (stride % 64 != 0)
    throw std::invalid_argument("runtime::tp_prefill_chunks: stride " + std::to_string(stride) +
                                " is not a multiple of 64");
  std::vector<std::array<uint32_t, 2>> out;
  size_t off = 0;
  while (off < n) {
    // A pair takes up to two chunks' rows; split evenly on the 64-grid so both halves of the wavefront have work,
    // A never shorter than B. Spec 17d: never past the next multiple of the hook's stride.
    size_t r = std::min<size_t>(n - off, size_t(2) * chunk);
    if (stride != 0) r = std::min<size_t>(r, stride - (pos + off) % stride);
    uint32_t a = uint32_t(r);
    if (r > 64) a = uint32_t(std::min<size_t>(chunk, (r / 2 + 63) / 64 * 64));
    out.push_back({pos + uint32_t(off), a});
    if (r > a) out.push_back({pos + uint32_t(off) + a, uint32_t(r - a)});
    off += r;
  }
  return out;
}

size_t tp_prefill_extra_bytes(const model::ModelDesc& card, uint32_t chunk_cap) {
  // TpPfCard: chunk B's residual (bf16), the two chunks' own halves and the four landing slots (fp32).
  return size_t(chunk_cap) * card.hidden * (2 + kTpPrefillChunks * 4 + kTpPrefillChunks * 2 * 4);
}

size_t tp_crossing_bytes(const model::ModelDesc& card, model::TpCrossing c) {
  const tp::Layout L = tp::layout(card, c);
  const size_t own = c == model::TpCrossing::Folded ? size_t(tp::kMRows) * card.hidden * 4 : 0;
  return L.total_bytes() + 2 * L.counter_bytes() + tp::kStateWords * 4 + own;
}

size_t tp_prefill_mtp_bytes(const model::ModelDesc& card, uint32_t chunk_cap) {
  return (size_t(chunk_cap) + 1) * card.hidden * 2 + size_t(2) * card.hidden * kernels::kPfSlabWidth * 2;
}

MemoryPlan tp_card_plan(const model::ModelDesc& card, model::TpCrossing c, uint32_t max_len, size_t model_bytes,
                        KvCache kv, const PrefillPath& path, bool mtp) {
  MemoryPlan p = plan(card, max_len, /*mtp=*/false, model_bytes, path, {}, kv);
  p.decode_state += tp_crossing_bytes(card, c);
  if (path.prefill) p.prefill_scratch += tp_prefill_extra_bytes(card, path.chunk_cap);
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
                              size_t reserve_bytes, uint32_t cap, KvCache kv, const PrefillPath& path, bool mtp) {
  // The crossings (and spec 17c's prefill buffers beside the one-card scratch plan() counts) do not grow with
  // max_len; spec 17d's head buffers do (its KV layer), so the search holds the whole card plan to each device.
  const auto fits = [&](uint32_t len) {
    for (uint32_t r = 0; r < model::kTpSize; ++r)
      if (tp_card_plan(card, c, len, model_bytes[r], kv, path, mtp).total() + reserve_bytes > device_bytes[r])
        return false;
    return true;
  };
  if (!mtp) {
    uint32_t best = cap;
    const size_t fixed = tp_crossing_bytes(card, c) + (path.prefill ? tp_prefill_extra_bytes(card, path.chunk_cap) : 0);
    for (uint32_t r = 0; r < model::kTpSize; ++r) {
      if (device_bytes[r] < fixed) return 0;
      best = std::min(best, max_len_that_fits(card, false, model_bytes[r], device_bytes[r] - fixed, reserve_bytes,
                                              cap, path, {}, kv));
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
