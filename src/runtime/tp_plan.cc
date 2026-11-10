#include "runtime/tp_plan.h"

#include <algorithm>
#include <stdexcept>

#include "runtime/tp_protocol.h"

namespace runtime {

size_t tp_crossing_bytes(const model::ModelDesc& card, model::TpCrossing c) {
  const tp::Layout L = tp::layout(card, c);
  const size_t own = c == model::TpCrossing::Folded ? size_t(tp::kMRows) * card.hidden * 4 : 0;
  return L.total_bytes() + 2 * L.counter_bytes() + tp::kStateWords * 4 + own;
}

MemoryPlan tp_card_plan(const model::ModelDesc& card, model::TpCrossing c, uint32_t max_len, size_t model_bytes,
                        KvCache kv) {
  PrefillPath none;
  none.prefill = false;
  MemoryPlan p = plan(card, max_len, /*mtp=*/false, model_bytes, none, {}, kv);
  p.decode_state += tp_crossing_bytes(card, c);
  return p;
}

uint32_t tp_max_len_that_fits(const model::ModelDesc& card, model::TpCrossing c,
                              const std::array<size_t, 2>& model_bytes, const std::array<size_t, 2>& device_bytes,
                              size_t reserve_bytes, uint32_t cap, KvCache kv) {
  uint32_t best = cap;
  const size_t crossing = tp_crossing_bytes(card, c);
  PrefillPath none;
  none.prefill = false;
  for (uint32_t r = 0; r < model::kTpSize; ++r) {
    // The crossing does not grow with max_len: it comes off the device like the reserve.
    if (device_bytes[r] < crossing) return 0;
    best = std::min(best, max_len_that_fits(card, false, model_bytes[r], device_bytes[r] - crossing, reserve_bytes, cap,
                                            none, {}, kv));
  }
  return best;
}

}  // namespace runtime
