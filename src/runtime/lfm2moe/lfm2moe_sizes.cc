#include "runtime/lfm2moe/lfm2moe_sizes.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace runtime::lfm2moe {

Plan plan(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev, uint32_t max_len,
         bool int8_head) {
  model::validate(p, d);
  if (max_len > d.trained_max_len)
    throw std::runtime_error("runtime::lfm2moe::plan: max_len " + std::to_string(max_len) +
                             " exceeds the trained context " + std::to_string(d.trained_max_len) +
                             " (max_position_embeddings)");
  Plan pl;
  pl.max_len = max_len;
  pl.weights = loader::lfm::device_weight_bytes(d, p, dev, int8_head);
  pl.kv = size_t(device_attn_layers(d, p, dev)) * max_len * d.kv_heads * d.head_dim * 2 * 2;
  pl.conv_rings = size_t(device_conv_layers(d, p, dev)) * loader::lfm::kConvRing * d.hidden * 2;
  pl.rope = loader::lfm::device_has_attention(d, p, dev) ? loader::lfm::rope_table_bytes(d, max_len) : 0;
  pl.reserve = size_t(kDefaultReserveGb * 1e9);
  return pl;
}

void require_fits(const Plan& p, size_t device_bytes) {
  if (p.total() <= device_bytes) return;
  char buf[512];
  std::snprintf(buf, sizeof buf,
               "runtime::lfm2moe: the plan at max_len %u does not fit %zu B (weights %zu, kv %zu, "
               "conv_rings %zu, rope %zu, decode_scratch %zu, prefill_scratch %zu, reserve %zu; total %zu)",
               p.max_len, device_bytes, p.weights, p.kv, p.conv_rings, p.rope, p.decode_scratch,
               p.prefill_scratch, p.reserve, p.total());
  throw std::runtime_error(buf);
}

uint32_t max_len_auto(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, size_t device_bytes,
                      double reserve_gb, bool int8_head) {
  model::validate(p, d);
  const uint32_t cap = d.trained_max_len;
  if (cap < kMaxLenQuantum)
    throw std::invalid_argument("runtime::lfm2moe::max_len_auto: the trained context " + std::to_string(cap) +
                                " is below one " + std::to_string(kMaxLenQuantum) + "-position quantum");
  const size_t reserve_bytes = size_t(reserve_gb * 1e9);
  const auto fits = [&](uint32_t len) {
    for (uint32_t dev = 0; dev < p.devices; ++dev) {
      Plan pl = plan(d, p, dev, len, int8_head);
      pl.reserve = reserve_bytes;
      if (pl.total() > device_bytes) return false;
    }
    return true;
  };
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

uint32_t pp_split(const model::Lfm2MoeDesc& d, bool int8_head) {
  if (d.layers < 2)
    throw std::invalid_argument(d.name + ": " + std::to_string(d.layers) + " layer(s), nothing to split");
  std::vector<size_t> layer_bytes(d.layers);
  for (uint32_t l = 0; l < d.layers; ++l) layer_bytes[l] = loader::lfm::layer_weight_bytes(d, l);
  const size_t dev0_fixed = loader::lfm::embed_tiled_bytes(d);
  const size_t dev1_fixed = size_t(d.hidden) * 4 +
                            (int8_head ? loader::lfm::head_int8_bytes(d) : loader::lfm::embed_tiled_bytes(d));
  return runtime::pp_balance(layer_bytes, dev0_fixed, dev1_fixed).split;
}

std::string describe(const Plan& p, size_t device_bytes) {
  char buf[400];
  std::snprintf(buf, sizeof buf,
               "max_len %u: weights %.3f GB, kv %.3f GB, conv rings %zu B, rope %zu B, reserve %.3f GB, "
               "total %.3f GB",
               p.max_len, p.weights / 1e9, p.kv / 1e9, p.conv_rings, p.rope, p.reserve / 1e9, p.total() / 1e9);
  std::string out = buf;
  if (device_bytes) {
    char tail[64];
    std::snprintf(tail, sizeof tail, " of %.3f GB", device_bytes / 1e9);
    out += tail;
  }
  return out;
}

}  // namespace runtime::lfm2moe
