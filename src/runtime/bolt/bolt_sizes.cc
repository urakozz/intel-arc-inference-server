#include "runtime/bolt/bolt_sizes.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace runtime::bolt {
namespace {

uint32_t device_blocks(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev, model::BoltBlock k) {
  uint32_t n = 0;
  for (uint32_t l = p.first(dev); l < p.end(dev); ++l) n += d.block.at(l) == k ? 1 : 0;
  return n;
}

[[noreturn]] void no_fit(const model::BoltDesc& d, const DevicePlan& p, uint32_t dev, uint32_t max_len, size_t device_bytes,
                         bool one_card) {
  char buf[900];
  std::snprintf(buf, sizeof buf,
                "runtime::bolt: the plan at max_len %u does not fit device %u's %zu B (weights %zu, kv %zu, ssm %zu, "
                "conv_rings %zu, decode_scratch %zu, prefill_scratch %zu, drafter %zu, drafter_embed_replica %zu, reserve "
                "%zu; total %zu)%s",
                max_len, dev, device_bytes, p.weights, p.kv, p.ssm, p.conv_rings, p.decode_scratch, p.prefill_scratch,
                p.drafter, p.drafter_embed_replica, p.reserve, p.total(),
                one_card ? " - " : "");
  std::string msg = buf;
  if (one_card)
    msg += d.name + " does not fit one 32 GB card (Decision 4: two cards by default, --pp 2; one card only through "
                    "spec 22's expert-offload tier, not built - spec 25 §4.2)";
  throw std::runtime_error(msg);
}

}  // namespace

DevicePlan device_plan(const model::BoltDesc& d, const model::BoltPlacement& p, uint32_t dev, uint32_t max_len, bool int8_head,
                       const BoltReserve& extra, size_t reserve_bytes) {
  model::validate(p, d);
  if (max_len > d.max_pos)
    throw std::runtime_error("runtime::bolt::plan: max_len " + std::to_string(max_len) +
                             " exceeds the trained context " + std::to_string(d.max_pos) + " (max_position_embeddings)");
  DevicePlan pl;
  pl.weights = loader::bolt::device_weight_bytes(d, p, dev, int8_head);
  const uint32_t attn = device_blocks(d, p, dev, model::BoltBlock::Attention);
  const uint32_t mamba = device_blocks(d, p, dev, model::BoltBlock::Mamba);
  pl.kv = size_t(attn) * max_len * d.kv_heads * d.head_dim * 2 * 2;                 // K and V, bf16
  pl.ssm = loader::bolt::ssm_state_bytes(d, mamba) * (1 + size_t(extra.verify_slots));
  pl.conv_rings = loader::bolt::conv_ring_bytes(d, mamba);
  pl.decode_scratch = extra.decode_scratch;
  pl.prefill_scratch = extra.prefill_scratch;
  if (dev + 1 == p.devices) {
    pl.drafter = extra.drafter;
    pl.drafter_embed_replica = extra.drafter_embed_replica;
  }
  pl.reserve = reserve_bytes;
  return pl;
}

void require_fits(const DevicePlan& p, uint32_t dev, uint32_t max_len, size_t device_bytes) {
  if (p.total() <= device_bytes) return;
  char buf[700];
  std::snprintf(buf, sizeof buf,
                "runtime::bolt: the plan at max_len %u does not fit device %u's %zu B (weights %zu, kv %zu, ssm %zu, "
                "conv_rings %zu, decode_scratch %zu, prefill_scratch %zu, drafter %zu, drafter_embed_replica %zu, reserve "
                "%zu; total %zu)",
                max_len, dev, device_bytes, p.weights, p.kv, p.ssm, p.conv_rings, p.decode_scratch, p.prefill_scratch,
                p.drafter, p.drafter_embed_replica, p.reserve, p.total());
  throw std::runtime_error(buf);
}

uint32_t pp_split(const model::BoltDesc& d, bool int8_head) {
  if (d.layers < 2)
    throw std::invalid_argument(d.name + ": " + std::to_string(d.layers) + " block(s), nothing to split");
  std::vector<size_t> layer_bytes(d.layers);
  for (uint32_t l = 0; l < d.layers; ++l) layer_bytes[l] = loader::bolt::layer_weight_bytes(d, l);
  const size_t dev0_fixed = loader::bolt::embed_tiled_bytes(d);
  const size_t dev1_fixed =
      loader::bolt::final_norm_bytes(d) + (int8_head ? loader::bolt::head_int8_bytes(d) : loader::bolt::head_bf16_bytes(d));
  return runtime::pp_balance(layer_bytes, dev0_fixed, dev1_fixed).split;
}

BoltPlan plan(const model::BoltDesc& d, uint32_t max_len, const std::vector<size_t>& device_free, const BoltReserve& extra,
              std::optional<uint32_t> split, size_t reserve_bytes, bool int8_head) {
  if (device_free.size() != 1 && device_free.size() != 2)
    throw std::invalid_argument("runtime::bolt::plan: " + std::to_string(device_free.size()) +
                                " devices (spec 25 §4.1 builds and gates two; one for synthetic / --layers checkpoints)");
  BoltPlan out;
  if (device_free.size() == 1) {
    if (split) throw std::invalid_argument("runtime::bolt::plan: a split on one device");
    out.place = model::BoltPlacement::one(d);
  } else {
    out.place = model::BoltPlacement::two(d, split ? *split : pp_split(d, int8_head));
  }
  model::validate(out.place, d);
  const bool one_card = device_free.size() == 1;
  // The largest max_len in [lo, hi] (multiples of 256) whose plan fits every device, or 0.
  const auto fits = [&](uint32_t len) {
    for (uint32_t dev = 0; dev < out.place.devices; ++dev)
      if (device_plan(d, out.place, dev, len, int8_head, extra, reserve_bytes).total() > device_free[dev]) return false;
    return true;
  };
  uint32_t len = max_len;
  if (len == 0) {
    const uint32_t cap = d.max_pos;
    if (cap < kMaxLenQuantum)
      throw std::invalid_argument("runtime::bolt::plan: the trained context " + std::to_string(cap) + " is below one " +
                                  std::to_string(kMaxLenQuantum) + "-position quantum");
    uint32_t lo = std::min(kMinAutoMaxLen, cap) / kMaxLenQuantum, hi = cap / kMaxLenQuantum;
    if (fits(lo * kMaxLenQuantum)) {
      while (lo < hi) {
        const uint32_t mid = lo + (hi - lo + 1) / 2;
        if (fits(mid * kMaxLenQuantum))
          lo = mid;
        else
          hi = mid - 1;
      }
      len = lo * kMaxLenQuantum;
    } else {
      len = std::min(kMinAutoMaxLen, cap);   // not even this fits: fall through to the naming refusal below
    }
  } else {
    if (len > d.max_pos)
      throw std::runtime_error("runtime::bolt::plan: --max-len " + std::to_string(len) + " exceeds the trained context " +
                               std::to_string(d.max_pos) + " (max_position_embeddings)");
    if (len % kMaxLenQuantum != 0)
      throw std::invalid_argument("runtime::bolt::plan: --max-len " + std::to_string(len) + " is not a multiple of " +
                                  std::to_string(kMaxLenQuantum) + " (the prefill attention's grid)");
  }
  out.max_len = len;
  for (uint32_t dev = 0; dev < out.place.devices; ++dev) {
    DevicePlan pl = device_plan(d, out.place, dev, len, int8_head, extra, reserve_bytes);
    if (pl.total() > device_free[dev]) no_fit(d, pl, dev, len, device_free[dev], one_card && d.layers == model::bolt().layers);
    out.device.push_back(pl);
  }
  return out;
}

std::string describe(const DevicePlan& p, size_t device_bytes) {
  char buf[512];
  std::snprintf(buf, sizeof buf,
                "weights %.3f GB, kv %.3f GB, ssm %.3f GB, conv rings %zu B, scratch %.3f GB, drafter %.3f GB, reserve %.3f "
                "GB, total %.3f GB",
                double(p.weights) / 1e9, double(p.kv) / 1e9, double(p.ssm) / 1e9, p.conv_rings,
                double(p.decode_scratch + p.prefill_scratch) / 1e9, double(p.drafter + p.drafter_embed_replica) / 1e9,
                double(p.reserve) / 1e9, double(p.total()) / 1e9);
  std::string s = buf;
  if (device_bytes) s += " of " + std::to_string(double(device_bytes) / 1e9) + " GB";
  return s;
}

}  // namespace runtime::bolt
