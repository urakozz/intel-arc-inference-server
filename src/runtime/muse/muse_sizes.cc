#include "runtime/muse/muse_sizes.h"

#include <cstdio>
#include <stdexcept>

#include "loader/muse_layout.h"
#include "runtime/memory_plan.h"     // kMaxLenQuantum, kMinAutoMaxLen, kDefaultReserveGb
#include "runtime/pipeline_plan.h"   // pp_balance

namespace runtime::muse {
namespace {

static_assert(kMuseReserveBytes == size_t(kDefaultReserveGb * 1e9), "the reserve is every planner's");

std::string gb(size_t b) {
  char s[32];
  std::snprintf(s, sizeof s, "%.3f GB", double(b) / 1e9);
  return s;
}

void check_capacity(uint32_t devices, const std::vector<size_t>& capacity) {
  if (devices != 1 && devices != 2)
    throw std::invalid_argument("plan_muse: " + std::to_string(devices) + " devices (one card, or --pp 2)");
  if (capacity.size() < devices)
    throw std::invalid_argument("plan_muse: " + std::to_string(capacity.size()) + " capacities for " +
                                std::to_string(devices) + " devices");
}

size_t scratch_of(const MuseReserve& r) { return r.decode_scratch + r.prefill_scratch; }

// The placement for `devices` at max_len: one card, the given split, or the split by bytes.
model::MusePlacement placement_at(const model::MuseGlimmerDesc& d, uint32_t devices, uint32_t max_len, bool int8_head,
                                  const MuseReserve& r, uint32_t split) {
  if (devices == 1) return model::MusePlacement::one(d);
  if (split != 0) {
    const model::MusePlacement p = model::MusePlacement::two(d, split);
    model::validate(p, d);
    return p;
  }
  if (d.layers < 2) throw std::invalid_argument("plan_muse: one layer cannot be split over two devices");
  const size_t dev0 = loader::muse_embed_bytes(d) + scratch_of(r);
  const size_t dev1 = loader::muse_final_norm_bytes(d) + loader::muse_lm_head_bytes(d, int8_head) + scratch_of(r) + r.drafter;
  return model::MusePlacement::two(d, runtime::pp_balance(pp_layer_bytes(d, max_len), dev0, dev1).split);
}

}  // namespace

std::vector<size_t> pp_layer_bytes(const model::MuseGlimmerDesc& d, uint32_t max_len) {
  const size_t w = loader::muse_layer_bytes(d).total();
  std::vector<size_t> b(d.layers);
  for (uint32_t l = 0; l < d.layers; ++l)
    b[l] = w + (d.is_full(l) ? loader::muse_full_layer_kv_bytes(d, max_len) : loader::muse_ring_bytes_per_layer(d));
  return b;
}

MusePlan plan_at(const model::MuseGlimmerDesc& d, const model::MusePlacement& p, uint32_t max_len, bool int8_head,
                 const MuseReserve& r, const std::vector<size_t>& capacity) {
  model::validate(p, d);
  check_capacity(p.devices, capacity);
  MusePlan plan;
  plan.placement = p;
  plan.max_len = max_len;
  plan.int8_head = int8_head;
  plan.fits = true;
  for (uint32_t dev = 0; dev < p.devices; ++dev) {
    MuseDevicePlan dp;
    dp.device = dev;
    dp.first = p.first(dev);
    dp.end = p.end(dev);
    dp.weights = loader::muse_device_weight_bytes(d, p, dev, int8_head);
    for (uint32_t l = dp.first; l < dp.end; ++l) {
      if (d.is_full(l))
        dp.kv += loader::muse_full_layer_kv_bytes(d, max_len);
      else
        dp.rings += loader::muse_ring_bytes_per_layer(d);
    }
    dp.scratch = scratch_of(r);
    dp.drafter = dev + 1 == p.devices ? r.drafter : 0;
    dp.reserve = kMuseReserveBytes;
    dp.capacity = capacity[dev];
    if (dp.total() > dp.capacity) {
      plan.fits = false;
      plan.why += (plan.why.empty() ? "" : "; ") + std::string("device ") + std::to_string(dev) + " layers [" +
                  std::to_string(dp.first) + ", " + std::to_string(dp.end) + ") at max_len " + std::to_string(max_len) +
                  " needs " + std::to_string(dp.total()) + " B (" + gb(dp.total()) + ": weights " + gb(dp.weights) +
                  ", KV " + gb(dp.kv) + ", rings " + gb(dp.rings) + ", scratch " + gb(dp.scratch) + ", drafter " +
                  gb(dp.drafter) + ", reserve " + gb(dp.reserve) + ") against " + std::to_string(dp.capacity) + " B (" +
                  gb(dp.capacity) + ")";
    }
    plan.devices.push_back(dp);
  }
  return plan;
}

MusePlan plan_muse(const model::MuseGlimmerDesc& d, uint32_t devices, uint32_t max_len, bool int8_head,
                   const MuseReserve& r, const std::vector<size_t>& capacity, uint32_t split) {
  check_capacity(devices, capacity);
  if (devices == 1 && split != 0 && split != d.layers)
    throw std::invalid_argument("plan_muse: a split (" + std::to_string(split) + ") on one device");
  if (max_len != 0) {
    if (max_len > d.max_pos)
      throw std::invalid_argument("plan_muse: max_len " + std::to_string(max_len) + " exceeds the model's " +
                                  std::to_string(d.max_pos) + " (max_position_embeddings) - spec 23 decision 6: "
                                  "contexts above 131072 are refused");
    if (max_len % kMaxLenQuantum != 0)
      throw std::invalid_argument("plan_muse: max_len " + std::to_string(max_len) + " is not a multiple of " +
                                  std::to_string(kMaxLenQuantum) + " (the prefill's and the attention blocks' grid)");
    return plan_at(d, placement_at(d, devices, max_len, int8_head, r, split), max_len, int8_head, r, capacity);
  }
  // auto: the largest length on the grid that fits every device (the split re-chosen at each length).
  const uint32_t top = d.max_pos / kMaxLenQuantum * kMaxLenQuantum;
  for (uint32_t L = top; L >= kMinAutoMaxLen && L >= kMaxLenQuantum; L -= kMaxLenQuantum) {
    MusePlan p = plan_at(d, placement_at(d, devices, L, int8_head, r, split), L, int8_head, r, capacity);
    if (p.fits) return p;
  }
  MusePlan p = plan_at(d, placement_at(d, devices, kMinAutoMaxLen, int8_head, r, split), kMinAutoMaxLen, int8_head, r,
                       capacity);
  p.why = "--max-len auto: not even " + std::to_string(kMinAutoMaxLen) + " fits - " + p.why;
  return p;
}

void require_fits(const MusePlan& p) {
  if (p.fits) return;
  throw std::runtime_error("Muse-Glimmer-30B does not fit: " + p.why);
}

std::string MusePlan::describe() const {
  std::string s = "Muse-Glimmer plan at max_len " + std::to_string(max_len) + ", " +
                  (placement.devices == 1 ? std::string("one card") : "split " + std::to_string(placement.split)) +
                  (int8_head ? ", int8 head" : ", bf16 head") + (fits ? "" : " - DOES NOT FIT") + ":";
  for (const MuseDevicePlan& d : devices)
    s += "\n  device " + std::to_string(d.device) + " layers [" + std::to_string(d.first) + ", " + std::to_string(d.end) +
         "): weights " + gb(d.weights) + ", KV " + gb(d.kv) + ", rings " + gb(d.rings) + ", scratch " + gb(d.scratch) +
         ", drafter " + gb(d.drafter) + ", reserve " + gb(d.reserve) + " = " + gb(d.total()) + " of " + gb(d.capacity);
  return s;
}

}  // namespace runtime::muse
