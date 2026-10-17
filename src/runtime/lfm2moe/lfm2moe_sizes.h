#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "loader/lfm2moe_layout.h"
#include "model/lfm2moe.h"
#include "runtime/memory_plan.h"     // kMaxLenQuantum, kMinAutoMaxLen, kDefaultReserveGb
#include "runtime/pipeline_plan.h"   // pp_balance

// Spec 24b Task 5: LFM2.5-24B-A2B's memory plan as device-free arithmetic - `--max-len auto` (the
// largest multiple of 256 that fits AND is at most the trained context, 131072, Review Focus 4) and
// the two-card split by bytes (runtime::pp_balance, spec 16b's rule - not a second one). Host only
// (tests/runtime/lfm2moe_plan_test.cc). 24c adds the decode scratch and the launch counts, 24d the
// prefill scratch; both fields exist here already (0 until then) so this Plan's shape does not
// change under them.
namespace runtime::lfm2moe {

// One device's plan: weights (loader::lfm::device_weight_bytes), kv (its attention layers' K and V
// at max_len), conv_rings (its conv layers' Bx ring), rope (its own table, when it holds an
// attention layer), decode_scratch / prefill_scratch (0 until 24c / 24d), reserve (the device bytes
// the plan deliberately does not count - runtime::kDefaultReserveGb by default; max_len_auto's
// `reserve_gb` argument overrides it for that check).
struct Plan {
  size_t weights = 0, kv = 0, conv_rings = 0, rope = 0, decode_scratch = 0, prefill_scratch = 0, reserve = 0;
  uint32_t max_len = 0;
  size_t total() const { return weights + kv + conv_rings + rope + decode_scratch + prefill_scratch + reserve; }
};

// A device's attention / conv layer counts under placement `p` (what `plan` below scales kv and
// conv_rings by).
inline uint32_t device_attn_layers(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev) {
  uint32_t n = 0;
  for (uint32_t l = p.first(dev); l < p.end(dev); ++l) n += d.is_attention(l) ? 1 : 0;
  return n;
}
inline uint32_t device_conv_layers(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev) {
  uint32_t n = 0;
  for (uint32_t l = p.first(dev); l < p.end(dev); ++l) n += d.is_attention(l) ? 0 : 1;
  return n;
}

// device `dev`'s plan at `max_len` (reserve defaults to runtime::kDefaultReserveGb; max_len_auto
// overrides `.reserve` on its own copy to try a caller-given reserve_gb).
Plan plan(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, uint32_t dev, uint32_t max_len, bool int8_head);

// Throws std::runtime_error naming every term of `p` (weights, kv, conv_rings, rope, decode_scratch,
// prefill_scratch, reserve, their sum) when p.total() exceeds device_bytes.
void require_fits(const Plan& p, size_t device_bytes);

// `--max-len auto`: the largest multiple of runtime::kMaxLenQuantum (256), at most
// d.trained_max_len (131072; Review Focus 4 - an explicit --max-len above it is the loader's own
// refusal, naming max_position_embeddings), whose plan at that length fits `device_bytes` on EVERY
// device of `p` (reserve_gb bytes held back on each). 0 when not even
// min(runtime::kMinAutoMaxLen, d.trained_max_len) does.
uint32_t max_len_auto(const model::Lfm2MoeDesc& d, const model::LfmPlacement& p, size_t device_bytes,
                      double reserve_gb, bool int8_head);

// The two-card split by LAYER WEIGHTS alone (runtime::pp_balance; spec 24 §4.5's two-card note: the
// hand-off is the residual row, so there is no per-length KV asymmetry to balance against) - the
// split whose heavier device holds the fewest layer bytes, device 0's embedding and the last
// device's final norm / head counted as each side's fixed overhead.
uint32_t pp_split(const model::Lfm2MoeDesc& d, bool int8_head);

// "weights W GB, kv K GB, conv rings C B, rope R B, reserve Z GB, total T GB" (+ " of D GB" when
// `device_bytes` is given).
std::string describe(const Plan& p, size_t device_bytes = 0);

}  // namespace runtime::lfm2moe
