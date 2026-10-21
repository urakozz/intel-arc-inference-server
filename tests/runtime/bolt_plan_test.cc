// Spec 25b Task 5: runtime::bolt's memory plan, max_len auto and the two-card split by bytes, host only. No total is typed:
// every figure is the sum of the planner's own per-layer outputs, CHECK_EQ'd against an independent accumulation, and printed.
#include <cstdio>
#include <string>
#include <vector>

#include "check.h"
#include "loader/bolt_layout.h"
#include "model/bolt.h"
#include "runtime/bolt/bolt_sizes.h"

namespace {

template <class F>
void refused(const char* key, F fn) {
  bool threw = false;
  try {
    fn();
  } catch (const std::exception& e) {
    threw = std::string(e.what()).find(key) != std::string::npos;
    if (!threw) std::fprintf(stderr, "refusal of %s named something else: %s\n", key, e.what());
  }
  if (!threw) std::fprintf(stderr, "not refused: %s\n", key);
  CHECK(threw);
}

}  // namespace

int main() {
  using namespace runtime::bolt;
  namespace bl = loader::bolt;
  const model::BoltDesc& d = model::bolt();
  const size_t card32 = size_t(32) << 30;   // a B70's 32 GiB (the lfm2moe plan test's figure; the box's memory_bytes confirms)

  // --- the split by bytes: pp_balance over the blocks' weights, the embedding / head as each side's fixed overhead -------
  const uint32_t split = pp_split(d);
  {
    std::vector<size_t> lb(d.layers);
    for (uint32_t l = 0; l < d.layers; ++l) lb[l] = bl::layer_weight_bytes(d, l);
    const size_t fixed0 = bl::embed_tiled_bytes(d), fixed1 = bl::final_norm_bytes(d) + bl::head_int8_bytes(d);
    // independent: the split whose heavier side is smallest, scanning every cut
    size_t best_mx = ~size_t(0);
    uint32_t best = 0;
    for (uint32_t s = 1; s < d.layers; ++s) {
      size_t a = fixed0, b = fixed1;
      for (uint32_t l = 0; l < d.layers; ++l) (l < s ? a : b) += lb[l];
      if (std::max(a, b) < best_mx) {
        best_mx = std::max(a, b);
        best = s;
      }
    }
    CHECK_EQ(split, best);
    CHECK_EQ(split, 48u);   // spec 25 §3: device 0 = blocks 0-47, device 1 = 48-87 (derived there, re-derived here)
    std::printf("bolt_plan_test: pp_split %u (heavier side %zu B)\n", split, best_mx);
  }

  // --- the real descriptor on two 32 GiB cards: 262144 positions fit with spec 26's drafter and replica reserved ---------
  {
    // The caller's terms (placeholders until 25c / 25d / spec 26 measure them): decode scratch < 100 MB, prefill scratch
    // ~1.5 GB a card, the int8 drafter 1.1 GB and its 1.07 GB embedding replica on device 1, 7 verify slots - ALL ESTIMATED
    // (spec 25 §3), the plan's job is only to place them.
    BoltReserve x;
    x.decode_scratch = size_t(100e6);
    x.prefill_scratch = size_t(1.5e9);
    x.drafter = size_t(1.1e9);
    x.drafter_embed_replica = bl::embed_tiled_bytes(d);
    x.verify_slots = 7;
    const BoltPlan p = plan(d, 0, {card32, card32}, x);
    CHECK_EQ(p.place.devices, 2u);
    CHECK_EQ(p.place.split, split);
    CHECK_EQ(p.max_len, d.max_pos);   // 262144: Decision 14's cap, and it fits
    CHECK_EQ(p.max_len % 256, 0u);
    CHECK_EQ(p.device.size(), size_t(2));
    for (uint32_t dev = 0; dev < 2; ++dev) {
      const DevicePlan& pl = p.device[dev];
      // each term from its own closed formula
      size_t w = 0;
      uint32_t attn = 0, mamba = 0;
      for (uint32_t l = p.place.first(dev); l < p.place.end(dev); ++l) {
        w += bl::layer_weight_bytes(d, l);
        attn += d.is_attention(l);
        mamba += d.is_mamba(l);
      }
      if (dev == 0) w += size_t(4096) * 131072 * 2;                                    // the embedding tiles
      if (dev == 1) w += size_t(4096) * 4 + 4096ull * 131072 + 131072ull * 4;        // norm_f, the int8 head + its scales
      CHECK_EQ(pl.weights, w);
      CHECK_EQ(pl.weights, bl::device_weight_bytes(d, p.place, dev, true));          // the loader's allocation formula
      CHECK_EQ(pl.kv, size_t(attn) * 262144 * (2 * 128) * 2 * 2);                     // heads x head_dim x {K, V} x bf16
      CHECK_EQ(pl.ssm, size_t(mamba) * (128ull * 64 * 128 * 4) * (1 + 7));            // 4 MiB fp32 a Mamba block, 8 slots
      CHECK_EQ(pl.conv_rings, size_t(mamba) * 16 * 10240 * 2);
      CHECK_EQ(pl.decode_scratch, size_t(100e6));
      CHECK_EQ(pl.prefill_scratch, size_t(1.5e9));
      CHECK_EQ(pl.drafter, dev == 1 ? size_t(1.1e9) : size_t(0));                    // the last device only
      CHECK_EQ(pl.drafter_embed_replica, dev == 1 ? bl::embed_tiled_bytes(d) : size_t(0));
      CHECK_EQ(pl.reserve, size_t(1.5e9));
      CHECK(pl.total() <= card32);
      std::printf("  device %u (blocks [%u, %u), %u attention, %u Mamba): %s\n", dev, p.place.first(dev), p.place.end(dev), attn,
                  mamba, describe(pl, card32).c_str());
    }
    CHECK_EQ(p.device[0].kv + p.device[1].kv, size_t(8) * 262144 * 1024);             // 8 attention blocks, 8,192 B a position
    // the two devices partition the model: Σ weights = the model's
    CHECK_EQ(p.device[0].weights + p.device[1].weights, bl::weight_bytes(d, true));

    // no drafter / scratch: the same split and length, and every term the placeholders add is exactly theirs
    const BoltPlan q = plan(d, 0, {card32, card32});
    CHECK_EQ(q.max_len, p.max_len);
    CHECK_EQ(p.device[1].total() - q.device[1].total(),
             x.decode_scratch + x.prefill_scratch + x.drafter + x.drafter_embed_replica + 7 * bl::ssm_state_bytes(d, 18));
  }

  // --- one 32 GiB card: refused naming the bytes and Decision 4 -----------------------------------------------------------
  {
    bool threw = false;
    try {
      plan(d, 0, {card32});
    } catch (const std::runtime_error& e) {
      const std::string m = e.what();
      threw = m.find("Decision 4") != std::string::npos && m.find("device 0") != std::string::npos &&
              m.find(std::to_string(bl::weight_bytes(d, true))) != std::string::npos;
      std::printf("  one card: %s\n", e.what());
    }
    CHECK(threw);
    refused("Decision 4", [&] { plan(d, 8192, {card32}); });
  }

  // --- a smaller pair reaches a smaller auto length (printed); it is the LARGEST multiple of 256 that fits --------------------
  {
    BoltReserve x;
    x.decode_scratch = size_t(100e6);
    x.prefill_scratch = size_t(1.5e9);
    const size_t card24 = size_t(24e9);
    const BoltPlan p = plan(d, 0, {card24, card24}, x);
    CHECK(p.max_len >= 4096 && p.max_len < d.max_pos && p.max_len % 256 == 0);
    for (uint32_t dev = 0; dev < 2; ++dev) CHECK(p.device[dev].total() <= card24);
    bool next_fits = true;
    try {
      plan(d, p.max_len + 256, {card24, card24}, x);
    } catch (const std::runtime_error&) { next_fits = false; }
    CHECK(!next_fits);   // p.max_len is the LARGEST that fits
    std::printf("  24 GB pair (scratch only): max_len auto = %u; device 1: %s\n", p.max_len,
                describe(p.device[1], card24).c_str());
    // the same pair with the drafter and its replica does not hold even 4096 positions: refused, naming a device
    x.drafter = size_t(1.1e9);
    x.drafter_embed_replica = bl::embed_tiled_bytes(d);
    refused("device 1", [&] { plan(d, 0, {card24, card24}, x); });
  }

  // --- explicit lengths ----------------------------------------------------------------------------------------------------
  refused("max_position_embeddings", [&] { plan(d, 262144 + 256, {card32, card32}); });
  refused("multiple of 256", [&] { plan(d, 8000, {card32, card32}); });
  {
    const BoltPlan p = plan(d, 65536, {card32, card32});
    CHECK_EQ(p.max_len, 65536u);
    CHECK_EQ(p.device[0].kv + p.device[1].kv, size_t(8) * 65536 * 1024);
    const BoltPlan s = plan(d, 65536, {card32, card32}, {}, 40u);   // a given split
    CHECK_EQ(s.place.split, 40u);
    CHECK_EQ(s.device[0].weights + s.device[1].weights, bl::weight_bytes(d, true));
    CHECK(s.device[0].weights < p.device[0].weights);
    refused("split", [&] { plan(d, 65536, {card32, card32}, {}, 88u); });
    refused("split", [&] { plan(d, 65536, {card32, card32}, {}, 0u); });
    refused("split", [&] { plan(d, 65536, {card32}, {}, 40u); });
    refused("devices", [&] { plan(d, 65536, {card32, card32, card32}); });
    // the bf16 head: no int8 copy, its own tiles (untied), 1,073,741,824 B against the int8 head's own
    const BoltPlan b = plan(d, 65536, {card32 * 2, card32 * 2}, {}, std::nullopt, size_t(1.5e9), false);
    CHECK_EQ(b.device[0].weights + b.device[1].weights, bl::weight_bytes(d, false));
    CHECK_EQ(bl::weight_bytes(d, false) - bl::weight_bytes(d, true), bl::head_bf16_bytes(d) - bl::head_int8_bytes(d));
  }

  // --- a synthetic / --layers checkpoint fits one card at the full context -------------------------------------------------
  {
    const model::BoltDesc t = model::bolt_truncated(d, 12);
    const BoltPlan p = plan(t, 0, {card32});
    CHECK_EQ(p.max_len, t.max_pos);
    CHECK_EQ(p.place.devices, 1u);
    CHECK_EQ(p.place.split, 12u);
    CHECK_EQ(p.device[0].weights, bl::weight_bytes(t, true));
    CHECK_EQ(p.device[0].kv, size_t(1) * 262144 * 1024);   // one attention block among the first 12
    CHECK_EQ(p.device[0].ssm, size_t(6) * 4194304);        // six Mamba blocks, fp32, one slot
    std::printf("  12 blocks on one card: %s\n", describe(p.device[0], card32).c_str());
  }

  std::puts("bolt_plan_test OK");
  return 0;
}
