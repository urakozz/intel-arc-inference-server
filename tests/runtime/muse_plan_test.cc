// Spec 23b Task 5: Muse-Glimmer-30B's memory plan, one card and two (runtime/muse/muse_sizes.h). Host only. Every
// number is DERIVED from the descriptor (spec 23 §2: one card holds the model with the drafter; decisions 6, 7).
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "loader/muse_layout.h"
#include "model/muse_glimmer.h"
#include "runtime/muse/muse_sizes.h"
#include "runtime/pipeline_plan.h"

namespace {
namespace rm = runtime::muse;
constexpr size_t kCard = 32530000000ull;   // memory_line()'s device total (spec 6 §8.4)

template <class E, class F>
bool throws_naming(const std::string& what, F f) {
  try {
    f();
  } catch (const E& e) {
    if (std::string(e.what()).find(what) != std::string::npos) {
      std::printf("  refused: %s\n", e.what());
      return true;
    }
    std::fprintf(stderr, "threw without '%s': %s\n", what.c_str(), e.what());
  }
  return false;
}
}  // namespace

int main() {
  const model::MuseGlimmerDesc d = model::muse_glimmer();
  const std::vector<size_t> one_card = {kCard}, two_cards = {kCard, kCard};
  const size_t W = loader::muse_device_weight_bytes(d, model::MusePlacement::one(d), 0, true);
  const size_t kv131k = size_t(13) * 131072 * 1024, rings = size_t(39) * 4194304;

  // --- one card, int8 head, no drafter: 131072 (decision 6) ---------------------------------------------------
  {
    const rm::MusePlan p = rm::plan_muse(d, 1, 0, true, {}, one_card);
    std::printf("%s\n", p.describe().c_str());
    CHECK(p.fits && p.max_len == 131072 && p.devices.size() == 1);
    CHECK_EQ(p.devices[0].weights, size_t(17409096960ull));
    CHECK_EQ(p.devices[0].weights, W);
    CHECK_EQ(p.devices[0].kv, kv131k);                 // 1.745 GB
    CHECK_EQ(p.devices[0].rings, rings);               // 163.6 MB
    CHECK_EQ(p.devices[0].reserve, size_t(1500000000ull));
    CHECK_EQ(p.devices[0].total(), size_t(17409096960ull) + kv131k + rings + 1500000000ull);   // ~19.3 GB + 1.5
  }
  // --- with the int4 drafter (1.358 GB, spec 23 §8.2): still 131072 (Review Focus 4) ---------------------------
  {
    rm::MuseReserve r;
    r.drafter = 1358000000ull;
    r.decode_scratch = 1000000000ull;   // placeholders (23c / 23d fill them): asserted ADDED, not their values
    r.prefill_scratch = 500000000ull;
    const rm::MusePlan p = rm::plan_muse(d, 1, 0, true, r, one_card);
    std::printf("%s\n", p.describe().c_str());
    CHECK(p.fits && p.max_len == 131072);
    CHECK_EQ(p.devices[0].drafter, r.drafter);
    CHECK_EQ(p.devices[0].scratch, r.decode_scratch + r.prefill_scratch);
    CHECK_EQ(p.devices[0].total(), W + kv131k + rings + r.drafter + 1500000000ull + 1500000000ull);
  }
  // --- bf16 head and the 5.112 GB bf16 drafter: still 131072 ----------------------------------------------------
  {
    rm::MuseReserve r;
    r.drafter = 5112000000ull;
    const rm::MusePlan p = rm::plan_muse(d, 1, 0, false, r, one_card);
    std::printf("%s\n", p.describe().c_str());
    CHECK(p.fits && p.max_len == 131072);
    CHECK_EQ(p.devices[0].weights, size_t(17409096960ull) - 1345639680ull + 2689662976ull);
  }
  // --- two cards: the split by bytes = runtime::pp_balance over the same terms; a given split -------------------
  {
    const rm::MusePlan p = rm::plan_muse(d, 2, 0, true, {}, two_cards);
    std::printf("%s\n", p.describe().c_str());
    CHECK(p.fits && p.max_len == 131072 && p.devices.size() == 2);
    const runtime::PpBalance b = runtime::pp_balance(rm::pp_layer_bytes(d, 131072), loader::muse_embed_bytes(d),
                                                     loader::muse_final_norm_bytes(d) + loader::muse_lm_head_bytes(d, true));
    CHECK_EQ(p.placement.split, b.split);
    CHECK_EQ(p.devices[0].weights + p.devices[0].kv + p.devices[0].rings, b.dev0);
    CHECK_EQ(p.devices[1].weights + p.devices[1].kv + p.devices[1].rings, b.dev1);
    CHECK_EQ(p.devices[0].weights + p.devices[1].weights, W);
    CHECK_EQ(p.devices[0].kv + p.devices[1].kv, kv131k);
    for (uint32_t dev = 0; dev < 2; ++dev)
      CHECK_EQ(p.devices[dev].weights, loader::muse_device_weight_bytes(d, p.placement, dev, true));
    const std::vector<size_t> lb = rm::pp_layer_bytes(d, 131072);
    CHECK_EQ(lb[0], loader::muse_layer_bytes(d).total() + size_t(4194304));
    CHECK_EQ(lb[3], loader::muse_layer_bytes(d).total() + size_t(131072) * 1024);
    const rm::MusePlan g = rm::plan_muse(d, 2, 32768, true, {}, two_cards, 20);   // --pp 2 --pipeline-split 20
    std::printf("%s\n", g.describe().c_str());
    CHECK(g.fits && g.placement.split == 20 && g.devices[0].end == 20 && g.devices[1].first == 20);
    CHECK_EQ(g.devices[0].weights, loader::muse_device_weight_bytes(d, g.placement, 0, true));
    CHECK(throws_naming<std::invalid_argument>("split", [&] { (void)rm::plan_muse(d, 2, 4096, true, {}, two_cards, 52); }));
  }
  // --- refusals: above the model's context (decision 6), off the grid, three devices ------------------------------
  CHECK(throws_naming<std::invalid_argument>("decision 6", [&] { (void)rm::plan_muse(d, 1, 262144, true, {}, one_card); }));
  CHECK(throws_naming<std::invalid_argument>("256", [&] { (void)rm::plan_muse(d, 1, 4000, true, {}, one_card); }));
  CHECK(throws_naming<std::invalid_argument>("devices", [&] { (void)rm::plan_muse(d, 3, 0, true, {}, {kCard, kCard, kCard}); }));
  // --- a 16 GB card: the weights alone do not fit - refused naming the bytes ---------------------------------------
  {
    const rm::MusePlan p = rm::plan_muse(d, 1, 0, true, {}, {16000000000ull});
    CHECK(!p.fits);
    CHECK(p.why.find("17409096960") == std::string::npos);   // the device total is named, not just the weights
    CHECK(p.why.find("16000000000") != std::string::npos);
    CHECK(throws_naming<std::runtime_error>("does not fit", [&] { rm::require_fits(p); }));
  }
  // --- a 20 GB card: the largest fitting max_len on the 256 grid ----------------------------------------------------
  {
    const size_t cap = 20000000000ull;
    const rm::MusePlan p = rm::plan_muse(d, 1, 0, true, {}, {cap});
    const size_t room = cap - W - rings - 1500000000ull;
    const uint32_t want = uint32_t(room / 13312 / 256 * 256);
    std::printf("  a 20 GB card: max_len %u (room %zu B / 13,312 B a position)\n", p.max_len, room);
    CHECK(p.fits && p.max_len == want && want < 131072);
    CHECK(!rm::plan_muse(d, 1, want + 256, true, {}, {cap}).fits);   // an explicit length past it does not fit
    rm::require_fits(p);
  }
  // --- spec 23c: the decode list's launches and scratch ---------------------------------------------------------
  {
    using rm::MuseAttn;
    using runtime::PpHandoff;
    const model::MusePlacement one = model::MusePlacement::one(d), two = model::MusePlacement::two(d, 26);
    CHECK_EQ(rm::decode_launches(d, one, MuseAttn::Flash, PpHandoff::Copy), size_t(733));   // 1 + 52 x 14 + 4
    CHECK_EQ(rm::decode_launches(d, one, MuseAttn::Eager, PpHandoff::Copy), size_t(681));   // 1 + 52 x 13 + 4
    CHECK_EQ(rm::decode_launches(d, two, MuseAttn::Flash, PpHandoff::Copy), size_t(733));   // the cut adds nothing
    CHECK_EQ(rm::decode_launches(d, two, MuseAttn::Flash, PpHandoff::Peer), size_t(735));   // pp_send + pp_recv
    CHECK_EQ(rm::decode_launches(d, two, MuseAttn::Eager, PpHandoff::Peer), size_t(683));
    CHECK_EQ(rm::device_launches(d, two, 0, MuseAttn::Flash, PpHandoff::Copy), size_t(1 + 26 * 14));
    CHECK_EQ(rm::device_launches(d, two, 1, MuseAttn::Flash, PpHandoff::Copy), size_t(26 * 14 + 4));
    CHECK(throws_naming<std::invalid_argument>("M 0", [&] { (void)rm::device_launches(d, one, 0, MuseAttn::Flash, PpHandoff::Copy, 0); }));
    // the decode scratch at 131072 (spec 23 §2 estimated 1.0-1.5 GB of scratch with prefill; decode alone is small)
    const rm::MuseScratchSizes s = rm::scratch_sizes(d, 131072, 1, MuseAttn::Flash);
    CHECK_EQ(s.partials, size_t(8) * 39936 * 4);           // gate||up S8 x 39936, the widest
    CHECK_EQ(s.attn_part, size_t(32) * 64 * 130 * 4);
    CHECK_EQ(s.rope, size_t(131072) * 128 * 4);            // 67.1 MB: the sliding layers' cos / sin
    CHECK_EQ(s.attn_s, size_t(0));
    CHECK_EQ(rm::decode_scratch_bytes(d, 131072, 1, MuseAttn::Flash), size_t(70374528));
    CHECK_EQ(rm::decode_scratch_bytes(d, 131072, 1, MuseAttn::Eager), size_t(70374528) + size_t(32) * 131072 * 4);
    CHECK_EQ(rm::decode_reserve(d, 0, 1, MuseAttn::Flash).decode_scratch, size_t(70374528));   // auto plans at 131072
    CHECK_EQ(rm::decode_reserve(d, 131072, 2, MuseAttn::Flash).decode_scratch,
             size_t(70374528) + rm::link_bytes(d, 1));
    CHECK(rm::link_bytes(d, 1) > rm::link_bytes(d, 0));
    // the device KV / rings are the planner's, per device
    const rm::MusePlan p2 = rm::plan_at(d, two, 131072, true, {}, two_cards);
    for (uint32_t dev = 0; dev < 2; ++dev) {
      CHECK_EQ(rm::device_kv_bytes(d, two, dev, 131072), p2.devices[dev].kv);
      CHECK_EQ(rm::device_ring_bytes(d, two, dev), p2.devices[dev].rings);
    }
    // one card still holds 131072 with the decode scratch reserved (and the eager score row)
    const rm::MusePlan p = rm::plan_muse(d, 1, 0, true, rm::decode_reserve(d, 0, 1, MuseAttn::Eager), one_card);
    CHECK(p.fits && p.max_len == 131072);
    // B70_MUSE_ATTN
    ::unsetenv("B70_MUSE_ATTN");
    CHECK(rm::muse_attn() == MuseAttn::Flash);
    ::setenv("B70_MUSE_ATTN", "eager", 1);
    CHECK(rm::muse_attn() == MuseAttn::Eager);
    ::setenv("B70_MUSE_ATTN", "fast", 1);
    CHECK(throws_naming<std::runtime_error>("flash or eager", [&] { (void)rm::muse_attn(); }));
    ::unsetenv("B70_MUSE_ATTN");
    std::printf("  23c: 733 / 681 launches a token (735 / 683 with peer), decode scratch %zu B at 131072 (+ %zu B eager)\n",
                rm::decode_scratch_bytes(d, 131072, 1, MuseAttn::Flash), size_t(32) * 131072 * 4);
  }
  // --- spec 23d: the prefill chunk's scratch, its launches, and the plan with it ---------------------------------
  using runtime::muse::MuseAttn;
  {
    const model::MusePlacement one = model::MusePlacement::one(d), two = model::MusePlacement::two(d, 26);
    CHECK_EQ(rm::kPfC, 2048u);
    // the slab walk: q||gate||k||v 8 x 1024 + 512, o_proj / down 6 x 1024 + 512, gate||up 39 x 1024
    CHECK_EQ(rm::pf_slabs(8704), 9u);
    CHECK_EQ(rm::pf_slabs(6656), 7u);
    CHECK_EQ(rm::pf_slabs(39936), 39u);
    CHECK_EQ(rm::pf_slab_width(8704, 8192), 512u);
    CHECK_EQ(rm::pf_ld(8704), 8704u);
    // a layer: norm, 9 x 2, prep, attention, gate, 7 x 2, fold _Z, post-add, norm, 39 x 2, silu, 7 x 2, fold _Z,
    // post-add = 134 (the same under eager: one attention launch either way)
    CHECK_EQ(rm::prefill_layer_launches(d), size_t(134));
    CHECK_EQ(rm::prefill_chunk_launches(d, one), size_t(1 + 52 * 134));   // + mg_pf_embed on device 0
    CHECK_EQ(rm::prefill_chunk_launches(d, two), size_t(1 + 52 * 134));   // the cut adds nothing
    CHECK_EQ(rm::prefill_device_launches(d, two, 0), size_t(1 + 26 * 134));
    CHECK_EQ(rm::prefill_device_launches(d, two, 1), size_t(26 * 134));
    CHECK_EQ(rm::kPrefillHeadLaunches, size_t(4));   // mg_final_norm on the last row, lm_head, argmax x 2
    const rm::PrefillSizes ps = rm::prefill_sizes(d);
    const size_t row = size_t(2048) * 6656 * 2;
    CHECK_EQ(ps.ids, size_t(2048) * 4);
    CHECK(ps.resid == row && ps.x == row && ps.a == row);
    CHECK_EQ(ps.partials, size_t(2048) * 39936 * 4);   // gate||up's fp32 rows, the widest linear
    CHECK_EQ(ps.slab, size_t(19968) * 1024 * 2);       // down's K
    CHECK(ps.sumsq_a == size_t(26) * 2048 * 4 && ps.sumsq_r == ps.sumsq_a);
    CHECK_EQ(ps.attn_q, size_t(2048) * 4096 * 4);
    CHECK(ps.gate == size_t(2048) * 4096 * 2 && ps.attn_out == ps.gate);
    CHECK_EQ(ps.h, size_t(2048) * 19968 * 2);
    CHECK_EQ(ps.total(), ps.ids + 3 * row + ps.partials + ps.slab + 2 * ps.sumsq_a + ps.attn_q + 2 * ps.gate + ps.h);
    const runtime::PpLandingLayout ll = rm::pf_landing_layout(d);
    CHECK(ll.resid_bytes == row && ll.sumsq_bytes == size_t(26) * 2048 * 4);
    CHECK(rm::pf_link_bytes(d, 1) > rm::pf_link_bytes(d, 0));
    CHECK_EQ(rm::prefill_reserve(d, 1), ps.total());
    CHECK_EQ(rm::prefill_reserve(d, 2), ps.total() + rm::pf_link_bytes(d, 1));
    // one card still reaches 131072 with the int8 head, the decode AND prefill scratch and the int4 drafter
    rm::MuseReserve r = rm::run_reserve(d, 0, 1, MuseAttn::Flash, true);
    CHECK_EQ(r.prefill_scratch, ps.total());
    CHECK_EQ(r.decode_scratch, rm::decode_reserve(d, 0, 1, MuseAttn::Flash).decode_scratch);
    CHECK_EQ(rm::run_reserve(d, 0, 1, MuseAttn::Flash, false).prefill_scratch, size_t(0));
    r.drafter = 1358000000ull;
    const rm::MusePlan p = rm::plan_muse(d, 1, 0, true, r, one_card);
    CHECK(p.fits && p.max_len == 131072);
    std::printf("  23d: %zu launches a chunk (134 a layer) + %zu at the head; prefill scratch %zu B (%.3f GB) a device; "
                "one card at 131072 with it, the int8 head and the int4 drafter: %.3f GB of %.3f\n",
                rm::prefill_chunk_launches(d, one), rm::kPrefillHeadLaunches, ps.total(), ps.total() / 1e9,
                p.devices[0].total() / 1e9, kCard / 1e9);
  }
  std::printf("muse_plan_test OK\n");
  return 0;
}
