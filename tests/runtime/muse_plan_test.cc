// Spec 23b Task 5: Muse-Glimmer-30B's memory plan, one card and two (runtime/muse/muse_sizes.h). Host only. Every
// number is DERIVED from the descriptor (spec 23 §2: one card holds the model with the drafter; decisions 6, 7).
#include <cstdio>
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
  std::printf("muse_plan_test OK\n");
  return 0;
}
