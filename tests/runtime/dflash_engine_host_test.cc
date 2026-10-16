// Spec 19c, host only: the arithmetic the DFlash engine loop rests on, without a device.
//
//   1. tap_sites (Task 1): Qwen3.8's drafter taps the outputs of layers 5 / 19 / 33 / 47 / 61 (vLLM's
//      target_layer_id i -> aux index i + 1), so the verify list copies them after the input fold of layers
//      6 / 20 / 34 / 48 / 62; a tap of the last layer goes to the final norm's fold (site = layers); an empty,
//      unordered, repeated or out-of-range list is refused.
//   2. TapVerifyDims (Task 1): 7 extra GDN slots of one gdn_state each (8 x 151 MB in all on Qwen3.8, slot 0 being
//      gdn_state) and the prefill staging [kC][taps][hidden] bf16.
//   3. The planner with a drafter (Task 2): a draft vocabulary without the MTP head is legal only with a drafter, and
//      it counts once (the compact head + ids); the verify slots and the prefill taps follow the drafter.
//   4. The adaptive-K cost row (Task 2, spec 19 decision 4): MtpCost::dflash - verify to M = 8, a draft row flat in K
//      - parses back from --spec-cost's syntax, and AdaptiveK over it picks K = 7 on agentic-tool-call acceptance
//      (0.95, 19a's A4), stays shallow on prose (0.5) and plain on noise (0.1); a draft vocabulary scales the head's
//      share of the draft row only.
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "check.h"
#include "model/model_desc.h"
#include "runtime/buffer_sizes.h"
#include "runtime/memory_plan.h"
#include "server/adaptive_k.h"

namespace {

void tap_sites_case() {
  const model::ModelDesc& d = model::qwen38();
  const std::vector<std::vector<uint32_t>> s = runtime::tap_sites(d, {5, 19, 33, 47, 61});
  CHECK(s.size() == size_t(d.layers) + 1);
  size_t n = 0;
  for (uint32_t site = 0; site <= d.layers; ++site) {
    n += s[site].size();
    const int want = site == 6 ? 0 : site == 20 ? 1 : site == 34 ? 2 : site == 48 ? 3 : site == 62 ? 4 : -1;
    if (want < 0) {
      CHECK(s[site].empty());
    } else {
      CHECK(s[site].size() == 1 && s[site][0] == uint32_t(want));
    }
  }
  CHECK(n == 5);
  // The last layer's output: the final norm's fold (site = layers); layer 0's output: layer 1's input fold.
  const auto last = runtime::tap_sites(d, {0, d.layers - 1});
  CHECK(last[1].size() == 1 && last[1][0] == 0);
  CHECK(last[d.layers].size() == 1 && last[d.layers][0] == 1);
  int refused = 0;
  for (const std::vector<uint32_t>& bad : std::vector<std::vector<uint32_t>>{
           {}, {19, 5}, {5, 5}, {5, d.layers}, {d.layers + 3}}) {
    try {
      (void)runtime::tap_sites(d, bad);
    } catch (const std::invalid_argument&) {
      ++refused;
    }
  }
  CHECK(refused == 5);
  std::printf("tap_sites: Qwen3.8's taps 5/19/33/47/61 at the input folds of 6/20/34/48/62, 5 bad lists refused\n");
}

void dims_case() {
  const model::ModelDesc& d = model::qwen38();
  CHECK(runtime::TapVerifyDims::kSlots == 8);
  // One slot: 48 GDN layers x 48 v-heads x 128 x 128 fp32 = 150,994,944 B (MtpDims' slot).
  CHECK(runtime::TapVerifyDims::gdn_spec_bytes(d) == size_t(7) * 150994944u);
  CHECK(runtime::TapVerifyDims::gdn_spec_bytes(d) ==
        runtime::MtpDims::sizes(16384, d, 0, runtime::KvCache::Bf16, 8).gdn_spec);
  CHECK(runtime::TapVerifyDims::prefill_taps_bytes(d, 5) == size_t(2048) * 5 * 5120 * 2);
  std::printf("TapVerifyDims: 7 slots %zu B, prefill taps %zu B\n", runtime::TapVerifyDims::gdn_spec_bytes(d),
              runtime::TapVerifyDims::prefill_taps_bytes(d, 5));
}

void plan_case() {
  const model::ModelDesc& d = model::qwen38();
  runtime::DrafterPlan dr;
  dr.weights = 2'053'464'064;   // 19b's int8 drafter
  dr.fixed = 53'000'000;
  dr.per_position = 512;
  dr.n_taps = 5;
  const runtime::DraftVocabPlan dv{32768, true};
  bool threw = false;
  try {
    (void)runtime::plan(d, 32768, false, 17'000'000'000ull, {}, dv);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);   // no head, no drafter: nothing drafts over V'
  const runtime::MemoryPlan with = runtime::plan(d, 32768, false, 17'000'000'000ull, {}, dv,
                                                 runtime::KvCache::Bf16, dr);
  const runtime::MemoryPlan without = runtime::plan(d, 32768, false, 17'000'000'000ull, {}, {},
                                                    runtime::KvCache::Bf16, dr);
  CHECK(with.draft_vocab > 0 && with.total() == without.total() + with.draft_vocab);
  CHECK(with.drafter_slots == runtime::TapVerifyDims::gdn_spec_bytes(d));
  const runtime::MemoryPlan decode_only = runtime::plan(d, 32768, false, 17'000'000'000ull,
                                                        runtime::PrefillPath{false}, {}, runtime::KvCache::Bf16, dr);
  CHECK(decode_only.drafter_taps == 0 && without.drafter_taps == runtime::TapVerifyDims::prefill_taps_bytes(d, 5));
  std::printf("plan with a drafter at 32768: %s\n", runtime::describe(with, 32'530'000'000ull, 1'500'000'000ull).c_str());
}

void cost_case() {
  using server::MtpCost;
  const MtpCost c = MtpCost::dflash(/*int8_head=*/true, /*int8_drafter=*/true);
  CHECK(c.verify.size() == 8 && c.draft.size() == 7 && c.max_k() == 7);
  for (size_t i = 1; i < c.verify.size(); ++i) CHECK(c.verify[i] > c.verify[i - 1]);
  for (double x : c.draft) CHECK(x == c.draft[0]);   // flat in K: one pass drafts the block
  CHECK(std::fabs(c.draft[0] - (MtpCost::kDFlashDrafterInt8 + MtpCost::kDFlashHeadInt8)) < 1e-12);
  const MtpCost b = MtpCost::dflash(false, false);
  CHECK(b.draft[0] > c.draft[0] && b.verify[7] < c.verify[7] + 0.2);
  // --spec-cost's syntax round-trips, a flat row written out in full
  const MtpCost p = MtpCost::parse("verify=1,1.2,1.5,1.8,2,2.2,2.4,2.6;draft=0.2,0.2,0.2,0.2,0.2,0.2,0.2", c);
  CHECK(p.max_k() == 7 && p.cost(7) == 2.6 + 0.2);
  // a draft vocabulary scales the head's share of the draft row only
  const MtpCost v = c.with_draft_vocab(MtpCost::dflash_head_share(true, true), 32768.0 / 248320.0);
  CHECK(v.verify == c.verify && v.draft[0] < c.draft[0] && v.draft[0] > MtpCost::kDFlashDrafterInt8);
  // the policy over it
  auto run = [&](double alpha) {
    server::AdaptiveKOptions o;
    o.max_k = 7;
    o.cost = c;
    server::AdaptiveK k(o);
    uint64_t x = 88172645463325252ull;
    for (int it = 0; it < 400; ++it) {
      const uint32_t want = k.next();
      uint32_t j = 0;
      while (j < want) {
        x ^= x << 13, x ^= x >> 7, x ^= x << 17;
        if (double(x % 1000000) / 1e6 >= alpha) break;
        ++j;
      }
      k.observe(want, j);
    }
    return k.current();
  };
  const uint32_t hi = run(0.95), mid = run(0.5), lo = run(0.1);
  std::printf("adaptive K over MtpCost::dflash (int8 head, int8 drafter): alpha 0.95 -> K %u, 0.5 -> K %u, "
              "0.1 -> K %u; draft row %.3f (32k V': %.3f)\n", hi, mid, lo, c.draft[0], v.draft[0]);
  CHECK(hi == 7);
  CHECK(mid >= 1 && mid <= 3);
  CHECK(lo == 0);
}

}  // namespace

int main() {
  tap_sites_case();
  dims_case();
  plan_case();
  cost_case();
  std::printf("dflash_engine_host_test OK\n");
  return 0;
}
