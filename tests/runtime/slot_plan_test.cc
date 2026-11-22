// Spec 13c (plan 13c Task 1): the planner's sequence slots on the host - runtime::plan_slots, runtime::slot_layout and
// runtime::slot0_len_that_fits (spec 13 §11.2's shape; --max-num-seqs N,  slot 0 the largest that
// fits beside N - 1 slots of 32768). No device: the sizes are
// runtime/buffer_sizes.h's functions, the weights memory_plan_test's derived byte counts.
//
// Pinned:
//   1. one slot is plan() itself, field for field (--max-len auto and an explicit length that leaves no room: today);
//   2. the layout §9.1 decided (1 x 131072 + 3 x 32768, int8 KV, MTP) reconciles byte for byte with spec 13 §9.1's
//      method (runtime::batch_memory) - plan_slots also counts the RoPE table at the summed
//      positions (the engine loads one table at max_len = the sum; batch_memory's at the longest slot), the further
//      slots' hh rows and the row lists' (MtpDims::hh_rows) and SlotDims' control blocks - and its total to §9.1's
//      printed precision. Int8 prefill's raw widening scratch is at the longest slot in both (3j: one sequence
//      prefills at a time, so the engine sizes it at PrefillScratch::kv8_len; 3e had it at the summed positions,
//      384 MiB more on this layout - 896 MiB vs 512 MiB);
//   3. --max-num-seqs' rule: slot 0 then N - 1 slots of 32768; under --max-len auto slot 0 is the largest multiple of
//      256 that fits with the reserve (one slot: max_len_that_fits); the layouts of the bf16 and int8 KV forms at
//      N = 1..4 (printed: they are the startup lines a user reads).
//   4. plan 19v3b Task 0, the slots' KV in pages of runtime::kKvPage: several slots allocate each slot's KV as whole
//      pages (runtime::slots_kv_len; plan_slots at it, the page table's u32 a page in SlotDims), one slot exactly its
//      length; the auto layouts at 1 / 2 / 4 slots, bf16 and int8 (printed with their pages), lose less than one page
//      a slot to the rounding; BatchLayout's KV bases are page boundaries.
//   5. spec 31, the KV page pool (plan_pool, pool_that_fits): check_pool's comment.
//   6. plan 31 Task 12, Ornith's fixed slots: check_ornith's comment.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "check.h"
#include "loader/moe_layout.h"
#include "model/model_desc.h"
#include "model/qwen35.h"
#include "runtime/batch_rows.h"
#include "runtime/buffer_sizes.h"
#include "runtime/memory_plan.h"

namespace {
using runtime::KvCache;

// memory_plan_test.cc's derived weights (the int8 head, the MTP head apart, no RoPE table).
constexpr size_t kLmBf16 = 2542796800ull, kLmInt8 = 1271398400ull + 993280ull;
constexpr size_t kQwenInt8Weights = 18086971392ull - 4194304ull - kLmBf16 + kLmInt8;   // 16,812,371,968
constexpr size_t kMtpWeights = 424673280ull * 2 + 5ull * 5120 * 4 + 2048;               // 849,451,008
// memory_plan_test.cc's Ornith with the int8 head (the int4 checkpoint's a||b), the MTP head apart.
constexpr size_t kOrnithInt8Weights = 30ull * 18637312 + 10ull * 14501888 + 40ull * 430604288 +
                                      1017118720ull + 509552640ull + 8192;   // 19,454,989,312
constexpr size_t kDevice = 32530000000ull;
constexpr size_t kReserve = size_t(runtime::kDefaultReserveGb * 1e9);
constexpr double kGB = 1e9;

void check_one_slot() {
  const model::ModelDesc& q = model::qwen38();
  for (KvCache kv : {KvCache::Bf16, KvCache::Int8})
    for (bool mtp : {false, true})
      for (uint32_t L : {4096u, 131072u}) {
        const size_t w = kQwenInt8Weights + (mtp ? kMtpWeights : 0);
        const runtime::MemoryPlan a = runtime::plan(q, L, mtp, w, {}, {}, kv);
        const runtime::MemoryPlan b = runtime::plan_slots(q, {L}, mtp, w, {}, {}, kv);
        CHECK_EQ(a.total(), b.total());
        CHECK_EQ(a.kv, b.kv);
        CHECK_EQ(a.decode_state, b.decode_state);
        CHECK_EQ(a.mtp_buffers, b.mtp_buffers);
        CHECK(b.slot_lens.empty());
        CHECK_EQ(b.slot_state, size_t{0});
      }
  // SlotDims and MtpDims at one sequence: nothing more than today.
  CHECK_EQ(runtime::SlotDims::sizes(q, 1, true).total(), size_t{0});
  CHECK_EQ(runtime::MtpDims::hh_rows(1), runtime::DecodeScratchDims::kDeepM + 1);   // spec 32
  CHECK_EQ(runtime::MtpDims::sizes(16384, q, 0, KvCache::Bf16, 4, 1).total(),
           runtime::MtpDims::sizes(16384, q).total());
}

void check_decided_layout() {
  const model::ModelDesc& q = model::qwen38();
  const std::vector<uint32_t> lens{131072, 32768, 32768, 32768};
  const runtime::MemoryPlan p =
      runtime::plan_slots(q, lens, true, kQwenInt8Weights + kMtpWeights, {}, {}, KvCache::Int8);
  CHECK_EQ(p.max_len, 229376u);
  CHECK(p.slot_lens == lens);
  const runtime::BatchMemory b = runtime::batch_memory(q, runtime::main_and_short(4, 131072, 32768, true),
                                                       KvCache::Int8, kQwenInt8Weights, kMtpWeights);
  // The engine sizes RoPE at the summed positions; its raw-widening buffer at the longest slot, as batch_memory (3j).
  const size_t rope = model::Qwen35::rope_table_bytes(229376) - model::Qwen35::rope_table_bytes(131072);
  // spec 32: one sequence's hh rows (65) now outnumber four slots' (60)
  const int64_t hh = (int64_t(runtime::MtpDims::hh_rows(4)) - runtime::MtpDims::hh_rows(1)) * q.hidden * 2;
  const size_t ctl = runtime::SlotDims::sizes(q, 4, true).controls;
  CHECK_EQ(ctl, size_t{10} * 768 + 3 * 256);   // Control 768 B (spec 32)
  const size_t pages = size_t(229376 / runtime::kKvPage) * 4;   // plan 19v3b Task 0: the page table, 112 pages
  CHECK_EQ(runtime::SlotDims::sizes(q, 4, true, 112).pages, pages);
  // spec 32: one sequence carries the deferred GDN rows (EngineSizes::gdn_pend); plan 31 Task 13: with MTP several
  // carry their own each
  const size_t deep = runtime::EngineDims::sizes(q, KvCache::Int8, 0, 1, true, false).gdn_pend;
  const size_t deep4 = runtime::EngineDims::sizes(q, KvCache::Int8, 0, 4, true, false).gdn_pend;
  CHECK_EQ(deep4, 4 * deep);
  CHECK_EQ(p.total(), size_t(int64_t(b.total() + rope + ctl + pages - deep + deep4) + hh));
  // 3e sized the widening scratch at the sum: 384 MiB (98304 positions x 4 KiB) more than the longest slot needs.
  CHECK_EQ(runtime::plan(q, 229376, true, kQwenInt8Weights + kMtpWeights, {}, {}, KvCache::Int8).prefill_lazy -
               p.prefill_lazy,
           size_t{402653184});
  CHECK_EQ(p.kv, b.kv - (runtime::MtpDims::sizes(229376, q, 0, KvCache::Int8).kv_k * 2));   // the head's KV is MTP's
  // The further slots' state: 3 x (4 GDN states + a ring) + the hh rows + the controls.
  // (spec 39: the deferred rows are plan()'s `deep`, not the slots' state)
  CHECK_EQ(p.slot_state, size_t(int64_t(size_t{3} * (4 * size_t{150994944} + 31457280) + ctl + pages) + hh));
  CHECK_EQ(p.deep, deep4);
  // §9.1: "total 29.10 GB, 1.93 GB spare" (l0-int8); on the l0 default 0.087 GB less (batch_rows_test's 29.01).
  // + the a||b's dense GEMM slabs (0.126 GB).
  CHECK_NEAR(p.total() / kGB,
             29.119 + double(size_t{131072} * 4 * 256 * 2 * 2 + runtime::route_slab_bytes(q, 0, q.layers) + deep4) / kGB,
             0.01);   // + the four sequences' deferred GDN rows (plan 31 Task 13)
  std::printf("plan_slots 131072 + 3 x 32768, int8 KV, MTP: %.3f GB (batch_memory %.3f + RoPE %.3f + hh %.6f + "
              "controls %zu B + page table %zu B), spare %.3f GB\n",
              p.total() / kGB, b.total() / kGB, rope / kGB, hh / kGB, ctl, pages,
              (double(kDevice - kReserve) - p.total()) / kGB);
  // Without MTP: one GDN state a further slot (SlotDims' states), no head KV, no hh rows.
  const runtime::MemoryPlan n = runtime::plan_slots(q, lens, false, kQwenInt8Weights, {}, {}, KvCache::Int8);
  CHECK_EQ(n.slot_state, size_t{3} * (size_t{150994944} + 31457280) + ctl + pages);
  // ... less the widening scratch's further 98304 positions (3j: at the longest slot, not the sum).
  CHECK_EQ(n.total() - n.slot_state,
           runtime::plan(q, 229376, false, kQwenInt8Weights, {}, {}, KvCache::Int8).total() - size_t{402653184});
  // bf16 KV widens nothing: plan() at the sum, unchanged by 3j.
  const runtime::MemoryPlan nb = runtime::plan_slots(q, lens, false, kQwenInt8Weights, {}, {}, KvCache::Bf16);
  CHECK_EQ(nb.total() - nb.slot_state, runtime::plan(q, 229376, false, kQwenInt8Weights, {}, {}, KvCache::Bf16).total());
}

void check_layout_rule() {
  const model::ModelDesc& q = model::qwen38();
  const size_t w = kQwenInt8Weights + kMtpWeights;
  constexpr uint32_t kTrained = 262144;
  const std::vector<uint32_t> d = runtime::slot_layout(131072, 4);
  CHECK(d == (std::vector<uint32_t>{131072, 32768, 32768, 32768}));
  CHECK(runtime::slot_layout(65536, 1) == (std::vector<uint32_t>{65536}));
  CHECK(runtime::slot_layout_name(d) == "131072 + 3 x 32768");
  CHECK(runtime::slot_layout_name({65536, 32768}) == "65536 + 32768");
  auto fit = [&](uint32_t seqs, KvCache kv) {
    return runtime::slot0_len_that_fits(q, seqs, true, w, kDevice, kReserve, kTrained, {}, kv);
  };
  // The decided layout fits at int8 KV; at bf16 KV slot 0 is shorter (§9: "at bf16 KV the slots are fewer").
  CHECK(fit(4, KvCache::Int8) >= 131072u);
  CHECK(fit(4, KvCache::Bf16) < 131072u);
  // The rule at every N: slot 0 a multiple of 256 whose layout fits with the reserve, 256 more does not (or the
  // trained context); one slot is max_len_that_fits.
  for (KvCache kv : {KvCache::Bf16, KvCache::Int8}) {
    CHECK_EQ(fit(1, kv), runtime::max_len_that_fits(q, true, w, kDevice, kReserve, kTrained, {}, {}, kv));
    for (uint32_t seqs = 1; seqs <= runtime::SlotDims::kMaxSeqs; ++seqs) {
      const uint32_t l0 = fit(seqs, kv);
      CHECK(l0 != 0 && l0 % runtime::kMaxLenQuantum == 0 && l0 <= kTrained);
      CHECK(runtime::plan_slots(q, runtime::slot_layout(l0, seqs), true, w, {}, {}, kv).total() + kReserve <= kDevice);
      if (l0 < kTrained)
        CHECK(runtime::plan_slots(q, runtime::slot_layout(l0 + runtime::kMaxLenQuantum, seqs), true, w, {}, {}, kv)
                      .total() +
                  kReserve >
              kDevice);
    }
  }
  std::printf("| --max-num-seqs | KV | slots (MTP, --max-len auto) | total GB | spare GB |\n");
  for (KvCache kv : {KvCache::Bf16, KvCache::Int8})
    for (uint32_t seqs = 1; seqs <= runtime::SlotDims::kMaxSeqs; ++seqs) {
      const std::vector<uint32_t> l = runtime::slot_layout(fit(seqs, kv), seqs);
      const double t = runtime::plan_slots(q, l, true, w, {}, {}, kv).total() / kGB;
      std::printf("| %u | %s | %s | %.3f | %.3f |\n", seqs, runtime::kv_cache_name(kv),
                  runtime::slot_layout_name(l).c_str(), t, double(kDevice - kReserve) / kGB - t);
    }
  // Refusals: no slot, five slots, a draft vocabulary beside slots.
  bool threw = false;
  try {
    runtime::plan_slots(q, {}, true, w);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  threw = false;
  try {
    runtime::plan_slots(q, {32768, 32768, 32768, 32768, 32768}, true, w);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  threw = false;
  try {
    runtime::plan_slots(q, {32768, 32768}, true, w, {}, runtime::DraftVocabPlan{32768, true});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}
void check_pages() {
  const model::ModelDesc& q = model::qwen38();
  const size_t w = kQwenInt8Weights + kMtpWeights;
  constexpr uint32_t kTrained = 262144, P = runtime::kKvPage;
  // One slot is its length (no page rounding, no table); several are whole pages each.
  CHECK_EQ(runtime::slots_kv_len({130000}), 130000u);
  CHECK_EQ(runtime::slots_kv_len({130000, 32768}), 131072u + 32768u);
  CHECK_EQ(runtime::slots_kv_len({131072, 32768, 32768, 32768}), 229376u);
  CHECK_EQ(runtime::plan_slots(q, {130000, 32768}, true, w).max_len, 131072u + 32768u);
  CHECK_EQ(runtime::plan_slots(q, {130000, 32768}, true, w).total(),
           runtime::plan_slots(q, {131072, 32768}, true, w).total());
  CHECK_EQ(runtime::SlotDims::sizes(q, 2, true, 80).pages, size_t{320});
  const runtime::BatchLayout lay{{{130000, true}, {32768, true}, {32768, true}}};
  CHECK_EQ(lay.kv_base(1), 131072u);
  CHECK_EQ(lay.kv_base(2), 131072u + 32768u);
  CHECK_EQ(lay.total(), runtime::slots_kv_len({130000, 32768, 32768}));
  std::printf("| --max-num-seqs | KV | slots (MTP, --max-len auto) | KV pages | positions lost to the pages |\n");
  for (KvCache kv : {KvCache::Bf16, KvCache::Int8})
    for (uint32_t seqs : {1u, 2u, 4u}) {
      const uint32_t l0 = runtime::slot0_len_that_fits(q, seqs, true, w, kDevice, kReserve, kTrained, {}, kv);
      const std::vector<uint32_t> l = runtime::slot_layout(l0, seqs);
      const uint32_t total = runtime::slots_kv_len(l);
      uint32_t lost = 0;
      for (uint32_t x : l) {
        const uint32_t r = seqs > 1 ? runtime::kv_pages(x) * P : x;
        CHECK(r - x < P);   // at most one page a slot, less one position
        lost += r - x;
      }
      CHECK_EQ(runtime::plan_slots(q, l, true, w, {}, {}, kv).max_len, total);
      std::printf("| %u | %s | %s | %s | %u |\n", seqs, runtime::kv_cache_name(kv), runtime::slot_layout_name(l).c_str(),
                  seqs > 1 ? std::to_string(total / P).c_str() : "-", lost);
    }
}
// Spec 31 (plan 31 Task 4): the KV page pool. The pool at the fixed slots' pages differs from plan_slots only by the
// page table (a cap-long table a sequence); the pool that fits uses the
// card as fully as the fixed slots, less one page a request and the cap-sized scratch; the sizes at 2 / 3 / 4
// requests, bf16 and int8, are printed (the startup lines' numbers).
void check_pool() {
  const model::ModelDesc& q = model::qwen38();
  const size_t w = kQwenInt8Weights + kMtpWeights;
  constexpr uint32_t kTrained = 262144, P = runtime::kKvPage;
  const std::vector<uint32_t> lens{131072, 32768, 32768, 32768};
  for (KvCache kv : {KvCache::Bf16, KvCache::Int8}) {
    const runtime::PoolShape s{4, 112, 131072};
    CHECK_EQ(runtime::pool_table_pages(s), 4u * 64);
    const runtime::MemoryPlan pool = runtime::plan_pool(q, s, true, w, {}, kv);
    const runtime::MemoryPlan fixed = runtime::plan_slots(q, lens, true, w, {}, {}, kv);
    CHECK_EQ(pool.max_len, 229376u);
    CHECK_EQ(pool.kv, fixed.kv);
    CHECK_EQ(pool.pool_pages, 112u);
    CHECK(pool.slot_lens == (std::vector<uint32_t>(4, 131072)));
    CHECK_EQ(pool.total(), fixed.total() + size_t(4 * 64 - 112) * 4);   // the longer page table only
  }
  // A page's bytes: every full-attention layer's and the MTP head's.
  const auto kv_bytes = [&](uint64_t positions, KvCache kv) {
    const runtime::PersistentSizes ps = runtime::PersistentDims::sizes(uint32_t(positions), q, kv);
    return ps.kv_k + ps.kv_v + 2 * runtime::MtpDims::sizes(uint32_t(positions), q, 0, kv).kv_k;
  };
  std::printf("| --max-num-seqs | KV | pool (MTP, --max-len auto) | one request up to | fixed slots | total GB | "
              "spare GB |\n");
  for (KvCache kv : {KvCache::Bf16, KvCache::Int8})
    for (uint32_t seqs = 2; seqs <= runtime::SlotDims::kMaxSeqs; ++seqs) {
      const runtime::PoolShape s = runtime::pool_that_fits(q, seqs, true, w, kDevice, kReserve, kTrained, {}, kv);
      CHECK(s.seqs == seqs && s.pages != 0);
      CHECK_EQ(s.cap, std::min<uint32_t>(s.pages * P, kTrained));
      const runtime::MemoryPlan p = runtime::plan_pool(q, s, true, w, {}, kv);
      CHECK(p.total() + kReserve <= kDevice);
      if (s.pages < seqs * (kTrained / P)) {   // one page more does not fit
        const runtime::PoolShape more{seqs, s.pages + 1, std::min<uint32_t>((s.pages + 1) * P, kTrained)};
        CHECK(runtime::plan_pool(q, more, true, w, {}, kv).total() + kReserve > kDevice);
      }
      // As full as the fixed slots: the pool's positions are not below their sum less one page a request and the
      // positions the pool's further cap-sized terms (the widening rows, the page table) take.
      const std::vector<uint32_t> l =
          runtime::slot_layout(runtime::slot0_len_that_fits(q, seqs, true, w, kDevice, kReserve, kTrained, {}, kv), seqs);
      const runtime::MemoryPlan f = runtime::plan_slots(q, l, true, w, {}, {}, kv);
      const size_t page = kv_bytes(P, kv);
      const int64_t more_scratch = int64_t(p.total() - kv_bytes(s.positions(), kv)) - int64_t(f.total() - kv_bytes(f.max_len, kv));
      const uint64_t scratch_pages = more_scratch > 0 ? (uint64_t(more_scratch) + page - 1) / page : 0;
      CHECK(s.positions() + uint64_t(seqs) * P + scratch_pages * P >= runtime::slots_kv_len(l));
      const double t = p.total() / kGB;
      std::printf("| %u | %s | %u pages of %u = %llu positions | %u | %s = %u | %.3f | %.3f |\n", seqs,
                  runtime::kv_cache_name(kv), s.pages, P, static_cast<unsigned long long>(s.positions()), s.cap,
                  runtime::slot_layout_name(l).c_str(), runtime::slots_kv_len(l), t, double(kDevice - kReserve) / kGB - t);
    }
  // One request plans byte for byte what plan() plans (the MTP deep term at one sequence, no draft vocabulary): the
  // pool never reaches --max-num-seqs 1.
  for (KvCache kv : {KvCache::Bf16, KvCache::Int8})
    for (const uint32_t L : {65536u, 262144u}) {
      const runtime::MemoryPlan one = runtime::plan_slots(q, {L}, true, w, {}, {}, kv);
      const runtime::MemoryPlan ref = runtime::plan(q, L, true, w, {}, {}, kv);
      CHECK_EQ(one.total(), ref.total());
      CHECK_EQ(one.mtp_buffers, ref.mtp_buffers);
      CHECK(one.slot_lens.empty() && one.pool_pages == 0);
    }
  // The startup breakdown names the pool.
  const std::string line = runtime::describe(runtime::plan_pool(q, {4, 70, 143360}, true, w, {}, KvCache::Bf16), kDevice,
                                             kReserve);
  CHECK(line.find("4 requests sharing a KV pool of 70 pages of 2048, one up to 143360") != std::string::npos);
  // Refusals: one sequence (plan() itself), five, a cap off the grid or past the pool; a limit below one quantum.
  const auto throws = [](auto f) {
    try {
      f();
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  CHECK(throws([&] { runtime::plan_pool(q, {1, 64, 131072}, true, w); }));
  CHECK(throws([&] { runtime::plan_pool(q, {5, 64, 131072}, true, w); }));
  CHECK(throws([&] { runtime::plan_pool(q, {4, 64, 131000}, true, w); }));
  CHECK(throws([&] { runtime::plan_pool(q, {4, 64, 131072 + 256}, true, w); }));
  CHECK(throws([&] { runtime::pool_that_fits(q, 4, true, w, kDevice, kReserve, 255); }));
  // An explicit --max-len caps a request: the pool stops at seqs requests of it.
  const runtime::PoolShape capped = runtime::pool_that_fits(q, 2, true, w, kDevice, kReserve, 32768, {}, KvCache::Int8);
  CHECK_EQ(capped.cap, 32768u);
  CHECK_EQ(capped.pages, 2u * 16);
}
// Plan 31 Task 12: Ornith's fixed slots - bf16 KV (Ornith has no int8 KV cache), MTP (its MoE head). One slot plans
// byte for byte what plan() plans; slot 0 reaches the trained 262144 beside three slots of 32768 (spec 31 §4a,
// derived there); the layouts at 1..4 slots are printed (the startup lines' numbers).
void check_ornith() {
  const model::ModelDesc& o = model::ornith();
  const size_t w = kOrnithInt8Weights + loader::mtp_head_bytes(o);
  constexpr uint32_t kTrained = 262144;
  for (const uint32_t L : {65536u, kTrained}) {
    const runtime::MemoryPlan one = runtime::plan_slots(o, {L}, true, w, {}, {}, KvCache::Bf16);
    const runtime::MemoryPlan ref = runtime::plan(o, L, true, w, {}, {}, KvCache::Bf16);
    CHECK_EQ(one.total(), ref.total());
    CHECK_EQ(one.decode_state, ref.decode_state);
    CHECK_EQ(one.mtp_buffers, ref.mtp_buffers);
    CHECK(one.slot_lens.empty() && one.slot_state == 0);
  }
  const auto fit = [&](uint32_t seqs) {
    return runtime::slot0_len_that_fits(o, seqs, true, w, kDevice, kReserve, kTrained, {}, KvCache::Bf16);
  };
  CHECK_EQ(fit(4), kTrained);
  // A further slot: its four GDN states (30 x 32 x 128 x 128 fp32 each) and its conv ring (30 x 32 x 8192 bf16).
  const std::vector<uint32_t> four = runtime::slot_layout(kTrained, 4);
  const runtime::MemoryPlan p = runtime::plan_slots(o, four, true, w, {}, {}, KvCache::Bf16);
  const runtime::SlotSizes ss = runtime::SlotDims::sizes(o, 4, true, p.max_len / runtime::kKvPage);
  CHECK_EQ(ss.rings, size_t{3} * 15728640);
  CHECK_EQ(runtime::MtpDims::sizes(p.max_len, o, 0, KvCache::Bf16, runtime::MtpDims::kSlots, 4).gdn_spec -
               runtime::MtpDims::sizes(p.max_len, o, 0, KvCache::Bf16).gdn_spec,
           size_t{3} * 4 * 62914560);
  std::printf("| Ornith --max-num-seqs | KV | slots (MTP, --max-len auto) | total GB | spare GB |\n");
  for (uint32_t seqs = 1; seqs <= runtime::SlotDims::kMaxSeqs; ++seqs) {
    const std::vector<uint32_t> l = runtime::slot_layout(fit(seqs), seqs);
    const double t = runtime::plan_slots(o, l, true, w, {}, {}, KvCache::Bf16).total() / kGB;
    std::printf("| %u | bf16 | %s | %.3f | %.3f |\n", seqs, runtime::slot_layout_name(l).c_str(), t,
                double(kDevice - kReserve) / kGB - t);
  }
  std::printf("Ornith 262144 + 3 x 32768, MTP: %s\n", runtime::describe(p, kDevice, kReserve).c_str());
  // Plan 31 Task 14: Ornith's pool at 2..4 requests - as full as the fixed slots (check_pool's bar: not below their
  // positions less one page a request and the pool's further cap-sized terms).
  constexpr uint32_t P = runtime::kKvPage;
  const auto kv_bytes = [&](uint64_t positions) {
    const runtime::PersistentSizes ps = runtime::PersistentDims::sizes(uint32_t(positions), o, KvCache::Bf16);
    return ps.kv_k + ps.kv_v + 2 * runtime::MtpDims::sizes(uint32_t(positions), o, 0, KvCache::Bf16).kv_k;
  };
  std::printf("| Ornith --max-num-seqs | pool (MTP, bf16, --max-len auto) | one request up to | fixed slots | total GB |\n");
  for (uint32_t seqs = 2; seqs <= runtime::SlotDims::kMaxSeqs; ++seqs) {
    const runtime::PoolShape ps = runtime::pool_that_fits(o, seqs, true, w, kDevice, kReserve, kTrained, {}, KvCache::Bf16);
    CHECK(ps.pages != 0 && ps.cap == std::min<uint32_t>(ps.pages * P, kTrained));
    const runtime::MemoryPlan pp = runtime::plan_pool(o, ps, true, w, {}, KvCache::Bf16);
    CHECK(pp.total() + kReserve <= kDevice);
    const std::vector<uint32_t> l = runtime::slot_layout(fit(seqs), seqs);
    const runtime::MemoryPlan f = runtime::plan_slots(o, l, true, w, {}, {}, KvCache::Bf16);
    const int64_t more = int64_t(pp.total() - kv_bytes(ps.positions())) - int64_t(f.total() - kv_bytes(f.max_len));
    const uint64_t more_pages = more > 0 ? (uint64_t(more) + kv_bytes(P) - 1) / kv_bytes(P) : 0;
    CHECK(ps.positions() + uint64_t(seqs) * P + more_pages * P >= runtime::slots_kv_len(l));
    std::printf("| %u | %u pages of %u = %llu positions | %u | %s = %u | %.3f |\n", seqs, ps.pages, P,
                static_cast<unsigned long long>(ps.positions()), ps.cap, runtime::slot_layout_name(l).c_str(),
                runtime::slots_kv_len(l), pp.total() / kGB);
  }
}
}  // namespace

int main() {
  check_one_slot();
  check_decided_layout();
  check_layout_rule();
  check_pages();
  check_pool();
  check_ornith();
  std::printf("slot_plan_test OK\n");
  return 0;
}
