// Spec 13c / spec 31: the planner's sequences of one card on the host - runtime::plan_pool and runtime::pool_that_fits
// (--max-num-seqs N: N requests sharing a KV page pool). No device: the sizes are runtime/buffer_sizes.h's functions,
// the weights memory_plan_test's derived byte counts.
//
// Pinned:
//   1. SlotDims and MtpDims at one sequence add nothing to plan();
//   2. four sequences over 229376 positions (int8 KV, MTP; spec 13 §9.1's 131072 + 3 x 32768) reconcile byte for byte
//      with spec 13 §9.1's method (runtime::batch_memory) - plan_pool also counts the RoPE table at the pool's
//      positions (the engine loads one table at max_len; batch_memory's at the longest slot), the further sequences'
//      hh rows and the row lists' (MtpDims::hh_rows), SlotDims' control blocks and the page table - and its total to
//      §9.1's printed precision. Int8 prefill's raw widening scratch is at the cap in both (3j: one sequence prefills
//      at a time, so the engine sizes it at PrefillScratch::kv8_len; 3e had it at the pool's positions);
//   3. the pool (plan_pool, pool_that_fits): check_pool's comment;
//   4. plan 31 Task 12 / 14, Ornith's pool: check_ornith's comment.
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

void check_one_sequence() {
  const model::ModelDesc& q = model::qwen38();
  // SlotDims and MtpDims at one sequence: nothing more than today.
  CHECK_EQ(runtime::SlotDims::sizes(q, 1, true).total(), size_t{0});
  CHECK_EQ(runtime::MtpDims::hh_rows(1), runtime::DecodeScratchDims::kDeepM + 1);   // spec 32
  CHECK_EQ(runtime::MtpDims::sizes(16384, q, 0, KvCache::Bf16, 4, 1).total(),
           runtime::MtpDims::sizes(16384, q).total());
}

void check_decided_layout() {
  const model::ModelDesc& q = model::qwen38();
  const runtime::PoolShape s{4, 112, 131072};   // 229376 positions, a request up to 131072
  const runtime::MemoryPlan p = runtime::plan_pool(q, s, true, kQwenInt8Weights + kMtpWeights, {}, KvCache::Int8);
  CHECK_EQ(p.max_len, 229376u);
  CHECK(p.slot_lens == std::vector<uint32_t>(4, 131072));
  const runtime::BatchMemory b = runtime::batch_memory(q, runtime::main_and_short(4, 131072, 32768, true),
                                                       KvCache::Int8, kQwenInt8Weights, kMtpWeights);
  // The engine sizes RoPE at the summed positions; its raw-widening buffer at the longest slot, as batch_memory (3j).
  const size_t rope = model::Qwen35::rope_table_bytes(229376) - model::Qwen35::rope_table_bytes(131072);
  // spec 32: one sequence's hh rows (65) now outnumber four slots' (60)
  const int64_t hh = (int64_t(runtime::MtpDims::hh_rows(4)) - runtime::MtpDims::hh_rows(1)) * q.hidden * 2;
  const size_t ctl = runtime::SlotDims::sizes(q, 4, true).controls;
  CHECK_EQ(ctl, size_t{10} * 768 + 3 * 1024);   // Control 768 B (spec 32), a row table 32 rows (plan 40a Task 3)
  // Plan 40a Task 3: the decode scratch at the 32-row lists' rows, not the 16 one sequence's int8 plan has
  const size_t wide = runtime::DecodeScratchDims::sizes(229376, q, 32).total() - runtime::DecodeScratchDims::sizes(229376, q).total();
  const size_t pages = size_t(runtime::pool_table_pages(s)) * 4;   // the page table, 4 x 64 entries
  CHECK_EQ(runtime::SlotDims::sizes(q, 4, true, 256).pages, pages);
  // spec 32: one sequence carries the deferred GDN rows (EngineSizes::gdn_pend); plan 31 Task 13: with MTP several
  // carry their own each
  const size_t deep = runtime::EngineDims::sizes(q, KvCache::Int8, 0, 1, true, false).gdn_pend;
  const size_t deep4 = runtime::EngineDims::sizes(q, KvCache::Int8, 0, 4, true, false).gdn_pend;
  CHECK_EQ(deep4, 4 * deep);
  CHECK_EQ(p.total(), size_t(int64_t(b.total() + rope + ctl + pages - deep + deep4 + wide) + hh));
  // 3e sized the widening scratch at the pool's positions: 384 MiB (98304 positions x 4 KiB) more than the cap needs.
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
             29.119 + double(size_t{131072} * 4 * 256 * 2 * 2 + runtime::route_slab_bytes(q, 0, q.layers) + deep4 + wide) / kGB,
             0.01);   // + the four sequences' deferred GDN rows (plan 31 Task 13)
  std::printf("plan_pool 4 up to 131072 in 229376 positions, int8 KV, MTP: %.3f GB (batch_memory %.3f + RoPE %.3f + hh %.6f + "
              "controls %zu B + page table %zu B), spare %.3f GB\n",
              p.total() / kGB, b.total() / kGB, rope / kGB, hh / kGB, ctl, pages,
              (double(kDevice - kReserve) - p.total()) / kGB);
  // Without MTP: one GDN state a further sequence (SlotDims' states), no head KV, no hh rows.
  const runtime::MemoryPlan n = runtime::plan_pool(q, s, false, kQwenInt8Weights, {}, KvCache::Int8);
  CHECK_EQ(n.slot_state, size_t{3} * (size_t{150994944} + 31457280) + ctl + pages);
  // ... less the widening scratch's further 98304 positions (3j: at the cap, not the pool's positions).
  CHECK_EQ(n.total() - n.slot_state,
           runtime::plan(q, 229376, false, kQwenInt8Weights, {}, {}, KvCache::Int8).total() - size_t{402653184});
  // bf16 KV widens nothing: plan() at the sum, unchanged by 3j.
  const runtime::MemoryPlan nb = runtime::plan_pool(q, s, false, kQwenInt8Weights, {}, KvCache::Bf16);
  CHECK_EQ(nb.total() - nb.slot_state, runtime::plan(q, 229376, false, kQwenInt8Weights, {}, {}, KvCache::Bf16).total());
}

// Spec 31 (plan 31 Task 4): the KV page pool. Its KV is plan()'s at the pool's positions; the pool that fits is the
// most pages that fit; the sizes at 2 / 3 / 4 requests, bf16 and int8, are printed (the startup lines' numbers).
void check_pool() {
  const model::ModelDesc& q = model::qwen38();
  const size_t w = kQwenInt8Weights + kMtpWeights;
  constexpr uint32_t kTrained = 262144, P = runtime::kKvPage;
  for (KvCache kv : {KvCache::Bf16, KvCache::Int8}) {
    const runtime::PoolShape s{4, 112, 131072};
    CHECK_EQ(runtime::pool_table_pages(s), 4u * 64);
    const runtime::MemoryPlan pool = runtime::plan_pool(q, s, true, w, {}, kv);
    CHECK_EQ(pool.max_len, 229376u);
    CHECK_EQ(pool.kv, runtime::plan(q, 229376, true, w, {}, {}, kv).kv);
    CHECK_EQ(pool.pool_pages, 112u);
    CHECK(pool.slot_lens == (std::vector<uint32_t>(4, 131072)));
  }
  std::printf("| --max-num-seqs | KV | pool (MTP, --max-len auto) | one request up to | total GB | spare GB |\n");
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
      const double t = p.total() / kGB;
      std::printf("| %u | %s | %u pages of %u = %llu positions | %u | %.3f | %.3f |\n", seqs, runtime::kv_cache_name(kv),
                  s.pages, P, static_cast<unsigned long long>(s.positions()), s.cap, t, double(kDevice - kReserve) / kGB - t);
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
// Plan 31 Task 12 / 14: Ornith's pool - bf16 KV, MTP (its MoE head). A further sequence's state: its four GDN states
// and its conv ring; the pools at 2..4 requests are printed (the startup lines' numbers).
void check_ornith() {
  const model::ModelDesc& o = model::ornith();
  const size_t w = kOrnithInt8Weights + loader::mtp_head_bytes(o);
  constexpr uint32_t kTrained = 262144, P = runtime::kKvPage;
  // A further sequence: its four GDN states (30 x 32 x 128 x 128 fp32 each) and its conv ring (30 x 32 x 8192 bf16).
  const runtime::PoolShape four{4, 256, kTrained};
  const runtime::MemoryPlan p = runtime::plan_pool(o, four, true, w, {}, KvCache::Bf16);
  const runtime::SlotSizes ss = runtime::SlotDims::sizes(o, 4, true, runtime::pool_table_pages(four));
  CHECK_EQ(ss.rings, size_t{3} * 15728640);
  CHECK_EQ(runtime::MtpDims::sizes(p.max_len, o, 0, KvCache::Bf16, runtime::MtpDims::kSlots, 4).gdn_spec -
               runtime::MtpDims::sizes(p.max_len, o, 0, KvCache::Bf16).gdn_spec,
           size_t{3} * 4 * 62914560);
  std::printf("Ornith 4 up to 262144 in 524288 positions, MTP: %s\n", runtime::describe(p, kDevice, kReserve).c_str());
  std::printf("| Ornith --max-num-seqs | pool (MTP, bf16, --max-len auto) | one request up to | total GB |\n");
  for (uint32_t seqs = 2; seqs <= runtime::SlotDims::kMaxSeqs; ++seqs) {
    const runtime::PoolShape ps = runtime::pool_that_fits(o, seqs, true, w, kDevice, kReserve, kTrained, {}, KvCache::Bf16);
    CHECK(ps.pages != 0 && ps.cap == std::min<uint32_t>(ps.pages * P, kTrained));
    const runtime::MemoryPlan pp = runtime::plan_pool(o, ps, true, w, {}, KvCache::Bf16);
    CHECK(pp.total() + kReserve <= kDevice);
    std::printf("| %u | %u pages of %u = %llu positions | %u | %.3f |\n", seqs, ps.pages, P,
                static_cast<unsigned long long>(ps.positions()), ps.cap, pp.total() / kGB);
  }
}
}  // namespace

int main() {
  check_one_sequence();
  check_decided_layout();
  check_pool();
  check_ornith();
  std::printf("slot_plan_test OK\n");
  return 0;
}
