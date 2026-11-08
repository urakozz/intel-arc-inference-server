// Spec 13c (plan 13c Task 1): the planner's sequence slots on the host - runtime::plan_slots and runtime::slot_layout
// (spec 13 §11.2: an asked --max-len's spare memory becomes 32768-position slots). No device: the sizes are
// runtime/buffer_sizes.h's functions, the weights memory_plan_test's derived byte counts.
//
// Pinned:
//   1. one slot is plan() itself, field for field (--max-len auto and an explicit length that leaves no room: today);
//   2. the layout §9.1 decided (1 x 131072 + 3 x 32768, int8 KV, MTP) reconciles byte for byte with spec 13 §9.1's
//      method (runtime::batch_memory) - plan_slots counts exactly three things more: the RoPE table at the summed
//      positions (the engine loads one table at max_len = the sum; batch_memory's at the longest slot), the further
//      slots' hh rows and the row lists' (MtpDims::hh_rows) and SlotDims' control blocks - and its total to §9.1's
//      printed precision (on the l0 prefill path: §9.1 priced l0-int8);
//   3. slot_layout's rule: slot 0 at the asked length, then 32768-position slots while they fit with the reserve, at
//      most four in all, never a shorter one; the layouts of the bf16 and int8 KV forms at the asked lengths an agentic
//      session would pick (printed: they are the startup lines the operator reads).
#include <cstdint>
#include <cstdio>
#include <vector>
#include "check.h"
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
  CHECK_EQ(runtime::MtpDims::hh_rows(1), runtime::DecodeScratchDims::kM + 1);
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
  // The three things plan_slots counts beyond §9.1's method, exactly.
  const size_t rope = model::Qwen35::rope_table_bytes(229376) - model::Qwen35::rope_table_bytes(131072);
  const size_t hh = size_t(runtime::MtpDims::hh_rows(4) - runtime::MtpDims::hh_rows(1)) * q.hidden * 2;
  const size_t ctl = runtime::SlotDims::sizes(q, 4, true).controls;
  CHECK_EQ(ctl, size_t{10} * 128 + 3 * 256);
  CHECK_EQ(p.total(), b.total() + rope + hh + ctl);
  CHECK_EQ(p.kv, b.kv - (runtime::MtpDims::sizes(229376, q, 0, KvCache::Int8).kv_k * 2));   // the head's KV is MTP's
  // The further slots' state: 3 x (4 GDN states + a ring) + the hh rows + the controls.
  CHECK_EQ(p.slot_state, size_t{3} * (4 * size_t{150994944} + 15728640) + hh + ctl);
  // §9.1: "total 29.10 GB, 1.93 GB spare" (l0-int8); on the l0 default 0.087 GB less (batch_rows_test's 29.01).
  CHECK_NEAR(p.total() / kGB, 29.03, 0.01);
  std::printf("plan_slots 131072 + 3 x 32768, int8 KV, MTP: %.3f GB (batch_memory %.3f + RoPE %.3f + hh %.6f + "
              "controls %zu B), spare %.3f GB\n",
              p.total() / kGB, b.total() / kGB, rope / kGB, hh / kGB, ctl, (double(kDevice - kReserve) - p.total()) / kGB);
  // Without MTP: one GDN state a further slot (SlotDims' states), no head KV, no hh rows.
  const runtime::MemoryPlan n = runtime::plan_slots(q, lens, false, kQwenInt8Weights, {}, {}, KvCache::Int8);
  CHECK_EQ(n.slot_state, size_t{3} * (size_t{150994944} + 15728640) + ctl);
  CHECK_EQ(n.total() - n.slot_state, runtime::plan(q, 229376, false, kQwenInt8Weights, {}, {}, KvCache::Int8).total());
}

void check_layout_rule() {
  const model::ModelDesc& q = model::qwen38();
  const size_t w = kQwenInt8Weights + kMtpWeights;
  auto lay = [&](uint32_t asked, KvCache kv) {
    return runtime::slot_layout(q, asked, true, w, kDevice, kReserve, {}, {}, kv);
  };
  const std::vector<uint32_t> d = lay(131072, KvCache::Int8);
  CHECK(d == (std::vector<uint32_t>{131072, 32768, 32768, 32768}));
  CHECK(runtime::slot_layout_name(d) == "131072 + 3 x 32768");
  CHECK(runtime::slot_layout_name({65536, 32768}) == "65536 + 32768");
  // bf16 KV: the decided layout does not fit (§9: "at bf16 KV the slots are fewer"); 131072 keeps one slot.
  CHECK(lay(131072, KvCache::Bf16) == (std::vector<uint32_t>{131072}));
  CHECK(lay(65536, KvCache::Bf16) == (std::vector<uint32_t>{65536, 32768, 32768}));
  CHECK(lay(32768, KvCache::Bf16) == (std::vector<uint32_t>{32768, 32768, 32768, 32768}));
  // The rule itself, at every asked length: each layout fits with the reserve, one more short slot does not (or four).
  for (KvCache kv : {KvCache::Bf16, KvCache::Int8})
    for (uint32_t asked = 4096; asked <= 196608; asked += 4096) {
      if (runtime::plan(q, asked, true, w, {}, {}, kv).total() + kReserve > kDevice) continue;
      const std::vector<uint32_t> l = lay(asked, kv);
      CHECK(!l.empty() && l[0] == asked && l.size() <= runtime::SlotDims::kMaxSeqs);
      for (size_t i = 1; i < l.size(); ++i) CHECK_EQ(l[i], runtime::SlotDims::kShortLen);
      CHECK(runtime::plan_slots(q, l, true, w, {}, {}, kv).total() + kReserve <= kDevice);
      if (l.size() < runtime::SlotDims::kMaxSeqs) {
        std::vector<uint32_t> more = l;
        more.push_back(runtime::SlotDims::kShortLen);
        CHECK(runtime::plan_slots(q, more, true, w, {}, {}, kv).total() + kReserve > kDevice);
      }
    }
  std::printf("| --max-len | KV | slots (MTP) | total GB | spare GB |\n");
  for (KvCache kv : {KvCache::Bf16, KvCache::Int8})
    for (uint32_t asked : {32768u, 65536u, 98304u, 131072u, 163840u}) {
      if (runtime::plan(q, asked, true, w, {}, {}, kv).total() + kReserve > kDevice) continue;
      const std::vector<uint32_t> l = lay(asked, kv);
      const double t = runtime::plan_slots(q, l, true, w, {}, {}, kv).total() / kGB;
      std::printf("| %u | %s | %s | %.3f | %.3f |\n", asked, runtime::kv_cache_name(kv),
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
}  // namespace

int main() {
  check_one_slot();
  check_decided_layout();
  check_layout_rule();
  std::printf("slot_plan_test OK\n");
  return 0;
}
