// Spec 13a (plan 13a Task 2 Step 1): runtime/batch_rows.h on the host - the row table's device layout, the rules a
// table keeps, the slots' KV bases, and the memory of spec 13 §9.1 / §9.2's layouts by the code's own formulas.
// No device. The weights are memory_plan_test's derived byte counts (the int8 head; the MTP head apart).
//
// Pinned: §9.1's per-slot rows (KV a position with the head's layer, the state a slot) exactly, and every layout
// total §9.1 / §9.2 quote to their printed precision (0.01 GB; "~" ones to 0.1 GB); the refusal's numbers - the
// largest batch that fits Agnes with and without MTP. It prints the table docs/probe-batching-2026-10-23.md §5 quotes.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include "check.h"
#include "kernels/batch_kernels.h"
#include "model/model_desc.h"
#include "runtime/batch_rows.h"

namespace {
using runtime::BatchLayout;
using runtime::BatchRow;
using runtime::RowRun;
using runtime::SlotSpec;

// memory_plan_test.cc's derived weights: the int8 head, without the MTP head and the RoPE table.
constexpr size_t kLmBf16 = 2542796800ull, kLmInt8 = 1271398400ull + 993280ull;
constexpr size_t kQwenInt8Weights = 18086971392ull - 4194304ull - kLmBf16 + kLmInt8;   // 16,812,371,968
constexpr size_t kAgnesInt8Weights = 14816378880ull + 926023680ull + 54ull * 983040 + 54ull * 327680 +
                                     (54ull * 205440 + 18ull * 43008 + 20480) + 2542796800ull + kLmInt8;
constexpr size_t kMtpWeights = 424673280ull * 2 + 5ull * 5120 * 4 + 2048;               // 849,451,008
// Spec 13 §9.1: 32.53 GB, the 1.5 GB reserve.
constexpr size_t kBudget = 32530000000ull - 1500000000ull;
constexpr double kGB = 1e9;

template <class F>
bool throws(F f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

void check_device_layout() {
  // The kernels' BROW_* words (src/kernels/CMakeLists.txt) mirror BatchRow's fields: capture.cc (B70_BATCH_PROBE)
  // static_asserts the pair; the values are repeated here so a moved field fails a host test too.
  CHECK_EQ(runtime::kBatchRowWords, 8u);
  CHECK_EQ(kernels::batch_row::kWords, runtime::kBatchRowWords);
  CHECK_EQ(kernels::batch_row::kPos, 0u);
  CHECK_EQ(kernels::batch_row::kKvBase, 1u);
  CHECK_EQ(kernels::batch_row::kSrc, 2u);
  CHECK_EQ(kernels::batch_row::kDst, 3u);
  CHECK_EQ(kernels::batch_row::kSlot, 4u);
  CHECK_EQ(sizeof(runtime::BatchRowTable), size_t{256});
}

void check_rules() {
  const BatchLayout lay{{{36864, true}, {8192, true}, {8192, true}, {8192, true}}};
  CHECK_EQ(lay.total(), 61440u);
  CHECK_EQ(lay.kv_base(0), 0u);
  CHECK_EQ(lay.kv_base(1), 36864u);
  CHECK_EQ(lay.kv_base(3), 36864u + 2 * 8192);
  // 2 slots x 4 rows (an MTP-shaped step): every row's base, src, dst, position.
  const std::vector<BatchRow> t = runtime::make_rows(lay, {{0, 32768, 4, 0, {4, 5, 6, 7}}, {2, 4096, 4, 2, {8, 9, 10, 11}}});
  CHECK_EQ(t.size(), size_t{8});
  CHECK_EQ(t[3].pos, 32771u);
  CHECK_EQ(t[4].kv_base, 36864u + 8192);
  CHECK_EQ(t[5].src, 2u);
  CHECK_EQ(t[7].dst, 11u);
  CHECK(runtime::run_lengths(t) == (std::vector<uint32_t>{4, 4}));
  // A run may write its own src (the tile is read before the first write: spec 8's in-place row 0).
  CHECK(!throws([&] { runtime::make_rows(lay, {{1, 10, 2, 1, {1, 5}}, {3, 10, 1, 3, {3}}}); }));
  // The rules.
  CHECK(throws([&] { runtime::make_rows(lay, {}); }));                                            // no row
  CHECK(throws([&] { runtime::make_rows(lay, {{0, 0, 5, 0, {4, 5, 6, 7, 8}}, {1, 0, 4, 1, {9, 10, 11, 12}}}); }));  // 9 rows
  CHECK(throws([&] { runtime::make_rows(lay, {{4, 0, 1, 4, {5}}}); }));                             // no slot 4
  CHECK(throws([&] { runtime::make_rows(lay, {{1, 8191, 2, 1, {5, 6}}}); }));                       // past the slot
  CHECK(throws([&] { runtime::make_rows(lay, {{1, 10, 1, 1, {5}}, {2, 10, 1, 2, {6}}, {1, 11, 1, 1, {7}}}); }));  // split
  CHECK(throws([&] { runtime::make_rows(lay, {{1, 10, 2, 1, {5, 5}}}); }));                         // one dst twice
  CHECK(throws([&] { runtime::make_rows(lay, {{1, 10, 1, 1, {2}}, {2, 10, 1, 2, {6}}}); }));          // another run's src
  CHECK(throws([&] { runtime::make_rows(lay, {{1, 10, 1, 3, {5}}, {2, 10, 1, 3, {6}}}); }));          // one src, two runs
  CHECK(throws([&] { runtime::make_rows(lay, {{1, 10, 2, 1, {5}}}); }));                            // dsts != rows
  // Rows built by hand: a position gap and a src change inside a run, a wrong base.
  std::vector<BatchRow> r = runtime::make_rows(lay, {{1, 10, 3, 1, {5, 6, 7}}});
  r[2].pos = 13;
  CHECK(throws([&] { runtime::check_rows(r, lay); }));
  r[2].pos = 12;
  r[1].src = 2;
  CHECK(throws([&] { runtime::check_rows(r, lay); }));
  r[1].src = 1;
  r[0].kv_base = 0;
  CHECK(throws([&] { runtime::check_rows(r, lay); }));
}

double gb(size_t b) { return double(b) / kGB; }

void print(const char* name, const runtime::BatchMemory& m) {
  const double spare = (double(kBudget) - double(m.total())) / kGB;
  std::printf("| %-56s | %6.2f | %5.2f | %6.2f | %5.2f | %6.2f | %+6.2f |\n", name, gb(m.kv), gb(m.state), gb(m.weights),
              gb(m.scratch), gb(m.total()), spare);
}

void check_memory() {
  const model::ModelDesc& q = model::qwen38();
  const model::ModelDesc& a = model::agnes();
  using runtime::KvCache;
  // §9.1's per-slot rows, exactly: KV a position with the head's layer (int8), the state of an MTP slot.
  const runtime::BatchMemory q1 = runtime::batch_memory(q, {{1, true}}, KvCache::Int8, 0, 0);
  const runtime::BatchMemory q0 = runtime::batch_memory(q, {{0, true}}, KvCache::Int8, 0, 0);
  CHECK_EQ(q1.kv - q0.kv, size_t{35088});
  CHECK_EQ(q0.state, size_t{4} * 150994944 + 15728640);   // 0.620 GB
  const runtime::BatchMemory a1 = runtime::batch_memory(a, {{1, true}}, KvCache::Int8, 0, 0);
  const runtime::BatchMemory a0 = runtime::batch_memory(a, {{0, true}}, KvCache::Int8, 0, 0);
  CHECK_EQ(a1.kv - a0.kv, size_t{39216});
  CHECK_NEAR(gb(a0.state), 0.697, 0.0005);
  // Without MTP: no head layer, one GDN state.
  CHECK_EQ(runtime::batch_memory(q, {{1, false}}, KvCache::Int8, 0, 0).kv -
               runtime::batch_memory(q, {{0, false}}, KvCache::Int8, 0, 0).kv,
           size_t{33024});
  CHECK_EQ(runtime::batch_memory(q, {{0, false}}, KvCache::Int8, 0, 0).state, size_t{150994944} + 15728640);

  std::printf("| %-56s | KV GB | state | weights | scratch | total | spare |\n", "layout (int8 KV unless said; DERIVED)");
  const auto mk = runtime::main_and_short;
  const runtime::BatchMemory qd = runtime::batch_memory(q, mk(4, 131072, 32768, true), KvCache::Int8, kQwenInt8Weights, kMtpWeights);
  print("Qwen3.8 1 x 131072 + 3 x 32768, MTP (decided)", qd);
  // 2026-10-27: the prefill default is l0 (spec 13 §9.1 priced l0-int8): the shared scratch loses
  // Int8State and keeps the slab, 0.087 GB less on Qwen3.8 and 0.099 GB less on Agnes than §9.1's rows.
  // 2026-10-31: the GDN prefill's w / u are fp32 (operator ruling: Q1-Q4 fp32), +0.050 GB of scratch on both.
  CHECK_NEAR(gb(qd.weights), 17.70, 0.015);
  CHECK_NEAR(gb(qd.scratch), 0.83, 0.015);
  CHECK_NEAR(gb(qd.total()), 29.06, 0.015);
  CHECK_NEAR((double(kBudget) - double(qd.total())) / kGB, 1.97, 0.015);
  const runtime::BatchMemory q4 = runtime::batch_memory(q, mk(4, 65536, 65536, true), KvCache::Int8, kQwenInt8Weights, kMtpWeights);
  print("Qwen3.8 4 x 65536, MTP", q4);
  CHECK_NEAR((double(kBudget) - double(q4.total())) / kGB, 0.84, 0.015);
  print("Qwen3.8 1 x 131072 + 3 x 32768, MTP, bf16 KV",
        runtime::batch_memory(q, mk(4, 131072, 32768, true), KvCache::Bf16, kQwenInt8Weights, kMtpWeights));
  print("Qwen3.8 1 x 131072 (today's engine), MTP",
        runtime::batch_memory(q, mk(1, 131072, 0, true), KvCache::Int8, kQwenInt8Weights, kMtpWeights));

  const runtime::BatchMemory ad = runtime::batch_memory(a, mk(2, 131072, 32768, true), KvCache::Int8, kAgnesInt8Weights, kMtpWeights);
  print("Agnes 1 x 131072 + 1 x 32768, MTP (decided)", ad);
  CHECK_NEAR(gb(ad.weights), 20.52, 0.015);
  CHECK_NEAR(gb(ad.scratch), 0.88, 0.015);
  CHECK_NEAR(gb(ad.total()), 29.22, 0.015);
  CHECK_NEAR((double(kBudget) - double(ad.total())) / kGB, 1.81, 0.015);
  const runtime::BatchMemory a4 = runtime::batch_memory(a, mk(4, 131072, 32768, true), KvCache::Int8, kAgnesInt8Weights, kMtpWeights);
  print("Agnes 1 x 131072 + 3 x 32768, MTP", a4);
  CHECK_NEAR(gb(a4.total()), 33.19, 0.015);
  CHECK_NEAR((double(kBudget) - double(a4.total())) / kGB, -2.16, 0.015);
  const runtime::BatchMemory a4off = runtime::batch_memory(a, mk(4, 131072, 32768, false), KvCache::Int8, kAgnesInt8Weights, kMtpWeights);
  print("Agnes 1 x 131072 + 3 x 32768, --mtp off", a4off);
  CHECK_NEAR(gb(a4off.total()), 29.80, 0.015);   // §9.2 says ~29.9 (its "~": an estimate, on l0-int8)
  CHECK_NEAR((double(kBudget) - double(a4off.total())) / kGB, 1.23, 0.015);   // §9.2: ~1.1
  std::vector<SlotSpec> main_only = mk(4, 131072, 32768, false);
  main_only[0].mtp = true;
  const runtime::BatchMemory a4main = runtime::batch_memory(a, main_only, KvCache::Int8, kAgnesInt8Weights, kMtpWeights);
  print("Agnes 1 x 131072 + 3 x 32768, MTP in the main slot only", a4main);
  CHECK_NEAR(gb(a4main.total()), 31.46, 0.015);
  CHECK_NEAR((double(kBudget) - double(a4main.total())) / kGB, -0.43, 0.015);
  std::vector<SlotSpec> three = mk(3, 131072, 32768, false);
  three[0].mtp = true;
  const runtime::BatchMemory a3main = runtime::batch_memory(a, three, KvCache::Int8, kAgnesInt8Weights, kMtpWeights);
  print("Agnes 1 x 131072 + 2 x 32768, MTP in the main slot only", a3main);
  CHECK_NEAR(gb(a3main.total()), 30.05, 0.015);   // §9.1's considered alternative (0.93 GB spare on l0-int8)
  print("Agnes 1 x 131072 + 2 x 32768, --mtp off",
        runtime::batch_memory(a, mk(3, 131072, 32768, false), KvCache::Int8, kAgnesInt8Weights, kMtpWeights));
  print("Agnes 1 x 131072 + 1 x 32768, MTP, bf16 KV",
        runtime::batch_memory(a, mk(2, 131072, 32768, true), KvCache::Bf16, kAgnesInt8Weights, kMtpWeights));

  // §9.2's refusal: the largest batch that fits, with MTP and without.
  CHECK_EQ(runtime::largest_batch(a, 4, 131072, 32768, true, KvCache::Int8, kAgnesInt8Weights, kMtpWeights, kBudget), 2u);
  CHECK_EQ(runtime::largest_batch(a, 4, 131072, 32768, false, KvCache::Int8, kAgnesInt8Weights, kMtpWeights, kBudget), 4u);
  CHECK_EQ(runtime::largest_batch(q, 4, 131072, 32768, true, KvCache::Int8, kQwenInt8Weights, kMtpWeights, kBudget), 4u);
  std::printf("largest batch (1 x 131072 + short 32768, int8 KV, one card): Qwen3.8 MTP 4; Agnes MTP 2, --mtp off 4\n");
}
}  // namespace

int main() {
  unsetenv("B70_DECODE_ATTN");   // the plan's attn_part is v2's (the default), as the engine allocates
  check_device_layout();
  check_rules();
  check_memory();
  std::printf("batch_rows_test OK\n");
  return 0;
}
