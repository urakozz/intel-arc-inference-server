// Spec 13a (plan 13a Task 2 Step 1): runtime/batch_rows.h on the host - the row table's device layout, the rules a
// table keeps, the slots' KV bases, and the memory of spec 13 §9.1 / §9.2's layouts by the code's own formulas.
// No device. The weights are memory_plan_test's derived byte counts (the int8 head; the MTP head apart).
//
// Pinned: §9.1's per-slot rows (KV a position with the head's layer, the state a slot) exactly, and every layout
// total §9.1 / §9.2 quote to their printed precision (0.01 GB; "~" ones to 0.1 GB); the largest batch that
// fits Qwen3.8 with MTP. It prints the table docs/probes/probe-batching-2026-10-23.md §5 quotes.
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
  using runtime::KvCache;
  // §9.1's per-slot rows, exactly: KV a position with the head's layer (int8), the state of an MTP slot.
  const runtime::BatchMemory q1 = runtime::batch_memory(q, {{1, true}}, KvCache::Int8, 0, 0);
  const runtime::BatchMemory q0 = runtime::batch_memory(q, {{0, true}}, KvCache::Int8, 0, 0);
  CHECK_EQ(q1.kv - q0.kv, size_t{35088});
  CHECK_EQ(q0.state, size_t{4} * 150994944 + 15728640);   // 0.620 GB
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
  // Int8State and keeps the slab, 0.087 GB less on Qwen3.8 than §9.1's rows.
  // 3e adds one layer's unscaled bf16 K/V temporary at the longest slot's capacity.
  const double raw128 = gb(size_t{131072} * 4 * 256 * 2 * 2);
  CHECK_NEAR(gb(qd.weights), 17.70, 0.015);
  CHECK_NEAR(gb(qd.scratch), 0.78 + raw128, 0.015);
  CHECK_NEAR(gb(qd.total()), 29.01 + raw128, 0.015);
  CHECK_NEAR((double(kBudget) - double(qd.total())) / kGB, 2.02 - raw128, 0.015);
  const runtime::BatchMemory q4 = runtime::batch_memory(q, mk(4, 65536, 65536, true), KvCache::Int8, kQwenInt8Weights, kMtpWeights);
  print("Qwen3.8 4 x 65536, MTP", q4);
  CHECK_NEAR((double(kBudget) - double(q4.total())) / kGB,
             0.89 - gb(size_t{65536} * 4 * 256 * 2 * 2), 0.015);
  print("Qwen3.8 1 x 131072 + 3 x 32768, MTP, bf16 KV",
        runtime::batch_memory(q, mk(4, 131072, 32768, true), KvCache::Bf16, kQwenInt8Weights, kMtpWeights));
  print("Qwen3.8 1 x 131072 (today's engine), MTP",
        runtime::batch_memory(q, mk(1, 131072, 0, true), KvCache::Int8, kQwenInt8Weights, kMtpWeights));

  // §9.2's refusal rule: the largest batch that fits.
  CHECK_EQ(runtime::largest_batch(q, 4, 131072, 32768, true, KvCache::Int8, kQwenInt8Weights, kMtpWeights, kBudget), 4u);
  std::printf("largest batch (1 x 131072 + short 32768, int8 KV, one card): Qwen3.8 MTP 4\n");
}
}  // namespace

int main() {
  check_device_layout();
  check_rules();
  check_memory();
  std::printf("batch_rows_test OK\n");
  return 0;
}
