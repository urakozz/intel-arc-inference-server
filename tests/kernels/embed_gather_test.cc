// embed_gather.cl - the first kernel of a decode step.
//
// The check that matters here is the **two-replay** one. Level Zero captures a
// kernel's arguments at `zeCommandListAppendLaunchKernel` time, not at execute
// time (tests/l0/arg_capture_test.cc proves exactly that), so a token id passed
// as a kernel argument would be frozen into the list at capture and every
// replay would gather the same row. This kernel instead reads the id out of the
// shared-memory control block, which the host (or, from the second token on,
// `argmax_stage2`) updates between replays. So: one closed list, executed
// twice, with nothing rebound and only `Control::cur_token[0]` changed -
// different rows must come out. That property is what the whole capture-once /
// replay-per-token design rests on.
//
// The embedding table here is a 64-row stand-in for the real
// [248320][5120] bf16 table (2.54 GB). That is faithful because the kernel's
// only use of the vocabulary size is the range guard: for an in-range id it
// touches exactly `embed[row*5120 .. +5120)` and nothing else.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/kernels.h"
#include "kernels/shape_suffix.h"   // kRefHidden: embed_gather_M1 is built at HIDDEN 5120
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "model/qwen35.h"
#include "runtime/control.h"

namespace {
using Q = model::Qwen35;

constexpr uint32_t kM = 1;
constexpr uint32_t kWG = 256;
constexpr uint32_t kRows = 64;           // stand-in table depth
constexpr uint16_t kSentinel = 0xA5A5;   // resid is prefilled with this
constexpr uint32_t kBadRow = 0xDEAD0001u;  // embed_gather's debug_flag value

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence fence{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
};

// gemv_harness.h's recipe: replay a closed list 8 times, drop the first 3
// (warm-up / clock ramp), median of the last 5 over the launches in it.
// Printed, never asserted.
double median_us(Dev& d, l0::CmdList& list, int iters) {
  std::vector<double> us;
  for (int rep = 0; rep < 8; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    d.q.execute(list, &d.fence);
    d.fence.wait();
    const double total =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    if (rep >= 3) us.push_back(total / iters);
  }
  std::sort(us.begin(), us.end());
  return us[us.size() / 2];
}

}  // namespace

int main() {
  Dev d;
  std::vector<uint16_t> table(size_t{kRows} * kernels::kRefHidden);
  {
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> u(-2.f, 2.f);
    for (auto& e : table) e = common::f32_to_bf16(u(rng));
  }

  l0::Mem embed(d.ctx, l0::MemKind::Device, table.size() * 2);
  d.imm.copy(embed.ptr(), table.data(), table.size() * 2);
  l0::Mem resid(d.ctx, l0::MemKind::Device, size_t{kM} * kernels::kRefHidden * 2);
  l0::Mem ctrl_mem(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  runtime::Control* ctrl = ctrl_mem.as<runtime::Control>();
  *ctrl = runtime::Control{};
  ctrl->n_active = kM;

  l0::Module mod(d.ctx, kernels::path(kernels::embed_gather_variant(kM)));
  l0::Kernel k = mod.kernel("embed_gather");
  k.group_size(kWG);
  k.arg_ptr(0, ctrl_mem.ptr());
  k.arg_ptr(1, embed.ptr());
  k.arg_ptr(2, resid.ptr());

  // Captured ONCE. Everything below is a replay of this list.
  l0::CmdList list = l0::CmdList::regular(d.ctx);
  list.launch(k, 1, kM);
  list.close();

  std::vector<uint16_t> got(size_t{kM} * kernels::kRefHidden);
  auto replay = [&](uint32_t row) {
    ctrl->cur_token[0] = row;
    ctrl->debug_flag = 0;
    d.imm.fill(resid.ptr(), 0xA5A5A5A5u, resid.size());
    d.q.execute(list, &d.fence);
    d.fence.wait();
    d.imm.copy(got.data(), resid.ptr(), got.size() * 2);
  };
  auto expect_row = [&](uint32_t row) {
    for (uint32_t j = 0; j < kernels::kRefHidden; ++j) CHECK_EQ(got[j], table[size_t{row} * kernels::kRefHidden + j]);
    CHECK_EQ(ctrl->debug_flag, 0u);
    std::printf("embed_gather cur_token[0]=%u -> resid == embed row %u (5120 bf16 exact)\n", row,
                row);
  };

  // The two-replay check: same closed list, nothing rebound, different id.
  replay(7);
  expect_row(7);
  replay(41);
  expect_row(41);
  // Edges of the stand-in table.
  replay(0);
  expect_row(0);
  replay(kRows - 1);
  expect_row(kRows - 1);

  // Out of range. An id >= kVocab cannot come from this engine (argmax masks at
  // kVocabUsed and the tokenizer's ids are smaller still), so it means the
  // control block was corrupted: the kernel refuses to read a row it cannot
  // prove is inside the table, writes nothing, and reports through debug_flag.
  for (uint32_t bad : {Q::kVocab, 0xFFFFFFFFu}) {
    replay(bad);
    for (uint32_t j = 0; j < kernels::kRefHidden; ++j) CHECK_EQ(got[j], kSentinel);
    CHECK_EQ(ctrl->debug_flag, kBadRow);
    std::printf("embed_gather cur_token[0]=%u (out of range) -> resid untouched, debug_flag=%#x\n",
                bad, ctrl->debug_flag);
  }

  // ... and the list still works afterwards.
  replay(23);
  expect_row(23);

  // Cost per token. The stand-in table is 655 KB and L2-resident here; the real
  // 2.54 GB table is not, so a token's row is a cold 10 KB read in the engine.
  {
    const int kIters = 512;
    l0::CmdList timed = l0::CmdList::regular(d.ctx);
    for (int i = 0; i < kIters; ++i) timed.launch(k, 1, kM);
    timed.close();
    std::printf("embed_gather: %.2f us per token (%u-row L2-resident table)\n",
                median_us(d, timed, kIters), kRows);
  }

  std::puts("embed_gather_test OK");
  return 0;
}
