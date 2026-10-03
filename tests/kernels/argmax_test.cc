// argmax.cl - the two-stage deterministic argmax over the lm_head logits.
//
// What this asserts is not "a maximum" but **which** maximum, and that the
// control block moved exactly the way the decode loop needs it to:
//
//   * the tail of the vocabulary (248077..248319) is masked - those rows exist
//     in `lm_head` only because 248320 is a round number for the tiling, and
//     the tokenizer can never emit them (model::Qwen35::kVocabUsed);
//   * an exact tie resolves to the LOWEST index, in both trees - the stage-1
//     work-group tree and the stage-2 tree over the 243 partials;
//   * `argmax_stage2` is the ONLY writer of `pos` and `cur_token` inside the
//     captured list: after every replay `out_token[0]` holds the sampled id,
//     `cur_token[0]` is the id the next step will gather, and `pos` has
//     advanced by `n_active`.
//
// One closed regular list is replayed once per case, with nothing rebound
// between replays: that is the decode list's shape, and it is what makes the
// pos-advance assertion meaningful (a fresh list per case would not show that
// the advance is per *execution*).
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

#include "check.h"
#include "kernels/kernels.h"
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

// The kernels index the control block as a flat `uint` array; the indices are
// baked as `-D CTRL_*` in src/kernels/CMakeLists.txt and mirrored on the host
// by kernels::ctrl_index. These catch a layout change in runtime::Control at
// compile time; that the *kernels'* baked numbers agree is proved end to end
// below, since every assertion reads or writes a named struct field.
static_assert(offsetof(runtime::Control, pos) == 4 * kernels::ctrl_index::kPos,
              "CTRL_POS disagrees with runtime::Control");
static_assert(offsetof(runtime::Control, n_active) == 4 * kernels::ctrl_index::kNActive,
              "CTRL_NACT disagrees with runtime::Control");
static_assert(offsetof(runtime::Control, cur_token) == 4 * kernels::ctrl_index::kCurToken,
              "CTRL_CUR disagrees with runtime::Control");
static_assert(offsetof(runtime::Control, out_token) == 4 * kernels::ctrl_index::kOutToken,
              "CTRL_OUT disagrees with runtime::Control");
static_assert(offsetof(runtime::Control, debug_flag) == 4 * kernels::ctrl_index::kDebugFlag,
              "CTRL_DEBUG disagrees with runtime::Control");

constexpr uint32_t kM = 1;
constexpr uint32_t kChunk = 1024;                                   // logits per stage-1 group
constexpr uint32_t kGroups = (Q::kVocab + kChunk - 1) / kChunk;     // 243
constexpr uint32_t kWG = 256;

// A background of distinct negative values: every planted peak is then the
// unambiguous winner, and no accidental tie can mask a real one.
std::vector<float> background(uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-4.f, -1.f);
  std::vector<float> v(Q::kVocab);
  for (auto& e : v) e = d(rng);
  return v;
}

// The kernels' comparator, on the host: strictly greater wins, so a tie keeps
// the lower index. Only the usable part of the vocabulary is scanned.
uint32_t ref_argmax(const std::vector<float>& v) {
  uint32_t bi = 0;
  float bv = v[0];
  for (uint32_t k = 1; k < Q::kVocabUsed; ++k)
    if (v[k] > bv) { bv = v[k]; bi = k; }
  return bi;
}

struct Case {
  const char* name;
  std::vector<float> logits;
  uint32_t expect;
};

std::vector<Case> cases() {
  std::vector<Case> cs;
  {
    auto v = background(11);
    v[123456] = 3.5f;
    cs.push_back({"unique max at 123456", std::move(v), 123456});
  }
  {
    // Two of the tied indices sit in the same stage-1 group (5000 and 5001 are
    // both in group 4, on different lanes) and the third in another (group 87),
    // so the tie is resolved once by the work-group tree and once by stage 2's.
    auto v = background(12);
    v[5000] = 7.0f;
    v[5001] = 7.0f;
    v[90000] = 7.0f;
    cs.push_back({"three-way exact tie -> lowest index", std::move(v), 5000});
  }
  {
    // The masked tail: the two largest logits in the row are unusable ids, one
    // of them exactly at the first masked index.
    auto v = background(13);
    v[Q::kVocabUsed] = 99.f;
    v[248200] = 100.f;
    v[1000] = 42.f;
    cs.push_back({"max in the masked tail is ignored", std::move(v), 1000});
  }
  {
    std::vector<float> v(Q::kVocab, 0.f);
    cs.push_back({"all equal -> index 0", std::move(v), 0});
  }
  {
    auto v = background(14);
    v[Q::kVocabUsed - 1] = 5.0f;
    cs.push_back({"max at the last usable index 248076", std::move(v), Q::kVocabUsed - 1});
  }
  {
    auto v = background(15);
    const uint32_t e = ref_argmax(v);
    cs.push_back({"random row vs the host reference", std::move(v), e});
  }
  return cs;
}

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence fence{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
};

// gemv_harness.h's recipe, minus the weight cycling: replay a closed list 8
// times, drop the first 3 (warm-up / clock ramp), take the median of the last 5
// and divide by the iterations recorded in it. Printed, never asserted - a
// timing assertion would fail on a busy box for no useful reason.
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
  l0::Mem logits(d.ctx, l0::MemKind::Device, size_t{kM} * Q::kVocab * 4);
  l0::Mem part(d.ctx, l0::MemKind::Device, size_t{kM} * kGroups * 2 * 4);
  l0::Mem ctrl_mem(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  runtime::Control* ctrl = ctrl_mem.as<runtime::Control>();

  l0::Module mod1(d.ctx, kernels::path(kernels::argmax_stage1_variant(kM, Q::kVocabUsed)));
  l0::Kernel k1 = mod1.kernel("argmax_stage1");
  k1.group_size(kWG);
  k1.arg_ptr(0, logits.ptr());
  k1.arg_ptr(1, part.ptr());

  l0::Module mod2(d.ctx, kernels::path(kernels::argmax_stage2_variant()));
  l0::Kernel k2 = mod2.kernel("argmax_stage2");
  k2.group_size(kWG);
  k2.arg_ptr(0, ctrl_mem.ptr());
  k2.arg_ptr(1, part.ptr());

  l0::CmdList list = l0::CmdList::regular(d.ctx);
  list.launch(k1, kGroups, kM);
  list.launch(k2, 1, 1);
  list.close();

  uint32_t pos = 17;   // an arbitrary non-zero start: the advance is what matters
  for (const Case& c : cases()) {
    d.imm.copy(logits.ptr(), c.logits.data(), c.logits.size() * 4);
    *ctrl = runtime::Control{};
    ctrl->pos = pos;
    ctrl->n_active = 1;
    ctrl->cur_token[0] = 0xFFFFFFFFu;   // sentinels: stage 2 must overwrite both
    ctrl->out_token[0] = 0xFFFFFFFFu;

    d.q.execute(list, &d.fence);
    d.fence.wait();

    std::printf("argmax %-38s -> out_token[0]=%u (expect %u), pos %u -> %u\n", c.name,
                ctrl->out_token[0], c.expect, pos, ctrl->pos);
    CHECK_EQ(ctrl->out_token[0], c.expect);
    CHECK_EQ(ctrl->cur_token[0], c.expect);   // fed straight back to embed_gather
    CHECK_EQ(ctrl->pos, pos + ctrl->n_active);
    CHECK_EQ(ctrl->debug_flag, 0u);
    pos = ctrl->pos;
  }

  // Cost of the pair, per token. The logits sit in one 993 KB buffer that is
  // re-read every iteration, so this is an L2-resident number - which is the
  // engine's case too, since lm_head has just written those logits.
  {
    const int kIters = 512;
    l0::CmdList timed = l0::CmdList::regular(d.ctx);
    for (int i = 0; i < kIters; ++i) {
      timed.launch(k1, kGroups, kM);
      timed.launch(k2, 1, 1);
    }
    timed.close();
    std::printf("argmax stage1(%u WGs) + stage2: %.2f us per token\n", kGroups,
                median_us(d, timed, kIters));
  }
  std::puts("argmax_test OK");
  return 0;
}
