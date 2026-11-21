// Spec 39 Task 1: runtime/gpu_state.h's rules and walks on a fixture (a fake allocation type, no device), and
// Flash-Next's reflected sums against the hand-written sums they replaced. Host only.
//
// The build-failure half is gpu_state_fail.cc (tests/CMakeLists.txt: gpu_state_unannotated_test,
// gpu_state_unplanned_test): a sizes member without an annotation, and a buffer without a sizes member (PERF-116's
// idx_q4), must not compile.
#include <cstdint>
#include <cstdio>
#include <array>
#include <memory>
#include <vector>

#include "check.h"
#include "model/qwen4exp.h"
#include "runtime/control.h"
#include "runtime/gpu_state.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

namespace gs = runtime::gpu_state;
namespace rq = runtime::qwen4exp;

namespace {
struct FakeMem {
  size_t bytes;
  size_t size() const { return bytes; }
};
enum class T : uint32_t { A, B };
using Snap = gs::Snapshot<T>;

struct Sizes {
  [[=gs::state, =gs::zero_on_reset, =gs::NotSnapshotted("fixture")]] size_t ctl = 0;
  [[=gs::kv, =gs::zero_on_reset, =Snap{T::A}]] size_t k = 0;
  [[=gs::state, =gs::KeptOnReset("fixture"), =Snap{T::B}]] size_t ring = 0;
  [[=gs::scratch]] size_t tmp = 0;
  [[=gs::constant]] size_t consts = 0;
};
struct Bufs {
  FakeMem ctl, k;
  std::unique_ptr<FakeMem> ring;
  FakeMem tmp, consts;
  uint32_t device = 0;   // not a buffer
};
static_assert(gs::sizes_problem<Sizes, T>().empty(), gs::sizes_problem<Sizes, T>());
static_assert(gs::snapshot_problem<T, Sizes>().empty(), gs::snapshot_problem<T, Sizes>());
static_assert(gs::buffers_problem<FakeMem, Bufs, Sizes>().empty(), gs::buffers_problem<FakeMem, Bufs, Sizes>());

// Each rule refuses its case.
struct NoClass {
  size_t a = 0;
};
struct TwoClasses {
  [[=gs::scratch, =gs::constant]] size_t a = 0;
};
struct ConstantZeroed {
  [[=gs::constant, =gs::zero_on_reset]] size_t a = 0;
};
struct ScratchSnapshot {
  [[=gs::scratch, =gs::NotSnapshotted("fixture")]] size_t a = 0;
};
struct ScratchZeroed {   // allowed: a reset may zero scratch
  [[=gs::scratch, =gs::zero_on_reset]] size_t a = 0;
};
struct NoResetSaid {
  [[=gs::state, =Snap{T::A}]] size_t a = 0;
};
struct NoSnapshotSaid {
  [[=gs::state, =gs::zero_on_reset]] size_t a = 0;
};
struct BothSnapshotSaid {
  [[=gs::state, =gs::zero_on_reset, =Snap{T::A}, =gs::NotSnapshotted("fixture")]] size_t a = 0;
};
static_assert(!gs::sizes_problem<NoClass, T>().empty());
static_assert(!gs::sizes_problem<TwoClasses, T>().empty());
static_assert(!gs::sizes_problem<ConstantZeroed, T>().empty());
static_assert(!gs::sizes_problem<ScratchSnapshot, T>().empty());
static_assert(gs::sizes_problem<ScratchZeroed, T>().empty());
static_assert(!gs::sizes_problem<NoResetSaid, T>().empty());
static_assert(!gs::sizes_problem<NoSnapshotSaid, T>().empty());
static_assert(!gs::sizes_problem<BothSnapshotSaid, T>().empty());
static_assert(!gs::snapshot_problem<T, NoResetSaid>().empty());   // B unclaimed
static_assert(!gs::snapshot_problem<T, Sizes, NoResetSaid>().empty());   // A claimed twice
struct Unsized {
  FakeMem ctl, k, extra;
  std::unique_ptr<FakeMem> ring;
  FakeMem tmp, consts;
};
struct Unbuffered {
  FakeMem ctl, k, tmp, consts;
};
static_assert(!gs::buffers_problem<FakeMem, Unsized, Sizes>().empty());
static_assert(!gs::buffers_problem<FakeMem, Unbuffered, Sizes>().empty());
// A container of allocations is refused unless it says why it is outside; an array of owned ones is a buffer.
struct Listed {
  FakeMem ctl, k;
  std::unique_ptr<FakeMem> ring;
  FakeMem tmp, consts;
  std::vector<FakeMem> more;
};
struct ListedOutside {
  FakeMem ctl, k;
  std::unique_ptr<FakeMem> ring;
  FakeMem tmp, consts;
  [[=gs::Outside("fixture")]] std::vector<FakeMem> more;
};
static_assert(!gs::buffers_problem<FakeMem, Listed, Sizes>().empty());
static_assert(gs::buffers_problem<FakeMem, ListedOutside, Sizes>().empty());
struct ChunkSizes {
  [[=gs::state, =gs::zero_on_reset, =gs::NotSnapshotted("fixture")]] size_t shadow = 0;
};
struct Chunks {
  std::array<std::unique_ptr<FakeMem>, 2> shadow;
};
struct CChunks {
  std::unique_ptr<FakeMem> shadow[2];
};
static_assert(gs::buffers_problem<FakeMem, Chunks, ChunkSizes>().empty());
static_assert(gs::buffers_problem<FakeMem, CChunks, ChunkSizes>().empty());
struct CListed {   // a C array of plain allocations: refused like any other container
  FakeMem shadow[2];
};
static_assert(!gs::buffers_problem<FakeMem, CListed, ChunkSizes>().empty());
}  // namespace

int main() {
  // The walks on the fixture.
  const Sizes s{.ctl = 1, .k = 10, .ring = 100, .tmp = 1000, .consts = 10000};
  CHECK_EQ(gs::bytes(s), size_t(11111));
  CHECK_EQ(gs::bytes(s, gs::Group::Kv), size_t(10));
  CHECK_EQ(gs::bytes(s, gs::Group::State), size_t(101));
  CHECK_EQ(gs::bytes(s, gs::Group::Scratch), size_t(11000));
  CHECK_EQ(gs::per_sequence_bytes(s), size_t(111));
  Bufs b{{1}, {10}, std::make_unique<FakeMem>(FakeMem{100}), {1000}, {10000}, 0};
  CHECK_EQ((gs::buffer_bytes<FakeMem, Sizes>(b, gs::Group::State)), size_t(101));
  std::vector<const FakeMem*> zeroed;
  gs::zero<FakeMem, Sizes>(b, [&](FakeMem& m) { zeroed.push_back(&m); });
  CHECK(zeroed.size() == 2 && zeroed[0] == &b.ctl && zeroed[1] == &b.k);   // ring is KeptOnReset
  CHECK((gs::snapshot_buffer<FakeMem, T, Sizes>(b, T::A) == &b.k));
  CHECK((gs::snapshot_buffer<FakeMem, T, Sizes>(b, T::B) == b.ring.get()));
  b.ring.reset();   // an owned buffer not allocated: not counted, not zeroed, no snapshot allocation
  CHECK_EQ((gs::buffer_bytes<FakeMem, Sizes>(b, gs::Group::State)), size_t(1));
  CHECK((gs::snapshot_buffer<FakeMem, T, Sizes>(b, T::B) == nullptr));
  Chunks c{{std::make_unique<FakeMem>(FakeMem{7}), nullptr}};   // one chunk's allocation made
  CHECK_EQ((gs::buffer_bytes<FakeMem, ChunkSizes>(c)), size_t(7));
  size_t zeroed_chunks = 0;
  gs::zero<FakeMem, ChunkSizes>(c, [&](FakeMem&) { ++zeroed_chunks; });
  CHECK_EQ(zeroed_chunks, size_t(1));

  // Flash-Next: the reflected sums equal the hand-written ones they replaced, at the real 48 layers on one and two
  // cards, with and without the head's verify rows.
  const model::Qwen4ExpDesc& d = model::qwen4exp();
  for (const model::Q4Placement& p : {model::Q4Placement::one(d), model::Q4Placement::two(d, 24)})
    for (uint32_t dev = 0; dev < p.devices; ++dev) {
      const rq::PersistentSizes ps = rq::persistent_sizes(d, p, dev, 32768);
      CHECK_EQ(ps.kv_total(), ps.kv + ps.idx_keys + ps.idx_tail);
      CHECK_EQ(ps.state(), ps.control + ps.gdn_state + ps.conv_ring + ps.ple);
      CHECK_EQ(ps.total(), ps.kv_total() + ps.state());
      const rq::VerifySizes vs = rq::verify_sizes(d, p, dev);
      CHECK_EQ(gs::bytes(vs), rq::gdn_pend_bytes(d, p, dev) + rq::ple_verify_bytes(d, p, dev));
      CHECK_EQ(gs::bytes(vs, gs::Group::State), gs::bytes(vs));
    }
  const rq::ScratchSizes ss = rq::scratch_sizes(d, 32768, rq::kDefaultQ4Attn, rq::kVerifyRows);
  CHECK_EQ(ss.total(), ss.H + ss.xn + ss.x + ss.partials + ss.down_f32 + ss.inj + ss.ab + ss.gdn_o + ss.idx_f32 +
                           ss.idx_q + ss.scores + ss.list + ss.diag + ss.attn_q + ss.attn_gate + ss.attn_part +
                           ss.attn_out + ss.logits_r + ss.routes + ss.moe_h + ss.y + ss.ple_e + ss.ple_kv + ss.ple_ids +
                           ss.ple_consts + ss.logits + ss.argmax_part);
  const rq::MtpSizes ms = rq::mtp_sizes(d, 32768);
  CHECK_EQ(ms.persistent(), ms.hctl + ms.kv + ms.idx_keys + ms.idx_tail + ms.hh);
  CHECK_EQ(ms.total(), ms.persistent() + ms.list + ms.diag + ms.xe + ms.xh + ms.fe + ms.fh + ms.logits + ms.routes);
  CHECK_EQ(gs::bytes(ms, gs::Group::Kv), ms.kv + ms.idx_keys + ms.idx_tail);
  CHECK_EQ(gs::bytes(ms, gs::Group::State), ms.hctl + ms.hh);
  // The tp card's own allocations (TpCardSizes): with the head the MoE's slots hold kVerifyRows rows and card 0 a draft
  // Control - the plan counted one row and no Control before.
  const model::Q4Placement tp = model::Q4Placement::tp2(d, model::Q4TpEnds::Split);
  const std::vector<rq::DevicePlan> off = rq::tp_plan(d, tp, 32768, true, false), on = rq::tp_plan(d, tp, 32768, true, true);
  for (uint32_t r = 0; r < 2; ++r)
    CHECK_EQ(on[r].link - off[r].link, size_t(d.top_k + 1) * (rq::kVerifyRows - 1) * d.hidden * 4 +
                                           (r == 0 ? sizeof(runtime::Control) : 0));
  std::puts("gpu_state_test OK");
  return 0;
}
