// gdn_step.cl vs the CPU reference in gdn_ref.h - the gated delta-rule decode
// step, the kernel 48 of the model's 64 layers run.
//
// The reference is the *same op chain in the same order* as the kernel, trees
// included (see gdn_ref.h's header), so what is left between them is exactly the
// libm functions OpenCL does not require to be correctly rounded: `exp` (3 ulp)
// in the decay, the sigmoid and the conv's SiLU, and `log1p` (2 ulp) in the
// softplus. Everything else - the bf16 roundings, the conv taps, `1.0f/sqrt`,
// the fp32 recurrence - is bit-reproducible, which is why the bars below are
// tight and why the ring is held to **bit-exact**:
//
//   * `conv_ring` - bit-exact over all 163840 words. The written slots carry
//     `rne_bf16` of the qkv linear's fp32 output, which is a rounding and not an
//     arithmetic; the untouched slots prove no work-group wrote outside the ones
//     it owns (see gdn_step.cl, "Ring ownership").
//   * `state` - relative error <= 1e-5.
//   * `gdn_o`  - relative error <= 1e-3 (it is `qᵀS` after the same chain, so it
//     carries every earlier ulp plus a 128-term contraction).
//
// "Relative error" here is per element, floored at the tensor's own RMS:
// `|got − ref| / max(|ref|, rms(ref))`. Flooring matters - `Δ = (v − kv)·β` is a
// difference of two same-sized numbers, so individual state cells can land
// arbitrarily close to zero by cancellation and a raw per-element ratio there
// would measure the cancellation, not the kernel. Above the RMS the floor does
// nothing and the bar is the ordinary relative error; the plain relative error
// over the elements at or above the RMS is printed alongside, so both numbers
// are on the record.
//
// Three things are checked, and the third does not go through the reference at
// all - a shared misreading of *which* ring slots the conv window covers would
// otherwise hide in both implementations:
//
//   1. `pos = 0` - the conv's whole 3-token history is below position zero and
//      must contribute zeros whatever the ring holds.
//   2. `pos = 5` - the history is ring slots 2, 3 and 4 and the write lands in
//      slot 5; all 16 slots are seeded with known bf16 words.
//   3. **slot ownership**, at `pos = 5`: zeroing slots 2..4 must change `gdn_o`,
//      and zeroing every *other* slot must leave it **bitwise unchanged**. That
//      pair pins the window to exactly those three slots, and pins the ring
//      write to a slot no reader touches, without consulting gdn_ref.h.
//
// Two more positions cover the window's edges: `pos = 1`, where the clamp is
// *partial* (positions -2 and -1 contribute zeros but position 0 is a real ring
// slot), and `pos = 15` at M = 2, where the write wraps - slots 15 and 0 - while
// the window reads 12, 13 and 14.
//
// A last case runs the M = 2 variant at n_active = 2. It is the only cover for
// the conv window's intra-step path: at `m = 1` the window is positions 3, 4, 5
// and 6, so **two of its four entries are this step's own raw values** - token
// 0's at position 5 and token 1's own at position 6 - while positions 3 and 4
// are still ring slots. Of the three *older* taps, exactly one is intra-step.
//
// Every case replays its launch from freshly re-uploaded inputs and requires
// `gdn_o`, `state` and `conv_ring` to be bitwise identical - the determinism the
// captured decode list rests on.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "gdn_ref.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "runtime/control.h"

namespace {

using gdn_ref::kAbStride;
using gdn_ref::kChunks;
using gdn_ref::kConvRows;
using gdn_ref::kDim;
using gdn_ref::kHeads;
using gdn_ref::kQkvzN;
using gdn_ref::kRing;

constexpr uint32_t kWG = 256;
constexpr size_t kSmallFloats = 164480 / 4;              // loader::kGdnBlockBytes / 4
constexpr size_t kStateElems = size_t(kHeads) * kDim * kDim;
constexpr size_t kRingElems = size_t(kRing) * kConvRows;

std::vector<float> random_f32(size_t n, uint32_t seed, float mean, float sigma) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(mean, sigma);
  std::vector<float> v(n);
  for (auto& e : v) e = d(rng);
  return v;
}

std::vector<float> uniform_f32(size_t n, uint32_t seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (auto& e : v) e = d(rng);
  return v;
}

std::vector<uint16_t> random_bf16(size_t n, uint32_t seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> v(n);
  for (auto& e : v) e = common::f32_to_bf16(d(rng));
  return v;
}

struct Err {
  double worst = 0.0;      // max |d| / max(|ref|, rms)  - the asserted number
  double plain = 0.0;      // max |d| / |ref| over the elements with |ref| >= rms
  double max_abs = 0.0;
  double rms = 0.0;
  size_t worst_i = 0;
};

Err compare_f32(const std::vector<float>& got, const std::vector<float>& ref) {
  CHECK_EQ(got.size(), ref.size());
  Err e;
  double sq = 0.0;
  for (float r : ref) sq += double(r) * double(r);
  e.rms = std::sqrt(sq / double(ref.size()));
  const double floor = e.rms > 0.0 ? e.rms : 1.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = std::fabs(double(got[i]) - double(ref[i])), a = std::fabs(double(ref[i]));
    if (d > e.max_abs) e.max_abs = d;
    const double rel = d / (a > floor ? a : floor);
    if (rel > e.worst) { e.worst = rel; e.worst_i = i; }
    if (a >= floor && d / a > e.plain) e.plain = d / a;
  }
  return e;
}

void require(const Err& e, double tol, const char* what) {
  if (!(e.worst <= tol))
    std::fprintf(stderr, "%s: relative error %.3e > tol %.1e at index %zu (max abs %.3e, rms %.3e)\n",
                 what, e.worst, tol, e.worst_i, e.max_abs, e.rms);
  CHECK(e.worst <= tol);
}

// Bit-exactness on the ring, with the first few mismatches named: a diverging
// ring is the failure mode that would be hardest to read from a summary.
void require_bits(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref,
                  const char* what) {
  CHECK_EQ(got.size(), ref.size());
  size_t bad = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (got[i] == ref[i]) continue;
    if (bad < 3)
      std::fprintf(stderr, "%s: slot %zu channel %zu: got 0x%04X ref 0x%04X\n", what,
                   i / kConvRows, i % kConvRows, got[i], ref[i]);
    ++bad;
  }
  if (bad) std::fprintf(stderr, "%s: %zu/%zu words differ\n", what, bad, ref.size());
  CHECK_EQ(bad, size_t{0});
}

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence fence{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
};

// Everything one launch needs, built from one seed.
struct Inputs {
  uint32_t M = 1;
  std::vector<float> qkvz, ab, small, state0;
  std::vector<uint16_t> ring0;
};

Inputs make_inputs(uint32_t M, uint32_t seed) {
  Inputs in;
  in.M = M;
  // The qkv‖z GEMV's partials (S = 1): only [0, 10240) is convolved here, but
  // the buffer is the real 16384-wide one so the indexing is the engine's.
  in.qkvz = random_f32(size_t(M) * kQkvzN, seed + 1, 0.f, 1.f);
  // a‖b, zero-padded 96 -> 128 by the GEMV; the kernel reads a at h, b at 48+h.
  in.ab = random_f32(size_t(M) * kAbStride, seed + 2, 0.f, 1.f);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t j = 96; j < kAbStride; ++j) in.ab[size_t(m) * kAbStride + j] = 0.f;

  // The layer's GDN small block: conv[10240][4] fp32 at 0, negA[48] at 163840,
  // dt_bias[48] at 164032 (loader/small_layout.h). The gated norm at 164224 is
  // bf16 and belongs to prep_gated_head - filled here, never read by this kernel.
  in.small = uniform_f32(kSmallFloats, seed + 3, -0.5f, 0.5f);
  std::mt19937 rng(seed + 4);
  std::uniform_real_distribution<float> alog(-4.0f, 0.5f), dtb(-1.0f, 1.0f);
  for (uint32_t h = 0; h < kHeads; ++h) {
    in.small[gdn_ref::kNegAOff + h] = -std::exp(alog(rng));   // the loader's NegExpFp32 bake
    in.small[gdn_ref::kDtBiasOff + h] = dtb(rng);
  }
  // Every ring slot seeded, so an out-of-window read would show up as noise.
  in.ring0 = random_bf16(kRingElems, seed + 5, -2.f, 2.f);
  in.state0 = random_f32(kStateElems, seed + 6, 0.f, 0.5f);
  return in;
}

struct Run {
  std::vector<uint16_t> ring;
  std::vector<float> state, o;
};

// The device side of one case: buffers, the kernel, and a closed one-launch list
// bound once. `go()` re-uploads the read-modify-write buffers (the ring and the
// state) and replays it, so two calls are a determinism test rather than two
// steps of the recurrence.
struct Bound {
  Dev& d;
  size_t o_elems, ring_elems, state_elems;
  l0::Mem ctrl_mem, qbuf, abuf, sbuf, rbuf, stbuf, obuf;
  l0::Module mod;
  l0::Kernel k;
  l0::CmdList list;

  Bound(Dev& dev, const Inputs& in, uint32_t pos, uint32_t n_act)
      : d(dev),
        o_elems(size_t(in.M) * kHeads * kDim),
        ring_elems(in.ring0.size()),
        state_elems(in.state0.size()),
        ctrl_mem(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control)),
        qbuf(d.ctx, l0::MemKind::Device, in.qkvz.size() * 4),
        abuf(d.ctx, l0::MemKind::Device, in.ab.size() * 4),
        sbuf(d.ctx, l0::MemKind::Device, in.small.size() * 4),
        rbuf(d.ctx, l0::MemKind::Device, ring_elems * 2),
        stbuf(d.ctx, l0::MemKind::Device, state_elems * 4),
        obuf(d.ctx, l0::MemKind::Device, o_elems * 4),
        mod(d.ctx, kernels::path(kernels::gdn_step_variant(in.M))),
        k(mod, "gdn_step"),
        list(l0::CmdList::regular(d.ctx)) {
    runtime::Control* ctrl = ctrl_mem.as<runtime::Control>();
    *ctrl = runtime::Control{};
    ctrl->pos = pos;
    ctrl->n_active = n_act;
    d.imm.copy(qbuf.ptr(), in.qkvz.data(), in.qkvz.size() * 4);
    d.imm.copy(abuf.ptr(), in.ab.data(), in.ab.size() * 4);
    d.imm.copy(sbuf.ptr(), in.small.data(), in.small.size() * 4);
    k.group_size(kWG);
    k.arg_ptr(0, ctrl_mem.ptr());
    k.arg_ptr(1, qbuf.ptr());
    k.arg_ptr(2, abuf.ptr());
    k.arg_ptr(3, sbuf.ptr());
    k.arg_ptr(4, rbuf.ptr());
    k.arg_ptr(5, stbuf.ptr());
    k.arg_ptr(6, obuf.ptr());
    list.launch(k, kHeads, kChunks);
    list.close();
  }

  Run go(const std::vector<uint16_t>& ring_in, const std::vector<float>& state_in) {
    d.imm.copy(rbuf.ptr(), ring_in.data(), ring_elems * 2);
    d.imm.copy(stbuf.ptr(), state_in.data(), state_elems * 4);
    d.imm.fill(obuf.ptr(), 0u, o_elems * 4);
    d.q.execute(list, &d.fence);
    d.fence.wait();
    Run r;
    r.ring.resize(ring_elems);
    r.state.resize(state_elems);
    r.o.resize(o_elems);
    d.imm.copy(r.ring.data(), rbuf.ptr(), ring_elems * 2);
    d.imm.copy(r.state.data(), stbuf.ptr(), state_elems * 4);
    d.imm.copy(r.o.data(), obuf.ptr(), o_elems * 4);
    return r;
  }
};

// Build the inputs, run the reference, run the kernel twice from identical
// uploads, and compare against the reference and against itself.
void run_case(Dev& d, uint32_t pos, uint32_t M, uint32_t n_act, uint32_t seed) {
  const Inputs in = make_inputs(M, seed);

  std::vector<uint16_t> ring_ref = in.ring0;
  std::vector<float> state_ref = in.state0, o_ref(size_t(M) * kHeads * kDim, 0.f);
  gdn_ref::step(pos, n_act, M, in.qkvz.data(), in.ab.data(), in.small.data(), ring_ref.data(),
                state_ref.data(), o_ref.data());

  Bound b(d, in, pos, n_act);
  const Run r0 = b.go(in.ring0, in.state0);
  const Run r1 = b.go(in.ring0, in.state0);

  size_t ring_written = 0;
  for (size_t i = 0; i < in.ring0.size(); ++i)
    if (ring_ref[i] != in.ring0[i]) ++ring_written;

  const Err es = compare_f32(r0.state, state_ref);
  const Err eo = compare_f32(r0.o, o_ref);
  std::printf("gdn_step pos=%u M=%u n_act=%u: state rel %.3e (>=rms %.3e, max abs %.3e), "
              "gdn_o rel %.3e (>=rms %.3e, max abs %.3e), ring %zu words rewritten\n",
              pos, M, n_act, es.worst, es.plain, es.max_abs, eo.worst, eo.plain, eo.max_abs,
              ring_written);
  require(es, 1e-5, "gdn_step state");
  require(eo, 1e-3, "gdn_step gdn_o");
  require_bits(r0.ring, ring_ref, "gdn_step conv_ring");

  CHECK(std::memcmp(r0.o.data(), r1.o.data(), o_ref.size() * 4) == 0);
  CHECK(std::memcmp(r0.state.data(), r1.state.data(), in.state0.size() * 4) == 0);
  CHECK(std::memcmp(r0.ring.data(), r1.ring.data(), in.ring0.size() * 2) == 0);
  std::printf("gdn_step pos=%u: replay bitwise identical (gdn_o, state, conv_ring)\n", pos);
}

// Which ring slots the conv window actually reads, asked of the device alone.
// At pos = 5 the window is slots 2, 3, 4 and the write is slot 5, so:
//   * blanking 2..4 must move `gdn_o` - otherwise the history is being ignored;
//   * blanking everything except 2..4 must leave `gdn_o` bit-identical -
//     otherwise some work-group is reading a slot it has no business in,
//     including the slot this very step writes.
void case_ring_slots(Dev& d, uint32_t pos, uint32_t seed) {
  CHECK(pos >= 3);   // below 3 the window is partly clamped and the split is not clean
  const Inputs in = make_inputs(1, seed);
  Bound b(d, in, pos, 1);

  std::vector<uint16_t> no_window = in.ring0, only_window = in.ring0;
  for (uint32_t slot = 0; slot < kRing; ++slot) {
    const bool in_window = slot == (pos - 3) % kRing || slot == (pos - 2) % kRing ||
                           slot == (pos - 1) % kRing;
    std::vector<uint16_t>& z = in_window ? no_window : only_window;
    for (uint32_t ch = 0; ch < kConvRows; ++ch) z[size_t(slot) * kConvRows + ch] = 0;
  }

  const Run full = b.go(in.ring0, in.state0);
  const Run without = b.go(no_window, in.state0);
  const Run only = b.go(only_window, in.state0);

  const size_t obytes = full.o.size() * 4;
  const bool moved = std::memcmp(full.o.data(), without.o.data(), obytes) != 0;
  const bool same = std::memcmp(full.o.data(), only.o.data(), obytes) == 0;
  std::printf("gdn_step pos=%u slot ownership: blanking slots %u,%u,%u %s gdn_o; "
              "blanking the other 13 %s it\n",
              pos, (pos - 3) % kRing, (pos - 2) % kRing, (pos - 1) % kRing,
              moved ? "moves" : "does NOT move", same ? "leaves" : "MOVES");
  CHECK(moved);
  CHECK(same);
}

}  // namespace

int main() {
  Dev d;
  // pos = 0: the conv's history is entirely below position zero, so all three
  // older taps must contribute 0 no matter what the ring holds.
  run_case(d, 0, 1, 1, 1000);
  // pos = 5: the history is ring slots 2, 3, 4 and the write lands in slot 5.
  run_case(d, 5, 1, 1, 2000);
  case_ring_slots(d, 5, 3000);
  // pos = 1: the clamp is partial - positions -2 and -1 contribute zeros, but
  // position 0 is a real ring slot, so the `p < 0` guard has to be per tap.
  run_case(d, 1, 1, 1, 5000);
  // pos = 15, M = 2: the write wraps (slots 15 and 0) while the window reads
  // slots 12, 13, 14 - the disjointness argument at the one place it could bite.
  run_case(d, 15, 2, 2, 6000);
  // M = 2 is compiled for spec 1 §9's M-loop rule; running it costs nothing and
  // is the ONLY cover for the conv window's intra-step path - at m = 1 exactly
  // one of the three older taps (position 5, token 0's raw value) comes from
  // this step rather than the ring; positions 3 and 4 are still ring slots.
  run_case(d, 5, 2, 2, 4000);
  std::puts("gdn_step_test OK");
  return 0;
}
