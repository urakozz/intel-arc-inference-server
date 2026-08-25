// attn.cl vs the CPU reference in attn_ref.h - the decode-attention trio that
// serves the model's 16 full-attention layers: `attn_prep` (norm + RoPE + the
// KV write), `attn_decode` (one flash-decode block per work-group, with the
// device-side early-out that makes a `max_len`-sized grid affordable) and
// `attn_reduce` (merge the blocks, divide, gate).
//
// The reference is the *same op chain in the same order* as the kernels - the
// norm tree, the score dot's lane tree, the online-softmax wave, the ascending
// block merge (attn_ref.h's header states all four, and attn.cl repeats them
// verbatim). What is left between host and device is therefore exactly the one
// libm function OpenCL does not require to be correctly rounded and this trio
// uses: **`exp`, 3 ulp** - in the softmax, in the block merge and in the final
// `sigmoid(gate)`. `1.0f/sqrt` is not among them (every kernel builds with
// `-cl-fp32-correctly-rounded-divide-sqrt`), which is why the norm, the RoPE
// and the KV cache are held to **bit-exact** and only `attn_out` and
// `attn_part` carry a tolerance:
//
//   * `kv_k` / `kv_v` - bit-exact over **all** of each cache (4.19 M words at
//     `max_len = 4096`, 16.8 M at 16384), not just the slots this step writes.
//     The untouched slots prove that the only positions `attn_prep` writes are
//     `pos … pos+n_active−1`.
//   * `attn_q` - bit-exact on the pass-through dims 64..255; the ruled bar on
//     the roped dims 0..63 is 2 ulp bf16 (the rotation is a rounded product
//     plus an `fma`, spelled identically on both sides, so it lands bit-exact
//     in practice and the printed number says so).
//   * `attn_part` - relative error ≤ 1e-3 on the finite entries, and **exactly
//     equal bit patterns** where the reference produced ±INF (a block that is
//     entirely beyond the causal bound for one `m` of an M > 1 step).
//   * `attn_out` - two bars, because neither alone is honest about a bf16
//     tensor built by cancellation.
//     **(a) Relative error ≤ 8e-3, floored at the tensor's RMS.** The floor is
//     the deviation from the plan's plain relative bar: `acc/sm` is a weighted
//     average of *signed* v values, so individual dims cancel to near zero and
//     an unfloored ratio there would measure the cancellation, not the kernel.
//     **(b) ≤ 2 bf16 ulp on every element with `|ref| >= rms/8`.** The floor in
//     (a) divides by the RMS, so (a) goes slack on exactly the elements a bf16
//     output can most easily flip - and `attn_out` *is* bf16. **The ulp
//     arithmetic, stated properly, because it is what sizes (a):** bf16 has
//     **7** explicit mantissa bits, so inside a binade [2^k, 2^k+1) the spacing
//     is 2^(k-7) and ONE ulp is between 2^-8 and 2^-7 of the element -
//     **3.9e-3 to 7.8e-3**, the top of that range at the *bottom* of a binade.
//     Two ulp is therefore **7.8e-3 to 1.5625e-2**. So a single
//     round-to-nearest boundary flip on a gated element reaches 7.8e-3 of
//     relative error. **(b) is therefore the
//     arbiter**: if a change ever trips (a), check (b) before believing the
//     kernel broke. The `rms/8` gate is what keeps (b)
//     meaningful - below it a ulp is not a unit of error (one dim cancels to
//     ~6.6e-10 against an RMS of ~0.07, where 3.5e-10 of wobble is 95 ulp of
//     nothing). The **2** is the arithmetic of the final chain and not a fudge:
//     `rne_bf16(f32(rne_bf16(acc/sm)) · sigmoid_f32(gate))` rounds to bf16
//     twice, and `exp`'s 3 ulp can push a boundary value one ulp at each of
//     those roundings and no further.
//
//     **(a) read 1e-3 until spec 1.5's lever L5, and 1e-3 was never consistent
//     with (b).** One ulp on a gated element reaches 7.8e-3, so (a) at 1e-3
//     could only ever pass while the device happened to reproduce the host bit
//     for bit on every gated element. At `ATTN_BLOCK` 256 it did, at every
//     depth. L5 reassociates the softmax partials (4 waves per block accumulate
//     where 16 did, and `attn_reduce` merges 4x as many), which moves the fp32
//     `acc/sm` in its last bits, and the first gated word to land the far side
//     of a bf16 rounding boundary took (a) with it - **1 ulp, against (b)'s
//     ruled 2**, with `attn_part` agreeing to ~6e-07 (fp32) underneath it. The
//     tripping element was **above** the rms/8 gate in both configurations
//     observed: at the intermediate `ATTN_BLOCK` 128 it was `pos = 254` index
//     504, |ref| in [0.0312, 0.0625) against rms 7.194e-02 and rms/8 8.99e-03;
//     at the shipped 64 it is L16384 `pos = 16383`, |ref| in [0.00195, 0.00391)
//     against rms 9.45e-03 and rms/8 1.18e-03. So the warrant is the
//     contradiction with (b), NOT the "below the gate a ulp is not a unit of
//     error" argument - these words carry real signal.
//
//     **(a) is now 8e-3 = 2^-7, one gated ulp at its worst. That narrows the
//     contradiction ~8x; it does NOT remove it.** A legitimate 2-ulp gated
//     difference lands anywhere in (7.8e-3, 1.5625e-2], so one in the residual
//     band **(8e-3, 1.5625e-2] would still trip (a) while (b) passes**. Nothing
//     measured is within 5x of that band (worst (a) anywhere is the L16384
//     case's 1.615e-3), and (b) is the arbiter, so the bar stays at 8e-3 rather
//     than being raised to 1.5625e-2 where it would stop covering anything.
//     (a)'s remaining job is the elements BELOW (b)'s `rms/8` gate, where it
//     still catches any absolute error over 2 ulp of the RMS.
//
// Eight depths pin the block edges and the early-out at `max_len = 4096`
// (64 blocks of `attn_ref::kBlock` = 64 - spec 1.5's lever L5 quartered the
// block; every geometry below is stated against the CURRENT `kBlock`, and the
// three cases marked (L5) were added because the retile moved the block edge
// out from under the ones that were here):
//
//   * `pos = 0`   - one valid position in the whole cache; 63 of the 64 blocks
//                   early-out, and the softmax's `mx` starts at −INF with a
//                   single finite score to find.
//   * `pos = 63`  - **(L5)** block 0 exactly full at the NEW block size: the
//                   last position of a block is the last valid one, and block 1
//                   must still early-out. This is what `pos = 255` used to test
//                   and no longer does.
//   * `pos = 127` - **(L5)** the second block exactly full - two live blocks
//                   and a two-step merge where 256-blocking had one block and
//                   no merge at all.
//   * `pos = 129` - **(L5)** the odd geometry the retile creates: three live
//                   blocks where 256-blocking had one, the third holding
//                   exactly TWO valid positions (wave 0, subgroups 0 and 1) and
//                   3 empty waves - the ragged tail `attn_reduce` now merges
//                   four times as often as it used to.
//   * `pos = 254` - block 3 partially valid, the 255th position masked.
//   * `pos = 255` - block 3 exactly full and block 4 early-outs. (Still a block
//                   edge after the retile: 256 = 4·64.)
//   * `pos = 256` - the first five-block case, block 4 holding exactly ONE
//                   valid position (its wave 0, subgroup 0) and 3 empty waves.
//   * `pos = 4095`- the cache full to `max_len`: all 64 blocks live, `nb = 64`
//                   merge steps, and the last position of the last block.
//
// Two M = 2 cases are **run**, not merely compiled (spec 1 §9 asks only that the
// M loop compile; the gdn_step precedent is to run it when it covers something
// nothing else does). They cover per-`m` causal masking, which no M = 1 case
// can reach because at M = 1 the causal bound is the same for every position in
// flight:
//
//   * `pos = 254, n_active = 2` - **inside one block**: at `m = 0` position 255
//     is masked, at `m = 1` it is valid, and both sit in block 3 (192…255), so
//     that block must produce two different partials for the same (q-head,
//     block).
//   * `pos = 255, n_active = 2` - **across the block edge**: block 4 is entirely
//     beyond `m = 0`'s bound (it writes `−INF, 0, 0` and `attn_reduce` must not
//     read it, since `nb(0) = 4`) while for `m = 1` it holds the one valid
//     position 256.
//
// One case runs at `max_len = 16384` - `pos = 16383, M = 1`. `MAXLEN` is a
// compile-time `-D` (attn_part's block stride and `NBLOCKS` both come from it),
// so L16384 is a **different binary**, and it is the one the captured decode
// list binds and the product ships; every case above tests a variant nothing
// runs in anger. One depth, at the far end of the cache, is what makes that
// binary's own arithmetic true: **256** live blocks and a 256-step merge in
// `attn_reduce`, four times the longest any L4096 case reaches - and, since
// L5, the case that proves `attn_reduce`'s `hmx[NBLOCKS]`/`hsm[NBLOCKS]` SLM
// arrays are still correct at four times their old length (256 floats each,
// 2 KB together of the 64 KB budget). It costs 67 MB of KV on the device and a
// few seconds of reference on the host, which is why it is one case and not a
// second sweep.
//
// The early-out is asserted directly rather than inferred: `attn_part` is
// canary-filled (1e30 in every word) *inside the replayed list*, so any block
// whose `kBlock·b ≥ pos + n_active` must come back still holding the canary - and
// `attn_out` matching the reference is what proves `attn_reduce` never read one
// of those blocks, since merging a 1e30 header would drive the output to 1.
//
// Every case replays its launch from freshly re-uploaded inputs and requires
// all six outputs to be bitwise identical - the determinism the captured decode
// list rests on.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "attn_ref.h"
#include "check.h"
#include "common/bf16.h"
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

using attn_ref::kBlock;
using attn_ref::kHeadDim;
using attn_ref::kKvHeads;
using attn_ref::kOutN;
using attn_ref::kPartStride;
using attn_ref::kQHeads;
using attn_ref::kQkvN;
using attn_ref::kRotDim;

constexpr uint32_t kWG = 256;
// Two compiled caches are exercised. `kMaxLen` is the cheap one every case
// below the last one uses: 64 blocks of 64, a 4 MB KV cache, a whole sweep of
// block edges for the price of one. `kProdLen` is the one the *product* runs -
// b70-decode's default `--max-len` and the only attention variant
// runtime::build binds (tests/CMakeLists.txt's B70_DECODE_LIST_KERNELS) - and it
// is a different compiled kernel, not the same kernel with a bigger grid:
// MAXLEN is a `-D` that sets attn_part's block stride and NBLOCKS. Running only
// L4096 left the shipped binary untested, so the last case below runs L16384
// once, at the far end of its cache (67 MB of KV, 6.3 MB of attn_part).
constexpr uint32_t kMaxLen = 4096;
constexpr uint32_t kProdLen = 16384;
constexpr uint32_t kFaSmallFloats = 2048 / 4;       // loader::kFaBlockBytes / 4
constexpr float kCanary = 1.0e30f;

constexpr uint32_t blocks_of(uint32_t max_len) { return max_len / kBlock; }
constexpr size_t kv_elems_of(uint32_t max_len) {
  return size_t(max_len) * kKvHeads * kHeadDim;
}

uint32_t canary_bits() {
  uint32_t u;
  const float f = kCanary;
  std::memcpy(&u, &f, 4);
  return u;
}

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

// The loader's RoPE table (src/loader/loader.cc, `rope_table`), recomputed here
// so the test does not need a checkpoint: cos/sin[p][0..1][i] for the 64 rotary
// dims of head_dim 256, theta 1e7, angles in double and stored float.
std::vector<float> rope_table(uint32_t max_len) {
  const uint32_t half = kRotDim / 2;
  std::vector<float> t(size_t(max_len) * 2 * half);
  std::vector<double> inv(half);
  for (uint32_t i = 0; i < half; ++i) inv[i] = std::pow(1e7, -2.0 * double(i) / double(kRotDim));
  for (uint32_t p = 0; p < max_len; ++p)
    for (uint32_t i = 0; i < half; ++i) {
      const double a = double(p) * inv[i];
      t[(size_t(p) * 2 + 0) * half + i] = float(std::cos(a));
      t[(size_t(p) * 2 + 1) * half + i] = float(std::sin(a));
    }
  return t;
}

struct Err {
  double worst = 0.0;    // max |d| / max(|ref|, rms) - the asserted number
  double plain = 0.0;    // max |d| / |ref| over the elements with |ref| >= rms
  double max_abs = 0.0;
  double rms = 0.0;
  size_t worst_i = 0;
  size_t counted = 0;
};

// Relative error floored at the tensor's own RMS, as in gdn_step_test: `acc` is
// a weighted sum of signed v values and individual dims can land arbitrarily
// close to zero by cancellation, where an unfloored ratio would measure the
// cancellation rather than the kernel. Non-finite reference entries are not
// compared here - `require_inf_bits` handles them.
Err compare_f32(const std::vector<float>& got, const std::vector<float>& ref) {
  CHECK_EQ(got.size(), ref.size());
  Err e;
  double sq = 0.0;
  size_t n = 0;
  for (float r : ref)
    if (std::isfinite(r)) { sq += double(r) * double(r); ++n; }
  e.rms = n ? std::sqrt(sq / double(n)) : 0.0;
  const double floor = e.rms > 0.0 ? e.rms : 1.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (!std::isfinite(ref[i])) continue;
    // Where the reference is finite the device must be too. Without this a NaN
    // or an Inf in `got` would sail through every ratio below: `fabs(NaN - r)`
    // is NaN and `NaN > e.worst` is false, so the worst-error scan would simply
    // decline to see it. The one thing a softmax kernel is most likely to get
    // wrong is exactly a NaN (exp(-INF - -INF)), so this is not a formality.
    CHECK(std::isfinite(got[i]));
    ++e.counted;
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
    std::fprintf(stderr,
                 "%s: relative error %.3e > tol %.1e at index %zu (max abs %.3e, rms %.3e)\n", what,
                 e.worst, tol, e.worst_i, e.max_abs, e.rms);
  CHECK(e.worst <= tol);
}

// Where the reference is non-finite (a block wholly beyond one m's causal
// bound writes mx = -INF), the device must produce the identical bit pattern.
void require_inf_bits(const std::vector<float>& got, const std::vector<float>& ref,
                      const char* what) {
  size_t bad = 0, n = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (std::isfinite(ref[i])) continue;
    ++n;
    uint32_t a, b;
    std::memcpy(&a, &got[i], 4);
    std::memcpy(&b, &ref[i], 4);
    if (a == b) continue;
    if (bad < 3)
      std::fprintf(stderr, "%s: index %zu: got 0x%08X ref 0x%08X\n", what, i, a, b);
    ++bad;
  }
  if (bad) std::fprintf(stderr, "%s: %zu/%zu non-finite entries differ\n", what, bad, n);
  CHECK_EQ(bad, size_t{0});
}

// bf16 as a monotone integer key, so "adjacent representable value" is a
// difference of 1: sign-magnitude reflected about zero.
int32_t bf16_key(uint16_t u) {
  return (u & 0x8000u) ? -int32_t(u & 0x7FFFu) : int32_t(u);
}

// The second `attn_out` bar, in the output's own dtype: **no element carrying
// real signal may sit more than `bar` bf16 ulp from the reference.**
//
// "Carrying real signal" is `|ref| >= rms/8`, and the gate is what makes the
// bar meaningful rather than either vacuous or false. `acc/sm` is a weighted
// average of *signed* v values, so individual dims cancel; one of them lands at
// ~6.6e-10 against a tensor RMS of ~0.07, where a 3.5e-10 wobble is 95 bf16 ulp
// of absolute nothing. Below the gate a ulp is not a unit of error. Above it,
// it is the only unit that means anything, because the floored relative bar
// divides by the RMS and so goes slack on exactly the elements a bf16 output
// can most easily flip.
//
// `bar` is **2**, and that is the arithmetic of the final chain rather than a
// fudge: `rne_bf16(f32(rne_bf16(acc/sm)) · sigmoid_f32(gate))` rounds to bf16
// **twice**, and `exp`'s 3 ulp of fp32 slack can push a boundary value one ulp
// at each of those roundings and no further. Three would mean an arithmetic
// difference, not a rounding cascade.
//
// The all-element count and worst distance are gathered too, and printed rather
// than asserted on - they say how *localised* a disagreement is (typically one
// word in 6144), which no aggregate can.
struct UlpErr {
  size_t off = 0;          // words differing at all
  int32_t worst = 0;       // worst ulp distance anywhere
  size_t checked = 0;      // elements at or above the rms/8 gate
  int32_t worst_gated = 0; // worst ulp distance among those - the asserted number
};

UlpErr require_ulp(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref, double rms,
                   int32_t bar, const char* what) {
  CHECK_EQ(got.size(), ref.size());
  UlpErr e;
  const double gate = rms / 8.0;
  size_t bad = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const bool signal = std::fabs(double(common::bf16_to_f32(ref[i]))) >= gate;
    if (signal) ++e.checked;
    if (got[i] == ref[i]) continue;
    ++e.off;
    int32_t dk = bf16_key(got[i]) - bf16_key(ref[i]);
    if (dk < 0) dk = -dk;
    if (dk > e.worst) e.worst = dk;
    if (!signal) continue;
    if (dk > e.worst_gated) e.worst_gated = dk;
    if (dk <= bar) continue;
    if (bad < 3)
      std::fprintf(stderr, "%s: element %zu: got 0x%04X ref 0x%04X (%d bf16 ulp, |ref| %.3e >= "
                           "rms/8 %.3e)\n",
                   what, i, got[i], ref[i], dk, std::fabs(double(common::bf16_to_f32(ref[i]))),
                   gate);
    ++bad;
  }
  if (bad)
    std::fprintf(stderr, "%s: %zu of the %zu words at |ref| >= rms/8 are more than %d bf16 ulp "
                         "out\n",
                 what, bad, e.checked, bar);
  CHECK_EQ(bad, size_t{0});
  return e;
}

void require_bits16(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref,
                    const char* what) {
  CHECK_EQ(got.size(), ref.size());
  size_t bad = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (got[i] == ref[i]) continue;
    if (bad < 3)
      std::fprintf(stderr, "%s: element %zu: got 0x%04X ref 0x%04X\n", what, i, got[i], ref[i]);
    ++bad;
  }
  if (bad) std::fprintf(stderr, "%s: %zu/%zu words differ\n", what, bad, ref.size());
  CHECK_EQ(bad, size_t{0});
}

void require_bits32(const std::vector<float>& got, const std::vector<float>& ref, size_t begin,
                    size_t end, size_t stride, size_t lo, size_t hi, const char* what) {
  size_t bad = 0, n = 0;
  for (size_t base = begin; base < end; base += stride)
    for (size_t d = lo; d < hi; ++d) {
      const size_t i = base + d;
      ++n;
      uint32_t a, b;
      std::memcpy(&a, &got[i], 4);
      std::memcpy(&b, &ref[i], 4);
      if (a == b) continue;
      if (bad < 3)
        std::fprintf(stderr, "%s: index %zu (dim %zu): got %.9g ref %.9g\n", what, i, d,
                     double(got[i]), double(ref[i]));
      ++bad;
    }
  if (bad) std::fprintf(stderr, "%s: %zu/%zu words differ\n", what, bad, n);
  CHECK_EQ(bad, size_t{0});
}

// The ruled bar for the roped dims: 2 ulp of bf16, i.e. 2 * 2^-8 relative.
double roped_rel(const std::vector<float>& got, const std::vector<float>& ref, uint32_t M) {
  double worst = 0.0;
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t h = 0; h < kQHeads; ++h)
      for (uint32_t d = 0; d < kRotDim; ++d) {
        const size_t i = (size_t(m) * kQHeads + h) * kHeadDim + d;
        const double a = std::fabs(double(ref[i]));
        if (a == 0.0) continue;
        const double r = std::fabs(double(got[i]) - double(ref[i])) / a;
        if (r > worst) worst = r;
      }
  return worst;
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
  uint32_t max_len = kMaxLen;
  std::vector<float> partials, fa_small, rope;
  std::vector<uint16_t> kv_k0, kv_v0;
};

Inputs make_inputs(uint32_t M, uint32_t seed, uint32_t max_len) {
  Inputs in;
  in.M = M;
  in.max_len = max_len;
  // The fused qkv GEMV's partials (S = 1). sigma 1 keeps the pre-norm head
  // vectors at a realistic scale; the norm removes it anyway.
  in.partials = random_f32(size_t(M) * kQkvN, seed + 1, 0.f, 1.f);
  // The layer's FA small block: q_norm fp32 (1+w)[256] at float 0, k_norm at
  // float 256 (loader/small_layout.h). The loader bakes 1 + w, so centre on 1.
  in.fa_small = uniform_f32(kFaSmallFloats, seed + 2, 0.75f, 1.25f);
  in.rope = rope_table(max_len);
  // The whole KV cache is seeded, not just the prefix: a read past the causal
  // bound would then show up as noise rather than as a convenient zero, and
  // the untouched-slot half of the bit-exact bar has something to prove.
  in.kv_k0 = random_bf16(kv_elems_of(max_len), seed + 3, -2.f, 2.f);
  in.kv_v0 = random_bf16(kv_elems_of(max_len), seed + 4, -2.f, 2.f);
  return in;
}

struct Run {
  std::vector<float> attn_q, attn_gate, attn_part;
  std::vector<uint16_t> kv_k, kv_v, attn_out;
};

// The device side of one case: buffers, the three kernels, and a closed list
// bound once - canary fill, prep, decode, reduce. The list is IN_ORDER
// (l0::CmdList::regular), so `attn_decode` sees `attn_prep`'s KV writes and
// `attn_reduce` sees `attn_decode`'s partials without any explicit barrier.
// `go()` re-uploads the read-modify-write buffers (the two KV caches) and
// replays, so two calls are a determinism test rather than two decode steps.
struct Bound {
  Dev& d;
  uint32_t M, max_len, n_blocks;
  size_t kv_elems, part_elems, qg_elems, out_elems;
  l0::Mem ctrl_mem, pbuf, sbuf, ropebuf, qbuf, gbuf, kbuf, vbuf, partbuf, obuf;
  l0::Module mod_prep, mod_dec, mod_red;
  l0::Kernel k_prep, k_dec, k_red;
  l0::CmdList list;

  Bound(Dev& dev, const Inputs& in, uint32_t pos, uint32_t n_act)
      : d(dev),
        M(in.M),
        max_len(in.max_len),
        n_blocks(blocks_of(in.max_len)),
        kv_elems(kv_elems_of(in.max_len)),
        part_elems(size_t(kQHeads) * n_blocks * in.M * kPartStride),
        qg_elems(size_t(in.M) * kQHeads * kHeadDim),
        out_elems(size_t(in.M) * kOutN),
        ctrl_mem(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control)),
        pbuf(d.ctx, l0::MemKind::Device, in.partials.size() * 4),
        sbuf(d.ctx, l0::MemKind::Device, in.fa_small.size() * 4),
        ropebuf(d.ctx, l0::MemKind::Device, in.rope.size() * 4),
        qbuf(d.ctx, l0::MemKind::Device, qg_elems * 4),
        gbuf(d.ctx, l0::MemKind::Device, qg_elems * 4),
        kbuf(d.ctx, l0::MemKind::Device, kv_elems * 2),
        vbuf(d.ctx, l0::MemKind::Device, kv_elems * 2),
        partbuf(d.ctx, l0::MemKind::Device, part_elems * 4),
        obuf(d.ctx, l0::MemKind::Device, out_elems * 2),
        mod_prep(d.ctx, kernels::path(kernels::attn_prep_variant(in.M))),
        mod_dec(d.ctx, kernels::path(kernels::attn_decode_variant(in.M, in.max_len, kBlock))),
        mod_red(d.ctx, kernels::path(kernels::attn_reduce_variant(in.M, in.max_len, kBlock))),
        k_prep(mod_prep, "attn_prep"),
        k_dec(mod_dec, "attn_decode"),
        k_red(mod_red, "attn_reduce"),
        list(l0::CmdList::regular(d.ctx)) {
    runtime::Control* ctrl = ctrl_mem.as<runtime::Control>();
    *ctrl = runtime::Control{};
    ctrl->pos = pos;
    ctrl->n_active = n_act;
    d.imm.copy(pbuf.ptr(), in.partials.data(), in.partials.size() * 4);
    d.imm.copy(sbuf.ptr(), in.fa_small.data(), in.fa_small.size() * 4);
    d.imm.copy(ropebuf.ptr(), in.rope.data(), in.rope.size() * 4);

    k_prep.group_size(kWG);
    k_prep.arg_ptr(0, ctrl_mem.ptr());
    k_prep.arg_ptr(1, pbuf.ptr());
    k_prep.arg_ptr(2, sbuf.ptr());
    k_prep.arg_ptr(3, ropebuf.ptr());
    k_prep.arg_ptr(4, qbuf.ptr());
    k_prep.arg_ptr(5, gbuf.ptr());
    k_prep.arg_ptr(6, kbuf.ptr());
    k_prep.arg_ptr(7, vbuf.ptr());

    k_dec.group_size(kWG);
    k_dec.arg_ptr(0, ctrl_mem.ptr());
    k_dec.arg_ptr(1, qbuf.ptr());
    k_dec.arg_ptr(2, kbuf.ptr());
    k_dec.arg_ptr(3, vbuf.ptr());
    k_dec.arg_ptr(4, partbuf.ptr());

    k_red.group_size(kWG);
    k_red.arg_ptr(0, ctrl_mem.ptr());
    k_red.arg_ptr(1, partbuf.ptr());
    k_red.arg_ptr(2, gbuf.ptr());
    k_red.arg_ptr(3, obuf.ptr());

    // The canary is inside the list so every replay starts from it.
    list.fill(partbuf.ptr(), canary_bits(), part_elems * 4);
    list.fill(qbuf.ptr(), 0u, qg_elems * 4);
    list.fill(gbuf.ptr(), 0u, qg_elems * 4);
    list.fill(obuf.ptr(), 0u, out_elems * 2);
    list.launch(k_prep, kQHeads + kKvHeads, M);
    list.launch(k_dec, kKvHeads, n_blocks);
    list.launch(k_red, kQHeads, M);
    list.close();
  }

  Run go(const Inputs& in) {
    d.imm.copy(kbuf.ptr(), in.kv_k0.data(), kv_elems * 2);
    d.imm.copy(vbuf.ptr(), in.kv_v0.data(), kv_elems * 2);
    d.q.execute(list, &d.fence);
    d.fence.wait();
    Run r;
    r.attn_q.resize(qg_elems);
    r.attn_gate.resize(qg_elems);
    r.attn_part.resize(part_elems);
    r.kv_k.resize(kv_elems);
    r.kv_v.resize(kv_elems);
    r.attn_out.resize(out_elems);
    d.imm.copy(r.attn_q.data(), qbuf.ptr(), qg_elems * 4);
    d.imm.copy(r.attn_gate.data(), gbuf.ptr(), qg_elems * 4);
    d.imm.copy(r.attn_part.data(), partbuf.ptr(), part_elems * 4);
    d.imm.copy(r.kv_k.data(), kbuf.ptr(), kv_elems * 2);
    d.imm.copy(r.kv_v.data(), vbuf.ptr(), kv_elems * 2);
    d.imm.copy(r.attn_out.data(), obuf.ptr(), out_elems * 2);
    return r;
  }
};

// Build the inputs, run the reference, run the trio twice from identical
// uploads, and compare against the reference and against itself.
void run_case(Dev& d, uint32_t pos, uint32_t M, uint32_t n_act, uint32_t seed,
              uint32_t max_len = kMaxLen) {
  CHECK(pos + n_act <= max_len);
  const uint32_t n_blocks = blocks_of(max_len);
  const size_t kv_elems = kv_elems_of(max_len);
  const Inputs in = make_inputs(M, seed, max_len);

  const size_t qg = size_t(M) * kQHeads * kHeadDim;
  const size_t part_elems = size_t(kQHeads) * n_blocks * M * kPartStride;
  std::vector<float> q_ref(qg, 0.f), g_ref(qg, 0.f), part_ref(part_elems, kCanary);
  std::vector<uint16_t> k_ref = in.kv_k0, v_ref = in.kv_v0, out_ref(size_t(M) * kOutN, 0);
  attn_ref::prep(pos, n_act, M, in.partials.data(), in.fa_small.data(), in.rope.data(),
                 q_ref.data(), g_ref.data(), k_ref.data(), v_ref.data());
  attn_ref::decode(pos, n_act, M, max_len, q_ref.data(), k_ref.data(), v_ref.data(),
                   part_ref.data());
  attn_ref::reduce(pos, n_act, M, max_len, part_ref.data(), g_ref.data(), out_ref.data());

  Bound b(d, in, pos, n_act);
  const Run r0 = b.go(in);
  const Run r1 = b.go(in);

  // attn_q: bit-exact on the pass-through dims, the ruled 2 ulp bf16 bar on the
  // roped ones. attn_gate is a pure rounding, so it is bit-exact everywhere.
  require_bits32(r0.attn_q, q_ref, 0, qg, kHeadDim, kRotDim, kHeadDim, "attn_q pass-through dims");
  const double q_rel = roped_rel(r0.attn_q, q_ref, M);
  require_bits32(r0.attn_gate, g_ref, 0, qg, kHeadDim, 0, kHeadDim, "attn_gate");
  require_bits16(r0.kv_k, k_ref, "kv_k");
  require_bits16(r0.kv_v, v_ref, "kv_v");

  // The early-out, asserted directly: a block whose start is at or beyond
  // pos + n_active must never have been written, so every one of its 258 words
  // is still the canary this list filled in.
  size_t canary_blocks = 0, canary_words = 0;
  for (uint32_t h = 0; h < kQHeads; ++h)
    for (uint32_t blk = 0; blk < n_blocks; ++blk) {
      if (blk * kBlock < pos + n_act) continue;
      ++canary_blocks;
      for (uint32_t m = 0; m < M; ++m) {
        const size_t base = (size_t(h) * n_blocks + blk) * M * kPartStride + size_t(m) * kPartStride;
        for (uint32_t w = 0; w < kPartStride; ++w) {
          CHECK(r0.attn_part[base + w] == kCanary);
          ++canary_words;
        }
      }
    }

  // The written blocks only: the canary is 1e30 and would swamp the RMS floor
  // (and prove nothing - the loop above already compared it exactly).
  std::vector<float> part_got, part_want;
  for (uint32_t h = 0; h < kQHeads; ++h)
    for (uint32_t blk = 0; blk < n_blocks; ++blk) {
      if (blk * kBlock >= pos + n_act) continue;
      for (uint32_t m = 0; m < M; ++m) {
        const size_t base = (size_t(h) * n_blocks + blk) * M * kPartStride + size_t(m) * kPartStride;
        for (uint32_t w = 0; w < kPartStride; ++w) {
          part_got.push_back(r0.attn_part[base + w]);
          part_want.push_back(part_ref[base + w]);
        }
      }
    }
  const Err ep = compare_f32(part_got, part_want);
  require_inf_bits(part_got, part_want, "attn_part (-INF entries)");
  require(ep, 1e-3, "attn_part");

  std::vector<float> out_f(r0.attn_out.size()), outr_f(out_ref.size());
  for (size_t i = 0; i < out_f.size(); ++i) {
    out_f[i] = common::bf16_to_f32(r0.attn_out[i]);
    outr_f[i] = common::bf16_to_f32(out_ref[i]);
  }
  const Err eo = compare_f32(out_f, outr_f);
  const UlpErr eu = require_ulp(r0.attn_out, out_ref, eo.rms, 2, "attn_out");

  std::printf("attn L%u pos=%u M=%u n_act=%u: attn_q roped rel %.3e (pass-through dims "
              "bit-exact), kv_k/kv_v bit-exact (%zu words each), attn_part rel %.3e (>=rms %.3e) "
              "over %zu finite words, attn_out rel %.3e (>=rms %.3e, max abs %.3e), %zu/%zu words "
              "differ (worst %d bf16 ulp anywhere; %d over the %zu at |ref| >= rms/8); %zu blocks "
              "x %u m (%zu words) still canary\n",
              max_len, pos, M, n_act, q_rel, kv_elems, ep.worst, ep.plain, ep.counted, eo.worst,
              eo.plain, eo.max_abs, eu.off, out_ref.size(), eu.worst, eu.worst_gated, eu.checked,
              canary_blocks, M, canary_words);

  CHECK(q_rel <= 2.0 / 256.0);   // 2 ulp of bf16
  // 8e-3 ~ 2^-7: ONE bf16 ulp at its worst (bottom of a binade), which the
  // 2-ulp bar below plainly permits. It is NOT the full 2-ulp reach - that is
  // 2^-6 = 1.5625e-2 - so the residual band (8e-3, 1.5625e-2] can still trip
  // this while `require_ulp` passes; the header argues why that is the right
  // place to stop. Tightening below 7.8e-3 would contradict the 2-ulp bar
  // outright; loosening to 1.5625e-2 would stop it covering the elements under
  // the rms/8 gate, which is the only thing it is still for.
  require(eo, 8e-3, "attn_out");

  CHECK(std::memcmp(r0.attn_q.data(), r1.attn_q.data(), qg * 4) == 0);
  CHECK(std::memcmp(r0.attn_gate.data(), r1.attn_gate.data(), qg * 4) == 0);
  CHECK(std::memcmp(r0.attn_part.data(), r1.attn_part.data(), part_elems * 4) == 0);
  CHECK(std::memcmp(r0.kv_k.data(), r1.kv_k.data(), kv_elems * 2) == 0);
  CHECK(std::memcmp(r0.kv_v.data(), r1.kv_v.data(), kv_elems * 2) == 0);
  CHECK(std::memcmp(r0.attn_out.data(), r1.attn_out.data(), out_ref.size() * 2) == 0);
  std::printf("attn L%u pos=%u M=%u: replay bitwise identical (attn_q, attn_gate, attn_part, "
              "kv_k, kv_v, attn_out)\n",
              max_len, pos, M);
}

}  // namespace

int main() {
  Dev d;
  // pos = 0: one valid position in the entire cache; 63 of 64 blocks early-out.
  run_case(d, 0, 1, 1, 1000);
  // pos = 63 / 127 / 129: the block edges lever L5 created by quartering kBlock
  // to 64 - block 0 exactly full with block 1 still early-outing, block 1
  // exactly full (a two-step merge where 256-blocking merged nothing), and then
  // the odd geometry the retile creates: three live blocks where 256-blocking
  // had one, the third holding just two valid positions (wave 0, subgroups 0
  // and 1). None of the three was a block boundary before L5.
  run_case(d, 63, 1, 1, 1300);
  run_case(d, 127, 1, 1, 1500);
  run_case(d, 129, 1, 1, 1700);
  // pos = 254 / 255 / 256: a later block edge from both sides - partially
  // valid, exactly full, and the first five-block case with one valid position
  // in it. 256 = 4·kBlock, so these three still straddle an edge after L5.
  run_case(d, 254, 1, 1, 2000);
  run_case(d, 255, 1, 1, 3000);
  run_case(d, 256, 1, 1, 4000);
  // pos = 4095: the cache full to max_len - 64 live blocks and a 64-step merge.
  run_case(d, 4095, 1, 1, 5000);
  // M = 2 is compiled for spec 1 §9's M-loop rule and RUN here because it is
  // the only cover for per-m causal masking: inside one block at pos = 254
  // (positions 254 and 255 are both in block 3; 255 is masked at m = 0 and
  // valid at m = 1) and across the block edge at pos = 255 (block 4 wholly
  // beyond m = 0's bound, so it writes -INF, 0, 0 and attn_reduce, whose nb(0)
  // is 4, must not read it).
  run_case(d, 254, 2, 2, 6000);
  run_case(d, 255, 2, 2, 7000);
  // L16384 - the ONLY attention variant the captured decode list binds, and
  // until now compiled but never executed (tests/CMakeLists.txt named it as a
  // dependency, which built it and proved nothing). MAXLEN is a compile-time
  // define, so this is a different binary from every case above: a different
  // NBLOCKS (256, not 64) and a different attn_part block stride. One depth,
  // at the far end of the cache, because that is where both differ most -
  // pos = 16383 makes all 256 blocks live and gives attn_reduce a 256-step
  // merge, four times the longest merge any L4096 case can reach - and it is
  // the only case that runs attn_reduce's hmx/hsm SLM arrays at their post-L5
  // length of NBLOCKS = 256 floats each. Same bars as the rest; ~67 MB of KV
  // and 6.3 MB of attn_part on the device.
  run_case(d, kProdLen - 1, 1, 1, 8000, kProdLen);
  std::puts("attn_test OK");
  return 0;
}
