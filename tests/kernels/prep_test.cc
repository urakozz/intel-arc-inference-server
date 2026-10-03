// prep.cl vs the CPU reference in prep_ref.h.
//
// The bar here is **bit-exactness**, not a tolerance. Every op in these kernels
// is a rounded scalar chain (bf16 in, fp32 inside one op, bf16 out) plus one
// fp32 variance tree whose order prep_ref.h reproduces exactly (lane i sums
// k = i, i+WG, … with `fma`, then the pairwise SLM tree 256->128->…->1). So:
//
//   * `prep_res_norm` - `x_out` AND the in-place-updated `resid` must be
//     bit-identical to the reference, at S_PREV = 0 and 16.
//   * `prep_silu_mul` / `prep_gated_head` - everything except the final `silu`
//     factor is likewise exact, but OpenCL allows 3 ulp on fp32 `exp` where the
//     host's `expf` is ~0.5, so the *final* value is compared at **2 bf16 ulp**.
//     To keep the exact bar on the rest of the chain, each of those kernels also
//     runs a case whose silu argument is 30.0f: `exp(-30) ≈ 9.4e-14` is far
//     below `2^-24`, so `1 + exp(-30)` is exactly 1.0f in fp32 on *any*
//     conforming implementation and `silu(30) = 30.0f` on both sides. Those
//     cases are asserted bit-exact, which pins the norm/mul chain (`n_b`, `t_b`,
//     `s_b`) that is otherwise not directly observable.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

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
#include "prep_ref.h"

namespace {

std::vector<float> random_f32(size_t n, uint32_t seed, float mean, float sigma) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(mean, sigma);
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

// bf16 words ordered as integers: the distance between two keys is the number
// of representable bf16 values between them (one "ulp" of the format).
int32_t bf16_key(uint16_t v) {
  return (v & 0x8000u) ? -int32_t(v & 0x7FFFu) : int32_t(v);
}

struct Cmp {
  uint32_t max_ulp = 0;
  size_t exact = 0, n = 0, worst = 0;
};

Cmp compare(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref) {
  Cmp c;
  c.n = ref.size();
  CHECK_EQ(got.size(), ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    if (got[i] == ref[i]) { ++c.exact; continue; }
    int32_t d = bf16_key(got[i]) - bf16_key(ref[i]);
    uint32_t u = uint32_t(d < 0 ? -d : d);
    if (u > c.max_ulp) { c.max_ulp = u; c.worst = i; }
  }
  return c;
}

void require(const Cmp& c, uint32_t tol, const char* what) {
  if (c.max_ulp > tol) {
    std::fprintf(stderr, "%s: max %u bf16 ulp > tol %u at index %zu (%zu/%zu exact)\n", what,
                 c.max_ulp, tol, c.worst, c.exact, c.n);
  }
  CHECK(c.max_ulp <= tol);
}

template <class T>
l0::Mem upload(l0::Context& ctx, l0::CmdList& imm, const std::vector<T>& v) {
  l0::Mem m(ctx, l0::MemKind::Device, v.size() * sizeof(T));
  imm.copy(m.ptr(), v.data(), v.size() * sizeof(T));
  return m;
}

template <class T>
void download(l0::CmdList& imm, std::vector<T>& v, const l0::Mem& m) {
  imm.copy(v.data(), m.ptr(), v.size() * sizeof(T));
}

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence fence{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  // One launch of one kernel, recorded in a regular list and executed once.
  void run(l0::Kernel& k, uint32_t gx, uint32_t gy) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    list.launch(k, gx, gy);
    list.close();
    q.execute(list, &fence);
    fence.wait();
  }

  // Two launches in ONE in-order list, which is how runtime::build appends the
  // two-stage prep pair: stage B must see stage A's `sumsq` and its rewritten
  // `resid`, and the list's in-order flag is the only thing that makes it so.
  // Running them as two lists would test a synchronisation the engine does not
  // use.
  void run2(l0::Kernel& a, uint32_t agx, uint32_t agy, l0::Kernel& b, uint32_t bgx,
            uint32_t bgy) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    list.launch(a, agx, agy);
    list.launch(b, bgx, bgy);
    list.close();
    q.execute(list, &fence);
    fence.wait();
  }
};

// ---------------------------------------------------------------------------

void case_res_norm(Dev& d, uint32_t M, uint32_t K, uint32_t S_PREV) {
  const size_t np = S_PREV ? size_t(S_PREV) * M * K : 1;
  std::vector<float> partials = random_f32(np, 100 + S_PREV, 0.f, 1.f);
  std::vector<uint16_t> resid = random_bf16(size_t(M) * K, 200 + S_PREV, -1.f, 1.f);
  std::vector<float> norm_w(K);
  {
    std::mt19937 rng(300 + S_PREV);
    std::uniform_real_distribution<float> d05(-0.05f, 0.05f);
    for (auto& w : norm_w) w = 1.0f + d05(rng);   // device holds (1 + w), fp32
  }

  std::vector<uint16_t> resid_ref = resid, x_ref(size_t(M) * K, 0);
  prep_ref::res_norm(partials.data(), resid_ref.data(), norm_w.data(), x_ref.data(), M, K, S_PREV);

  l0::Mem pbuf = upload(d.ctx, d.imm, partials);
  l0::Mem rbuf = upload(d.ctx, d.imm, resid);
  l0::Mem wbuf = upload(d.ctx, d.imm, norm_w);
  l0::Mem xbuf(d.ctx, l0::MemKind::Device, size_t(M) * K * 2);
  l0::Module mod(d.ctx, kernels::path(kernels::prep_res_norm_variant(M, K, S_PREV)));
  l0::Kernel k = mod.kernel("prep_res_norm");
  k.group_size(256);
  k.arg_ptr(0, pbuf.ptr());
  k.arg_ptr(1, rbuf.ptr());
  k.arg_ptr(2, wbuf.ptr());
  k.arg_ptr(3, xbuf.ptr());
  d.run(k, 1, M);

  std::vector<uint16_t> x_got(size_t(M) * K), resid_got(size_t(M) * K);
  download(d.imm, x_got, xbuf);
  download(d.imm, resid_got, rbuf);

  Cmp cx = compare(x_got, x_ref), cr = compare(resid_got, resid_ref);
  std::printf("prep_res_norm M=%u K=%u SP=%u: x_out %zu/%zu exact (max %u ulp), "
              "resid %zu/%zu exact (max %u ulp)\n",
              M, K, S_PREV, cx.exact, cx.n, cx.max_ulp, cr.exact, cr.n, cr.max_ulp);
  require(cx, 0, "prep_res_norm x_out");
  require(cr, 0, "prep_res_norm resid");
}

// ---------------------------------------------------------------------------
// The two-stage prep_res_norm - spec 1.5's lever L1 (prep_res_fold +
// prep_norm_finish). Three bars, and they are not the same bar:
//
//   1. `resid` **bit-exact against the SINGLE-stage reference**. The lever's
//      whole numerics claim is that the per-element fold chain is untouched, and
//      this is that claim: the same s = 0..15 order, the same two RNE steps,
//      just distributed differently. Note what it does NOT buy: this is a
//      per-launch property for one set of inputs, not a promise about the
//      engine's tensors. The reordered Sigma below moves `x` in the last ulp on
//      real rows, so the golden gate's per-layer taps DO move (measured, both
//      directions, docs/14) even though every `resid` write here is exact.
//   2. `x` and `sumsq` **bit-exact against the TWO-stage reference**, which
//      models the new Σ tree (G chunk trees, then ascending g). Nothing in this
//      pair is allowed to be "close": the reference reproduces the tree, so the
//      only reason for a difference would be a transcription bug.
//   3. `x` against the single-stage reference at the **Task-5 ruled bar**: <= 2
//      bf16 ulp wherever |ref| >= rms/8. This is the ONE quantity the lever
//      moves, it moves it by reordering a fp32 sum, and the printed max ulp is
//      the number the report carries. The arbiter for the change is the golden
//      gate, not this tolerance (docs/12, `prep_res_norm` -> Measured).
//
// `norm_wgs` is stage B's own work-group count: 20 is what the runtime binds,
// 1 is the plan's literal single-work-group finish. Both are run and both must
// produce the same bits, because they fold the same G partials in the same
// order - the grid changes who rescales which element, not the arithmetic.
//
// **How that equality is actually established, precisely.** Each grid is
// compared to the SAME host reference and each is required to be bit-exact
// against it, so their equality to each other is *transitive through the
// reference*, not asserted directly. That is strictly as strong (both equal a
// third fixed thing) and it is worth naming, because a reader looking for a
// `CHECK(x_got_W1 == x_got_W20)` will not find one. Note also the coverage
// asymmetry: `W = 1` is exercised at **SP4 only** (main() below), because the
// stage-B kernel does not read `partials` at all and so cannot distinguish the
// prep modes - SP0 and SP4 bind the identical `prep_norm_finish` binary.
void case_res_norm_two_stage(Dev& d, uint32_t M, uint32_t K, uint32_t S_PREV, uint32_t G,
                             uint32_t norm_wgs) {
  const size_t np = S_PREV ? size_t(S_PREV) * M * K : 1;
  // The SAME seeds case_res_norm uses, so the single-stage reference below is
  // literally the reference that kernel is graded against.
  std::vector<float> partials = random_f32(np, 100 + S_PREV, 0.f, 1.f);
  std::vector<uint16_t> resid = random_bf16(size_t(M) * K, 200 + S_PREV, -1.f, 1.f);
  std::vector<float> norm_w(K);
  {
    std::mt19937 rng(300 + S_PREV);
    std::uniform_real_distribution<float> d05(-0.05f, 0.05f);
    for (auto& w : norm_w) w = 1.0f + d05(rng);
  }

  std::vector<uint16_t> resid_one = resid, x_one(size_t(M) * K, 0);
  prep_ref::res_norm(partials.data(), resid_one.data(), norm_w.data(), x_one.data(), M, K, S_PREV);

  std::vector<uint16_t> resid_two = resid, x_two(size_t(M) * K, 0);
  std::vector<float> sumsq_ref(size_t(G) * M, 0.f);
  prep_ref::res_fold(partials.data(), resid_two.data(), sumsq_ref.data(), M, K, S_PREV, G);
  prep_ref::norm_finish(sumsq_ref.data(), resid_two.data(), norm_w.data(), x_two.data(), M, K, G);

  // The reference's own consistency check, before the device is asked anything:
  // the two host paths must agree on `resid` bit for bit, or the claim being
  // tested is false in the reference and the device comparison proves nothing.
  CHECK(resid_one == resid_two);

  l0::Mem pbuf = upload(d.ctx, d.imm, partials);
  l0::Mem rbuf = upload(d.ctx, d.imm, resid);
  l0::Mem wbuf = upload(d.ctx, d.imm, norm_w);
  l0::Mem sbuf(d.ctx, l0::MemKind::Device, size_t(G) * M * sizeof(float));
  l0::Mem xbuf(d.ctx, l0::MemKind::Device, size_t(M) * K * 2);
  l0::Module amod(d.ctx, kernels::path(kernels::prep_res_fold_variant(M, K, S_PREV, G)));
  l0::Kernel ka = amod.kernel("prep_res_fold");
  ka.group_size(256);
  ka.arg_ptr(0, pbuf.ptr());
  ka.arg_ptr(1, rbuf.ptr());
  ka.arg_ptr(2, sbuf.ptr());
  l0::Module bmod(d.ctx, kernels::path(kernels::prep_norm_finish_variant(M, K, G, norm_wgs)));
  l0::Kernel kb = bmod.kernel("prep_norm_finish");
  kb.group_size(256);
  kb.arg_ptr(0, sbuf.ptr());
  kb.arg_ptr(1, rbuf.ptr());
  kb.arg_ptr(2, wbuf.ptr());
  kb.arg_ptr(3, xbuf.ptr());
  d.run2(ka, G, M, kb, norm_wgs, M);

  std::vector<uint16_t> x_got(size_t(M) * K), resid_got(size_t(M) * K);
  std::vector<float> sumsq_got(size_t(G) * M);
  download(d.imm, x_got, xbuf);
  download(d.imm, resid_got, rbuf);
  download(d.imm, sumsq_got, sbuf);

  Cmp cr = compare(resid_got, resid_one);      // bar 1
  Cmp cx2 = compare(x_got, x_two);             // bar 2
  size_t sum_exact = 0;
  for (size_t i = 0; i < sumsq_got.size(); ++i)
    if (sumsq_got[i] == sumsq_ref[i]) ++sum_exact;

  // Bar 3, with the magnitude guard: the rms of the single-stage reference row,
  // over the same K the kernel divides by.
  double sq = 0.0;
  for (size_t i = 0; i < x_one.size(); ++i) {
    const double v = prep_ref::f32(x_one[i]);
    sq += v * v;
  }
  const double rms = std::sqrt(sq / double(x_one.size()));
  const double guard = rms / 8.0;
  uint32_t worst_ulp = 0;
  size_t graded = 0, exact_vs_one = 0;
  for (size_t i = 0; i < x_one.size(); ++i) {
    if (x_got[i] == x_one[i]) ++exact_vs_one;
    if (std::fabs(double(prep_ref::f32(x_one[i]))) < guard) continue;
    ++graded;
    const int32_t diff = bf16_key(x_got[i]) - bf16_key(x_one[i]);
    const uint32_t u = uint32_t(diff < 0 ? -diff : diff);
    if (u > worst_ulp) worst_ulp = u;
  }

  std::printf("prep_res_fold+norm_finish M=%u K=%u SP=%u G=%u W=%u:\n"
              "  resid vs single-stage ref  %zu/%zu exact (max %u ulp)   [bar: bit-exact]\n"
              "  sumsq vs two-stage ref     %zu/%zu exact                [bar: bit-exact]\n"
              "  x     vs two-stage ref     %zu/%zu exact (max %u ulp)   [bar: bit-exact]\n"
              "  x     vs single-stage ref  %zu/%zu exact, max %u ulp over %zu of %zu graded"
              " (|ref| >= rms/8 = %.6g)   [bar: <= 2 ulp]\n",
              M, K, S_PREV, G, norm_wgs, cr.exact, cr.n, cr.max_ulp, sum_exact, sumsq_ref.size(),
              cx2.exact, cx2.n, cx2.max_ulp, exact_vs_one, x_one.size(), worst_ulp, graded,
              x_one.size(), guard);

  require(cr, 0, "prep_res_fold resid (vs single-stage reference)");
  CHECK_EQ(sum_exact, sumsq_ref.size());
  require(cx2, 0, "prep_norm_finish x_out (vs two-stage reference)");
  CHECK(worst_ulp <= 2);
  CHECK(graded > 0);
}

// `exact_silu`: force every gate column to sum to 30.0f, making silu(gate)
// provably 30.0f on host and device alike, so the whole chain is bit-exact.
void case_silu_mul(Dev& d, uint32_t M, bool exact_silu) {
  const uint32_t S = prep_ref::kSiluS, FN = prep_ref::kSiluFusedN, N = prep_ref::kSiluN;
  std::vector<float> partials = random_f32(size_t(S) * M * FN, exact_silu ? 401 : 400, 0.f, 1.f);
  if (exact_silu)
    for (size_t s = 0; s < S; ++s)
      for (size_t m = 0; m < M; ++m)
        for (size_t j = 0; j < FN; ++j)
          if ((j / 16) % 2 == 0)
            partials[(s * M + m) * FN + j] = 30.0f / float(S);  // all slices -> 30.0

  std::vector<uint16_t> x_ref(size_t(M) * N, 0);
  prep_ref::silu_mul(partials.data(), x_ref.data(), M);

  l0::Mem pbuf = upload(d.ctx, d.imm, partials);
  l0::Mem xbuf(d.ctx, l0::MemKind::Device, size_t(M) * N * 2);
  l0::Module mod(d.ctx, kernels::path(kernels::prep_silu_mul_variant(M, prep_ref::kSiluN)));
  l0::Kernel k = mod.kernel("prep_silu_mul");
  k.group_size(256);
  k.arg_ptr(0, pbuf.ptr());
  k.arg_ptr(1, xbuf.ptr());
  d.run(k, (N + 4095) / 4096, M);

  std::vector<uint16_t> x_got(size_t(M) * N);
  download(d.imm, x_got, xbuf);
  Cmp c = compare(x_got, x_ref);
  const uint32_t tol = exact_silu ? 0 : 2;
  std::printf("prep_silu_mul M=%u %s: %zu/%zu exact (max %u ulp, tol %u)\n", M,
              exact_silu ? "silu(30)=30 exact-case" : "random", c.exact, c.n, c.max_ulp, tol);
  require(c, tol, "prep_silu_mul x_out");
}

// `exact_silu`: z = 30.0f everywhere - see the note above; this is what makes
// the gated norm's own chain (o_b -> var tree -> n_b -> t_b) observable exactly.
void case_gated_head(Dev& d, uint32_t M, bool exact_silu) {
  const uint32_t S = prep_ref::kGatedS, QN = prep_ref::kQkvzN, H = prep_ref::kGatedHeads;
  const uint32_t D = prep_ref::kHeadDim, ON = prep_ref::kGatedOutN;
  std::vector<float> qkvz = random_f32(size_t(S) * M * QN, exact_silu ? 501 : 500, 0.f, 1.f);
  if (exact_silu)
    for (size_t i = 0; i < qkvz.size(); ++i) qkvz[i] = 30.0f;
  std::vector<float> gdn_o = random_f32(size_t(M) * H * D, 600, 0.f, 1.f);
  std::vector<uint16_t> gated_w = random_bf16(D, 700, -0.5f, 0.5f);   // plain w, bf16

  std::vector<uint16_t> x_ref(size_t(M) * ON, 0);
  prep_ref::gated_head(qkvz.data(), gdn_o.data(), gated_w.data(), x_ref.data(), M);

  l0::Mem qbuf = upload(d.ctx, d.imm, qkvz);
  l0::Mem obuf = upload(d.ctx, d.imm, gdn_o);
  l0::Mem wbuf = upload(d.ctx, d.imm, gated_w);
  l0::Mem xbuf(d.ctx, l0::MemKind::Device, size_t(M) * ON * 2);
  l0::Module mod(d.ctx, kernels::path(kernels::prep_gated_head_variant(M)));
  l0::Kernel k = mod.kernel("prep_gated_head");
  k.group_size(128);
  k.arg_ptr(0, qbuf.ptr());
  k.arg_ptr(1, obuf.ptr());
  k.arg_ptr(2, wbuf.ptr());
  k.arg_ptr(3, xbuf.ptr());
  d.run(k, H, M);

  std::vector<uint16_t> x_got(size_t(M) * ON);
  download(d.imm, x_got, xbuf);
  Cmp c = compare(x_got, x_ref);
  const uint32_t tol = exact_silu ? 0 : 2;
  std::printf("prep_gated_head M=%u %s: %zu/%zu exact (max %u ulp, tol %u)\n", M,
              exact_silu ? "silu(30)=30 exact-case" : "random", c.exact, c.n, c.max_ulp, tol);
  require(c, tol, "prep_gated_head x_out");
}

}  // namespace

int main() {
  Dev d;
  case_res_norm(d, 1, 5120, 0);
  case_res_norm(d, 1, 5120, 4);
  // The two-stage pair the runtime binds (G = 20), at both prep modes and at
  // both stage-B grids - the 20-work-group finish the engine uses and the
  // single-work-group one kept as its measurement control.
  case_res_norm_two_stage(d, 1, 5120, 0, 20, 20);
  case_res_norm_two_stage(d, 1, 5120, 4, 20, 20);
  case_res_norm_two_stage(d, 1, 5120, 4, 20, 1);
  case_silu_mul(d, 1, false);
  case_silu_mul(d, 1, true);
  case_gated_head(d, 1, false);
  case_gated_head(d, 1, true);
  std::puts("prep_test OK");
  return 0;
}
