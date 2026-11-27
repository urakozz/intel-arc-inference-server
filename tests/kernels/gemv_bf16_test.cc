// gemv_bf16 correctness vs the CPU reference: the padded a||b projection
// (N = 128) and lm_head (N = 248320, 2.5 GB of weights) - plus, for the a||b
// shape, the retiled variant the runtime binds since spec 1.5's lever L2
// (COLS_PER_WG 16, KSPLIT 16), over the SAME host inputs as the `{64, 1}`
// build. The split gives each column's K to KSPLIT subgroups and merges them in
// a fixed SLM tree, so the two builds are NOT byte-equal and must not be
// asserted to be: the bars here are the reference tolerance for each, and the
// same tolerance between them. Element-exact tokens are the golden gate's job -
// and note the `ksplit == 1` branch below, which still demands the bytes if the
// shipped tiling ever goes back to an unsplit one.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>
#include "check.h"
#include "common/bf16.h"
#include "gemv_harness.h"
#include "gemv_ref.h"
#include "kernels/kernels.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"

// The kernel's SIMD width: one lane per output column (gemv_bf16.cl's SG).
static constexpr uint32_t kGemvSg = 16;

// The inputs of one shape, built once on the host: the same `w`/`x`/`ref` can
// then be handed to more than one compiled variant, which is what makes a
// comparison between two of them mean anything.
struct Inputs {
  std::vector<uint16_t> w_rm, x;
  std::vector<float> ref;
  common::Bf16Tiled w;
};

static Inputs make_inputs(uint32_t M, uint32_t K, uint32_t N) {
  std::vector<uint16_t> w_rm = random_bf16(size_t(N) * K, 7 + N, -0.05f, 0.05f);  // [N][K]
  std::vector<uint16_t> x = random_bf16(size_t(M) * K, 11);
  std::vector<float> ref;
  gemv_bf16_ref(w_rm.data(), K, N, x, M, ref);
  common::Bf16Tiled w = common::Bf16Tiled::from_rowmajor(w_rm.data(), K, N);
  return Inputs{std::move(w_rm), std::move(x), std::move(ref), std::move(w)};
}

// One (shape, tiling) pair against the host reference. Returns the device
// output so a caller can compare two variants of the same shape directly.
static std::vector<float> run_case(l0::Context& ctx, l0::Queue& q, l0::Fence& f,
                                   const Inputs& in, uint32_t M, kernels::GemvBf16Tiling t) {
  GemvResult r = run_gemv_bf16(ctx, q, f, in.w, in.x, M, 0, t);
  double err = max_abs_err(r.out, in.ref), tol = tol_for(in.ref);
  std::printf("gemv_bf16 M=%u K=%u N=%u cols/WG=%2u KSPLIT=%u (%u work-groups, %u subgroups)"
              "  max_abs_err=%.3g tol=%.3g\n",
              M, in.w.K, in.w.N, t.cols, t.ksplit, in.w.N / t.cols,
              in.w.N / kGemvSg * t.ksplit, err, tol);
  CHECK(err <= tol);
  return r.out;
}

int main() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence f(q);

  // a||b, both tilings, from ONE set of host inputs.
  {
    const kernels::GemvBf16Tiling shipped = kernels::gemv_bf16_tiling(128);
    const kernels::GemvBf16Tiling base{kernels::kGemvBf16Cols, 1};
    const Inputs in = make_inputs(1, 5120, 128);
    const std::vector<float> wide = run_case(ctx, q, f, in, 1, base);
    const std::vector<float> tiled = run_case(ctx, q, f, in, 1, shipped);
    CHECK_EQ(wide.size(), tiled.size());
    // The two builds against each other. `ksplit > 1` reorders the sum, so this
    // is a tolerance and not a memcmp - but at `ksplit == 1` the tilings differ
    // only in which subgroup owns a column, and then the bytes MUST match.
    const double pair_err = max_abs_err(tiled, wide), tol = tol_for(in.ref);
    std::printf("gemv_bf16 a||b: {%u,%u} vs {%u,%u}  max_abs_diff=%.3g tol=%.3g\n", base.cols,
                base.ksplit, shipped.cols, shipped.ksplit, pair_err, tol);
    CHECK(pair_err <= tol);
    if (shipped.ksplit == base.ksplit)
      CHECK(std::memcmp(wide.data(), tiled.data(), wide.size() * sizeof(float)) == 0);
    // The runtime's own choice for this shape must be the retiled one; for
    // lm_head's it must be the default. (kernel_table_test proves both exist.)
    CHECK_EQ(shipped.cols, kernels::kGemvBf16TinyCols);
    CHECK_EQ(kernels::gemv_bf16_tiling(248320).cols, kernels::kGemvBf16Cols);
    CHECK_EQ(kernels::gemv_bf16_tiling(248320).ksplit, 1u);
  }

  // The verify rows on the DPAS (BF16_DPAS: 8 rows an operand at every M): at a||b and the MTP head's fc and q||k||v,
  // every row of the M = 2..8 builds and of the wide / deep lists' ROWG build (item 40: 8 rows a group on grid
  // dimension 0, 16 rows as two groups) byte for byte the one-row build's.
  for (const auto& [K, N] : {std::pair<uint32_t, uint32_t>{5120, 128}, {10240, 5120}, {5120, 14336}}) {
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(N);
    const Inputs in = make_inputs(16, K, N);
    std::vector<float> one(size_t(16) * N);
    for (uint32_t r = 0; r < 16; ++r) {
      const std::vector<uint16_t> xr(in.x.begin() + size_t(r) * K, in.x.begin() + size_t(r + 1) * K);
      const std::vector<float> o = run_gemv_bf16(ctx, q, f, in.w, xr, 1, 0, t).out;
      std::copy(o.begin(), o.end(), one.begin() + size_t(r) * N);
    }
    CHECK(max_abs_err(one, in.ref) <= tol_for(in.ref));
    bool same = true;
    for (uint32_t M = 2; M <= 8; ++M) {
      const std::vector<uint16_t> xm(in.x.begin(), in.x.begin() + size_t(M) * K);
      const std::vector<float> o = run_gemv_bf16(ctx, q, f, in.w, xm, M, 0, t).out;
      same = same && std::memcmp(o.data(), one.data(), o.size() * sizeof(float)) == 0;
    }
    const std::vector<float> rg = run_gemv_bf16(ctx, q, f, in.w, in.x, 8, 0, t, 2).out;
    CHECK_EQ(one.size(), rg.size());
    same = same && std::memcmp(one.data(), rg.data(), one.size() * sizeof(float)) == 0;
    std::printf("gemv_bf16 K=%u N=%u: the 2..8-row builds and ROWG 2 x 8 rows vs the one-row build %s\n", K, N,
                same ? "bitwise" : "DIFFER");
    CHECK(same);
  }

  // lm_head: 2.5 GB of weights, one tiling - the grid fills the device on N.
  {
    const Inputs in = make_inputs(1, 5120, 248320);
    run_case(ctx, q, f, in, 1, kernels::gemv_bf16_tiling(248320));
  }
  std::puts("gemv_bf16_test OK");
  return 0;
}
