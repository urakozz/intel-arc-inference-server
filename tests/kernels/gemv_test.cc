// gemv correctness vs the CPU reference, a few representative variants.
// probe_gemv (Task 9) runs the full matrix; this keeps ctest fast.
#include <cstdio>
#include <cstring>
#include "check.h"
#include "common/int4.h"
#include "gemv_harness.h"
#include "gemv_ref.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"

static void run_case(l0::Context& ctx, l0::Queue& q, l0::Fence& f, GemvCase c) {
  common::Int4Gptq w = common::Int4Gptq::random(c.K, c.N, 1234 + c.N);
  std::vector<uint16_t> x = random_bf16(size_t(c.M) * c.K, 99);
  std::vector<float> ref;
  gemv_ref(w, x, c.M, ref);
  GemvResult r = run_gemv(ctx, q, f, w, x, c, /*timed_launches=*/0);
  double err = max_abs_err(r.out, ref), tol = tol_for(ref);
  std::printf("gemv M=%u K=%u N=%u S=%u L=%u  max_abs_err=%.3g tol=%.3g\n", c.M, c.K, c.N, c.S, c.L, err, tol);
  CHECK(err <= tol);
}

// The tuned out/o_proj path combines layout 0, one 8-row block2D read, and the
// xor+arithmetic-shift dequant. Alternate two words so all 16 signed nibbles
// are exercised, then require device bit-identity with the old subtract/mask
// control at the same layout and S.
static void run_tuned_identity(l0::Context& ctx, l0::Queue& q, l0::Fence& f) {
  common::Int4Gptq w;
  w.K = 6144;
  w.N = 5120;
  w.qweight.resize(size_t(w.K / 8) * w.N);
  for (uint32_t row = 0; row < w.K / 8; ++row)
    for (uint32_t n = 0; n < w.N; ++n)
      w.qweight[size_t(row) * w.N + n] = row & 1u ? 0xFEDCBA98u : 0x76543210u;
  w.scales.assign(size_t(w.K / 64) * w.N, common::f32_to_f16(0.03125f));
  std::vector<uint16_t> x = random_bf16(w.K, 199);
  GemvResult tuned = run_gemv(ctx, q, f, w, x, {1, w.K, w.N, 4, 0}, 0);
  GemvResult control = run_gemv(
      ctx, q, f, w, x,
      {1, w.K, w.N, 4, 0, "gemv_control_M1_K6144_N5120_S4_L0"}, 0);
  CHECK_EQ(tuned.out.size(), control.out.size());
  CHECK(std::memcmp(tuned.out.data(), control.out.data(), tuned.out.size() * sizeof(float)) == 0);
}

int main() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence f(q);
  run_tuned_identity(ctx, q, f);
  run_case(ctx, q, f, {1, 6144, 5120, 4, 0});
  run_case(ctx, q, f, {1, 5120, 14336, 2, 0});
  run_case(ctx, q, f, {1, 5120, 16384, 1, 1});
  run_case(ctx, q, f, {1, 5120, 34816, 8, 0});
  run_case(ctx, q, f, {1, 17408, 5120, 4, 0});
  run_case(ctx, q, f, {2, 6144, 5120, 1, 0});
  // Spec 14 G1 (card): Agnes's folded gate'||up' and down' at their PROVISIONAL cells.
  run_case(ctx, q, f, {1, 5120, 38912, 8, 0});
  run_case(ctx, q, f, {1, 19456, 5120, 4, 0});
  std::puts("gemv_test OK");
  return 0;
}
