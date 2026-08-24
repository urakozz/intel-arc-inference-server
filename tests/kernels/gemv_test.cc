// gemv correctness vs the CPU reference, a few representative variants.
// probe_gemv (Task 9) runs the full matrix; this keeps ctest fast.
#include <cstdio>
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

int main() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence f(q);
  run_case(ctx, q, f, {1, 6144, 5120, 1, 0});
  run_case(ctx, q, f, {1, 6144, 5120, 4, 1});
  run_case(ctx, q, f, {1, 17408, 5120, 8, 0});
  run_case(ctx, q, f, {1, 5120, 34816, 2, 1});
  run_case(ctx, q, f, {2, 6144, 5120, 1, 0});
  std::puts("gemv_test OK");
  return 0;
}
