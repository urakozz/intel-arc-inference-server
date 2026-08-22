// gemv_bf16 correctness vs the CPU reference: the padded a||b projection
// (N = 128) and lm_head (N = 248320, 2.5 GB of weights).
#include <cstdio>
#include "check.h"
#include "common/bf16.h"
#include "gemv_harness.h"
#include "gemv_ref.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"

static void run_case(l0::Context& ctx, l0::Queue& q, l0::Fence& f, uint32_t M, uint32_t K, uint32_t N) {
  std::vector<uint16_t> w_rm = random_bf16(size_t(N) * K, 7 + N, -0.05f, 0.05f);   // [N][K]
  std::vector<uint16_t> x = random_bf16(size_t(M) * K, 11);
  std::vector<float> ref;
  gemv_bf16_ref(w_rm.data(), K, N, x, M, ref);
  common::Bf16Tiled w = common::Bf16Tiled::from_rowmajor(w_rm.data(), K, N);
  GemvResult r = run_gemv_bf16(ctx, q, f, w, x, M, 0);
  double err = max_abs_err(r.out, ref), tol = tol_for(ref);
  std::printf("gemv_bf16 M=%u K=%u N=%u  max_abs_err=%.3g tol=%.3g\n", M, K, N, err, tol);
  CHECK(err <= tol);
}

int main() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence f(q);
  run_case(ctx, q, f, 1, 5120, 128);
  run_case(ctx, q, f, 1, 5120, 248320);
  std::puts("gemv_bf16_test OK");
  return 0;
}
